//! Square-root unscented Kalman filter.

use nalgebra::{Cholesky, ComplexField, SMatrix, SVector};

use crate::cholupdate::{downdate, is_singular, update};
use crate::error::KalmanError;
use crate::model::{MeasurementModel, ProcessModel};
use crate::scalar::Float;
use crate::ukf::{SigmaPoints, UkfParams, Weights};
use crate::update::all_finite;

/// A square-root unscented Kalman filter over an `N`-dimensional state.
///
/// It propagates the lower-triangular Cholesky factor `S` of the covariance (`P = S Sᵀ`)
/// instead of `P` itself. That keeps the covariance symmetric positive-semidefinite by
/// construction and halves the dynamic range the arithmetic has to handle, which matters most
/// in `f32`.
///
/// Noise is supplied as square roots too: any matrix `G` with `Q = G Gᵀ` (or `R = G Gᵀ`), such
/// as the Cholesky factor. `G` may have any number of columns, so singular noise is fine.
/// The filter is left unchanged on any error.
#[derive(Debug, Clone, PartialEq)]
pub struct SqrtUkf<const N: usize, T: Float = f64> {
    x: SVector<T, N>,
    s: SMatrix<T, N, N>,
    weights: Weights<T>,
}

impl<const N: usize, T: Float> SqrtUkf<N, T> {
    /// Creates a filter with initial state `x`, covariance `p` and the default [`UkfParams`].
    ///
    /// Returns [`KalmanError::CovarianceNotPositiveDefinite`] if `p` can't be factored.
    pub fn new(x: SVector<T, N>, p: SMatrix<T, N, N>) -> Result<Self, KalmanError> {
        Self::with_weights(x, p, Weights::unchecked::<N>(&UkfParams::default()))
    }

    /// Creates a filter with custom unscented-transform parameters.
    ///
    /// Returns [`KalmanError::InvalidInput`] for invalid parameters (see
    /// [`Ukf::with_params`](crate::ukf::Ukf::with_params)), or
    /// [`KalmanError::CovarianceNotPositiveDefinite`] if `p` can't be factored.
    pub fn with_params(
        x: SVector<T, N>,
        p: SMatrix<T, N, N>,
        params: UkfParams,
    ) -> Result<Self, KalmanError> {
        Self::with_weights(x, p, Weights::new::<N>(&params)?)
    }

    fn with_weights(
        x: SVector<T, N>,
        p: SMatrix<T, N, N>,
        weights: Weights<T>,
    ) -> Result<Self, KalmanError> {
        let s = Cholesky::new(p)
            .ok_or(KalmanError::CovarianceNotPositiveDefinite)?
            .unpack();
        Ok(Self { x, s, weights })
    }

    /// Returns the current state estimate.
    pub fn state(&self) -> &SVector<T, N> {
        &self.x
    }

    /// Returns the lower-triangular square root `S` of the state covariance.
    pub fn sqrt_covariance(&self) -> &SMatrix<T, N, N> {
        &self.s
    }

    /// Returns the state covariance `S Sᵀ`.
    pub fn covariance(&self) -> SMatrix<T, N, N> {
        self.s * self.s.transpose()
    }

    /// Propagates the state and its covariance factor through `model`, with process noise
    /// `Q = q_sqrt q_sqrtᵀ`.
    pub fn predict<P: ProcessModel<N, T>, const K: usize>(
        &mut self,
        model: &P,
        q_sqrt: &SMatrix<T, N, K>,
        dt: T,
    ) -> Result<(), KalmanError> {
        let w = &self.weights;
        let points = SigmaPoints::from_factor(&self.x, &self.s, w).map(|s| model.predict(s, dt));
        let x = points.mean(w);
        let s = weighted_factor(&points, &x, q_sqrt, w)
            .ok_or(KalmanError::CovarianceNotPositiveDefinite)?;

        if !all_finite(x.as_slice()) || !all_finite(s.as_slice()) {
            return Err(KalmanError::NumericalFailure);
        }
        if is_singular(&s) {
            return Err(KalmanError::CovarianceNotPositiveDefinite);
        }
        self.x = x;
        self.s = s;
        Ok(())
    }

    /// Corrects the estimate with measurement `z` and measurement noise
    /// `R = r_sqrt r_sqrtᵀ`, using a Joseph-form update of the covariance factor.
    ///
    /// Returns the normalized innovation squared (NIS).
    pub fn update<H: MeasurementModel<N, M, T>, const M: usize, const K: usize>(
        &mut self,
        model: &H,
        z: &SVector<T, M>,
        r_sqrt: &SMatrix<T, M, K>,
    ) -> Result<T, KalmanError> {
        if !all_finite(z.as_slice()) || !all_finite(r_sqrt.as_slice()) {
            return Err(KalmanError::InvalidInput);
        }

        let w = &self.weights;
        let points = SigmaPoints::from_factor(&self.x, &self.s, w);
        // Measurement deviations go through the model's residual, so wrapped quantities work.
        let (z_pred, dz) = points
            .map(|s| model.measure(s))
            .residual_moments(w, |a, b| model.residual(a, b));
        let zero_z = SVector::<T, M>::zeros();
        let s_z =
            weighted_factor(&dz, &zero_z, r_sqrt, w).ok_or(KalmanError::SingularInnovation)?;
        let p_xz = points.cross_covariance(&self.x, &dz, &zero_z, w);

        // K = Pxz (Sz Szᵀ)⁻¹, computed as (Sz⁻ᵀ Sz⁻¹ Pxzᵀ)ᵀ with two triangular solves.
        let k = s_z
            .solve_lower_triangular(&p_xz.transpose())
            .and_then(|t| s_z.tr_solve_lower_triangular(&t))
            .ok_or(KalmanError::SingularInnovation)?
            .transpose();
        let y = model.residual(z, &z_pred);
        let whitened = s_z
            .solve_lower_triangular(&y)
            .ok_or(KalmanError::SingularInnovation)?;
        let nis = whitened.norm_squared();
        let x = self.x + k * y;

        // Joseph form over the sigma points (see `SigmaPoints::corrected_deviations`), built with rank-one
        // updates only. The textbook P - (K Sz)(K Sz)ᵀ needs downdates, which cancel
        // catastrophically when a precise measurement shrinks the covariance by more than the
        // precision can resolve.
        let corrected = points.corrected_deviations(&self.x, &dz, &zero_z, &k);
        let s = weighted_factor(&corrected, &SVector::zeros(), &(k * r_sqrt), w)
            .ok_or(KalmanError::CovarianceNotPositiveDefinite)?;

        if !all_finite(x.as_slice()) || !all_finite(s.as_slice()) || !nis.is_finite() {
            return Err(KalmanError::NumericalFailure);
        }
        if is_singular(&s) {
            return Err(KalmanError::CovarianceNotPositiveDefinite);
        }
        self.x = x;
        self.s = s;
        Ok(nis)
    }
}

/// Returns a lower-triangular `L` with `L Lᵀ` equal to the weighted covariance of `points`
/// around `mean`, plus `noise_sqrt noise_sqrtᵀ`.
///
/// Returns `None` if a negative center weight would make the result indefinite.
fn weighted_factor<const D: usize, const N: usize, const K: usize, T: Float>(
    points: &SigmaPoints<D, N, T>,
    mean: &SVector<T, D>,
    noise_sqrt: &SMatrix<T, D, K>,
    w: &Weights<T>,
) -> Option<SMatrix<T, D, D>> {
    let mut l = SMatrix::<T, D, D>::zeros();
    let scale = ComplexField::sqrt(w.rest);
    for i in 0..N {
        update(&mut l, (points.plus.column(i) - mean) * scale);
        update(&mut l, (points.minus.column(i) - mean) * scale);
    }
    for j in 0..K {
        update(&mut l, noise_sqrt.column(j).into_owned());
    }

    let center = points.center - mean;
    if w.cov0 >= T::zero() {
        update(&mut l, center * ComplexField::sqrt(w.cov0));
    } else {
        downdate(&mut l, center * ComplexField::sqrt(-w.cov0))?;
    }
    Some(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ukf::Ukf;
    use nalgebra::{Matrix1, Matrix2, Vector1, Vector2};

    /// A pendulum: angle and angular velocity.
    struct Pendulum;

    impl ProcessModel<2> for Pendulum {
        fn predict(&self, x: &Vector2<f64>, dt: f64) -> Vector2<f64> {
            Vector2::new(x[0] + x[1] * dt, x[1] - 9.81 * x[0].sin() * dt)
        }
    }

    /// Horizontal position of the bob on a unit-length arm.
    struct BobPosition;

    impl MeasurementModel<2, 1> for BobPosition {
        fn measure(&self, x: &Vector2<f64>) -> Vector1<f64> {
            Vector1::new(x[0].sin())
        }
    }

    fn assert_matches_ukf(params: UkfParams) {
        let dt = 0.01;
        let q = Matrix2::new(1e-6, 0.0, 0.0, 1e-4);
        let r = Matrix1::new(0.01);
        let q_sqrt = Cholesky::new(q).unwrap().unpack();
        let r_sqrt = Cholesky::new(r).unwrap().unpack();
        let (x0, p0) = (Vector2::new(0.5, 0.0), Matrix2::identity() * 0.1);

        let mut ukf = Ukf::with_params(x0, p0, params).unwrap();
        let mut sr = SqrtUkf::with_params(x0, p0, params).unwrap();
        for k in 0..1000 {
            let t = k as f64 * dt;
            let z = Vector1::new((0.5 * (3.13 * t).cos()).sin());

            ukf.predict(&Pendulum, &q, dt).unwrap();
            sr.predict(&Pendulum, &q_sqrt, dt).unwrap();
            let nis_ukf = ukf.update(&BobPosition, &z, &r).unwrap();
            let nis_sr = sr.update(&BobPosition, &z, &r_sqrt).unwrap();

            approx::assert_relative_eq!(nis_ukf, nis_sr, epsilon = 1e-9);
            approx::assert_relative_eq!(ukf.state(), sr.state(), epsilon = 1e-9);
            approx::assert_relative_eq!(*ukf.covariance(), sr.covariance(), epsilon = 1e-9);
        }
    }

    #[test]
    fn matches_ukf_on_a_nonlinear_model() {
        assert_matches_ukf(UkfParams::default());
        assert_matches_ukf(UkfParams {
            alpha: 0.5,
            beta: 2.0,
            kappa: 1.0,
        });
    }

    #[test]
    fn negative_center_weight_matches_ukf() {
        // For N = 2 these give a covariance center weight of -2.25, which exercises the
        // downdate path in `weighted_factor`.
        assert_matches_ukf(UkfParams {
            alpha: 0.5,
            beta: 0.0,
            kappa: 0.0,
        });
    }

    #[test]
    fn accepts_a_rectangular_noise_square_root() {
        // Q = g gᵀ with a single column: noise on velocity only.
        let g = nalgebra::Matrix2x1::new(0.0, 0.1);
        let mut sr = SqrtUkf::new(Vector2::new(0.5, 0.0), Matrix2::identity() * 0.1).unwrap();
        sr.predict(&Pendulum, &g, 0.01).unwrap();
        assert!(Cholesky::new(sr.covariance()).is_some());
    }

    #[test]
    fn non_positive_definite_initial_covariance_is_rejected() {
        let result = SqrtUkf::new(Vector2::zeros(), Matrix2::new(1.0, 2.0, 2.0, 1.0));
        assert_eq!(result, Err(KalmanError::CovarianceNotPositiveDefinite));
    }

    #[test]
    fn nan_measurement_returns_error_and_keeps_state() {
        let mut sr = SqrtUkf::new(Vector2::zeros(), Matrix2::identity()).unwrap();
        let before = sr.clone();

        let result = sr.update(&BobPosition, &Vector1::new(f64::NAN), &Matrix1::new(0.1));
        assert_eq!(result, Err(KalmanError::InvalidInput));
        assert_eq!(sr, before);
    }
}
