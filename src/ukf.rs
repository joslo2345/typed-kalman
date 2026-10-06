//! Unscented Kalman filter.

use nalgebra::{Cholesky, ComplexField, SMatrix, SVector};

use crate::error::KalmanError;
use crate::model::{MeasurementModel, ProcessModel};
use crate::update::{all_finite, symmetrize};

/// Parameters of the scaled unscented transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UkfParams {
    /// Spread of the sigma points around the mean. Must be positive.
    pub alpha: f64,
    /// Prior knowledge of the distribution. `2.0` is optimal for Gaussians.
    pub beta: f64,
    /// Secondary scaling parameter. `alpha² (N + kappa)` must be positive.
    pub kappa: f64,
}

impl Default for UkfParams {
    /// Returns `alpha = 1`, `beta = 2`, `kappa = 0`.
    ///
    /// These keep every sigma-point weight non-negative, which avoids the loss of
    /// positive-definiteness that small `alpha` values can cause in floating point.
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 2.0,
            kappa: 0.0,
        }
    }
}

/// Sigma-point weights derived from [`UkfParams`] for an `N`-dimensional state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Weights {
    /// Mean weight of the center point.
    pub mean0: f64,
    /// Covariance weight of the center point.
    pub cov0: f64,
    /// Mean and covariance weight of every other point.
    pub rest: f64,
    /// Distance of the outer points from the mean, in standard deviations.
    pub gamma: f64,
}

impl Weights {
    /// Validates `params` and derives the weights.
    pub(crate) fn new<const N: usize>(params: &UkfParams) -> Result<Self, KalmanError> {
        let UkfParams { alpha, beta, kappa } = *params;
        let n_plus_lambda = alpha * alpha * (N as f64 + kappa);
        let valid = alpha.is_finite() && beta.is_finite() && n_plus_lambda.is_finite();
        if !valid || alpha <= 0.0 || n_plus_lambda <= 0.0 {
            return Err(KalmanError::InvalidInput);
        }
        Ok(Self::unchecked::<N>(params))
    }

    /// Derives the weights without validating `params`.
    pub(crate) fn unchecked<const N: usize>(params: &UkfParams) -> Self {
        let UkfParams { alpha, beta, kappa } = *params;
        let n_plus_lambda = alpha * alpha * (N as f64 + kappa);
        let mean0 = (n_plus_lambda - N as f64) / n_plus_lambda;
        Self {
            mean0,
            cov0: mean0 + 1.0 - alpha * alpha + beta,
            rest: 0.5 / n_plus_lambda,
            // Through nalgebra so it uses libm when `std` is off.
            gamma: ComplexField::sqrt(n_plus_lambda),
        }
    }
}

/// The `2N + 1` sigma points of an `N`-dimensional distribution, each mapped into `D` dimensions.
///
/// Point `i` of `plus` and `minus` lies at `±gamma` times column `i` of the covariance's
/// Cholesky factor. Storing them as two `D × N` matrices avoids needing `2N + 1` as a const
/// generic, which stable Rust doesn't support.
pub(crate) struct SigmaPoints<const D: usize, const N: usize> {
    pub center: SVector<f64, D>,
    pub plus: SMatrix<f64, D, N>,
    pub minus: SMatrix<f64, D, N>,
}

impl<const N: usize> SigmaPoints<N, N> {
    /// Draws sigma points around mean `x` with covariance `p`.
    fn draw(x: &SVector<f64, N>, p: &SMatrix<f64, N, N>, w: &Weights) -> Result<Self, KalmanError> {
        let l = Cholesky::new(*p)
            .ok_or(KalmanError::CovarianceNotPositiveDefinite)?
            .unpack();
        Ok(Self::from_factor(x, &l, w))
    }

    /// Draws sigma points around mean `x` from a square root `l` of the covariance
    /// (any matrix with `P = L Lᵀ`).
    pub(crate) fn from_factor(x: &SVector<f64, N>, l: &SMatrix<f64, N, N>, w: &Weights) -> Self {
        let l = l * w.gamma;
        let mut plus = SMatrix::<f64, N, N>::zeros();
        let mut minus = SMatrix::<f64, N, N>::zeros();
        for i in 0..N {
            plus.set_column(i, &(x + l.column(i)));
            minus.set_column(i, &(x - l.column(i)));
        }
        Self {
            center: *x,
            plus,
            minus,
        }
    }
}

impl<const D: usize, const N: usize> SigmaPoints<D, N> {
    /// Passes every point through `g`.
    pub(crate) fn map<const E: usize>(
        &self,
        g: impl Fn(&SVector<f64, D>) -> SVector<f64, E>,
    ) -> SigmaPoints<E, N> {
        let mut plus = SMatrix::<f64, E, N>::zeros();
        let mut minus = SMatrix::<f64, E, N>::zeros();
        for i in 0..N {
            plus.set_column(i, &g(&self.plus.column(i).into_owned()));
            minus.set_column(i, &g(&self.minus.column(i).into_owned()));
        }
        SigmaPoints {
            center: g(&self.center),
            plus,
            minus,
        }
    }

    /// Returns the weighted mean of the points.
    pub(crate) fn mean(&self, w: &Weights) -> SVector<f64, D> {
        let mut sum = SVector::<f64, D>::zeros();
        for i in 0..N {
            sum += self.plus.column(i) + self.minus.column(i);
        }
        self.center * w.mean0 + sum * w.rest
    }

    /// Returns the weighted cross-covariance between these points (around `mean`) and
    /// `other` (around `other_mean`).
    pub(crate) fn cross_covariance<const E: usize>(
        &self,
        mean: &SVector<f64, D>,
        other: &SigmaPoints<E, N>,
        other_mean: &SVector<f64, E>,
        w: &Weights,
    ) -> SMatrix<f64, D, E> {
        let outer = |a: SVector<f64, D>, b: SVector<f64, E>| a * b.transpose();
        let mut sum = SMatrix::<f64, D, E>::zeros();
        for i in 0..N {
            sum += outer(
                self.plus.column(i) - mean,
                other.plus.column(i) - other_mean,
            );
            sum += outer(
                self.minus.column(i) - mean,
                other.minus.column(i) - other_mean,
            );
        }
        outer(self.center - mean, other.center - other_mean) * w.cov0 + sum * w.rest
    }
}

/// An unscented Kalman filter over an `N`-dimensional state.
///
/// Unlike the [`Ekf`](crate::ekf::Ekf), it needs no Jacobians: it propagates `2N + 1` sigma
/// points through the models directly. Every method that draws sigma points returns
/// [`KalmanError::CovarianceNotPositiveDefinite`] if the covariance can't be factored, and the
/// filter is left unchanged on any error.
#[derive(Debug, Clone, PartialEq)]
pub struct Ukf<const N: usize> {
    x: SVector<f64, N>,
    p: SMatrix<f64, N, N>,
    weights: Weights,
}

impl<const N: usize> Ukf<N> {
    /// Creates a filter with initial state `x`, covariance `p` and the default
    /// [`UkfParams`].
    pub fn new(x: SVector<f64, N>, p: SMatrix<f64, N, N>) -> Self {
        Self {
            x,
            p,
            weights: Weights::unchecked::<N>(&UkfParams::default()),
        }
    }

    /// Creates a filter with custom unscented-transform parameters.
    ///
    /// Returns [`KalmanError::InvalidInput`] if `alpha` isn't positive or
    /// `alpha² (N + kappa)` isn't positive.
    pub fn with_params(
        x: SVector<f64, N>,
        p: SMatrix<f64, N, N>,
        params: UkfParams,
    ) -> Result<Self, KalmanError> {
        Ok(Self {
            x,
            p,
            weights: Weights::new::<N>(&params)?,
        })
    }

    /// Returns the current state estimate.
    pub fn state(&self) -> &SVector<f64, N> {
        &self.x
    }

    /// Returns the current state covariance.
    pub fn covariance(&self) -> &SMatrix<f64, N, N> {
        &self.p
    }

    /// Propagates the state and covariance through `model` with process noise `q`.
    pub fn predict<P: ProcessModel<N>>(
        &mut self,
        model: &P,
        q: &SMatrix<f64, N, N>,
        dt: f64,
    ) -> Result<(), KalmanError> {
        let w = &self.weights;
        let points = SigmaPoints::draw(&self.x, &self.p, w)?.map(|s| model.predict(s, dt));
        let x = points.mean(w);
        let p = symmetrize(points.cross_covariance(&x, &points, &x, w) + q);

        if !all_finite(x.as_slice()) || !all_finite(p.as_slice()) {
            return Err(KalmanError::NumericalFailure);
        }
        self.x = x;
        self.p = p;
        Ok(())
    }

    /// Corrects the estimate with measurement `z` and measurement noise `r`.
    ///
    /// Returns the normalized innovation squared (NIS).
    pub fn update<H: MeasurementModel<N, M>, const M: usize>(
        &mut self,
        model: &H,
        z: &SVector<f64, M>,
        r: &SMatrix<f64, M, M>,
    ) -> Result<f64, KalmanError> {
        if !all_finite(z.as_slice()) || !all_finite(r.as_slice()) {
            return Err(KalmanError::InvalidInput);
        }

        let w = &self.weights;
        let points = SigmaPoints::draw(&self.x, &self.p, w)?;
        let measured = points.map(|s| model.measure(s));
        let z_pred = measured.mean(w);
        let s = measured.cross_covariance(&z_pred, &measured, &z_pred, w) + r;
        let s_chol = Cholesky::new(s).ok_or(KalmanError::SingularInnovation)?;
        let p_xz = points.cross_covariance(&self.x, &measured, &z_pred, w);

        // K = Pxz S⁻¹, computed as (S⁻¹ Pxzᵀ)ᵀ since S is symmetric.
        let k = s_chol.solve(&p_xz.transpose()).transpose();
        let y = z - z_pred;
        let x = self.x + k * y;
        let p = symmetrize(self.p - k * s * k.transpose());
        let nis = y.dot(&s_chol.solve(&y));

        if !all_finite(x.as_slice()) || !all_finite(p.as_slice()) || !nis.is_finite() {
            return Err(KalmanError::NumericalFailure);
        }
        // P - K S Kᵀ can lose positive-definiteness in floating point. Reject it here so the
        // next predict can always draw sigma points.
        if Cholesky::new(p).is_none() {
            return Err(KalmanError::CovarianceNotPositiveDefinite);
        }
        self.x = x;
        self.p = p;
        Ok(nis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::LinearKf;
    use nalgebra::{Matrix1, Matrix2, Matrix2x4, Matrix4, Vector1, Vector2, Vector4};

    struct Square;

    impl ProcessModel<1> for Square {
        fn predict(&self, x: &Vector1<f64>, _dt: f64) -> Vector1<f64> {
            Vector1::new(x[0] * x[0])
        }
    }

    #[test]
    fn predict_is_exact_for_squaring_a_gaussian() {
        // For x ~ N(mu, sigma²): E[x²] = mu² + sigma², Var[x²] = 4 mu² sigma² + 2 sigma⁴.
        let (mu, sigma) = (3.0, 0.5);
        let mut ukf = Ukf::new(Vector1::new(mu), Matrix1::new(sigma * sigma));
        ukf.predict(&Square, &Matrix1::zeros(), 0.0).unwrap();

        let s2 = sigma * sigma;
        approx::assert_relative_eq!(ukf.state()[0], mu * mu + s2, epsilon = 1e-12);
        approx::assert_relative_eq!(
            ukf.covariance()[0],
            4.0 * mu * mu * s2 + 2.0 * s2 * s2,
            epsilon = 1e-12
        );
    }

    struct Linear {
        f: Matrix4<f64>,
        h: Matrix2x4<f64>,
    }

    impl ProcessModel<4> for Linear {
        fn predict(&self, x: &Vector4<f64>, _dt: f64) -> Vector4<f64> {
            self.f * x
        }
    }

    impl MeasurementModel<4, 2> for Linear {
        fn measure(&self, x: &Vector4<f64>) -> Vector2<f64> {
            self.h * x
        }
    }

    fn assert_matches_linear_kf(params: UkfParams) {
        let dt = 0.1;
        #[rustfmt::skip]
        let f = Matrix4::new(
            1.0, 0.0, dt,  0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let h = Matrix2x4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        );
        let q = Matrix4::identity() * 1e-3;
        let r = Matrix2::identity() * 0.25;
        let model = Linear { f, h };

        let mut kf = LinearKf::<4>::default();
        let mut ukf = Ukf::with_params(Vector4::zeros(), Matrix4::identity(), params).unwrap();
        for k in 0..1000 {
            let t = k as f64 * dt;
            let z = Vector2::new(t.sin() * 10.0, t.cos() * 5.0);

            kf.predict(&f, &q);
            ukf.predict(&model, &q, dt).unwrap();
            let nis_kf = kf.update(&h, &z, &r).unwrap();
            let nis_ukf = ukf.update(&model, &z, &r).unwrap();

            approx::assert_relative_eq!(nis_kf, nis_ukf, epsilon = 1e-9);
            approx::assert_relative_eq!(kf.state(), ukf.state(), epsilon = 1e-9);
            approx::assert_relative_eq!(kf.covariance(), ukf.covariance(), epsilon = 1e-9);
        }
    }

    #[test]
    fn matches_linear_kf_on_linear_models() {
        // The unscented transform is exact for linear models, whatever the parameters.
        assert_matches_linear_kf(UkfParams::default());
        assert_matches_linear_kf(UkfParams {
            alpha: 0.5,
            beta: 2.0,
            kappa: 1.0,
        });
    }

    #[test]
    fn invalid_params_are_rejected() {
        let params = UkfParams {
            alpha: 1.0,
            beta: 2.0,
            kappa: -1.0,
        };
        let result = Ukf::<1>::with_params(Vector1::zeros(), Matrix1::identity(), params);
        assert_eq!(result, Err(KalmanError::InvalidInput));
    }

    struct Position;

    impl MeasurementModel<2, 1> for Position {
        fn measure(&self, x: &Vector2<f64>) -> Vector1<f64> {
            Vector1::new(x[0])
        }
    }

    #[test]
    fn nan_measurement_returns_error_and_keeps_state() {
        let mut ukf = Ukf::new(Vector2::zeros(), Matrix2::identity());
        let before = ukf.clone();

        let result = ukf.update(&Position, &Vector1::new(f64::NAN), &Matrix1::new(1.0));
        assert_eq!(result, Err(KalmanError::InvalidInput));
        assert_eq!(ukf, before);
    }

    #[test]
    fn non_positive_definite_covariance_returns_error_and_keeps_state() {
        let mut ukf = Ukf::new(Vector2::zeros(), Matrix2::new(1.0, 2.0, 2.0, 1.0));
        let before = ukf.clone();

        let result = ukf.update(&Position, &Vector1::new(1.0), &Matrix1::new(1.0));
        assert_eq!(result, Err(KalmanError::CovarianceNotPositiveDefinite));
        assert_eq!(ukf, before);
    }
}
