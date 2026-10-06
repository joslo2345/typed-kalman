//! Square-root linear Kalman filter.

use nalgebra::{Cholesky, SMatrix, SVector};

use crate::cholupdate::{is_singular, update_columns};
use crate::error::KalmanError;
use crate::scalar::Float;
use crate::update::all_finite;

/// A linear Kalman filter that stores the lower-triangular Cholesky factor `S` of its
/// covariance (`P = S Sᵀ`) instead of `P`.
///
/// `S` has the square root of `P`'s condition number, so this filter keeps working on problems
/// where `P` itself is too ill-conditioned for the number format and every covariance-form
/// filter, [`LinearKf`](crate::linear::LinearKf) included, loses positive-definiteness. Both
/// steps build the new factor from positive rank-one updates only (predict from `[F S, Q½]`,
/// update from the Joseph form `[(I - K H) S, K R½]`), so nothing can cancel to an indefinite
/// result. It costs more per step than `LinearKf`.
///
/// Noise is supplied as square roots: any matrix `G` with `Q = G Gᵀ` (or `R = G Gᵀ`), such as
/// the Cholesky factor. `G` may have any number of columns, so singular noise is fine. The
/// filter is left unchanged on any error.
#[derive(Debug, Clone, PartialEq)]
pub struct SqrtKf<const N: usize, T: Float = f64> {
    x: SVector<T, N>,
    s: SMatrix<T, N, N>,
}

impl<const N: usize, T: Float> SqrtKf<N, T> {
    /// Creates a filter with initial state `x` and covariance `p`.
    ///
    /// Returns [`KalmanError::CovarianceNotPositiveDefinite`] if `p` can't be factored.
    pub fn new(x: SVector<T, N>, p: SMatrix<T, N, N>) -> Result<Self, KalmanError> {
        let s = Cholesky::new(p)
            .ok_or(KalmanError::CovarianceNotPositiveDefinite)?
            .unpack();
        Ok(Self { x, s })
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

    /// Propagates the state with transition matrix `f` and process noise
    /// `Q = q_sqrt q_sqrtᵀ`.
    pub fn predict<const K: usize>(
        &mut self,
        f: &SMatrix<T, N, N>,
        q_sqrt: &SMatrix<T, N, K>,
    ) -> Result<(), KalmanError> {
        let x = f * self.x;
        // P⁻ = (F S)(F S)ᵀ + Q½ Q½ᵀ.
        let mut s = SMatrix::<T, N, N>::zeros();
        update_columns(&mut s, &(f * self.s));
        update_columns(&mut s, q_sqrt);

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

    /// Corrects the estimate with measurement `z`, observation matrix `h` and measurement
    /// noise `R = r_sqrt r_sqrtᵀ`, using the Joseph form of the factor update.
    ///
    /// Returns the normalized innovation squared (NIS).
    pub fn update<const M: usize, const K: usize>(
        &mut self,
        h: &SMatrix<T, M, N>,
        z: &SVector<T, M>,
        r_sqrt: &SMatrix<T, M, K>,
    ) -> Result<T, KalmanError> {
        if !all_finite(z.as_slice()) || !all_finite(r_sqrt.as_slice()) {
            return Err(KalmanError::InvalidInput);
        }

        // Innovation covariance factor: S_z S_zᵀ = (H S)(H S)ᵀ + R½ R½ᵀ.
        let hs = h * self.s;
        let mut s_z = SMatrix::<T, M, M>::zeros();
        update_columns(&mut s_z, &hs);
        update_columns(&mut s_z, r_sqrt);
        if is_singular(&s_z) {
            return Err(KalmanError::SingularInnovation);
        }

        // K = S (H S)ᵀ (S_z S_zᵀ)⁻¹, so Kᵀ = S_z⁻ᵀ S_z⁻¹ (H S) Sᵀ: two triangular solves.
        let k = s_z
            .solve_lower_triangular(&(hs * self.s.transpose()))
            .and_then(|t| s_z.tr_solve_lower_triangular(&t))
            .ok_or(KalmanError::SingularInnovation)?
            .transpose();
        let y = z - h * self.x;
        let nis = s_z
            .solve_lower_triangular(&y)
            .ok_or(KalmanError::SingularInnovation)?
            .norm_squared();
        let x = self.x + k * y;

        // Joseph form: P⁺ = (I - K H) P (I - K H)ᵀ + K R Kᵀ, as factor columns.
        let mut s = SMatrix::<T, N, N>::zeros();
        update_columns(&mut s, &(self.s - k * hs));
        update_columns(&mut s, &(k * r_sqrt));

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::LinearKf;
    use nalgebra::{Matrix1, Matrix1x2, Matrix2, Matrix2x4, Matrix4, Vector1, Vector2, Vector4};

    #[test]
    fn scalar_update_matches_hand_calculation() {
        let mut kf = SqrtKf::new(Vector2::new(0.0, 0.0), Matrix2::identity()).unwrap();
        let nis = kf
            .update(
                &Matrix1x2::new(1.0, 0.0),
                &Vector1::new(2.0),
                &Matrix1::new(1.0),
            )
            .unwrap();

        // S = 2, K = [0.5, 0], y = 2.
        approx::assert_relative_eq!(nis, 2.0, epsilon = 1e-12);
        approx::assert_relative_eq!(*kf.state(), Vector2::new(1.0, 0.0), epsilon = 1e-12);
        approx::assert_relative_eq!(
            kf.covariance(),
            Matrix2::new(0.5, 0.0, 0.0, 1.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn matches_linear_kf() {
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
        let q = Matrix4::from_diagonal(&Vector4::new(1e-4, 1e-4, 1e-2, 1e-2));
        let r = Matrix2::new(0.25, 0.05, 0.05, 0.3);
        let q_sqrt = Cholesky::new(q).unwrap().unpack();
        let r_sqrt = Cholesky::new(r).unwrap().unpack();

        let mut kf = LinearKf::<4>::default();
        let mut sr = SqrtKf::new(Vector4::zeros(), Matrix4::identity()).unwrap();
        for k in 0..1000 {
            let t = k as f64 * dt;
            let z = Vector2::new(t.sin() * 10.0, t.cos() * 5.0);

            kf.predict(&f, &q);
            sr.predict(&f, &q_sqrt).unwrap();
            let nis_kf = kf.update(&h, &z, &r).unwrap();
            let nis_sr = sr.update(&h, &z, &r_sqrt).unwrap();

            approx::assert_relative_eq!(nis_kf, nis_sr, epsilon = 1e-9);
            approx::assert_relative_eq!(kf.state(), sr.state(), epsilon = 1e-9);
            approx::assert_relative_eq!(*kf.covariance(), sr.covariance(), epsilon = 1e-9);
        }
    }

    #[test]
    fn accepts_rectangular_and_singular_noise() {
        // Process noise on velocity only: Q = g gᵀ is singular.
        let g = nalgebra::Matrix2x1::new(0.0, 0.1);
        let mut kf = SqrtKf::new(Vector2::zeros(), Matrix2::identity()).unwrap();
        kf.predict(&Matrix2::new(1.0, 0.1, 0.0, 1.0), &g).unwrap();
        approx::assert_relative_eq!(
            kf.covariance(),
            Matrix2::new(1.01, 0.1, 0.1, 1.01),
            epsilon = 1e-12
        );
    }

    #[test]
    fn invalid_inputs_return_errors_and_keep_state() {
        let mut kf = SqrtKf::new(Vector2::zeros(), Matrix2::identity()).unwrap();
        let before = kf.clone();
        let h = Matrix1x2::new(1.0, 0.0);

        let nan = kf.update(&h, &Vector1::new(f64::NAN), &Matrix1::new(1.0));
        assert_eq!(nan, Err(KalmanError::InvalidInput));
        assert_eq!(kf, before);

        // A measurement of a state with no uncertainty and no noise: S_z = 0.
        let mut certain = SqrtKf::new(Vector2::zeros(), Matrix2::identity()).unwrap();
        certain.s = Matrix2::zeros();
        let before = certain.clone();
        let singular = certain.update(&h, &Vector1::new(1.0), &Matrix1::new(0.0));
        assert_eq!(singular, Err(KalmanError::SingularInnovation));
        assert_eq!(certain, before);

        assert_eq!(
            SqrtKf::new(Vector2::zeros(), Matrix2::new(1.0, 2.0, 2.0, 1.0)),
            Err(KalmanError::CovarianceNotPositiveDefinite)
        );
    }
}
