//! Linear Kalman filter with the Joseph-form covariance update.

use nalgebra::{SMatrix, SVector};

use crate::error::KalmanError;
use crate::scalar::Float;
use crate::update::{self, symmetrize};

/// A linear Kalman filter over an `N`-dimensional state.
///
/// The model matrices are passed to each [`predict`](Self::predict) and
/// [`update`](Self::update) call, so time-varying models need no extra setup.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearKf<const N: usize, T: Float = f64> {
    x: SVector<T, N>,
    p: SMatrix<T, N, N>,
}

impl<const N: usize, T: Float> LinearKf<N, T> {
    /// Creates a filter with initial state `x` and covariance `p`.
    pub fn new(x: SVector<T, N>, p: SMatrix<T, N, N>) -> Self {
        Self { x, p }
    }

    /// Returns the current state estimate.
    pub fn state(&self) -> &SVector<T, N> {
        &self.x
    }

    /// Returns the current state covariance.
    pub fn covariance(&self) -> &SMatrix<T, N, N> {
        &self.p
    }

    /// Propagates the state with transition matrix `f` and process noise `q`.
    pub fn predict(&mut self, f: &SMatrix<T, N, N>, q: &SMatrix<T, N, N>) {
        self.x = f * self.x;
        self.p = symmetrize(f * self.p * f.transpose() + q);
    }

    /// Corrects the estimate with measurement `z`, observation matrix `h` and measurement
    /// noise `r`, using the Joseph-form covariance update.
    ///
    /// Returns the normalized innovation squared (NIS). On error the filter is unchanged.
    pub fn update<const M: usize>(
        &mut self,
        h: &SMatrix<T, M, N>,
        z: &SVector<T, M>,
        r: &SMatrix<T, M, M>,
    ) -> Result<T, KalmanError> {
        let y = z - h * self.x;
        let corrected = update::joseph(&self.x, &self.p, h, &y, r)?;
        self.x = corrected.x;
        self.p = corrected.p;
        Ok(corrected.nis)
    }
}

impl<const N: usize, T: Float> Default for LinearKf<N, T> {
    /// Returns a filter with a zero state and an identity covariance.
    fn default() -> Self {
        Self::new(SVector::zeros(), SMatrix::identity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ekf::Ekf;
    use crate::model::{MeasurementJacobian, MeasurementModel, ProcessJacobian, ProcessModel};
    use nalgebra::{Matrix1, Matrix1x2, Matrix2, Matrix2x4, Matrix4, Vector1, Vector2, Vector4};

    #[test]
    fn scalar_update_matches_hand_calculation() {
        let mut kf = LinearKf::new(Vector2::new(0.0, 0.0), Matrix2::identity());
        let nis = kf
            .update(
                &Matrix1x2::new(1.0, 0.0),
                &Vector1::new(2.0),
                &Matrix1::new(1.0),
            )
            .unwrap();

        // S = 2, K = [0.5, 0], y = 2.
        approx::assert_relative_eq!(nis, 2.0);
        approx::assert_relative_eq!(*kf.state(), Vector2::new(1.0, 0.0));
        approx::assert_relative_eq!(*kf.covariance(), Matrix2::new(0.5, 0.0, 0.0, 1.0));
    }

    #[test]
    fn predict_propagates_covariance() {
        let mut kf = LinearKf::new(Vector2::new(1.0, 2.0), Matrix2::identity());
        kf.predict(&Matrix2::new(1.0, 0.5, 0.0, 1.0), &Matrix2::zeros());

        approx::assert_relative_eq!(*kf.state(), Vector2::new(2.0, 2.0));
        approx::assert_relative_eq!(*kf.covariance(), Matrix2::new(1.25, 0.5, 0.5, 1.0));
    }

    #[test]
    fn nan_measurement_returns_error_and_keeps_state() {
        let mut kf = LinearKf::<2>::default();
        let before = kf.clone();

        let result = kf.update(
            &Matrix1x2::new(1.0, 0.0),
            &Vector1::new(f64::NAN),
            &Matrix1::new(1.0),
        );
        assert_eq!(result, Err(KalmanError::InvalidInput));
        assert_eq!(kf, before);
    }

    #[test]
    fn singular_innovation_returns_error_and_keeps_state() {
        let mut kf = LinearKf::new(Vector2::zeros(), Matrix2::zeros());
        let before = kf.clone();

        let result = kf.update(
            &Matrix1x2::new(1.0, 0.0),
            &Vector1::new(1.0),
            &Matrix1::new(0.0),
        );
        assert_eq!(result, Err(KalmanError::SingularInnovation));
        assert_eq!(kf, before);
    }

    /// The 2D constant-velocity model, as an EKF model, for comparing against the linear filter.
    struct Linear {
        f: Matrix4<f64>,
        h: Matrix2x4<f64>,
    }

    impl ProcessModel<4> for Linear {
        fn predict(&self, x: &Vector4<f64>, _dt: f64) -> Vector4<f64> {
            self.f * x
        }
    }

    impl ProcessJacobian<4> for Linear {
        fn jacobian(&self, _x: &Vector4<f64>, _dt: f64) -> Matrix4<f64> {
            self.f
        }
    }

    impl MeasurementModel<4, 2> for Linear {
        fn measure(&self, x: &Vector4<f64>) -> Vector2<f64> {
            self.h * x
        }
    }

    impl MeasurementJacobian<4, 2> for Linear {
        fn jacobian(&self, _x: &Vector4<f64>) -> Matrix2x4<f64> {
            self.h
        }
    }

    #[test]
    fn matches_ekf_with_linear_models() {
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
        let mut ekf = Ekf::new(Vector4::zeros(), Matrix4::identity());
        for k in 0..1000 {
            let t = k as f64 * dt;
            let z = Vector2::new(t.sin() * 10.0, t.cos() * 5.0);

            kf.predict(&f, &q);
            ekf.predict(&model, &q, dt);
            let nis_kf = kf.update(&h, &z, &r).unwrap();
            let nis_ekf = ekf.update(&model, &z, &r).unwrap();

            approx::assert_relative_eq!(nis_kf, nis_ekf, epsilon = 1e-12);
            approx::assert_relative_eq!(kf.state(), ekf.state(), epsilon = 1e-12);
            approx::assert_relative_eq!(kf.covariance(), ekf.covariance(), epsilon = 1e-12);
        }
    }
}
