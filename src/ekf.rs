//! Extended Kalman filter with user-supplied or automatic Jacobians.

use nalgebra::{Cholesky, SMatrix, SVector};

use crate::error::KalmanError;
use crate::model::{MeasurementModel, ProcessModel};

/// An extended Kalman filter over an `N`-dimensional state.
#[derive(Debug, Clone, PartialEq)]
pub struct Ekf<const N: usize> {
    x: SVector<f64, N>,
    p: SMatrix<f64, N, N>,
}

impl<const N: usize> Ekf<N> {
    /// Creates a filter with initial state `x` and covariance `p`.
    pub fn new(x: SVector<f64, N>, p: SMatrix<f64, N, N>) -> Self {
        Self { x, p }
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
    pub fn predict<P: ProcessModel<N>>(&mut self, model: &P, q: &SMatrix<f64, N, N>, dt: f64) {
        let f = model.jacobian(&self.x, dt);
        self.x = model.predict(&self.x, dt);
        self.p = symmetrize(f * self.p * f.transpose() + q);
    }

    /// Corrects the estimate with measurement `z` and measurement noise `r`, using the
    /// Joseph-form covariance update.
    ///
    /// Returns the normalized innovation squared (NIS). On error the filter is unchanged.
    pub fn update<H: MeasurementModel<N, M>, const M: usize>(
        &mut self,
        model: &H,
        z: &SVector<f64, M>,
        r: &SMatrix<f64, M, M>,
    ) -> Result<f64, KalmanError> {
        if !all_finite(z.as_slice()) || !all_finite(r.as_slice()) {
            return Err(KalmanError::InvalidInput);
        }

        let h = model.jacobian(&self.x);
        let y = z - model.measure(&self.x);
        let s = h * self.p * h.transpose() + r;
        let s_chol = Cholesky::new(s).ok_or(KalmanError::SingularInnovation)?;

        // K = P Hᵀ S⁻¹, computed as (S⁻¹ H P)ᵀ since S and P are symmetric.
        let k = s_chol.solve(&(h * self.p)).transpose();
        let i_kh = SMatrix::<f64, N, N>::identity() - k * h;
        let x = self.x + k * y;
        let p = symmetrize(i_kh * self.p * i_kh.transpose() + k * r * k.transpose());
        let nis = y.dot(&s_chol.solve(&y));

        if !all_finite(x.as_slice()) || !all_finite(p.as_slice()) || !nis.is_finite() {
            return Err(KalmanError::NumericalFailure);
        }
        self.x = x;
        self.p = p;
        Ok(nis)
    }
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn symmetrize<const N: usize>(p: SMatrix<f64, N, N>) -> SMatrix<f64, N, N> {
    (p + p.transpose()) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix1, Matrix2, Vector1, Vector2};

    /// Position and velocity, moving at constant velocity.
    struct ConstantVelocity;

    impl ProcessModel<2> for ConstantVelocity {
        fn predict(&self, x: &Vector2<f64>, dt: f64) -> Vector2<f64> {
            self.jacobian(x, dt) * x
        }
        fn jacobian(&self, _x: &Vector2<f64>, dt: f64) -> Matrix2<f64> {
            Matrix2::new(1.0, dt, 0.0, 1.0)
        }
    }

    /// Measures position only.
    struct Position;

    impl MeasurementModel<2, 1> for Position {
        fn measure(&self, x: &Vector2<f64>) -> Vector1<f64> {
            Vector1::new(x[0])
        }
        fn jacobian(&self, _x: &Vector2<f64>) -> nalgebra::Matrix1x2<f64> {
            nalgebra::Matrix1x2::new(1.0, 0.0)
        }
    }

    #[test]
    fn scalar_update_matches_hand_calculation() {
        let mut kf = Ekf::new(Vector2::new(0.0, 0.0), Matrix2::identity());
        let nis = kf
            .update(&Position, &Vector1::new(2.0), &Matrix1::new(1.0))
            .unwrap();

        // S = 2, K = [0.5, 0], y = 2.
        approx::assert_relative_eq!(nis, 2.0);
        approx::assert_relative_eq!(*kf.state(), Vector2::new(1.0, 0.0));
        approx::assert_relative_eq!(*kf.covariance(), Matrix2::new(0.5, 0.0, 0.0, 1.0));
    }

    #[test]
    fn predict_propagates_covariance() {
        let mut kf = Ekf::new(Vector2::new(1.0, 2.0), Matrix2::identity());
        kf.predict(&ConstantVelocity, &Matrix2::zeros(), 0.5);

        approx::assert_relative_eq!(*kf.state(), Vector2::new(2.0, 2.0));
        approx::assert_relative_eq!(*kf.covariance(), Matrix2::new(1.25, 0.5, 0.5, 1.0));
    }

    #[test]
    fn nan_measurement_returns_error_and_keeps_state() {
        let mut kf = Ekf::new(Vector2::new(0.0, 0.0), Matrix2::identity());
        let before = kf.clone();

        let result = kf.update(&Position, &Vector1::new(f64::NAN), &Matrix1::new(1.0));
        assert_eq!(result, Err(KalmanError::InvalidInput));
        assert_eq!(kf, before);
    }

    #[test]
    fn singular_innovation_returns_error_and_keeps_state() {
        let mut kf = Ekf::new(Vector2::new(0.0, 0.0), Matrix2::zeros());
        let before = kf.clone();

        let result = kf.update(&Position, &Vector1::new(1.0), &Matrix1::new(0.0));
        assert_eq!(result, Err(KalmanError::SingularInnovation));
        assert_eq!(kf, before);
    }
}
