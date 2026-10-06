//! Traits that describe how the state evolves and how it is measured.
//!
//! The UKF needs only [`ProcessModel`] and [`MeasurementModel`]. The EKF also needs their
//! Jacobians, through [`ProcessJacobian`] and [`MeasurementJacobian`]. The scalar type `T`
//! defaults to `f64`; see [`Float`].

use nalgebra::{SMatrix, SVector};

use crate::scalar::Float;

/// Describes how an `N`-dimensional state evolves over time.
pub trait ProcessModel<const N: usize, T: Float = f64> {
    /// Propagates the state `x` forward by `dt`.
    fn predict(&self, x: &SVector<T, N>, dt: T) -> SVector<T, N>;

    /// Returns the difference `a - b` between two states.
    ///
    /// The default subtracts. Override it when a state component wraps around, such as a
    /// heading: wrap that component with [`wrap_angle`]. The UKFs use it to average their
    /// propagated sigma points, so a heading near ±π isn't averaged to 0. To keep the estimate
    /// itself in range, also wrap the heading in [`predict`](Self::predict).
    fn state_residual(&self, a: &SVector<T, N>, b: &SVector<T, N>) -> SVector<T, N> {
        a - b
    }
}

/// A [`ProcessModel`] that can also supply its Jacobian.
pub trait ProcessJacobian<const N: usize, T: Float = f64>: ProcessModel<N, T> {
    /// Returns the Jacobian of [`predict`](ProcessModel::predict) with respect to the state,
    /// evaluated at `x`.
    fn jacobian(&self, x: &SVector<T, N>, dt: T) -> SMatrix<T, N, N>;
}

/// Describes how an `N`-dimensional state maps to an `M`-dimensional measurement.
pub trait MeasurementModel<const N: usize, const M: usize, T: Float = f64> {
    /// Returns the measurement expected for the state `x`.
    fn measure(&self, x: &SVector<T, N>) -> SVector<T, M>;

    /// Returns the difference `a - b` between two measurements.
    ///
    /// The default subtracts. Override it when a component wraps around, such as a bearing:
    /// the difference between 179° and -179° is 2°, not 358°. [`wrap_angle`] does that for
    /// radians. Every filter computes its innovation with this, and the UKFs also use it to
    /// average and spread their sigma points, so wrapped measurements work near the seam.
    fn residual(&self, a: &SVector<T, M>, b: &SVector<T, M>) -> SVector<T, M> {
        a - b
    }
}

/// Wraps an angle in radians into `[-π, π)`.
///
/// For use in [`MeasurementModel::residual`] when a measurement component is an angle.
pub fn wrap_angle<T: Float>(angle: T) -> T {
    let two_pi = T::two_pi();
    angle - two_pi * ((angle + T::pi()) / two_pi).floor()
}

/// A [`MeasurementModel`] that can also supply its Jacobian.
pub trait MeasurementJacobian<const N: usize, const M: usize, T: Float = f64>:
    MeasurementModel<N, M, T>
{
    /// Returns the Jacobian of [`measure`](MeasurementModel::measure) with respect to the state,
    /// evaluated at `x`.
    fn jacobian(&self, x: &SVector<T, N>) -> SMatrix<T, M, N>;
}
