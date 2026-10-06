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
}

/// A [`MeasurementModel`] that can also supply its Jacobian.
pub trait MeasurementJacobian<const N: usize, const M: usize, T: Float = f64>:
    MeasurementModel<N, M, T>
{
    /// Returns the Jacobian of [`measure`](MeasurementModel::measure) with respect to the state,
    /// evaluated at `x`.
    fn jacobian(&self, x: &SVector<T, N>) -> SMatrix<T, M, N>;
}
