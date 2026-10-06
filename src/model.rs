//! Traits that describe how the state evolves and how it is measured.
//!
//! The UKF needs only [`ProcessModel`] and [`MeasurementModel`]. The EKF also needs their
//! Jacobians, through [`ProcessJacobian`] and [`MeasurementJacobian`].

use nalgebra::{SMatrix, SVector};

/// Describes how an `N`-dimensional state evolves over time.
pub trait ProcessModel<const N: usize> {
    /// Propagates the state `x` forward by `dt`.
    fn predict(&self, x: &SVector<f64, N>, dt: f64) -> SVector<f64, N>;
}

/// A [`ProcessModel`] that can also supply its Jacobian.
pub trait ProcessJacobian<const N: usize>: ProcessModel<N> {
    /// Returns the Jacobian of [`predict`](ProcessModel::predict) with respect to the state,
    /// evaluated at `x`.
    fn jacobian(&self, x: &SVector<f64, N>, dt: f64) -> SMatrix<f64, N, N>;
}

/// Describes how an `N`-dimensional state maps to an `M`-dimensional measurement.
pub trait MeasurementModel<const N: usize, const M: usize> {
    /// Returns the measurement expected for the state `x`.
    fn measure(&self, x: &SVector<f64, N>) -> SVector<f64, M>;
}

/// A [`MeasurementModel`] that can also supply its Jacobian.
pub trait MeasurementJacobian<const N: usize, const M: usize>: MeasurementModel<N, M> {
    /// Returns the Jacobian of [`measure`](MeasurementModel::measure) with respect to the state,
    /// evaluated at `x`.
    fn jacobian(&self, x: &SVector<f64, N>) -> SMatrix<f64, M, N>;
}
