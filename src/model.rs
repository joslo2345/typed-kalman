//! Traits that describe how the state evolves and how it is measured.

use nalgebra::{SMatrix, SVector};

/// Describes how an `N`-dimensional state evolves over time.
pub trait ProcessModel<const N: usize> {
    /// Propagates the state `x` forward by `dt`.
    fn predict(&self, x: &SVector<f64, N>, dt: f64) -> SVector<f64, N>;

    /// Returns the Jacobian of [`predict`](Self::predict) with respect to the state, evaluated at `x`.
    fn jacobian(&self, x: &SVector<f64, N>, dt: f64) -> SMatrix<f64, N, N>;
}

/// Describes how an `N`-dimensional state maps to an `M`-dimensional measurement.
pub trait MeasurementModel<const N: usize, const M: usize> {
    /// Returns the measurement expected for the state `x`.
    fn measure(&self, x: &SVector<f64, N>) -> SVector<f64, M>;

    /// Returns the Jacobian of [`measure`](Self::measure) with respect to the state, evaluated at `x`.
    fn jacobian(&self, x: &SVector<f64, N>) -> SMatrix<f64, M, N>;
}
