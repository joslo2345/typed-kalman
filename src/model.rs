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

/// A process driven by a known control input `u` with `U` components, such as a commanded
/// acceleration or a gyro reading.
///
/// Filters take a [`ProcessModel`]; wrap a controlled model and this step's input in
/// [`WithInput`] to get one. That works with every filter that takes a process model.
pub trait ControlledProcess<const N: usize, const U: usize, T: Float = f64> {
    /// Propagates the state `x` forward by `dt` under input `u`.
    fn predict(&self, x: &SVector<T, N>, u: &SVector<T, U>, dt: T) -> SVector<T, N>;

    /// Returns the difference `a - b` between two states; see
    /// [`ProcessModel::state_residual`].
    fn state_residual(&self, a: &SVector<T, N>, b: &SVector<T, N>) -> SVector<T, N> {
        a - b
    }
}

/// A [`ControlledProcess`] that can also supply its Jacobian with respect to the state.
pub trait ControlledJacobian<const N: usize, const U: usize, T: Float = f64>:
    ControlledProcess<N, U, T>
{
    /// Returns the Jacobian of [`predict`](ControlledProcess::predict) with respect to the
    /// state, evaluated at `x` and `u`.
    fn jacobian(&self, x: &SVector<T, N>, u: &SVector<T, U>, dt: T) -> SMatrix<T, N, N>;
}

/// A [`ControlledProcess`] together with one step's input, usable wherever a
/// [`ProcessModel`] (or, for a [`ControlledJacobian`], a [`ProcessJacobian`]) is expected.
///
/// ```
/// use typed_kalman::{ControlledJacobian, ControlledProcess, Ekf, WithInput};
/// use nalgebra::{Matrix2, Vector1, Vector2};
///
/// /// Position and velocity, driven by a commanded acceleration.
/// struct Cart;
///
/// impl ControlledProcess<2, 1> for Cart {
///     fn predict(&self, x: &Vector2<f64>, u: &Vector1<f64>, dt: f64) -> Vector2<f64> {
///         Vector2::new(x[0] + x[1] * dt + 0.5 * u[0] * dt * dt, x[1] + u[0] * dt)
///     }
/// }
///
/// impl ControlledJacobian<2, 1> for Cart {
///     fn jacobian(&self, _x: &Vector2<f64>, _u: &Vector1<f64>, dt: f64) -> Matrix2<f64> {
///         Matrix2::new(1.0, dt, 0.0, 1.0)
///     }
/// }
///
/// let mut ekf = Ekf::new(Vector2::zeros(), Matrix2::identity());
/// let acceleration = Vector1::new(2.0);
/// ekf.predict(&WithInput::new(&Cart, acceleration), &(Matrix2::identity() * 1e-3), 0.5);
/// assert!((ekf.state()[1] - 1.0).abs() < 1e-12); // v = a t
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WithInput<'a, Model, const U: usize, T: Float = f64> {
    model: &'a Model,
    input: SVector<T, U>,
}

impl<'a, Model, const U: usize, T: Float> WithInput<'a, Model, U, T> {
    /// Pairs `model` with this step's input `input`.
    pub fn new(model: &'a Model, input: SVector<T, U>) -> Self {
        Self { model, input }
    }
}

impl<Model: ControlledProcess<N, U, T>, const N: usize, const U: usize, T: Float> ProcessModel<N, T>
    for WithInput<'_, Model, U, T>
{
    fn predict(&self, x: &SVector<T, N>, dt: T) -> SVector<T, N> {
        self.model.predict(x, &self.input, dt)
    }

    fn state_residual(&self, a: &SVector<T, N>, b: &SVector<T, N>) -> SVector<T, N> {
        self.model.state_residual(a, b)
    }
}

impl<Model: ControlledJacobian<N, U, T>, const N: usize, const U: usize, T: Float>
    ProcessJacobian<N, T> for WithInput<'_, Model, U, T>
{
    fn jacobian(&self, x: &SVector<T, N>, dt: T) -> SMatrix<T, N, N> {
        self.model.jacobian(x, &self.input, dt)
    }
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
