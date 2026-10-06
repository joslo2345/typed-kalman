#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

//! Type-safe Kalman filters for desktop and embedded targets.
//!
//! Dimensions are const generics on [`nalgebra`] static matrices, so a measurement of the wrong
//! size doesn't compile. Nothing allocates, the whole crate works without `std`, and every
//! filter runs in `f32` or `f64`.
//!
//! # Quick start
//!
//! Track a target moving at constant velocity, `[position, velocity]`, from noisy position
//! measurements:
//!
//! ```
//! use typed_kalman::LinearKf;
//! use nalgebra::{Matrix1, Matrix1x2, Matrix2, Vector1, Vector2};
//!
//! let dt = 0.1;
//! let f = Matrix2::new(1.0, dt, 0.0, 1.0); // state transition
//! let q = Matrix2::new(1e-4, 0.0, 0.0, 1e-2); // process noise
//! let h = Matrix1x2::new(1.0, 0.0); // we measure position only
//! let r = Matrix1::new(0.25); // measurement noise
//!
//! let mut kf = LinearKf::new(Vector2::new(0.0, 0.0), Matrix2::identity());
//! for k in 1..=50 {
//!     let z = Vector1::new(2.0 * k as f64 * dt); // truly moving at 2 units/s
//!     kf.predict(&f, &q);
//!     let nis = kf.update(&h, &z, &r)?; // normalized innovation squared, for diagnostics
//!     assert!(nis.is_finite());
//! }
//! assert!((kf.state()[1] - 2.0).abs() < 0.1); // the velocity has been estimated
//! # Ok::<(), typed_kalman::KalmanError>(())
//! ```
//!
//! # Choosing a filter
//!
//! | Filter | Models | Use it when |
//! |---|---|---|
//! | [`LinearKf`] | matrices `F`, `H` | the system is linear; fastest |
//! | [`SqrtKf`] | matrices, noise as square roots | the covariance is too ill-conditioned for `LinearKf` (very precise sensors, `f32`); 2–3× slower |
//! | [`Ekf`] | [`ProcessJacobian`], [`MeasurementJacobian`] | the system is mildly nonlinear and you have Jacobians, or let the `autodiff` feature compute them |
//! | [`Ukf`] | [`ProcessModel`], [`MeasurementModel`] | the system is nonlinear and Jacobians are awkward |
//! | [`SqrtUkf`] | the same, noise as square roots | as the UKF, with a covariance factor that can't lose positive-definiteness |
//!
//! [`smoother::smooth`] runs a Rauch-Tung-Striebel smoother over a recorded history (for the
//! UKF, [`smoother::smooth_with_cross`] with [`Ukf::predict_with_cross`]), and
//! [`diagnostics`] checks filter consistency with NEES and chi-squared bounds.
//!
//! # Errors
//!
//! Filters never panic on bad data. A NaN or infinite input, a singular innovation covariance,
//! or a covariance that stops being positive-definite returns a [`KalmanError`] and leaves the
//! filter exactly as it was, so the caller can skip that measurement and carry on.
//!
//! # Features
//!
//! - `std` (default): links the standard library. Turn it off with `default-features = false`
//!   for bare-metal targets; nothing else changes.
//! - `autodiff`: Jacobians computed exactly with dual numbers, in the `autodiff` module.
//!
//! # Embedded use
//!
//! Without `std`, the filters work unchanged, in single precision:
//!
//! ```
//! use typed_kalman::LinearKf;
//! use nalgebra::{Matrix1, Matrix1x2, Matrix2, Vector1, Vector2};
//!
//! fn step(kf: &mut LinearKf<2, f32>, position: f32) -> Option<f32> {
//!     let f = Matrix2::new(1.0, 0.01, 0.0, 1.0);
//!     let q = Matrix2::new(1e-6, 0.0, 0.0, 1e-4);
//!     kf.predict(&f, &q);
//!     // On a bad reading the filter is unchanged, so just skip it.
//!     kf.update(&Matrix1x2::new(1.0, 0.0), &Vector1::new(position), &Matrix1::new(0.01)).ok()?;
//!     Some(kf.state()[1]) // estimated velocity
//! }
//! # fn main() {
//! # let mut kf = LinearKf::new(Vector2::new(0.0f32, 0.0), Matrix2::identity());
//! # assert!(step(&mut kf, 0.5).is_some());
//! # assert!(step(&mut kf, f32::NAN).is_none());
//! # }
//! ```

#[cfg(feature = "autodiff")]
pub mod autodiff;
mod cholupdate;
pub mod diagnostics;
pub mod ekf;
pub mod error;
pub mod linear;
pub mod model;
pub mod scalar;
pub mod smoother;
pub mod sqrt_kf;
pub mod sqrt_ukf;
pub mod ukf;
mod update;

/// Compiles and runs the README's examples as doctests, so they can't go stale.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

pub use ekf::Ekf;
pub use error::KalmanError;
pub use linear::LinearKf;
pub use model::{
    wrap_angle, ControlledJacobian, ControlledProcess, MeasurementJacobian, MeasurementModel,
    ProcessJacobian, ProcessModel, WithInput,
};
pub use scalar::Float;
pub use smoother::{CrossStep, Estimate, RtsStep};
pub use sqrt_kf::SqrtKf;
pub use sqrt_ukf::SqrtUkf;
pub use ukf::{Ukf, UkfParams};
