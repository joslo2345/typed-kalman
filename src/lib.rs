#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

//! Type-safe Kalman filters for desktop and embedded targets.

pub mod diagnostics;
pub mod ekf;
pub mod error;
pub mod linear;
pub mod model;
pub mod smoother;
pub mod sqrt_ukf;
pub mod ukf;
mod update;

pub use ekf::Ekf;
pub use error::KalmanError;
pub use linear::LinearKf;
pub use model::{MeasurementJacobian, MeasurementModel, ProcessJacobian, ProcessModel};
pub use sqrt_ukf::SqrtUkf;
pub use ukf::{Ukf, UkfParams};
