#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

//! Type-safe Kalman filters for desktop and embedded targets.

pub mod diagnostics;
pub mod ekf;
pub mod error;
pub mod linear;
pub mod smoother;
pub mod sqrt_ukf;
pub mod ukf;
