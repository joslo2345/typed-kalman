//! Error type returned by filter operations.

use core::fmt;

/// An error from a filter operation. The filter state is left unchanged when one is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KalmanError {
    /// An input (measurement or noise covariance) contained NaN or infinity.
    InvalidInput,
    /// The innovation covariance was not symmetric positive-definite, so it could not be inverted.
    SingularInnovation,
    /// The state covariance was not symmetric positive-definite, so sigma points could not be
    /// drawn from it, or an update would have made it so.
    CovarianceNotPositiveDefinite,
    /// The update produced a non-finite state or covariance.
    NumericalFailure,
}

impl fmt::Display for KalmanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidInput => "input contains NaN or infinity",
            Self::SingularInnovation => "innovation covariance is not positive-definite",
            Self::CovarianceNotPositiveDefinite => "state covariance is not positive-definite",
            Self::NumericalFailure => "update produced a non-finite state or covariance",
        })
    }
}

impl core::error::Error for KalmanError {}
