//! Measurement update shared by the linear and extended filters.

use nalgebra::{Cholesky, SMatrix, SVector};

use crate::error::KalmanError;
use crate::scalar::Float;

/// A corrected state and covariance, plus the normalized innovation squared.
pub(crate) struct Corrected<const N: usize, T: Float> {
    pub x: SVector<T, N>,
    pub p: SMatrix<T, N, N>,
    pub nis: T,
}

/// Corrects `x` and `p` with innovation `y`, measurement Jacobian `h` and measurement noise `r`,
/// using the Joseph-form covariance update.
///
/// Does not modify anything, so callers can keep their state unchanged on error.
#[inline]
pub(crate) fn joseph<const N: usize, const M: usize, T: Float>(
    x: &SVector<T, N>,
    p: &SMatrix<T, N, N>,
    h: &SMatrix<T, M, N>,
    y: &SVector<T, M>,
    r: &SMatrix<T, M, M>,
) -> Result<Corrected<N, T>, KalmanError> {
    if !all_finite(y.as_slice()) || !all_finite(r.as_slice()) {
        return Err(KalmanError::InvalidInput);
    }

    let s = h * p * h.transpose() + r;
    let s_chol = Cholesky::new(s).ok_or(KalmanError::SingularInnovation)?;

    // S is only M × M, so inverting it once through its Cholesky factor and reusing the
    // inverse for the gain and the NIS is about 8% faster per step (measured on S2) than
    // solving with H P's N columns and then again with y.
    let s_inv = s_chol.inverse();
    let k = p * h.transpose() * s_inv;
    let i_kh = SMatrix::<T, N, N>::identity() - k * h;
    let x = x + k * y;
    let p = symmetrize(i_kh * p * i_kh.transpose() + k * r * k.transpose());
    let nis = y.dot(&(s_inv * y));

    if !all_finite(x.as_slice()) || !all_finite(p.as_slice()) || !nis.is_finite() {
        return Err(KalmanError::NumericalFailure);
    }
    Ok(Corrected { x, p, nis })
}

/// Makes `p` exactly symmetric by copying its lower triangle over its upper one, removing the
/// asymmetry that rounding errors introduce.
///
/// Mirroring is cheaper than averaging with the transpose, and Cholesky reads only the lower
/// triangle anyway, so this keeps what the filter actually uses.
#[inline]
pub(crate) fn symmetrize<const N: usize, T: Float>(mut p: SMatrix<T, N, N>) -> SMatrix<T, N, N> {
    p.fill_upper_triangle_with_lower_triangle();
    p
}

/// Returns whether every value is neither NaN nor infinite.
///
/// Folds over every value instead of stopping at the first bad one, which lets the compiler
/// vectorize it; values are almost always finite, so early exit buys nothing.
#[inline]
pub(crate) fn all_finite<T: Float>(values: &[T]) -> bool {
    values.iter().fold(true, |ok, v| ok & v.is_finite())
}
