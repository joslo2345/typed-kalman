//! Measurement update shared by the linear and extended filters.

use nalgebra::{Cholesky, SMatrix, SVector};

use crate::error::KalmanError;

/// A corrected state and covariance, plus the normalized innovation squared.
pub(crate) struct Corrected<const N: usize> {
    pub x: SVector<f64, N>,
    pub p: SMatrix<f64, N, N>,
    pub nis: f64,
}

/// Corrects `x` and `p` with innovation `y`, measurement Jacobian `h` and measurement noise `r`,
/// using the Joseph-form covariance update.
///
/// Does not modify anything, so callers can keep their state unchanged on error.
pub(crate) fn joseph<const N: usize, const M: usize>(
    x: &SVector<f64, N>,
    p: &SMatrix<f64, N, N>,
    h: &SMatrix<f64, M, N>,
    y: &SVector<f64, M>,
    r: &SMatrix<f64, M, M>,
) -> Result<Corrected<N>, KalmanError> {
    if !all_finite(y.as_slice()) || !all_finite(r.as_slice()) {
        return Err(KalmanError::InvalidInput);
    }

    let s = h * p * h.transpose() + r;
    let s_chol = Cholesky::new(s).ok_or(KalmanError::SingularInnovation)?;

    // K = P Hᵀ S⁻¹, computed as (S⁻¹ H P)ᵀ since S and P are symmetric.
    let k = s_chol.solve(&(h * p)).transpose();
    let i_kh = SMatrix::<f64, N, N>::identity() - k * h;
    let x = x + k * y;
    let p = symmetrize(i_kh * p * i_kh.transpose() + k * r * k.transpose());
    let nis = y.dot(&s_chol.solve(y));

    if !all_finite(x.as_slice()) || !all_finite(p.as_slice()) || !nis.is_finite() {
        return Err(KalmanError::NumericalFailure);
    }
    Ok(Corrected { x, p, nis })
}

/// Returns `(p + pᵀ) / 2`, removing the asymmetry that rounding errors introduce.
pub(crate) fn symmetrize<const N: usize>(p: SMatrix<f64, N, N>) -> SMatrix<f64, N, N> {
    (p + p.transpose()) * 0.5
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}
