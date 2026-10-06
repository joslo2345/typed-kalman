//! A textbook Kalman filter with no numerical safeguards, to compare stability against.
//!
//! It uses the short covariance update `P = (I - K H) P`, inverts `S` explicitly, and never
//! symmetrizes, as many hand-written filters do.

use kalman_rs::Float;
use nalgebra::{SMatrix, SVector};

/// An unstabilized linear Kalman filter.
pub struct NaiveKf<const N: usize, T: Float> {
    pub x: SVector<T, N>,
    pub p: SMatrix<T, N, N>,
}

impl<const N: usize, T: Float> NaiveKf<N, T> {
    /// Predicts with transition `f` and process noise `q`.
    pub fn predict(&mut self, f: &SMatrix<T, N, N>, q: &SMatrix<T, N, N>) {
        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + q;
    }

    /// Updates with measurement `z`. Returns `false` if `S` can't be inverted.
    pub fn update<const M: usize>(
        &mut self,
        h: &SMatrix<T, M, N>,
        z: &SVector<T, M>,
        r: &SMatrix<T, M, M>,
    ) -> bool {
        let s = h * self.p * h.transpose() + r;
        let Some(s_inv) = s.try_inverse() else {
            return false;
        };
        let k = self.p * h.transpose() * s_inv;
        self.x += k * (z - h * self.x);
        self.p = (SMatrix::<T, N, N>::identity() - k * h) * self.p;
        true
    }
}
