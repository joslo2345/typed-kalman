//! Rank-one updates and downdates of lower-triangular Cholesky factors.
//!
//! These build and modify covariance square roots without forming the covariance, which
//! squares the condition number. The square-root filters build each new factor from zero, one
//! column at a time, instead of with a QR factorization of a `D × (columns)` matrix whose size
//! can't be expressed with const generics.

use nalgebra::{ComplexField, SMatrix, SVector};

use crate::scalar::Float;

/// Replaces lower-triangular `l` with the factor of `l lᵀ + c cᵀ` summed over the columns `c`
/// of `columns`.
#[inline]
pub(crate) fn update_columns<const D: usize, const C: usize, T: Float>(
    l: &mut SMatrix<T, D, D>,
    columns: &SMatrix<T, D, C>,
) {
    for j in 0..C {
        update(l, columns.column(j).into_owned());
    }
}

/// Replaces lower-triangular `l` with the factor of `l lᵀ + v vᵀ`, using Givens rotations.
///
/// Unlike the textbook update, this works when `l` has zeros on its diagonal, so a factor can
/// be built up from a zero matrix.
#[inline]
pub(crate) fn update<const D: usize, T: Float>(l: &mut SMatrix<T, D, D>, mut v: SVector<T, D>) {
    for k in 0..D {
        let (lkk, vk) = (l[(k, k)], v[k]);
        let r = ComplexField::sqrt(lkk * lkk + vk * vk);
        if r == T::zero() {
            continue;
        }
        let (c, s) = (lkk / r, vk / r);
        l[(k, k)] = r;
        for i in k + 1..D {
            let (lik, vi) = (l[(i, k)], v[i]);
            l[(i, k)] = c * lik + s * vi;
            v[i] = c * vi - s * lik;
        }
    }
}

/// Replaces lower-triangular `l` with the factor of `l lᵀ - v vᵀ`, using hyperbolic rotations.
///
/// Returns `None`, leaving `l` partially modified, if the result wouldn't be positive-definite.
#[inline]
pub(crate) fn downdate<const D: usize, T: Float>(
    l: &mut SMatrix<T, D, D>,
    mut v: SVector<T, D>,
) -> Option<()> {
    for k in 0..D {
        let (lkk, vk) = (l[(k, k)], v[k]);
        let r2 = lkk * lkk - vk * vk;
        if !r2.is_finite() || r2 <= T::zero() {
            return None;
        }
        let r = ComplexField::sqrt(r2);
        let (c, s) = (r / lkk, vk / lkk);
        l[(k, k)] = r;
        for i in k + 1..D {
            l[(i, k)] = (l[(i, k)] - s * v[i]) / c;
            v[i] = c * v[i] - s * l[(i, k)];
        }
    }
    Some(())
}

/// Returns whether lower-triangular `s` has a zero on its diagonal, so `S Sᵀ` is singular.
///
/// This happens in low precision when the sigma-point spread falls below the resolution of the
/// state's magnitude, so every point rounds to the mean.
#[inline]
pub(crate) fn is_singular<const N: usize, T: Float>(s: &SMatrix<T, N, N>) -> bool {
    (0..N).any(|i| s[(i, i)] == T::zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Cholesky, Matrix2, Matrix3, Vector2, Vector3};

    #[rustfmt::skip]
    fn spd3() -> Matrix3<f64> {
        Matrix3::new(
            4.0, 1.0, 0.5,
            1.0, 3.0, 0.2,
            0.5, 0.2, 2.0,
        )
    }

    #[test]
    fn update_matches_direct_factorization() {
        let v = Vector3::new(0.3, -1.2, 0.7);
        let mut l = Cholesky::new(spd3()).unwrap().unpack();
        update(&mut l, v);
        approx::assert_relative_eq!(
            l * l.transpose(),
            spd3() + v * v.transpose(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn update_builds_a_factor_from_zero() {
        let mut l = Matrix3::zeros();
        let (a, b, c) = (
            Vector3::new(1.0, 2.0, 0.0),
            Vector3::new(0.0, 1.0, -1.0),
            Vector3::new(0.5, 0.0, 3.0),
        );
        for v in [a, b, c] {
            update(&mut l, v);
        }
        let expected = a * a.transpose() + b * b.transpose() + c * c.transpose();
        approx::assert_relative_eq!(l * l.transpose(), expected, epsilon = 1e-12);
    }

    #[test]
    fn downdate_matches_direct_factorization() {
        let v = Vector3::new(0.3, -0.5, 0.4);
        let mut l = Cholesky::new(spd3()).unwrap().unpack();
        downdate(&mut l, v).unwrap();
        approx::assert_relative_eq!(
            l * l.transpose(),
            spd3() - v * v.transpose(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn downdate_rejects_an_indefinite_result() {
        let mut l = Matrix2::identity();
        assert!(downdate(&mut l, Vector2::new(1.0, 0.0)).is_none());
    }

    #[test]
    fn update_columns_matches_the_sum_of_outer_products() {
        let columns = nalgebra::Matrix3x2::new(1.0, 0.5, -2.0, 0.0, 0.3, 4.0);
        let mut l = Matrix3::zeros();
        update_columns(&mut l, &columns);
        approx::assert_relative_eq!(
            l * l.transpose(),
            columns * columns.transpose(),
            epsilon = 1e-12
        );
    }
}
