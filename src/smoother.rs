//! Rauch-Tung-Striebel smoother.
//!
//! The smoother runs backward over a history the caller records during forward filtering. It
//! works on slices, so the history can live in a `Vec` or, without `std`, in a fixed-size
//! array.
//!
//! It applies to the [`LinearKf`](crate::linear::LinearKf), and to the
//! [`Ekf`](crate::ekf::Ekf) if each step records the process Jacobian as `f`.

use nalgebra::{Cholesky, SMatrix, SVector};

use crate::error::KalmanError;
use crate::update::{all_finite, symmetrize};

/// A state estimate and its covariance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate<const N: usize> {
    /// State estimate.
    pub x: SVector<f64, N>,
    /// State covariance.
    pub p: SMatrix<f64, N, N>,
}

/// What the smoother needs from one step of forward filtering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtsStep<const N: usize> {
    /// The transition matrix that predicted this step from the previous one. Ignored for the
    /// first step.
    pub f: SMatrix<f64, N, N>,
    /// The estimate after predicting this step, before its measurement update. For the first
    /// step, the prior.
    pub predicted: Estimate<N>,
    /// The estimate after this step's measurement update. If a step had no measurement, this
    /// equals `predicted`.
    pub filtered: Estimate<N>,
}

/// Runs the RTS smoother backward over `steps`, writing the smoothed estimate of each step to
/// the same index of `out`.
///
/// Returns [`KalmanError::InvalidInput`] if `out` and `steps` differ in length, or
/// [`KalmanError::CovarianceNotPositiveDefinite`] if a predicted covariance can't be inverted.
/// The contents of `out` are unspecified after an error.
pub fn smooth<const N: usize>(
    steps: &[RtsStep<N>],
    out: &mut [Estimate<N>],
) -> Result<(), KalmanError> {
    if steps.len() != out.len() {
        return Err(KalmanError::InvalidInput);
    }
    let Some(last) = steps.last() else {
        return Ok(());
    };
    out[out.len() - 1] = last.filtered;

    for k in (0..steps.len() - 1).rev() {
        let (current, next) = (&steps[k], &steps[k + 1]);
        let smoothed_next = out[k + 1];

        // C = P_filt Fᵀ P_pred⁻¹, computed as (P_pred⁻¹ F P_filt)ᵀ since both are symmetric.
        let p_pred =
            Cholesky::new(next.predicted.p).ok_or(KalmanError::CovarianceNotPositiveDefinite)?;
        let c = p_pred.solve(&(next.f * current.filtered.p)).transpose();

        let x = current.filtered.x + c * (smoothed_next.x - next.predicted.x);
        let p = symmetrize(
            current.filtered.p + c * (smoothed_next.p - next.predicted.p) * c.transpose(),
        );
        if !all_finite(x.as_slice()) || !all_finite(p.as_slice()) {
            return Err(KalmanError::NumericalFailure);
        }
        out[k] = Estimate { x, p };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::LinearKf;
    #[cfg(feature = "std")]
    use nalgebra::{DMatrix, DVector};
    use nalgebra::{Matrix1, Matrix1x2, Matrix2, Vector1, Vector2};

    const STEPS: usize = 12;

    struct Problem {
        f: Matrix2<f64>,
        h: Matrix1x2<f64>,
        q: Matrix2<f64>,
        r: Matrix1<f64>,
        prior: Estimate<2>,
        zs: [Vector1<f64>; STEPS],
    }

    fn problem() -> Problem {
        let zs = core::array::from_fn(|k| Vector1::new((k as f64 * 0.7).sin() * 3.0 + k as f64));
        Problem {
            f: Matrix2::new(1.0, 0.5, 0.0, 1.0),
            h: Matrix1x2::new(1.0, 0.0),
            q: Matrix2::new(0.02, 0.01, 0.01, 0.05),
            r: Matrix1::new(0.4),
            prior: Estimate {
                x: Vector2::new(0.0, 1.0),
                p: Matrix2::new(2.0, 0.3, 0.3, 1.0),
            },
            zs,
        }
    }

    /// Filters forward, recording the history the smoother needs.
    fn filter(pr: &Problem) -> [RtsStep<2>; STEPS] {
        let mut kf = LinearKf::new(pr.prior.x, pr.prior.p);
        core::array::from_fn(|k| {
            if k > 0 {
                kf.predict(&pr.f, &pr.q);
            }
            let predicted = Estimate {
                x: *kf.state(),
                p: *kf.covariance(),
            };
            kf.update(&pr.h, &pr.zs[k], &pr.r).unwrap();
            RtsStep {
                f: pr.f,
                predicted,
                filtered: Estimate {
                    x: *kf.state(),
                    p: *kf.covariance(),
                },
            }
        })
    }

    /// Computes the exact posterior of all states given all measurements, by conditioning the
    /// joint Gaussian of states and measurements directly.
    #[cfg(feature = "std")]
    fn batch_posterior(pr: &Problem) -> (DVector<f64>, DMatrix<f64>) {
        let (n, d) = (STEPS, 2);

        // Prior means and covariances of each state, then the joint state covariance:
        // Cov(x_i, x_j) = P_i (F^(j-i))ᵀ for j ≥ i.
        let mut means = vec![pr.prior.x];
        let mut covs = vec![pr.prior.p];
        for k in 1..n {
            means.push(pr.f * means[k - 1]);
            covs.push(pr.f * covs[k - 1] * pr.f.transpose() + pr.q);
        }
        let mut mu = DVector::zeros(n * d);
        let mut sigma = DMatrix::zeros(n * d, n * d);
        for i in 0..n {
            mu.rows_mut(i * d, d).copy_from(&means[i]);
            let mut f_pow = Matrix2::identity();
            for j in i..n {
                let block = covs[i] * f_pow.transpose();
                sigma.view_mut((i * d, j * d), (d, d)).copy_from(&block);
                sigma
                    .view_mut((j * d, i * d), (d, d))
                    .copy_from(&block.transpose());
                f_pow = pr.f * f_pow;
            }
        }

        // Z = H_blk X + V.
        let mut h_blk = DMatrix::zeros(n, n * d);
        let mut z = DVector::zeros(n);
        for k in 0..n {
            h_blk.view_mut((k, k * d), (1, d)).copy_from(&pr.h);
            z[k] = pr.zs[k][0];
        }
        let s_zz = &h_blk * &sigma * h_blk.transpose() + DMatrix::identity(n, n) * pr.r[0];
        let gain = &sigma * h_blk.transpose() * s_zz.try_inverse().unwrap();
        let mean = &mu + &gain * (z - &h_blk * &mu);
        let cov = &sigma - &gain * &h_blk * &sigma;
        (mean, cov)
    }

    #[test]
    #[cfg(feature = "std")]
    fn matches_batch_posterior() {
        let pr = problem();
        let steps = filter(&pr);
        let mut out = [steps[0].filtered; STEPS];
        smooth(&steps, &mut out).unwrap();

        let (mean, cov) = batch_posterior(&pr);
        for (k, est) in out.iter().enumerate() {
            let expected_x = Vector2::from_column_slice(mean.rows(k * 2, 2).as_slice());
            let expected_p = Matrix2::from_fn(|i, j| cov[(k * 2 + i, k * 2 + j)]);
            approx::assert_relative_eq!(est.x, expected_x, epsilon = 1e-9);
            approx::assert_relative_eq!(est.p, expected_p, epsilon = 1e-9);
        }
    }

    #[test]
    fn last_step_equals_filtered_and_earlier_steps_tighten() {
        let pr = problem();
        let steps = filter(&pr);
        let mut out = [steps[0].filtered; STEPS];
        smooth(&steps, &mut out).unwrap();

        assert_eq!(out[STEPS - 1], steps[STEPS - 1].filtered);
        for k in 0..STEPS - 1 {
            assert!(out[k].p.trace() < steps[k].filtered.p.trace());
        }
    }

    #[test]
    fn empty_history_is_fine() {
        smooth::<2>(&[], &mut []).unwrap();
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        let pr = problem();
        let steps = filter(&pr);
        let mut out = [steps[0].filtered; STEPS - 1];
        assert_eq!(smooth(&steps, &mut out), Err(KalmanError::InvalidInput));
    }

    #[test]
    fn singular_predicted_covariance_is_rejected() {
        let pr = problem();
        let mut steps = filter(&pr);
        steps[3].predicted.p = Matrix2::zeros();
        let mut out = [steps[0].filtered; STEPS];
        assert_eq!(
            smooth(&steps, &mut out),
            Err(KalmanError::CovarianceNotPositiveDefinite)
        );
    }
}
