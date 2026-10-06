//! NIS and NEES consistency diagnostics.
//!
//! A consistent filter's errors match the covariance it reports. Two statistics check this:
//!
//! - The normalized innovation squared (NIS), returned by every filter's `update`, is
//!   chi-squared with `M` degrees of freedom for an `M`-dimensional measurement.
//! - The normalized estimation error squared ([`nees`]), which needs the true state, is
//!   chi-squared with `N` degrees of freedom for an `N`-dimensional state.
//!
//! Average either statistic over Monte Carlo runs (or time steps) and compare the average with
//! [`chi_squared_bounds`].

use nalgebra::{Cholesky, ComplexField, SMatrix, SVector};

use crate::error::KalmanError;

/// Returns the normalized estimation error squared, `(x_true - x_est)ᵀ P⁻¹ (x_true - x_est)`.
///
/// Returns [`KalmanError::CovarianceNotPositiveDefinite`] if `p` can't be inverted.
pub fn nees<const N: usize>(
    x_true: &SVector<f64, N>,
    x_est: &SVector<f64, N>,
    p: &SMatrix<f64, N, N>,
) -> Result<f64, KalmanError> {
    let chol = Cholesky::new(*p).ok_or(KalmanError::CovarianceNotPositiveDefinite)?;
    let e = x_true - x_est;
    Ok(e.dot(&chol.solve(&e)))
}

/// A two-sided acceptance interval for an averaged consistency statistic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsistencyBounds {
    /// Lower bound of the interval.
    pub lower: f64,
    /// Upper bound of the interval.
    pub upper: f64,
}

impl ConsistencyBounds {
    /// Returns whether `value` lies inside the interval, bounds included.
    pub fn contains(&self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }
}

/// Returns the interval that the average of `runs` independent chi-squared statistics, each
/// with `dof` degrees of freedom, falls inside with probability `confidence`.
///
/// For example, `chi_squared_bounds(4, 200, 0.95)` bounds the average NEES of a 4-state filter
/// over 200 Monte Carlo runs. Returns [`KalmanError::InvalidInput`] if `dof` or `runs` is zero,
/// or `confidence` isn't strictly between 0 and 1.
pub fn chi_squared_bounds(
    dof: usize,
    runs: usize,
    confidence: f64,
) -> Result<ConsistencyBounds, KalmanError> {
    let valid = runs > 0 && confidence > 0.0 && confidence < 1.0;
    if !valid {
        return Err(KalmanError::InvalidInput);
    }
    // The sum of the statistics is chi-squared with dof × runs degrees of freedom. Multiplying
    // as floats avoids integer overflow. A zero `dof` is rejected by the quantile.
    let runs = runs as f64;
    let total_dof = dof as f64 * runs;
    Ok(ConsistencyBounds {
        lower: chi_squared_quantile(total_dof, (1.0 - confidence) / 2.0)? / runs,
        upper: chi_squared_quantile(total_dof, (1.0 + confidence) / 2.0)? / runs,
    })
}

/// Returns the value below which a chi-squared variable with `dof` degrees of freedom falls
/// with probability `p`.
///
/// Returns [`KalmanError::InvalidInput`] if `dof` isn't positive and finite, or `p` isn't
/// strictly between 0 and 1.
pub fn chi_squared_quantile(dof: f64, p: f64) -> Result<f64, KalmanError> {
    // Written as a positive check so NaN arguments are rejected too.
    let valid = dof > 0.0 && dof.is_finite() && p > 0.0 && p < 1.0;
    if !valid {
        return Err(KalmanError::InvalidInput);
    }
    let cdf = |x: f64| regularized_lower_gamma(dof / 2.0, x / 2.0);

    // Bracket the quantile, then bisect. The CDF is monotonic, so this always converges.
    let mut lo = 0.0;
    let mut hi = dof.max(1.0);
    while cdf(hi) < p {
        lo = hi;
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-14 * hi {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Returns the regularized lower incomplete gamma function `P(a, x)`, using a series for
/// `x < a + 1` and a continued fraction otherwise (Numerical Recipes, section 6.2).
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    const EPS: f64 = 1e-15;
    const MAX_ITER: usize = 10_000;
    const TINY: f64 = 1e-300;

    if x <= 0.0 {
        return 0.0;
    }
    let prefactor = ComplexField::exp(-x + a * ComplexField::ln(x) - ln_gamma(a));

    if x < a + 1.0 {
        let (mut ap, mut term, mut sum) = (a, 1.0 / a, 1.0 / a);
        for _ in 0..MAX_ITER {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if ComplexField::abs(term) < ComplexField::abs(sum) * EPS {
                break;
            }
        }
        sum * prefactor
    } else {
        // Modified Lentz's method for the continued fraction of Q(a, x) = 1 - P(a, x).
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..MAX_ITER {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if ComplexField::abs(d) < TINY {
                d = TINY;
            }
            c = b + an / c;
            if ComplexField::abs(c) < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if ComplexField::abs(delta - 1.0) < EPS {
                break;
            }
        }
        1.0 - prefactor * h
    }
}

/// Returns `ln Γ(x)` for `x > 0`, using the Lanczos approximation (g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const HALF_LN_TWO_PI: f64 = 0.918_938_533_204_672_8;

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }
    let t = x + G + 0.5;
    HALF_LN_TWO_PI + (x + 0.5) * ComplexField::ln(t) - t + ComplexField::ln(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::LinearKf;
    use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};

    #[test]
    fn nees_matches_hand_calculation() {
        let p = Matrix2::new(4.0, 0.0, 0.0, 1.0);
        let value = nees(&Vector2::new(2.0, 1.0), &Vector2::zeros(), &p).unwrap();
        approx::assert_relative_eq!(value, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn nees_rejects_a_singular_covariance() {
        let result = nees(&Vector2::zeros(), &Vector2::zeros(), &Matrix2::zeros());
        assert_eq!(result, Err(KalmanError::CovarianceNotPositiveDefinite));
    }

    #[test]
    fn ln_gamma_matches_factorials() {
        // Γ(n) = (n - 1)!
        approx::assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-14);
        approx::assert_relative_eq!(ln_gamma(10.0), ComplexField::ln(362_880.0), epsilon = 1e-12);
        // Γ(1/2) = √π
        approx::assert_relative_eq!(
            ln_gamma(0.5),
            0.5 * ComplexField::ln(core::f64::consts::PI),
            epsilon = 1e-14
        );
    }

    #[test]
    fn quantiles_match_published_tables() {
        // (dof, p, quantile), from standard chi-squared tables.
        let table = [
            (1.0, 0.025, 0.000_982_069),
            (1.0, 0.975, 5.023_886),
            (4.0, 0.025, 0.484_418_6),
            (4.0, 0.975, 11.143_29),
            (10.0, 0.025, 3.246_973),
            (10.0, 0.975, 20.483_18),
            (100.0, 0.025, 74.221_93),
            (100.0, 0.975, 129.561_2),
        ];
        for (dof, p, expected) in table {
            let q = chi_squared_quantile(dof, p).unwrap();
            approx::assert_relative_eq!(q, expected, max_relative = 1e-6);
        }
    }

    #[test]
    fn bounds_scale_with_runs() {
        let bounds = chi_squared_bounds(2, 50, 0.95).unwrap();
        approx::assert_relative_eq!(bounds.lower, 74.221_93 / 50.0, max_relative = 1e-6);
        approx::assert_relative_eq!(bounds.upper, 129.561_2 / 50.0, max_relative = 1e-6);
        assert!(bounds.contains(2.0));
        assert!(!bounds.contains(3.0));
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        assert_eq!(
            chi_squared_quantile(0.0, 0.5),
            Err(KalmanError::InvalidInput)
        );
        assert_eq!(
            chi_squared_quantile(2.0, 1.0),
            Err(KalmanError::InvalidInput)
        );
        assert_eq!(
            chi_squared_bounds(0, 10, 0.95),
            Err(KalmanError::InvalidInput)
        );
        assert_eq!(
            chi_squared_bounds(2, 0, 0.95),
            Err(KalmanError::InvalidInput)
        );
        assert_eq!(
            chi_squared_bounds(2, 10, f64::NAN),
            Err(KalmanError::InvalidInput)
        );
    }

    /// A small deterministic Gaussian generator (xorshift64* and Box-Muller), so the
    /// Monte Carlo test needs no extra dependencies.
    struct Gaussian(u64);

    impl Gaussian {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
            (bits as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn sample(&mut self) -> f64 {
            let (u1, u2) = (self.uniform(), self.uniform());
            ComplexField::sqrt(-2.0 * ComplexField::ln(u1))
                * ComplexField::cos(2.0 * core::f64::consts::PI * u2)
        }

        /// Draws from N(0, L Lᵀ).
        fn vector<const D: usize>(&mut self, l: &SMatrix<f64, D, D>) -> SVector<f64, D> {
            l * SVector::<f64, D>::from_fn(|_, _| self.sample())
        }
    }

    #[test]
    fn linear_kf_is_consistent_in_monte_carlo() {
        const RUNS: usize = 200;
        const STEPS: usize = 50;
        let dt = 0.1;
        #[rustfmt::skip]
        let f = Matrix4::new(
            1.0, 0.0, dt,  0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let h = Matrix2x4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        );
        let q = Matrix4::from_diagonal(&Vector4::new(1e-4, 1e-4, 1e-2, 1e-2));
        let r = Matrix2::identity() * 0.25;
        let p0 = Matrix4::identity();
        let chol = |m: Matrix4<f64>| Cholesky::new(m).unwrap().unpack();
        let (q_l, p0_l) = (chol(q), chol(p0));
        let r_l = Cholesky::new(r).unwrap().unpack();

        let mut rng = Gaussian(0x9e37_79b9_7f4a_7c15);
        let (mut nees_sum, mut nis_sum) = (0.0, 0.0);
        for _ in 0..RUNS {
            let mut truth = rng.vector(&p0_l);
            let mut kf = LinearKf::new(Vector4::zeros(), p0);
            let mut nis = 0.0;
            for _ in 0..STEPS {
                truth = f * truth + rng.vector(&q_l);
                let z = h * truth + rng.vector(&r_l);
                kf.predict(&f, &q);
                nis = kf.update(&h, &z, &r).unwrap();
            }
            nees_sum += nees(&truth, kf.state(), kf.covariance()).unwrap();
            nis_sum += nis;
        }

        let nees_bounds = chi_squared_bounds(4, RUNS, 0.95).unwrap();
        let nis_bounds = chi_squared_bounds(2, RUNS, 0.95).unwrap();
        let (avg_nees, avg_nis) = (nees_sum / RUNS as f64, nis_sum / RUNS as f64);
        assert!(
            nees_bounds.contains(avg_nees),
            "NEES {avg_nees} outside {nees_bounds:?}"
        );
        assert!(
            nis_bounds.contains(avg_nis),
            "NIS {avg_nis} outside {nis_bounds:?}"
        );
    }
}
