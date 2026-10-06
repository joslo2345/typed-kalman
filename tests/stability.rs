//! Our covariance stays symmetric positive-definite on problems that break naive filters.

mod common;

use common::naive::NaiveKf;
use common::rng::Rng;
use common::{scenarios, FromScenario};
use kalman_rs::linear::LinearKf;
use kalman_rs::Float;
use nalgebra::{Cholesky, SMatrix};
use proptest::prelude::*;

/// Returns whether `p` is finite, symmetric to within `tol` relative to its largest entry, and
/// positive-definite.
fn is_spd<const N: usize, T: Float>(p: &SMatrix<T, N, N>, tol: T) -> bool {
    let finite = p.iter().all(|v| v.is_finite());
    let asymmetry = (p - p.transpose()).abs().max();
    finite && asymmetry <= tol * p.abs().max() && Cholesky::new(*p).is_some()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// The guide asks for `log_r` down to -12, but at that scale the covariance itself reaches
    /// condition numbers around 1e17, beyond what `f64` can resolve, and Cholesky fails on any
    /// filter that stores P (adskalman's Joseph form fails first). Down to -8 the condition
    /// number stays below about 1e12, so this tests the filter rather than the number format.
    #[test]
    fn covariance_stays_symmetric_positive_definite(
        seed in any::<u64>(),
        log_r in -8.0f64..2.0,
    ) {
        let sc = scenarios::random_linear(seed, 10f64.powf(log_r), 2000);
        let mut kf = LinearKf::from_scenario(&sc);

        for z in &sc.zs {
            kf.predict(&sc.f, &sc.q);
            kf.update(&sc.h, z, &sc.r).unwrap();

            let p = kf.covariance();
            prop_assert!((p - p.transpose()).abs().max() < 1e-10);
            prop_assert!(p.cholesky().is_some());
        }
    }
}

/// Runs a million single-precision steps of the ill-conditioned scenario (S4). Our filter must
/// stay stable throughout. The naive filter's first failure, if any, is printed for the
/// comparison report (run with `--nocapture` to see it).
///
/// The square-root UKF isn't included: in `f32`, once the state's magnitude grows enough that
/// the sigma-point spread falls below its resolution, every point rounds to the mean and the
/// filter reports `CovarianceNotPositiveDefinite`. That's a limit of sigma-point filters in low
/// precision, not of the covariance update.
#[test]
fn f32_million_steps() {
    const STEPS: usize = 1_000_000;
    let sc = scenarios::ill_conditioned();
    let (f, h, q, r) = (
        sc.f.cast::<f32>(),
        sc.h.cast::<f32>(),
        sc.q.cast::<f32>(),
        sc.r.cast::<f32>(),
    );
    let (x0, p0) = (sc.x0.cast::<f32>(), sc.p0.cast::<f32>());
    let tol = 1e-4f32;

    let mut ours = LinearKf::new(x0, p0);
    let mut naive = NaiveKf { x: x0, p: p0 };
    let mut naive_failed_at = None;

    let mut rng = Rng::new(4);
    let mut sim = sc.simulator(&mut rng);
    for step in 1..=STEPS {
        let (_, z) = sim.next_step(&mut rng);
        let z = z.cast::<f32>();

        ours.predict(&f, &q);
        ours.update(&h, &z, &r)
            .unwrap_or_else(|e| panic!("LinearKf failed at step {step}: {e}"));
        assert!(
            is_spd(ours.covariance(), tol),
            "LinearKf lost SPD at step {step}"
        );

        if naive_failed_at.is_none() {
            naive.predict(&f, &q);
            if !naive.update(&h, &z, &r) || !is_spd(&naive.p, tol) {
                naive_failed_at = Some(step);
            }
        }
    }

    match naive_failed_at {
        Some(step) => println!("report: S4 naive steps_to_failure = {step}"),
        None => println!("report: S4 naive stayed stable for {STEPS} steps"),
    }
    println!("report: S4 kalman-rs steps_to_failure > {STEPS}");
    assert!(
        naive_failed_at.is_some(),
        "S4 no longer breaks the naive filter, so it doesn't test stability"
    );
}
