//! Our covariance stays symmetric positive-definite on problems that break naive filters.

mod common;

use common::naive::NaiveKf;
use common::rng::Rng;
use common::{scenarios, FromScenario};
use kalman_rs::linear::LinearKf;
use kalman_rs::Float;
use kalman_rs::SqrtKf;
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// The guide's full range, which `LinearKf` can't cover: the square-root filter's factor
    /// has the square root of P's condition number, so it stays representable. A strictly
    /// positive diagonal on the lower-triangular factor proves `S Sᵀ` positive-definite.
    #[test]
    fn sqrt_kf_factor_stays_valid_over_the_full_range(
        seed in any::<u64>(),
        log_r in -12.0f64..2.0,
    ) {
        let sc = scenarios::random_linear(seed, 10f64.powf(log_r), 2000);
        let q_sqrt = Cholesky::new(sc.q).unwrap().unpack();
        let r_sqrt = Cholesky::new(sc.r).unwrap().unpack();
        let mut kf = SqrtKf::new(sc.x0, sc.p0).unwrap();

        for (step, z) in sc.zs.iter().enumerate() {
            let result = kf.predict(&sc.f, &q_sqrt).and_then(|_| kf.update(&sc.h, z, &r_sqrt));
            prop_assert!(result.is_ok(), "step {}: {:?}", step, result);

            let s = kf.sqrt_covariance();
            prop_assert!(s.iter().all(|v| v.is_finite()));
            prop_assert!((0..4).all(|i| s[(i, i)] > 0.0), "step {}: diagonal {}", step, s.diagonal());
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

/// A case found by proptest where P's condition number grows past 1e17, so Cholesky fails for
/// `LinearKf`, the naive filter and adskalman alike (each within a few hundred steps). The square-root filter
/// must get through all 2,000 steps.
#[test]
fn sqrt_kf_survives_a_case_that_breaks_covariance_filters() {
    let sc = scenarios::random_linear(
        6_892_070_046_071_684_633,
        10f64.powf(-11.843_856_017_616_23),
        2000,
    );
    let q_sqrt = Cholesky::new(sc.q).unwrap().unpack();
    let r_sqrt = Cholesky::new(sc.r).unwrap().unpack();

    let mut sr = SqrtKf::new(sc.x0, sc.p0).unwrap();
    let mut kf = LinearKf::from_scenario(&sc);
    let mut kf_failed_at = None;
    for (step, z) in sc.zs.iter().enumerate() {
        sr.predict(&sc.f, &q_sqrt).unwrap();
        sr.update(&sc.h, z, &r_sqrt)
            .unwrap_or_else(|e| panic!("SqrtKf failed at step {step}: {e}"));
        let s = sr.sqrt_covariance();
        assert!(
            (0..4).all(|i| s[(i, i)] > 0.0),
            "SqrtKf factor degenerate at step {step}"
        );

        if kf_failed_at.is_none() {
            kf.predict(&sc.f, &sc.q);
            if kf.update(&sc.h, z, &sc.r).is_err() || kf.covariance().cholesky().is_none() {
                kf_failed_at = Some(step);
            }
        }
    }
    println!("report: f64 extreme case: LinearKf first non-PD at {kf_failed_at:?}, SqrtKf ok");
}

/// A million single-precision steps of a problem whose covariance exceeds `f32`'s range
/// (`scenarios::beyond_f32`). `LinearKf` loses positive-definiteness almost at once; `SqrtKf`
/// must not.
#[test]
fn sqrt_kf_handles_what_f32_covariance_filters_cannot() {
    const STEPS: usize = 1_000_000;
    let sc = scenarios::beyond_f32();
    let (f, h, q, r) = (
        sc.f.cast::<f32>(),
        sc.h.cast::<f32>(),
        sc.q.cast::<f32>(),
        sc.r.cast::<f32>(),
    );
    let q_sqrt = Cholesky::new(sc.q).unwrap().unpack().cast::<f32>();
    let r_sqrt = Cholesky::new(sc.r).unwrap().unpack().cast::<f32>();

    let mut sr = SqrtKf::new(sc.x0.cast::<f32>(), sc.p0.cast::<f32>()).unwrap();
    let mut kf = LinearKf::new(sc.x0.cast::<f32>(), sc.p0.cast::<f32>());
    let mut kf_failed_at = None;

    let mut rng = Rng::new(4);
    let mut sim = sc.simulator(&mut rng);
    for step in 1..=STEPS {
        let (_, z) = sim.next_step(&mut rng);
        let z = z.cast::<f32>();

        sr.predict(&f, &q_sqrt)
            .and_then(|_| sr.update(&h, &z, &r_sqrt))
            .unwrap_or_else(|e| panic!("SqrtKf failed at step {step}: {e}"));
        let s = sr.sqrt_covariance();
        assert!(
            s.iter().all(|v| v.is_finite()) && (0..4).all(|i| s[(i, i)] > 0.0),
            "SqrtKf factor degenerate at step {step}"
        );

        if kf_failed_at.is_none() {
            kf.predict(&f, &q);
            if kf.update(&h, &z, &r).is_err() || !is_spd(kf.covariance(), 1e-4) {
                kf_failed_at = Some(step);
            }
        }
    }
    match kf_failed_at {
        Some(step) => println!("report: beyond-f32 LinearKf steps_to_failure = {step}"),
        None => println!("report: beyond-f32 LinearKf stayed stable for {STEPS} steps"),
    }
    println!("report: beyond-f32 SqrtKf steps_to_failure > {STEPS}");
    assert!(
        kf_failed_at.is_some(),
        "beyond_f32 no longer breaks LinearKf, so it doesn't show what SqrtKf adds"
    );
}
