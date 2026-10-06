//! Our filters produce the same estimates as adskalman, the established baseline.
//!
//! For a linear-Gaussian problem the Kalman filter is optimal, so two correct implementations
//! must agree. Matching is the goal here, not beating.

mod common;

use common::baseline::{self, Method, Models};
use common::{scenarios, FromScenario};
use nalgebra::Cholesky;
use typed_kalman::linear::LinearKf;
use typed_kalman::smoother::{smooth, Estimate, RtsStep};
use typed_kalman::SqrtKf;

#[test]
fn matches_adskalman_on_constant_velocity() {
    let sc = scenarios::constant_velocity(1000, 42);
    let theirs = baseline::adskalman_filter(&sc);

    let mut ours = LinearKf::from_scenario(&sc);
    for (k, z) in sc.zs.iter().enumerate() {
        ours.predict(&sc.f, &sc.q);
        ours.update(&sc.h, z, &sc.r).unwrap();
        approx::assert_relative_eq!(ours.state(), theirs[k].state(), epsilon = 1e-9);
        approx::assert_relative_eq!(ours.covariance(), theirs[k].covariance(), epsilon = 1e-9);
    }
}

#[test]
fn matches_adskalman_on_random_problems() {
    for seed in 0..20 {
        let sc = scenarios::random_linear(seed, 1.0, 500);
        let theirs = baseline::adskalman_filter(&sc);

        let mut ours = LinearKf::from_scenario(&sc);
        for (k, z) in sc.zs.iter().enumerate() {
            ours.predict(&sc.f, &sc.q);
            ours.update(&sc.h, z, &sc.r).unwrap();
            approx::assert_relative_eq!(
                ours.state(),
                theirs[k].state(),
                epsilon = 1e-9,
                max_relative = 1e-9
            );
        }
    }
}

#[test]
fn optimal_gain_methods_agree_on_a_well_conditioned_problem() {
    // On an easy problem, adskalman's faster covariance updates should also agree with our
    // Joseph form. Only Joseph form stays this close on hard problems.
    let sc = scenarios::constant_velocity(1000, 7);
    let models = Models::<4, 2>::new(&sc);
    for method in [Method::OptimalKalman, Method::OptimalKalmanForcedSymmetric] {
        let mut ours = LinearKf::from_scenario(&sc);
        let mut zs = sc.zs.iter();
        models.run(&sc.zs, method, |theirs| {
            ours.predict(&sc.f, &sc.q);
            ours.update(&sc.h, zs.next().unwrap(), &sc.r).unwrap();
            approx::assert_relative_eq!(ours.state(), theirs.state(), epsilon = 1e-9);
        });
    }
}

#[test]
fn smoother_matches_adskalman() {
    let sc = scenarios::constant_velocity(300, 3);
    let theirs = Models::<4, 2>::new(&sc).smooth(&sc.zs);

    let mut kf = LinearKf::from_scenario(&sc);
    let history: Vec<RtsStep<4>> = sc
        .zs
        .iter()
        .map(|z| {
            kf.predict(&sc.f, &sc.q);
            let predicted = Estimate {
                x: *kf.state(),
                p: *kf.covariance(),
            };
            kf.update(&sc.h, z, &sc.r).unwrap();
            RtsStep {
                f: sc.f,
                predicted,
                filtered: Estimate {
                    x: *kf.state(),
                    p: *kf.covariance(),
                },
            }
        })
        .collect();
    let mut ours = vec![history[0].filtered; history.len()];
    smooth(&history, &mut ours).unwrap();

    for (o, t) in ours.iter().zip(&theirs) {
        approx::assert_relative_eq!(o.x, *t.state(), epsilon = 1e-9);
        approx::assert_relative_eq!(o.p, *t.covariance(), epsilon = 1e-9);
    }
}

#[test]
fn sqrt_kf_matches_adskalman() {
    for seed in [42, 43, 44] {
        let sc = scenarios::constant_velocity(1000, seed);
        let theirs = baseline::adskalman_filter(&sc);
        let q_sqrt = Cholesky::new(sc.q).unwrap().unpack();
        let r_sqrt = Cholesky::new(sc.r).unwrap().unpack();

        let mut ours = SqrtKf::new(sc.x0, sc.p0).unwrap();
        for (k, z) in sc.zs.iter().enumerate() {
            ours.predict(&sc.f, &q_sqrt).unwrap();
            ours.update(&sc.h, z, &r_sqrt).unwrap();
            approx::assert_relative_eq!(ours.state(), theirs[k].state(), epsilon = 1e-9);
            approx::assert_relative_eq!(ours.covariance(), *theirs[k].covariance(), epsilon = 1e-9);
        }
    }
}
