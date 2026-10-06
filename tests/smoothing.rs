//! The unscented RTS smoother: exact against the linear RTS smoother on a linear problem, and
//! better than filtering on a nonlinear one.

mod common;

use common::rng::Rng;
use common::{scenarios, FromScenario};
use nalgebra::{Cholesky, Matrix1, Matrix2, Matrix2x4, Matrix4, Vector1, Vector2, Vector4};
use typed_kalman::diagnostics::{chi_squared_bounds, nees};
use typed_kalman::smoother::{smooth, smooth_with_cross, CrossStep, Estimate, RtsStep};
use typed_kalman::{LinearKf, MeasurementModel, ProcessModel, Ukf};

struct Linear {
    f: Matrix4<f64>,
    h: Matrix2x4<f64>,
}

impl ProcessModel<4> for Linear {
    fn predict(&self, x: &Vector4<f64>, _dt: f64) -> Vector4<f64> {
        self.f * x
    }
}

impl MeasurementModel<4, 2> for Linear {
    fn measure(&self, x: &Vector4<f64>) -> Vector2<f64> {
        self.h * x
    }
}

fn estimate<const N: usize>(
    x: &nalgebra::SVector<f64, N>,
    p: &nalgebra::SMatrix<f64, N, N>,
) -> Estimate<N> {
    Estimate { x: *x, p: *p }
}

#[test]
fn unscented_smoother_matches_rts_on_a_linear_problem() {
    let sc = scenarios::constant_velocity(300, 3);
    let model = Linear { f: sc.f, h: sc.h };

    let mut kf = LinearKf::from_scenario(&sc);
    let mut ukf = Ukf::new(sc.x0, sc.p0);
    let (mut rts, mut cross_steps) = (Vec::new(), Vec::new());
    for z in &sc.zs {
        let before = *ukf.covariance();
        kf.predict(&sc.f, &sc.q);
        let cross = ukf.predict_with_cross(&model, &sc.q, 0.1).unwrap();
        // For a linear transition the cross-covariance is P Fᵀ.
        approx::assert_relative_eq!(cross, before * sc.f.transpose(), epsilon = 1e-9);
        let (kf_pred, ukf_pred) = (
            estimate(kf.state(), kf.covariance()),
            estimate(ukf.state(), ukf.covariance()),
        );
        kf.update(&sc.h, z, &sc.r).unwrap();
        ukf.update(&model, z, &sc.r).unwrap();
        rts.push(RtsStep {
            f: sc.f,
            predicted: kf_pred,
            filtered: estimate(kf.state(), kf.covariance()),
        });
        cross_steps.push(CrossStep {
            cross,
            predicted: ukf_pred,
            filtered: estimate(ukf.state(), ukf.covariance()),
        });
    }

    let mut from_rts = vec![rts[0].filtered; rts.len()];
    let mut from_cross = vec![cross_steps[0].filtered; cross_steps.len()];
    smooth(&rts, &mut from_rts).unwrap();
    smooth_with_cross(&cross_steps, &mut from_cross).unwrap();
    for (a, b) in from_rts.iter().zip(&from_cross) {
        approx::assert_relative_eq!(a.x, b.x, epsilon = 1e-9);
        approx::assert_relative_eq!(a.p, b.p, epsilon = 1e-9);
    }
}

/// A pendulum, `[angle, angular velocity]`, measured by its bob's horizontal position.
struct Pendulum;

const DT: f64 = 0.02;

impl ProcessModel<2> for Pendulum {
    fn predict(&self, x: &Vector2<f64>, dt: f64) -> Vector2<f64> {
        Vector2::new(x[0] + x[1] * dt, x[1] - 9.81 * x[0].sin() * dt)
    }
}

impl MeasurementModel<2, 1> for Pendulum {
    fn measure(&self, x: &Vector2<f64>) -> Vector1<f64> {
        Vector1::new(x[0].sin())
    }
}

#[test]
fn unscented_smoother_beats_filtering_on_a_pendulum() {
    const RUNS: usize = 50;
    const STEPS: usize = 300;
    let q = Matrix2::new(1e-6, 0.0, 0.0, 1e-4);
    let r = Matrix1::new(0.01);
    let p0 = Matrix2::identity() * 0.05;
    let (q_l, r_l, p0_l) = (
        Cholesky::new(q).unwrap().unpack(),
        Cholesky::new(r).unwrap().unpack(),
        Cholesky::new(p0).unwrap().unpack(),
    );

    let (mut filtered_sq, mut smoothed_sq, mut smoothed_nees) = (0.0, 0.0, 0.0);
    for run in 0..RUNS {
        let mut rng = Rng::new(100 + run as u64);
        let x0 = Vector2::new(0.6, 0.0);
        let mut truth = x0 + rng.correlated(&p0_l);
        let mut ukf = Ukf::new(x0, p0);
        let (mut steps, mut truths) = (Vec::new(), Vec::new());
        for _ in 0..STEPS {
            truth = Pendulum.predict(&truth, DT) + rng.correlated(&q_l);
            let z = Pendulum.measure(&truth) + rng.correlated(&r_l);
            let cross = ukf.predict_with_cross(&Pendulum, &q, DT).unwrap();
            let predicted = estimate(ukf.state(), ukf.covariance());
            ukf.update(&Pendulum, &z, &r).unwrap();
            steps.push(CrossStep {
                cross,
                predicted,
                filtered: estimate(ukf.state(), ukf.covariance()),
            });
            truths.push(truth);
        }
        let mut smoothed = vec![steps[0].filtered; STEPS];
        smooth_with_cross(&steps, &mut smoothed).unwrap();
        for k in 0..STEPS {
            filtered_sq += (steps[k].filtered.x - truths[k]).norm_squared();
            smoothed_sq += (smoothed[k].x - truths[k]).norm_squared();
            smoothed_nees += nees(&truths[k], &smoothed[k].x, &smoothed[k].p).unwrap();
        }
    }

    let n = (RUNS * STEPS) as f64;
    let (filtered_rmse, smoothed_rmse) = ((filtered_sq / n).sqrt(), (smoothed_sq / n).sqrt());
    let average_nees = smoothed_nees / n;
    println!(
        "report: pendulum UKF RMSE filtered {filtered_rmse:.4}, smoothed {smoothed_rmse:.4}, \
         smoothed NEES {average_nees:.2}"
    );
    assert!(
        smoothed_rmse < 0.8 * filtered_rmse,
        "smoothing didn't help: RMSE {smoothed_rmse} vs filtered {filtered_rmse}"
    );
    // The smoothed covariance must be honest about the smaller error. Steps within a run are
    // correlated, so bound with the number of runs, not of steps.
    let bounds = chi_squared_bounds(2, RUNS, 0.99).unwrap();
    assert!(
        bounds.contains(average_nees),
        "smoothed NEES {average_nees} outside {bounds:?}"
    );
}
