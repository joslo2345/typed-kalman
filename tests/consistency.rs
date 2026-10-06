//! Every filter's reported covariance matches its actual errors.
//!
//! Over 500 Monte Carlo runs with known truth, the average NEES and NIS at the final step must
//! fall inside their 95% chi-squared bounds.

mod common;

use common::scenarios;
use kalman_rs::diagnostics::{chi_squared_bounds, nees};
use kalman_rs::{Ekf, LinearKf, MeasurementJacobian, MeasurementModel, ProcessJacobian};
use kalman_rs::{KalmanError, ProcessModel, SqrtUkf, Ukf};
use nalgebra::{Cholesky, Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};

const RUNS: usize = 500;
const STEPS: usize = 50;

struct Linear {
    f: Matrix4<f64>,
    h: Matrix2x4<f64>,
}

impl ProcessModel<4> for Linear {
    fn predict(&self, x: &Vector4<f64>, _dt: f64) -> Vector4<f64> {
        self.f * x
    }
}

impl ProcessJacobian<4> for Linear {
    fn jacobian(&self, _x: &Vector4<f64>, _dt: f64) -> Matrix4<f64> {
        self.f
    }
}

impl MeasurementModel<4, 2> for Linear {
    fn measure(&self, x: &Vector4<f64>) -> Vector2<f64> {
        self.h * x
    }
}

impl MeasurementJacobian<4, 2> for Linear {
    fn jacobian(&self, _x: &Vector4<f64>) -> Matrix2x4<f64> {
        self.h
    }
}

/// One filter step on a scenario: predict, then update with `z`, returning the NIS.
type Step<F> = fn(&mut F, &scenarios::Scenario<4, 2>, &Vector2<f64>) -> Result<f64, KalmanError>;

/// Runs `RUNS` independent trials and asserts that the averaged final NEES and NIS are inside
/// their 95% bounds.
fn assert_consistent<F>(
    name: &str,
    new: impl Fn(&scenarios::Scenario<4, 2>) -> F,
    step: Step<F>,
    estimate: impl Fn(&F) -> (Vector4<f64>, Matrix4<f64>),
) {
    let (mut nees_sum, mut nis_sum) = (0.0, 0.0);
    for run in 0..RUNS {
        let sc = scenarios::constant_velocity(STEPS, 1000 + run as u64);
        let mut filter = new(&sc);
        let mut nis = 0.0;
        for z in &sc.zs {
            nis = step(&mut filter, &sc, z).unwrap();
        }
        let (x, p) = estimate(&filter);
        nees_sum += nees(sc.truth.last().unwrap(), &x, &p).unwrap();
        nis_sum += nis;
    }

    let (avg_nees, avg_nis) = (nees_sum / RUNS as f64, nis_sum / RUNS as f64);
    let nees_bounds = chi_squared_bounds(4, RUNS, 0.95).unwrap();
    let nis_bounds = chi_squared_bounds(2, RUNS, 0.95).unwrap();
    println!("report: {name} avg NEES = {avg_nees:.3}, avg NIS = {avg_nis:.3}");
    assert!(
        nees_bounds.contains(avg_nees),
        "{name}: NEES {avg_nees} outside {nees_bounds:?}"
    );
    assert!(
        nis_bounds.contains(avg_nis),
        "{name}: NIS {avg_nis} outside {nis_bounds:?}"
    );
}

fn model(sc: &scenarios::Scenario<4, 2>) -> Linear {
    Linear { f: sc.f, h: sc.h }
}

#[test]
fn linear_kf_is_consistent() {
    assert_consistent(
        "LinearKf",
        |sc| LinearKf::new(sc.x0, sc.p0),
        |kf, sc, z| {
            kf.predict(&sc.f, &sc.q);
            kf.update(&sc.h, z, &sc.r)
        },
        |kf| (*kf.state(), *kf.covariance()),
    );
}

#[test]
fn ekf_is_consistent() {
    assert_consistent(
        "Ekf",
        |sc| Ekf::new(sc.x0, sc.p0),
        |ekf, sc, z| {
            ekf.predict(&model(sc), &sc.q, 0.1);
            ekf.update(&model(sc), z, &sc.r)
        },
        |ekf| (*ekf.state(), *ekf.covariance()),
    );
}

#[test]
fn ukf_is_consistent() {
    assert_consistent(
        "Ukf",
        |sc| Ukf::new(sc.x0, sc.p0),
        |ukf, sc, z| {
            ukf.predict(&model(sc), &sc.q, 0.1)?;
            ukf.update(&model(sc), z, &sc.r)
        },
        |ukf| (*ukf.state(), *ukf.covariance()),
    );
}

#[test]
fn sqrt_ukf_is_consistent() {
    assert_consistent(
        "SqrtUkf",
        |sc| SqrtUkf::new(sc.x0, sc.p0).unwrap(),
        |sr, sc, z| {
            let q_sqrt = Cholesky::new(sc.q).unwrap().unpack();
            let r_sqrt: Matrix2<f64> = Cholesky::new(sc.r).unwrap().unpack();
            sr.predict(&model(sc), &q_sqrt, 0.1)?;
            sr.update(&model(sc), z, &r_sqrt)
        },
        |sr| (*sr.state(), sr.covariance()),
    );
}
