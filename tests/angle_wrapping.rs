//! Bearing measurements across the ±π seam.
//!
//! A target passes behind the sensor along the negative x-axis, so its bearing jumps from +π
//! to -π halfway through. With a wrapping `residual`, every nonlinear filter tracks it
//! smoothly; without one, the innovation near the seam is about 2π and the filter is thrown off.

mod common;

use common::rng::Rng;
use nalgebra::{Cholesky, Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};
use std::f64::consts::PI;
use typed_kalman::diagnostics::chi_squared_bounds;
use typed_kalman::{
    wrap_angle, Ekf, KalmanError, MeasurementJacobian, MeasurementModel, ProcessJacobian,
    ProcessModel, SqrtUkf, Ukf,
};

const STEPS: usize = 100;
const DT: f64 = 1.0;

/// Constant velocity, measured by range and bearing from the origin. `wrap` chooses whether
/// the residual wraps the bearing.
struct Tracker {
    f: Matrix4<f64>,
    wrap: bool,
}

impl ProcessModel<4> for Tracker {
    fn predict(&self, x: &Vector4<f64>, _dt: f64) -> Vector4<f64> {
        self.f * x
    }
}

impl ProcessJacobian<4> for Tracker {
    fn jacobian(&self, _x: &Vector4<f64>, _dt: f64) -> Matrix4<f64> {
        self.f
    }
}

impl MeasurementModel<4, 2> for Tracker {
    fn measure(&self, x: &Vector4<f64>) -> Vector2<f64> {
        Vector2::new(x[0].hypot(x[1]), x[1].atan2(x[0]))
    }

    fn residual(&self, a: &Vector2<f64>, b: &Vector2<f64>) -> Vector2<f64> {
        let d = a - b;
        if self.wrap {
            Vector2::new(d[0], wrap_angle(d[1]))
        } else {
            d
        }
    }
}

impl MeasurementJacobian<4, 2> for Tracker {
    fn jacobian(&self, x: &Vector4<f64>) -> Matrix2x4<f64> {
        let r2 = x[0] * x[0] + x[1] * x[1];
        let r = r2.sqrt();
        #[rustfmt::skip]
        let h = Matrix2x4::new(
            x[0] / r,  x[1] / r, 0.0, 0.0,
            -x[1] / r2, x[0] / r2, 0.0, 0.0,
        );
        h
    }
}

struct Problem {
    f: Matrix4<f64>,
    q: Matrix4<f64>,
    r: Matrix2<f64>,
    x0: Vector4<f64>,
    p0: Matrix4<f64>,
    truth: Vec<Vector4<f64>>,
    zs: Vec<Vector2<f64>>,
}

/// A target at x = -5000 m moving from y = +300 m to y = -300 m, so its bearing crosses ±π
/// at step 50.
fn problem() -> Problem {
    #[rustfmt::skip]
    let f = Matrix4::new(
        1.0, 0.0, DT,  0.0,
        0.0, 1.0, 0.0, DT,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    let q = Matrix4::from_diagonal(&Vector4::new(1e-4, 1e-4, 1e-4, 1e-4));
    let r = Matrix2::new(25.0, 0.0, 0.0, 4e-6); // 5 m, 2 mrad
    let p0 = Matrix4::from_diagonal(&Vector4::new(100.0, 100.0, 1.0, 1.0));

    let mut rng = Rng::new(7);
    let q_l = Cholesky::new(q).unwrap().unpack();
    let r_l = Cholesky::new(r).unwrap().unpack();
    let p0_l = Cholesky::new(p0).unwrap().unpack();
    let mut x = Vector4::new(-5000.0, 300.0, 0.0, -6.0);
    let x0 = x + rng.correlated(&p0_l);
    let tracker = Tracker { f, wrap: true };
    let (mut truth, mut zs) = (Vec::new(), Vec::new());
    for _ in 0..STEPS {
        x = f * x + rng.correlated(&q_l);
        let z = tracker.measure(&x) + rng.correlated(&r_l);
        zs.push(Vector2::new(z[0], wrap_angle(z[1])));
        truth.push(x);
    }
    Problem {
        f,
        q,
        r,
        x0,
        p0,
        truth,
        zs,
    }
}

/// Runs one filter step (predict, then update) and returns the NIS.
type Step<'a> = Box<dyn FnMut(&Tracker, &Vector2<f64>) -> Result<f64, KalmanError> + 'a>;

/// Runs a filter over the problem, returning each step's NIS and the final position error.
fn run(
    pr: &Problem,
    wrap: bool,
    mut step: Step,
    state: impl Fn() -> Vector4<f64>,
) -> (Vec<f64>, f64) {
    let model = Tracker { f: pr.f, wrap };
    let nis: Vec<f64> = pr
        .zs
        .iter()
        .map(|z| step(&model, z).unwrap_or(f64::INFINITY))
        .collect();
    let error = (state().fixed_rows::<2>(0) - pr.truth[STEPS - 1].fixed_rows::<2>(0)).norm();
    (nis, error)
}

/// Asserts a filter tracked the target consistently across the seam.
fn assert_tracks(name: &str, nis: &[f64], error: f64) {
    let average = nis.iter().sum::<f64>() / nis.len() as f64;
    let bounds = chi_squared_bounds(2, nis.len(), 0.99).unwrap();
    assert!(
        bounds.contains(average),
        "{name}: average NIS {average} outside {bounds:?}"
    );
    assert!(error < 50.0, "{name}: final position error {error} m");
}

#[test]
fn the_problem_crosses_the_seam() {
    let pr = problem();
    let bearings: Vec<f64> = pr.zs.iter().map(|z| z[1]).collect();
    assert!(bearings[..STEPS / 2 - 5].iter().all(|&b| b > PI - 0.1));
    assert!(bearings[STEPS / 2 + 5..].iter().all(|&b| b < -PI + 0.1));
}

#[test]
fn every_filter_tracks_across_the_seam_with_a_wrapping_residual() {
    let pr = problem();

    let ekf = std::cell::RefCell::new(Ekf::new(pr.x0, pr.p0));
    let (nis, error) = run(
        &pr,
        true,
        Box::new(|m, z| {
            let mut f = ekf.borrow_mut();
            f.predict(m, &pr.q, DT);
            f.update(m, z, &pr.r)
        }),
        || *ekf.borrow().state(),
    );
    assert_tracks("Ekf", &nis, error);

    let ukf = std::cell::RefCell::new(Ukf::new(pr.x0, pr.p0));
    let (nis, error) = run(
        &pr,
        true,
        Box::new(|m, z| {
            let mut f = ukf.borrow_mut();
            f.predict(m, &pr.q, DT)?;
            f.update(m, z, &pr.r)
        }),
        || *ukf.borrow().state(),
    );
    assert_tracks("Ukf", &nis, error);

    let (q_sqrt, r_sqrt) = (
        Cholesky::new(pr.q).unwrap().unpack(),
        Cholesky::new(pr.r).unwrap().unpack(),
    );
    let sr = std::cell::RefCell::new(SqrtUkf::new(pr.x0, pr.p0).unwrap());
    let (nis, error) = run(
        &pr,
        true,
        Box::new(|m, z| {
            let mut f = sr.borrow_mut();
            f.predict(m, &q_sqrt, DT)?;
            f.update(m, z, &r_sqrt)
        }),
        || *sr.borrow().state(),
    );
    assert_tracks("SqrtUkf", &nis, error);
}

#[test]
fn without_wrapping_the_seam_throws_the_filter_off() {
    // The control case: shows the test above really exercises the seam.
    let pr = problem();
    let ukf = std::cell::RefCell::new(Ukf::new(pr.x0, pr.p0));
    let (nis, _) = run(
        &pr,
        false,
        Box::new(|m, z| {
            let mut f = ukf.borrow_mut();
            f.predict(m, &pr.q, DT)?;
            f.update(m, z, &pr.r)
        }),
        || *ukf.borrow().state(),
    );
    let worst = nis.iter().cloned().fold(0.0, f64::max);
    assert!(worst > 1e6, "largest NIS without wrapping was only {worst}");
}

#[test]
fn wrap_angle_maps_into_minus_pi_to_pi() {
    for (angle, expected) in [
        (0.0, 0.0),
        (PI - 0.1, PI - 0.1),
        (PI + 0.1, -PI + 0.1),
        (-PI - 0.1, PI - 0.1),
        (3.0 * PI, -PI),
        (-7.0 * PI + 0.5, -PI + 0.5),
    ] {
        let got = wrap_angle(angle);
        assert!(
            (got - expected).abs() < 1e-12,
            "wrap_angle({angle}) = {got}, expected {expected}"
        );
        assert!((-PI..PI).contains(&got));
    }
    // Works in f32 too.
    assert!((wrap_angle(3.5f32) - (3.5 - 2.0 * std::f32::consts::PI)).abs() < 1e-6);
}
