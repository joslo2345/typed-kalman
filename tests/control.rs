//! Known control inputs: `predict_with_input` on the matrix filters, and `WithInput` for every
//! model-based filter.

use nalgebra::{Cholesky, Matrix1, Matrix1x2, Matrix2, Matrix2x1, Vector1, Vector2};
use typed_kalman::{
    wrap_angle, ControlledJacobian, ControlledProcess, Ekf, KalmanError, LinearKf,
    MeasurementJacobian, MeasurementModel, ProcessModel, SqrtKf, SqrtUkf, Ukf, WithInput,
};

const DT: f64 = 0.1;

/// A cart: position and velocity, driven by a commanded acceleration. Linear, so every filter
/// must agree exactly.
struct Cart;

impl ControlledProcess<2, 1> for Cart {
    fn predict(&self, x: &Vector2<f64>, u: &Vector1<f64>, dt: f64) -> Vector2<f64> {
        f(dt) * x + b(dt) * u
    }
}

impl ControlledJacobian<2, 1> for Cart {
    fn jacobian(&self, _x: &Vector2<f64>, _u: &Vector1<f64>, dt: f64) -> Matrix2<f64> {
        f(dt)
    }
}

impl MeasurementModel<2, 1> for Cart {
    fn measure(&self, x: &Vector2<f64>) -> Vector1<f64> {
        h() * x
    }
}

impl MeasurementJacobian<2, 1> for Cart {
    fn jacobian(&self, _x: &Vector2<f64>) -> Matrix1x2<f64> {
        h()
    }
}

fn f(dt: f64) -> Matrix2<f64> {
    Matrix2::new(1.0, dt, 0.0, 1.0)
}

fn b(dt: f64) -> Matrix2x1<f64> {
    Matrix2x1::new(0.5 * dt * dt, dt)
}

fn h() -> Matrix1x2<f64> {
    Matrix1x2::new(1.0, 0.0)
}

#[test]
fn linear_kf_adds_the_input_to_the_prediction() {
    let mut kf = LinearKf::new(Vector2::new(1.0, 2.0), Matrix2::identity());
    let mut plain = kf.clone();
    let q = Matrix2::identity() * 1e-3;
    let u = Vector1::new(3.0);

    kf.predict_with_input(&f(DT), &b(DT), &u, &q);
    plain.predict(&f(DT), &q);

    approx::assert_relative_eq!(*kf.state(), f(DT) * Vector2::new(1.0, 2.0) + b(DT) * u);
    // A known input adds no uncertainty.
    assert_eq!(kf.covariance(), plain.covariance());
}

#[test]
fn every_filter_agrees_on_a_controlled_cart() {
    let q = Matrix2::new(1e-4, 0.0, 0.0, 1e-3);
    let r = Matrix1::new(0.25);
    let (q_sqrt, r_sqrt) = (
        Cholesky::new(q).unwrap().unpack(),
        Cholesky::new(r).unwrap().unpack(),
    );
    let (x0, p0) = (Vector2::zeros(), Matrix2::identity());

    let mut kf = LinearKf::new(x0, p0);
    let mut skf = SqrtKf::new(x0, p0).unwrap();
    let mut ekf = Ekf::new(x0, p0);
    let mut ukf = Ukf::new(x0, p0);
    let mut sukf = SqrtUkf::new(x0, p0).unwrap();

    for k in 0..200 {
        let t = k as f64 * DT;
        let u = Vector1::new((0.3 * t).sin());
        let z = Vector1::new(0.5 * t * t * 0.1 + (2.0 * t).cos());
        let model = WithInput::new(&Cart, u);

        kf.predict_with_input(&f(DT), &b(DT), &u, &q);
        skf.predict_with_input(&f(DT), &b(DT), &u, &q_sqrt).unwrap();
        ekf.predict(&model, &q, DT);
        ukf.predict(&model, &q, DT).unwrap();
        sukf.predict(&model, &q_sqrt, DT).unwrap();

        kf.update(&h(), &z, &r).unwrap();
        skf.update(&h(), &z, &r_sqrt).unwrap();
        ekf.update(&Cart, &z, &r).unwrap();
        ukf.update(&Cart, &z, &r).unwrap();
        sukf.update(&Cart, &z, &r_sqrt).unwrap();

        for (name, x) in [
            ("SqrtKf", *skf.state()),
            ("Ekf", *ekf.state()),
            ("Ukf", *ukf.state()),
            ("SqrtUkf", *sukf.state()),
        ] {
            assert!(
                (x - kf.state()).abs().max() < 1e-9,
                "{name} differs at step {k}: {x} vs {}",
                kf.state()
            );
        }
    }
}

#[test]
fn a_non_finite_input_is_rejected_and_keeps_state() {
    let mut skf = SqrtKf::new(Vector2::zeros(), Matrix2::identity()).unwrap();
    let before = skf.clone();
    let result = skf.predict_with_input(
        &f(DT),
        &b(DT),
        &Vector1::new(f64::NAN),
        &Matrix2::identity(),
    );
    assert_eq!(result, Err(KalmanError::InvalidInput));
    assert_eq!(skf, before);
}

/// A heading driven by a commanded turn rate, with a wrapping state residual.
struct Turn;

impl ControlledProcess<1, 1> for Turn {
    fn predict(&self, x: &Vector1<f64>, u: &Vector1<f64>, dt: f64) -> Vector1<f64> {
        Vector1::new(wrap_angle(x[0] + u[0] * dt))
    }

    fn state_residual(&self, a: &Vector1<f64>, b: &Vector1<f64>) -> Vector1<f64> {
        Vector1::new(wrap_angle(a[0] - b[0]))
    }
}

#[test]
fn with_input_forwards_the_state_residual() {
    let model = WithInput::new(&Turn, Vector1::new(1.0));
    let d = model.state_residual(&Vector1::new(3.1), &Vector1::new(-3.1));
    approx::assert_relative_eq!(d[0], 6.2 - 2.0 * std::f64::consts::PI, epsilon = 1e-12);
}
