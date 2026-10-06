//! Bad inputs return errors and leave the filter unchanged; nothing panics.

mod common;

use common::scenarios;
use kalman_rs::{Ekf, KalmanError, LinearKf, MeasurementJacobian, MeasurementModel};
use kalman_rs::{SqrtUkf, Ukf};
use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};

/// Measures position, or NaN when `broken` is set, as a model failing at some states might.
struct Position {
    h: Matrix2x4<f64>,
    broken: bool,
}

impl MeasurementModel<4, 2> for Position {
    fn measure(&self, x: &Vector4<f64>) -> Vector2<f64> {
        if self.broken {
            Vector2::repeat(f64::NAN)
        } else {
            self.h * x
        }
    }
}

impl MeasurementJacobian<4, 2> for Position {
    fn jacobian(&self, _x: &Vector4<f64>) -> Matrix2x4<f64> {
        self.h
    }
}

/// Measurements and noise matrices that must be rejected as invalid input.
fn invalid_inputs() -> Vec<(&'static str, Vector2<f64>, Matrix2<f64>)> {
    let good_z = Vector2::new(1.0, 2.0);
    let good_r = Matrix2::identity();
    let mut nan_r = good_r;
    nan_r[(0, 1)] = f64::NAN;
    vec![
        ("NaN measurement", Vector2::new(f64::NAN, 1.0), good_r),
        (
            "infinite measurement",
            Vector2::new(1.0, f64::INFINITY),
            good_r,
        ),
        ("NaN in R", good_z, nan_r),
        ("infinite R", good_z, good_r * f64::INFINITY),
    ]
}

/// Asserts that `update` fails with `expected` and leaves `filter` unchanged.
fn assert_rejected<F: Clone + PartialEq + std::fmt::Debug>(
    name: &str,
    filter: &mut F,
    expected: KalmanError,
    update: impl FnOnce(&mut F) -> Result<f64, KalmanError>,
) {
    let before = filter.clone();
    assert_eq!(update(filter), Err(expected), "{name}");
    assert_eq!(*filter, before, "{name}: filter changed");
}

#[test]
fn invalid_inputs_are_rejected_by_every_filter() {
    let sc = scenarios::constant_velocity(1, 0);
    let model = Position {
        h: sc.h,
        broken: false,
    };

    for (name, z, r) in invalid_inputs() {
        let r_sqrt = r.map(|v| if v.is_finite() { v.sqrt() } else { v });
        let expected = KalmanError::InvalidInput;

        let mut kf = LinearKf::new(sc.x0, sc.p0);
        assert_rejected(name, &mut kf, expected, |f| f.update(&sc.h, &z, &r));
        let mut ekf = Ekf::new(sc.x0, sc.p0);
        assert_rejected(name, &mut ekf, expected, |f| f.update(&model, &z, &r));
        let mut ukf = Ukf::new(sc.x0, sc.p0);
        assert_rejected(name, &mut ukf, expected, |f| f.update(&model, &z, &r));
        let mut sr = SqrtUkf::new(sc.x0, sc.p0).unwrap();
        assert_rejected(name, &mut sr, expected, |f| f.update(&model, &z, &r_sqrt));
    }
}

#[test]
fn a_model_returning_nan_is_rejected() {
    let sc = scenarios::constant_velocity(1, 0);
    let broken = Position {
        h: sc.h,
        broken: true,
    };
    let (z, r) = (Vector2::new(1.0, 2.0), Matrix2::identity());

    // The EKF sees NaN in the innovation; the UKFs see it in every sigma point.
    let mut ekf = Ekf::new(sc.x0, sc.p0);
    assert_rejected("EKF", &mut ekf, KalmanError::InvalidInput, |f| {
        f.update(&broken, &z, &r)
    });
    for name in ["UKF", "SqrtUkf"] {
        let result = match name {
            "UKF" => {
                let mut ukf = Ukf::new(sc.x0, sc.p0);
                let before = ukf.clone();
                let result = ukf.update(&broken, &z, &r);
                assert_eq!(ukf, before, "{name}: filter changed");
                result
            }
            _ => {
                let mut sr = SqrtUkf::new(sc.x0, sc.p0).unwrap();
                let before = sr.clone();
                let result = sr.update(&broken, &z, &r);
                assert_eq!(sr, before, "{name}: filter changed");
                result
            }
        };
        assert!(result.is_err(), "{name} accepted a NaN measurement model");
    }
}

#[test]
fn singular_innovation_is_rejected() {
    // With zero state covariance and zero measurement noise, S = 0.
    let sc = scenarios::constant_velocity(1, 0);
    let model = Position {
        h: sc.h,
        broken: false,
    };
    let (z, r) = (Vector2::new(1.0, 2.0), Matrix2::zeros());
    let expected = KalmanError::SingularInnovation;

    let mut kf = LinearKf::new(sc.x0, Matrix4::zeros());
    assert_rejected("LinearKf", &mut kf, expected, |f| f.update(&sc.h, &z, &r));
    let mut ekf = Ekf::new(sc.x0, Matrix4::zeros());
    assert_rejected("Ekf", &mut ekf, expected, |f| f.update(&model, &z, &r));
}

#[test]
fn non_positive_definite_covariance_is_rejected_by_sigma_point_filters() {
    let sc = scenarios::constant_velocity(1, 0);
    let model = Position {
        h: sc.h,
        broken: false,
    };
    let indefinite = Matrix4::from_diagonal(&Vector4::new(1.0, -1.0, 1.0, 1.0));
    let (z, r) = (Vector2::new(1.0, 2.0), Matrix2::identity());

    let mut ukf = Ukf::new(sc.x0, indefinite);
    assert_rejected(
        "Ukf",
        &mut ukf,
        KalmanError::CovarianceNotPositiveDefinite,
        |f| f.update(&model, &z, &r),
    );
    assert_eq!(
        SqrtUkf::new(sc.x0, indefinite),
        Err(KalmanError::CovarianceNotPositiveDefinite)
    );
}
