//! The scalar type that filters are generic over.

use nalgebra::RealField;

/// A floating-point scalar the filters can run on: `f32` or `f64`.
///
/// Every filter, model trait and helper takes it as a type parameter that defaults to `f64`,
/// so `LinearKf<4>` means `LinearKf<4, f64>`. Use `f32` on microcontrollers that only have a
/// single-precision FPU. Math functions go through nalgebra's [`RealField`], which uses `libm`
/// when `std` is off.
pub trait Float: RealField + Copy {}

impl<T: RealField + Copy> Float for T {}

/// Converts an `f64` constant to `T`.
pub(crate) fn lit<T: Float>(value: f64) -> T {
    nalgebra::convert(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::nees;
    use crate::ekf::Ekf;
    use crate::linear::LinearKf;
    use crate::model::{MeasurementJacobian, MeasurementModel, ProcessJacobian, ProcessModel};
    use crate::smoother::{smooth, Estimate, RtsStep};
    use crate::sqrt_ukf::SqrtUkf;
    use crate::ukf::Ukf;
    use nalgebra::{Matrix1, Matrix1x2, Matrix2, Vector1, Vector2};

    const STEPS: usize = 200;

    /// Constant velocity, written once for any scalar type.
    struct ConstantVelocity;

    impl<T: Float> ProcessModel<2, T> for ConstantVelocity {
        fn predict(&self, x: &Vector2<T>, dt: T) -> Vector2<T> {
            Vector2::new(x[0] + x[1] * dt, x[1])
        }
    }

    impl<T: Float> ProcessJacobian<2, T> for ConstantVelocity {
        fn jacobian(&self, _x: &Vector2<T>, dt: T) -> Matrix2<T> {
            Matrix2::new(T::one(), dt, T::zero(), T::one())
        }
    }

    /// Measures position.
    struct Position;

    impl<T: Float> MeasurementModel<2, 1, T> for Position {
        fn measure(&self, x: &Vector2<T>) -> Vector1<T> {
            Vector1::new(x[0])
        }
    }

    impl<T: Float> MeasurementJacobian<2, 1, T> for Position {
        fn jacobian(&self, _x: &Vector2<T>) -> Matrix1x2<T> {
            Matrix1x2::new(T::one(), T::zero())
        }
    }

    /// Final estimates from each filter, the first smoothed estimate, and the final NEES.
    struct Results<T: Float> {
        finals: [Vector2<T>; 4],
        smoothed_first: Vector2<T>,
        nees: T,
    }

    /// Runs every filter on the same problem in precision `T`.
    fn run<T: Float>() -> Results<T> {
        let dt = lit::<T>(0.1);
        let f = Matrix2::new(T::one(), dt, T::zero(), T::one());
        let h = Matrix1x2::new(T::one(), T::zero());
        let q = Matrix2::new(lit(1e-4), T::zero(), T::zero(), lit(1e-3));
        let r = Matrix1::new(lit::<T>(0.25));
        let q_sqrt = Matrix2::new(
            lit(1e-2),
            T::zero(),
            T::zero(),
            lit(0.031_622_776_601_683_79),
        );
        let r_sqrt = Matrix1::new(lit::<T>(0.5));
        let (x0, p0) = (Vector2::zeros(), Matrix2::identity());

        let mut kf = LinearKf::new(x0, p0);
        let mut ekf = Ekf::new(x0, p0);
        let mut ukf = Ukf::new(x0, p0);
        let mut sr = SqrtUkf::new(x0, p0).unwrap();
        let mut history = [RtsStep {
            f,
            predicted: Estimate { x: x0, p: p0 },
            filtered: Estimate { x: x0, p: p0 },
        }; STEPS];

        for (k, step) in history.iter_mut().enumerate() {
            let t = k as f64 * 0.1;
            let z = Vector1::new(lit::<T>(2.0 * t + 0.3 * (3.0 * t).sin()));

            kf.predict(&f, &q);
            ekf.predict(&ConstantVelocity, &q, dt);
            ukf.predict(&ConstantVelocity, &q, dt).unwrap();
            sr.predict(&ConstantVelocity, &q_sqrt, dt).unwrap();
            step.predicted = Estimate {
                x: *kf.state(),
                p: *kf.covariance(),
            };

            kf.update(&h, &z, &r).unwrap();
            ekf.update(&Position, &z, &r).unwrap();
            ukf.update(&Position, &z, &r).unwrap();
            sr.update(&Position, &z, &r_sqrt).unwrap();
            step.filtered = Estimate {
                x: *kf.state(),
                p: *kf.covariance(),
            };
        }

        let mut smoothed = [history[0].filtered; STEPS];
        smooth(&history, &mut smoothed).unwrap();
        let truth = Vector2::new(lit(2.0 * (STEPS - 1) as f64 * 0.1), lit(2.0));
        Results {
            finals: [*kf.state(), *ekf.state(), *ukf.state(), *sr.state()],
            smoothed_first: smoothed[0].x,
            nees: nees(&truth, kf.state(), kf.covariance()).unwrap(),
        }
    }

    #[test]
    fn f32_tracks_f64() {
        let (single, double) = (run::<f32>(), run::<f64>());
        let widen = |v: &Vector2<f32>| v.map(f64::from);

        for (s, d) in single.finals.iter().zip(&double.finals) {
            approx::assert_relative_eq!(widen(s), *d, max_relative = 1e-4);
        }
        approx::assert_relative_eq!(
            widen(&single.smoothed_first),
            double.smoothed_first,
            max_relative = 1e-3,
            epsilon = 1e-4
        );
        approx::assert_relative_eq!(f64::from(single.nees), double.nees, max_relative = 1e-3);
    }
}
