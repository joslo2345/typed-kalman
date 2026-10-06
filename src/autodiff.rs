//! Automatic Jacobians using forward-mode dual numbers.
//!
//! Write the model once, generic over [`Real`], by implementing [`AutoProcess`] or
//! [`AutoMeasurement`]. Wrapping it in [`AutoDiff`] then gives it the
//! [`ProcessJacobian`] or [`MeasurementJacobian`] the [`Ekf`](crate::ekf::Ekf) needs, with
//! Jacobians exact to floating-point precision. The wrapped model also works with the UKFs.
//!
//! ```
//! use typed_kalman::autodiff::{AutoDiff, AutoMeasurement, Real};
//! use typed_kalman::MeasurementJacobian;
//! use nalgebra::{SVector, Vector2, Vector4};
//!
//! /// Range and bearing to a target at (x, y), with state [x, y, vx, vy].
//! struct RangeBearing;
//!
//! impl AutoMeasurement<4, 2> for RangeBearing {
//!     fn measure<T: Real>(&self, s: &SVector<T, 4>) -> SVector<T, 2> {
//!         let (x, y) = (s[0], s[1]);
//!         Vector2::new((x * x + y * y).sqrt(), y.atan2(x))
//!     }
//! }
//!
//! let h = AutoDiff(RangeBearing).jacobian(&Vector4::new(3.0, 4.0, 0.0, 0.0));
//! assert!((h[(0, 0)] - 0.6).abs() < 1e-12); // ∂range/∂x = x / range
//! ```
//!
//! Only the state is differentiated. Model parameters and `dt` are plain numbers of the filter's
//! scalar type (`f64` by default), and mix with `T` through `T + f64`, `T * f64` and so on
//! (write the constant on the right). For an `f32` filter, implement `AutoProcess<N, f32>` /
//! `AutoMeasurement<N, M, f32>` and write constants as `f32`.

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Sub};

use nalgebra::{ComplexField, RealField, SMatrix, SVector};

use crate::model::{MeasurementJacobian, MeasurementModel, ProcessJacobian, ProcessModel};
use crate::scalar::{lit, Float};

/// A real scalar that models are written over: the filter's scalar `S` (`f64` or `f32`), or a
/// [`Dual`] number when differentiating. Constants of type `S` mix in on the right
/// (`T * S`, `T + S`, ...).
pub trait Real<S: Float = f64>:
    nalgebra::Scalar
    + Copy
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<S, Output = Self>
    + Sub<S, Output = Self>
    + Mul<S, Output = Self>
    + Div<S, Output = Self>
{
    /// Converts a constant, which has zero derivative.
    fn constant(value: S) -> Self;
    /// Returns the value, discarding any derivative.
    fn value(self) -> S;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Tangent.
    fn tan(self) -> Self;
    /// Arcsine.
    fn asin(self) -> Self;
    /// Arccosine.
    fn acos(self) -> Self;
    /// Arctangent.
    fn atan(self) -> Self;
    /// Four-quadrant arctangent of `self / x`.
    fn atan2(self, x: Self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// `e` raised to `self`.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// `self` raised to an integer power.
    fn powi(self, n: i32) -> Self;
    /// Absolute value. Its derivative at zero is taken as that of `+self`.
    fn abs(self) -> Self;
}

// The plain scalars. Math goes through nalgebra's traits so it uses libm when `std` is off.
macro_rules! impl_real_for_scalar {
    ($($t:ty),*) => {$(
        impl Real<$t> for $t {
            fn constant(value: $t) -> Self {
                value
            }
            fn value(self) -> $t {
                self
            }
            fn sin(self) -> Self {
                ComplexField::sin(self)
            }
            fn cos(self) -> Self {
                ComplexField::cos(self)
            }
            fn tan(self) -> Self {
                ComplexField::tan(self)
            }
            fn asin(self) -> Self {
                ComplexField::asin(self)
            }
            fn acos(self) -> Self {
                ComplexField::acos(self)
            }
            fn atan(self) -> Self {
                ComplexField::atan(self)
            }
            fn atan2(self, x: Self) -> Self {
                RealField::atan2(self, x)
            }
            fn sqrt(self) -> Self {
                ComplexField::sqrt(self)
            }
            fn exp(self) -> Self {
                ComplexField::exp(self)
            }
            fn ln(self) -> Self {
                ComplexField::ln(self)
            }
            fn powi(self, n: i32) -> Self {
                ComplexField::powi(self, n)
            }
            fn abs(self) -> Self {
                ComplexField::abs(self)
            }
        }
    )*};
}

impl_real_for_scalar!(f32, f64);

/// A dual number: a value of type `S` plus its gradient with respect to `N` variables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<const N: usize, S: Float = f64> {
    /// The value.
    pub re: S,
    /// The gradient of the value with respect to each variable.
    pub eps: SVector<S, N>,
}

impl<const N: usize, S: Float> Dual<N, S> {
    /// Returns variable `i` with value `value`, whose gradient is the `i`-th unit vector.
    pub fn variable(value: S, i: usize) -> Self {
        let mut eps = SVector::zeros();
        eps[i] = S::one();
        Self { re: value, eps }
    }

    /// Applies a function with value `f` and derivative `df` at `self.re` (the chain rule).
    fn chain(self, f: S, df: S) -> Self {
        Self {
            re: f,
            eps: self.eps * df,
        }
    }
}

impl<const N: usize, S: Float> Add for Dual<N, S> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            eps: self.eps + rhs.eps,
        }
    }
}

impl<const N: usize, S: Float> Sub for Dual<N, S> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            eps: self.eps - rhs.eps,
        }
    }
}

impl<const N: usize, S: Float> Mul for Dual<N, S> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re,
            eps: self.eps * rhs.re + rhs.eps * self.re,
        }
    }
}

impl<const N: usize, S: Float> Div for Dual<N, S> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            re: self.re / rhs.re,
            eps: (self.eps * rhs.re - rhs.eps * self.re) / (rhs.re * rhs.re),
        }
    }
}

impl<const N: usize, S: Float> Neg for Dual<N, S> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            eps: -self.eps,
        }
    }
}

impl<const N: usize, S: Float> Add<S> for Dual<N, S> {
    type Output = Self;
    fn add(self, rhs: S) -> Self {
        Self {
            re: self.re + rhs,
            eps: self.eps,
        }
    }
}

impl<const N: usize, S: Float> Sub<S> for Dual<N, S> {
    type Output = Self;
    fn sub(self, rhs: S) -> Self {
        Self {
            re: self.re - rhs,
            eps: self.eps,
        }
    }
}

impl<const N: usize, S: Float> Mul<S> for Dual<N, S> {
    type Output = Self;
    fn mul(self, rhs: S) -> Self {
        Self {
            re: self.re * rhs,
            eps: self.eps * rhs,
        }
    }
}

impl<const N: usize, S: Float> Div<S> for Dual<N, S> {
    type Output = Self;
    fn div(self, rhs: S) -> Self {
        Self {
            re: self.re / rhs,
            eps: self.eps / rhs,
        }
    }
}

// Derivatives through nalgebra's traits, so they use libm when `std` is off.
impl<const N: usize, S: Float> Real<S> for Dual<N, S> {
    fn constant(value: S) -> Self {
        Self {
            re: value,
            eps: SVector::zeros(),
        }
    }
    fn value(self) -> S {
        self.re
    }
    fn sin(self) -> Self {
        self.chain(ComplexField::sin(self.re), ComplexField::cos(self.re))
    }
    fn cos(self) -> Self {
        self.chain(ComplexField::cos(self.re), -ComplexField::sin(self.re))
    }
    fn tan(self) -> Self {
        let t = ComplexField::tan(self.re);
        self.chain(t, S::one() + t * t)
    }
    fn asin(self) -> Self {
        let d = S::one() / ComplexField::sqrt(S::one() - self.re * self.re);
        self.chain(ComplexField::asin(self.re), d)
    }
    fn acos(self) -> Self {
        let d = -S::one() / ComplexField::sqrt(S::one() - self.re * self.re);
        self.chain(ComplexField::acos(self.re), d)
    }
    fn atan(self) -> Self {
        let d = S::one() / (S::one() + self.re * self.re);
        self.chain(ComplexField::atan(self.re), d)
    }
    fn atan2(self, x: Self) -> Self {
        // d atan2(y, x) = (x dy - y dx) / (x² + y²)
        let r2 = x.re * x.re + self.re * self.re;
        Self {
            re: RealField::atan2(self.re, x.re),
            eps: (self.eps * x.re - x.eps * self.re) / r2,
        }
    }
    fn sqrt(self) -> Self {
        let s = ComplexField::sqrt(self.re);
        self.chain(s, lit::<S>(0.5) / s)
    }
    fn exp(self) -> Self {
        let e = ComplexField::exp(self.re);
        self.chain(e, e)
    }
    fn ln(self) -> Self {
        self.chain(ComplexField::ln(self.re), S::one() / self.re)
    }
    fn powi(self, n: i32) -> Self {
        let d = lit::<S>(f64::from(n)) * ComplexField::powi(self.re, n - 1);
        self.chain(ComplexField::powi(self.re, n), d)
    }
    fn abs(self) -> Self {
        if self.re < S::zero() {
            -self
        } else {
            self
        }
    }
}

/// A process model written generically over [`Real`], so [`AutoDiff`] can differentiate it.
///
/// `S` is the filter's scalar: implement `AutoProcess<N>` for `f64` filters, or
/// `AutoProcess<N, f32>` for `f32` ones, with constants of that type.
pub trait AutoProcess<const N: usize, S: Float = f64> {
    /// Propagates the state `x` forward by `dt`.
    fn predict<T: Real<S>>(&self, x: &SVector<T, N>, dt: S) -> SVector<T, N>;
}

/// A measurement model written generically over [`Real`], so [`AutoDiff`] can differentiate
/// it. `S` is the filter's scalar, as for [`AutoProcess`].
pub trait AutoMeasurement<const N: usize, const M: usize, S: Float = f64> {
    /// Returns the measurement expected for the state `x`.
    fn measure<T: Real<S>>(&self, x: &SVector<T, N>) -> SVector<T, M>;

    /// Returns the difference `a - b` between two measurements; see
    /// [`MeasurementModel::residual`]. Override it for wrapped components such as bearings.
    fn residual(&self, a: &SVector<S, M>, b: &SVector<S, M>) -> SVector<S, M> {
        a - b
    }
}

/// Wraps an [`AutoProcess`] or [`AutoMeasurement`] model, implementing the filter model traits
/// with Jacobians computed by dual numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoDiff<Model>(pub Model);

/// Returns `x` as dual numbers, with `x[i]` seeded as variable `i`.
fn seed<const N: usize, S: Float>(x: &SVector<S, N>) -> SVector<Dual<N, S>, N> {
    SVector::from_fn(|i, _| Dual::variable(x[i], i))
}

/// Stacks the gradients of `y` as the rows of a Jacobian.
fn jacobian_of<const N: usize, const M: usize, S: Float>(
    y: &SVector<Dual<N, S>, M>,
) -> SMatrix<S, M, N> {
    SMatrix::from_fn(|i, j| y[i].eps[j])
}

impl<Model: AutoProcess<N, S>, const N: usize, S: Float + Real<S>> ProcessModel<N, S>
    for AutoDiff<Model>
{
    fn predict(&self, x: &SVector<S, N>, dt: S) -> SVector<S, N> {
        self.0.predict(x, dt)
    }
}

impl<Model: AutoProcess<N, S>, const N: usize, S: Float + Real<S>> ProcessJacobian<N, S>
    for AutoDiff<Model>
{
    fn jacobian(&self, x: &SVector<S, N>, dt: S) -> SMatrix<S, N, N> {
        jacobian_of(&self.0.predict(&seed(x), dt))
    }
}

impl<Model: AutoMeasurement<N, M, S>, const N: usize, const M: usize, S: Float + Real<S>>
    MeasurementModel<N, M, S> for AutoDiff<Model>
{
    fn measure(&self, x: &SVector<S, N>) -> SVector<S, M> {
        self.0.measure(x)
    }

    fn residual(&self, a: &SVector<S, M>, b: &SVector<S, M>) -> SVector<S, M> {
        self.0.residual(a, b)
    }
}

impl<Model: AutoMeasurement<N, M, S>, const N: usize, const M: usize, S: Float + Real<S>>
    MeasurementJacobian<N, M, S> for AutoDiff<Model>
{
    fn jacobian(&self, x: &SVector<S, N>) -> SMatrix<S, M, N> {
        jacobian_of(&self.0.measure(&seed(x)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ekf::Ekf;
    use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};

    /// Returns the derivative of `f` at `a`, computed with a one-variable dual number.
    fn derivative(f: impl Fn(Dual<1>) -> Dual<1>, a: f64) -> f64 {
        f(Dual::variable(a, 0)).eps[0]
    }

    /// A named function of one dual variable and its expected derivative.
    type Case = (&'static str, fn(Dual<1>) -> Dual<1>, f64);

    #[test]
    fn elementary_derivatives_match_calculus() {
        let a: f64 = 0.4;
        let cases: [Case; 12] = [
            ("sin", |x| Real::sin(x), Real::cos(a)),
            ("cos", |x| Real::cos(x), -Real::sin(a)),
            ("tan", |x| Real::tan(x), 1.0 / (Real::cos(a) * Real::cos(a))),
            ("asin", |x| Real::asin(x), 1.0 / Real::sqrt(1.0 - a * a)),
            ("acos", |x| Real::acos(x), -1.0 / Real::sqrt(1.0 - a * a)),
            ("atan", |x| Real::atan(x), 1.0 / (1.0 + a * a)),
            ("sqrt", |x| Real::sqrt(x), 0.5 / Real::sqrt(a)),
            ("exp", |x| Real::exp(x), Real::exp(a)),
            ("ln", |x| Real::ln(x), 1.0 / a),
            ("powi", |x| Real::powi(x, 3), 3.0 * a * a),
            ("abs", |x| Real::abs(-x), 1.0),
            ("quotient", |x| Dual::constant(1.0) / (x * x + 1.0), {
                -2.0 * a / ((a * a + 1.0) * (a * a + 1.0))
            }),
        ];
        for (name, f, expected) in cases {
            let got = derivative(f, a);
            assert!(
                Real::abs(got - expected) < 1e-14,
                "{name}: {got} != {expected}"
            );
        }
    }

    #[test]
    fn atan2_gradient_matches_calculus() {
        let (y, x) = (Dual::<2>::variable(4.0, 0), Dual::<2>::variable(3.0, 1));
        let angle = y.atan2(x);
        approx::assert_relative_eq!(angle.re, Real::atan2(4.0f64, 3.0));
        // ∂/∂y = x / r², ∂/∂x = -y / r², with r² = 25.
        approx::assert_relative_eq!(angle.eps, Vector2::new(3.0 / 25.0, -4.0 / 25.0));
    }

    /// Constant velocity with a quadratic drag on each velocity component.
    struct Drag {
        k: f64,
    }

    impl AutoProcess<4> for Drag {
        fn predict<T: Real>(&self, s: &SVector<T, 4>, dt: f64) -> SVector<T, 4> {
            let (vx, vy) = (s[2], s[3]);
            let speed = Real::sqrt(vx * vx + vy * vy);
            Vector4::new(
                s[0] + vx * dt,
                s[1] + vy * dt,
                vx - speed * vx * (self.k * dt),
                vy - speed * vy * (self.k * dt),
            )
        }
    }

    /// Range and bearing to the target.
    struct RangeBearing;

    impl AutoMeasurement<4, 2> for RangeBearing {
        fn measure<T: Real>(&self, s: &SVector<T, 4>) -> SVector<T, 2> {
            let (x, y) = (s[0], s[1]);
            Vector2::new(Real::sqrt(x * x + y * y), y.atan2(x))
        }
    }

    /// The same models with hand-derived Jacobians.
    struct Manual {
        k: f64,
    }

    impl ProcessModel<4> for Manual {
        fn predict(&self, x: &Vector4<f64>, dt: f64) -> Vector4<f64> {
            Drag { k: self.k }.predict(x, dt)
        }
    }

    impl ProcessJacobian<4> for Manual {
        fn jacobian(&self, s: &Vector4<f64>, dt: f64) -> Matrix4<f64> {
            let (vx, vy) = (s[2], s[3]);
            let v = Real::sqrt(vx * vx + vy * vy);
            let c = self.k * dt;
            #[rustfmt::skip]
            let j = Matrix4::new(
                1.0, 0.0, dt,                           0.0,
                0.0, 1.0, 0.0,                          dt,
                0.0, 0.0, 1.0 - c * (v + vx * vx / v), -c * vx * vy / v,
                0.0, 0.0, -c * vx * vy / v,            1.0 - c * (v + vy * vy / v),
            );
            j
        }
    }

    impl MeasurementModel<4, 2> for Manual {
        fn measure(&self, x: &Vector4<f64>) -> Vector2<f64> {
            RangeBearing.measure(x)
        }
    }

    impl MeasurementJacobian<4, 2> for Manual {
        fn jacobian(&self, s: &Vector4<f64>) -> Matrix2x4<f64> {
            let (x, y) = (s[0], s[1]);
            let r2 = x * x + y * y;
            let r = Real::sqrt(r2);
            #[rustfmt::skip]
            let j = Matrix2x4::new(
                x / r,   y / r,  0.0, 0.0,
                -y / r2, x / r2, 0.0, 0.0,
            );
            j
        }
    }

    #[test]
    fn jacobians_match_hand_derivation() {
        let s = Vector4::new(3.0, 4.0, 1.5, -0.5);
        let manual = Manual { k: 0.1 };
        approx::assert_relative_eq!(
            AutoDiff(Drag { k: 0.1 }).jacobian(&s, 0.2),
            ProcessJacobian::jacobian(&manual, &s, 0.2),
            epsilon = 1e-14
        );
        approx::assert_relative_eq!(
            AutoDiff(RangeBearing).jacobian(&s),
            MeasurementJacobian::jacobian(&manual, &s),
            epsilon = 1e-14
        );
    }

    #[test]
    fn ekf_with_autodiff_matches_hand_jacobians() {
        let (dt, k) = (0.1, 0.05);
        let q = Matrix4::from_diagonal(&Vector4::new(1e-4, 1e-4, 1e-3, 1e-3));
        let r = Matrix2::new(0.01, 0.0, 0.0, 1e-4);
        let x0 = Vector4::new(10.0, 5.0, -1.0, 0.5);
        let manual = Manual { k };
        let (process, measurement) = (AutoDiff(Drag { k }), AutoDiff(RangeBearing));

        let mut by_hand = Ekf::new(x0, Matrix4::identity());
        let mut automatic = by_hand.clone();
        for step in 0..500 {
            let t = step as f64 * dt;
            let z = Vector2::new(11.0 + Real::sin(t), 0.46 + 0.01 * Real::cos(t));

            by_hand.predict(&manual, &q, dt);
            automatic.predict(&process, &q, dt);
            let nis_hand = by_hand.update(&manual, &z, &r).unwrap();
            let nis_auto = automatic.update(&measurement, &z, &r).unwrap();

            approx::assert_relative_eq!(nis_hand, nis_auto, epsilon = 1e-9);
            approx::assert_relative_eq!(by_hand.state(), automatic.state(), epsilon = 1e-9);
            approx::assert_relative_eq!(
                by_hand.covariance(),
                automatic.covariance(),
                epsilon = 1e-9
            );
        }
    }

    /// Range and bearing, written once for any scalar: no constants, so it's generic over `S`.
    struct AnyPrecisionRangeBearing;

    impl<S: Float> AutoMeasurement<4, 2, S> for AnyPrecisionRangeBearing {
        fn measure<T: Real<S>>(&self, s: &SVector<T, 4>) -> SVector<T, 2> {
            let (x, y) = (s[0], s[1]);
            Vector2::new(Real::sqrt(x * x + y * y), y.atan2(x))
        }
    }

    /// The drag model in single precision, with `f32` constants mixed in as `T * f32`.
    struct DragF32 {
        k: f32,
    }

    impl AutoProcess<4, f32> for DragF32 {
        fn predict<T: Real<f32>>(&self, s: &SVector<T, 4>, dt: f32) -> SVector<T, 4> {
            let (vx, vy) = (s[2], s[3]);
            let speed = Real::sqrt(vx * vx + vy * vy);
            SVector::<T, 4>::new(
                s[0] + vx * dt,
                s[1] + vy * dt,
                vx - speed * vx * (self.k * dt),
                vy - speed * vy * (self.k * dt),
            )
        }
    }

    #[test]
    fn f32_jacobians_match_f64() {
        let s = Vector4::new(3.0, 4.0, 1.5, -0.5);
        let single: nalgebra::Matrix2x4<f32> =
            MeasurementJacobian::jacobian(&AutoDiff(AnyPrecisionRangeBearing), &s.cast::<f32>());
        let double: Matrix2x4<f64> =
            MeasurementJacobian::jacobian(&AutoDiff(AnyPrecisionRangeBearing), &s);
        approx::assert_relative_eq!(single.cast::<f64>(), double, max_relative = 1e-6);

        let single =
            ProcessJacobian::jacobian(&AutoDiff(DragF32 { k: 0.1 }), &s.cast::<f32>(), 0.2);
        let double = ProcessJacobian::jacobian(&Manual { k: 0.1 }, &s, 0.2);
        approx::assert_relative_eq!(single.cast::<f64>(), double, max_relative = 1e-6);
    }

    #[test]
    fn f32_ekf_with_autodiff_tracks_its_f64_twin() {
        let (dt, k) = (0.1, 0.05);
        let q = Matrix4::from_diagonal(&Vector4::new(1e-4, 1e-4, 1e-3, 1e-3));
        let r = Matrix2::new(0.01, 0.0, 0.0, 1e-4);
        let x0 = Vector4::new(10.0, 5.0, -1.0, 0.5);

        let mut double = Ekf::new(x0, Matrix4::identity());
        let mut single = Ekf::new(x0.cast::<f32>(), Matrix4::<f32>::identity());
        let (process, measurement) = (
            AutoDiff(DragF32 { k: k as f32 }),
            AutoDiff(AnyPrecisionRangeBearing),
        );
        for step in 0..200 {
            let t = step as f64 * dt;
            let z = Vector2::new(11.0 + t.sin(), 0.46 + 0.01 * t.cos());

            double.predict(&Manual { k }, &q, dt);
            double.update(&Manual { k }, &z, &r).unwrap();
            single.predict(&process, &q.cast(), dt as f32);
            single.update(&measurement, &z.cast(), &r.cast()).unwrap();
        }
        approx::assert_relative_eq!(
            single.state().cast::<f64>(),
            *double.state(),
            max_relative = 1e-3
        );
    }
}
