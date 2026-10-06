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
//! Only the state is differentiated. Model parameters and `dt` stay `f64`, and mix with `T`
//! through `T + f64`, `T * f64` and so on (write the `f64` on the right).

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Sub};

use nalgebra::{ComplexField, RealField, SMatrix, SVector};

use crate::model::{MeasurementJacobian, MeasurementModel, ProcessJacobian, ProcessModel};

/// A real scalar that models are written over: `f64`, or a [`Dual`] number when
/// differentiating.
pub trait Real:
    nalgebra::Scalar
    + Copy
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// Converts a constant, which has zero derivative.
    fn constant(value: f64) -> Self;
    /// Returns the value, discarding any derivative.
    fn value(self) -> f64;
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

// Through nalgebra's traits so the math uses libm when `std` is off.
impl Real for f64 {
    fn constant(value: f64) -> Self {
        value
    }
    fn value(self) -> f64 {
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

/// A dual number: a value plus its gradient with respect to `N` variables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<const N: usize> {
    /// The value.
    pub re: f64,
    /// The gradient of the value with respect to each variable.
    pub eps: SVector<f64, N>,
}

impl<const N: usize> Dual<N> {
    /// Returns variable `i` with value `value`, whose gradient is the `i`-th unit vector.
    pub fn variable(value: f64, i: usize) -> Self {
        let mut eps = SVector::zeros();
        eps[i] = 1.0;
        Self { re: value, eps }
    }

    /// Applies a function with value `f` and derivative `df` at `self.re` (the chain rule).
    fn chain(self, f: f64, df: f64) -> Self {
        Self {
            re: f,
            eps: self.eps * df,
        }
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            eps: self.eps + rhs.eps,
        }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            eps: self.eps - rhs.eps,
        }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re,
            eps: self.eps * rhs.re + rhs.eps * self.re,
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            re: self.re / rhs.re,
            eps: (self.eps * rhs.re - rhs.eps * self.re) / (rhs.re * rhs.re),
        }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            eps: -self.eps,
        }
    }
}

impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Self {
            re: self.re + rhs,
            eps: self.eps,
        }
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        Self {
            re: self.re - rhs,
            eps: self.eps,
        }
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            eps: self.eps * rhs,
        }
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self {
            re: self.re / rhs,
            eps: self.eps / rhs,
        }
    }
}

impl<const N: usize> Real for Dual<N> {
    fn constant(value: f64) -> Self {
        Self {
            re: value,
            eps: SVector::zeros(),
        }
    }
    fn value(self) -> f64 {
        self.re
    }
    fn sin(self) -> Self {
        self.chain(Real::sin(self.re), Real::cos(self.re))
    }
    fn cos(self) -> Self {
        self.chain(Real::cos(self.re), -Real::sin(self.re))
    }
    fn tan(self) -> Self {
        let t = Real::tan(self.re);
        self.chain(t, 1.0 + t * t)
    }
    fn asin(self) -> Self {
        self.chain(
            Real::asin(self.re),
            1.0 / Real::sqrt(1.0 - self.re * self.re),
        )
    }
    fn acos(self) -> Self {
        self.chain(
            Real::acos(self.re),
            -1.0 / Real::sqrt(1.0 - self.re * self.re),
        )
    }
    fn atan(self) -> Self {
        self.chain(Real::atan(self.re), 1.0 / (1.0 + self.re * self.re))
    }
    fn atan2(self, x: Self) -> Self {
        // d atan2(y, x) = (x dy - y dx) / (x² + y²)
        let r2 = x.re * x.re + self.re * self.re;
        Self {
            re: Real::atan2(self.re, x.re),
            eps: (self.eps * x.re - x.eps * self.re) / r2,
        }
    }
    fn sqrt(self) -> Self {
        let s = Real::sqrt(self.re);
        self.chain(s, 0.5 / s)
    }
    fn exp(self) -> Self {
        let e = Real::exp(self.re);
        self.chain(e, e)
    }
    fn ln(self) -> Self {
        self.chain(Real::ln(self.re), 1.0 / self.re)
    }
    fn powi(self, n: i32) -> Self {
        self.chain(
            Real::powi(self.re, n),
            n as f64 * Real::powi(self.re, n - 1),
        )
    }
    fn abs(self) -> Self {
        if self.re < 0.0 {
            -self
        } else {
            self
        }
    }
}

/// A process model written generically over [`Real`], so [`AutoDiff`] can differentiate it.
pub trait AutoProcess<const N: usize> {
    /// Propagates the state `x` forward by `dt`.
    fn predict<T: Real>(&self, x: &SVector<T, N>, dt: f64) -> SVector<T, N>;
}

/// A measurement model written generically over [`Real`], so [`AutoDiff`] can differentiate
/// it.
pub trait AutoMeasurement<const N: usize, const M: usize> {
    /// Returns the measurement expected for the state `x`.
    fn measure<T: Real>(&self, x: &SVector<T, N>) -> SVector<T, M>;
}

/// Wraps an [`AutoProcess`] or [`AutoMeasurement`] model, implementing the filter model traits
/// with Jacobians computed by dual numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoDiff<Model>(pub Model);

/// Returns `x` as dual numbers, with `x[i]` seeded as variable `i`.
fn seed<const N: usize>(x: &SVector<f64, N>) -> SVector<Dual<N>, N> {
    SVector::from_fn(|i, _| Dual::variable(x[i], i))
}

/// Stacks the gradients of `y` as the rows of a Jacobian.
fn jacobian_of<const N: usize, const M: usize>(y: &SVector<Dual<N>, M>) -> SMatrix<f64, M, N> {
    SMatrix::from_fn(|i, j| y[i].eps[j])
}

impl<Model: AutoProcess<N>, const N: usize> ProcessModel<N> for AutoDiff<Model> {
    fn predict(&self, x: &SVector<f64, N>, dt: f64) -> SVector<f64, N> {
        self.0.predict(x, dt)
    }
}

impl<Model: AutoProcess<N>, const N: usize> ProcessJacobian<N> for AutoDiff<Model> {
    fn jacobian(&self, x: &SVector<f64, N>, dt: f64) -> SMatrix<f64, N, N> {
        jacobian_of(&self.0.predict(&seed(x), dt))
    }
}

impl<Model: AutoMeasurement<N, M>, const N: usize, const M: usize> MeasurementModel<N, M>
    for AutoDiff<Model>
{
    fn measure(&self, x: &SVector<f64, N>) -> SVector<f64, M> {
        self.0.measure(x)
    }
}

impl<Model: AutoMeasurement<N, M>, const N: usize, const M: usize> MeasurementJacobian<N, M>
    for AutoDiff<Model>
{
    fn jacobian(&self, x: &SVector<f64, N>) -> SMatrix<f64, M, N> {
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
}
