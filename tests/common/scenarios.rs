//! Problem generators with fixed seeds, shared by every test and benchmark.

use nalgebra::{Cholesky, Matrix2, Matrix2x4, Matrix4, SMatrix, SVector, Vector4};

use super::rng::Rng;

/// A linear-Gaussian problem: the model, the prior, and simulated truth and measurements.
#[derive(Debug, Clone)]
pub struct Scenario<const N: usize, const M: usize> {
    /// State transition matrix.
    pub f: SMatrix<f64, N, N>,
    /// Observation matrix.
    pub h: SMatrix<f64, M, N>,
    /// Process noise covariance.
    pub q: SMatrix<f64, N, N>,
    /// Measurement noise covariance.
    pub r: SMatrix<f64, M, M>,
    /// Prior mean.
    pub x0: SVector<f64, N>,
    /// Prior covariance.
    pub p0: SMatrix<f64, N, N>,
    /// True state at each step, after that step's transition.
    pub truth: Vec<SVector<f64, N>>,
    /// Measurement at each step.
    pub zs: Vec<SVector<f64, M>>,
}

impl<const N: usize, const M: usize> Scenario<N, M> {
    /// Builds a scenario and simulates `steps` steps. The true initial state is drawn from the
    /// prior, so the filter's initial covariance is correct.
    #[allow(clippy::too_many_arguments)]
    fn simulate(
        f: SMatrix<f64, N, N>,
        h: SMatrix<f64, M, N>,
        q: SMatrix<f64, N, N>,
        r: SMatrix<f64, M, M>,
        x0: SVector<f64, N>,
        p0: SMatrix<f64, N, N>,
        steps: usize,
        rng: &mut Rng,
    ) -> Self {
        let mut sc = Self {
            f,
            h,
            q,
            r,
            x0,
            p0,
            truth: Vec::with_capacity(steps),
            zs: Vec::with_capacity(steps),
        };
        let mut sim = sc.simulator(rng);
        for _ in 0..steps {
            let (x, z) = sim.next_step(rng);
            sc.truth.push(x);
            sc.zs.push(z);
        }
        sc
    }

    /// Returns a simulator that generates truth and measurements one step at a time, for runs
    /// too long to store.
    pub fn simulator(&self, rng: &mut Rng) -> Simulator<N, M> {
        let p0_l = factor(&self.p0);
        Simulator {
            f: self.f,
            h: self.h,
            q_l: factor(&self.q),
            r_l: factor(&self.r),
            x: self.x0 + rng.correlated(&p0_l),
        }
    }
}

/// Simulates a scenario's truth and measurements step by step.
pub struct Simulator<const N: usize, const M: usize> {
    f: SMatrix<f64, N, N>,
    h: SMatrix<f64, M, N>,
    q_l: SMatrix<f64, N, N>,
    r_l: SMatrix<f64, M, M>,
    x: SVector<f64, N>,
}

impl<const N: usize, const M: usize> Simulator<N, M> {
    /// Advances the truth one step and returns it with its measurement.
    pub fn next_step(&mut self, rng: &mut Rng) -> (SVector<f64, N>, SVector<f64, M>) {
        self.x = self.f * self.x + rng.correlated(&self.q_l);
        let z = self.h * self.x + rng.correlated(&self.r_l);
        (self.x, z)
    }
}

/// Returns the lower Cholesky factor, panicking on bad test data.
fn factor<const D: usize>(m: &SMatrix<f64, D, D>) -> SMatrix<f64, D, D> {
    Cholesky::new(*m)
        .expect("scenario covariance must be positive-definite")
        .unpack()
}

/// The 2D constant-velocity transition for state `[x, y, vx, vy]`.
pub fn constant_velocity_f(dt: f64) -> Matrix4<f64> {
    #[rustfmt::skip]
    let f = Matrix4::new(
        1.0, 0.0, dt,  0.0,
        0.0, 1.0, 0.0, dt,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    f
}

/// Position-only observation of `[x, y, vx, vy]`.
pub fn position_h() -> Matrix2x4<f64> {
    #[rustfmt::skip]
    let h = Matrix2x4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    );
    h
}

/// A 2D constant-velocity target observed in position, like benchmark scenario S2.
pub fn constant_velocity(steps: usize, seed: u64) -> Scenario<4, 2> {
    let dt = 0.1;
    let mut rng = Rng::new(seed);
    Scenario::simulate(
        constant_velocity_f(dt),
        position_h(),
        Matrix4::from_diagonal(&Vector4::new(1e-4, 1e-4, 1e-2, 1e-2)),
        Matrix2::identity() * 0.25,
        Vector4::new(0.0, 0.0, 1.0, 0.5),
        Matrix4::from_diagonal(&Vector4::new(1.0, 1.0, 0.1, 0.1)),
        steps,
        &mut rng,
    )
}

/// A random, ill-conditioned linear problem with measurement noise of scale `r_scale`.
///
/// The transition is constant velocity with a random time step, the observation matrix is
/// random, and the process noise and prior variances span many orders of magnitude.
///
/// Every eigenvalue of R is at least `r_scale`, and prior variances are at most 10, so for
/// `r_scale ≥ 1e-12` the posterior covariance's condition number stays below about 1e13. Beyond
/// roughly 1e15, `f64` rounding makes Cholesky fail on any filter's covariance (adskalman's
/// included), so the test would measure the number format rather than the filter.
pub fn random_linear(seed: u64, r_scale: f64, steps: usize) -> Scenario<4, 2> {
    let mut rng = Rng::new(seed);
    let dt = rng.log_uniform(-2.0, 0.0);
    let h = Matrix2x4::from_fn(|_, _| rng.normal());
    let q = Matrix4::from_diagonal(&Vector4::from_fn(|_, _| rng.log_uniform(-8.0, 0.0)));
    let g = Matrix2::from_fn(|_, _| rng.normal());
    let r = (g * g.transpose() + Matrix2::identity()) * r_scale;
    let p0 = Matrix4::from_diagonal(&Vector4::from_fn(|_, _| rng.log_uniform(-6.0, 1.0)));
    Scenario::simulate(
        constant_velocity_f(dt),
        h,
        q,
        r,
        Vector4::zeros(),
        p0,
        steps,
        &mut rng,
    )
}

/// An ill-conditioned problem for long single-precision runs, like benchmark scenario S4.
///
/// A vague prior (variance 1e4) meets precise measurements (variance 1e-6), so the first
/// update shrinks the covariance by ten orders of magnitude, more than `f32` can resolve. The
/// short form `P = (I - K H) P` loses positive-definiteness there; the Joseph form doesn't.
pub fn ill_conditioned() -> Scenario<4, 2> {
    Scenario::simulate(
        constant_velocity_f(0.01),
        position_h(),
        Matrix4::from_diagonal(&Vector4::new(1e-8, 1e-8, 1e-5, 1e-5)),
        Matrix2::identity() * 1e-6,
        Vector4::zeros(),
        Matrix4::identity() * 1e4,
        0,
        &mut Rng::new(4),
    )
}

/// A problem beyond what any covariance-form filter can represent in `f32`.
///
/// One-second steps with nearly perfect measurements (variance 1e-8) against a prior of 1e4
/// make P's condition number far exceed `f32`'s ~1e7 range, so `LinearKf` loses
/// positive-definiteness within two steps. Its square root stays within range, so `SqrtKf`
/// keeps going.
pub fn beyond_f32() -> Scenario<4, 2> {
    Scenario::simulate(
        constant_velocity_f(1.0),
        position_h(),
        Matrix4::from_diagonal(&Vector4::new(1e-12, 1e-12, 1e-9, 1e-9)),
        Matrix2::identity() * 1e-8,
        Vector4::zeros(),
        Matrix4::identity() * 1e4,
        0,
        &mut Rng::new(4),
    )
}

/// Discrete white-noise-acceleration process noise for one constant-velocity axis, with
/// acceleration spectral density `q`: `q [[dt³/3, dt²/2], [dt²/2, dt]]`.
fn wna_axis(q: f64, dt: f64) -> Matrix2<f64> {
    let (dt2, dt3) = (dt * dt, dt * dt * dt);
    Matrix2::new(dt3 / 3.0, dt2 / 2.0, dt2 / 2.0, dt) * q
}

/// S1: a 1D constant-velocity target observed in position. State `[x, vx]`.
pub fn s1(steps: usize, seed: u64) -> Scenario<2, 1> {
    let dt = 0.1;
    Scenario::simulate(
        Matrix2::new(1.0, dt, 0.0, 1.0),
        nalgebra::Matrix1x2::new(1.0, 0.0),
        wna_axis(0.1, dt),
        nalgebra::Matrix1::new(0.25),
        nalgebra::Vector2::new(0.0, 1.0),
        Matrix2::identity(),
        steps,
        &mut Rng::new(seed),
    )
}

/// S2: a 2D constant-velocity target observed in position. State `[x, y, vx, vy]`.
pub fn s2(steps: usize, seed: u64) -> Scenario<4, 2> {
    constant_velocity(steps, seed)
}

/// Range and bearing from a sensor at the origin to a target at `[x, y, ...]`.
pub fn range_bearing(x: &Vector4<f64>) -> nalgebra::Vector2<f64> {
    // Through libm so the generated S3 files are identical on every platform.
    nalgebra::Vector2::new(libm::hypot(x[0], x[1]), libm::atan2(x[1], x[0]))
}

/// S3: a 2D constant-velocity target tracked by range and bearing. State `[x, y, vx, vy]`.
///
/// The motion is linear and only the measurement is nonlinear, so `h` is unused (zero) and
/// `zs` come from [`range_bearing`]. The target stays at `x > 0`, so bearings never come near
/// ±π and no angle wrapping is needed.
pub fn s3(steps: usize, seed: u64) -> Scenario<4, 2> {
    let dt = 1.0;
    let mut q = Matrix4::zeros();
    let axis = wna_axis(0.01, dt);
    for (i, j) in [(0, 2), (1, 3)] {
        q[(i, i)] = axis[(0, 0)];
        q[(i, j)] = axis[(0, 1)];
        q[(j, i)] = axis[(1, 0)];
        q[(j, j)] = axis[(1, 1)];
    }
    let mut sc = Scenario {
        f: constant_velocity_f(dt),
        h: Matrix2x4::zeros(),
        q,
        r: Matrix2::new(25.0, 0.0, 0.0, 1e-4),
        x0: Vector4::new(10_000.0, 0.0, -1.0, 2.0),
        p0: Matrix4::from_diagonal(&Vector4::new(100.0, 100.0, 1.0, 1.0)),
        truth: Vec::with_capacity(steps),
        zs: Vec::with_capacity(steps),
    };
    let mut rng = Rng::new(seed);
    let (p0_l, q_l, r_l) = (factor(&sc.p0), factor(&sc.q), factor(&sc.r));
    let mut x = sc.x0 + rng.correlated(&p0_l);
    for _ in 0..steps {
        x = sc.f * x + rng.correlated(&q_l);
        sc.truth.push(x);
        sc.zs.push(range_bearing(&x) + rng.correlated(&r_l));
    }
    sc
}

/// S4: the ill-conditioned single-precision stability problem ([`ill_conditioned`]).
pub fn s4() -> Scenario<4, 2> {
    ill_conditioned()
}

/// Returns `[a×]`, the matrix with `[a×] b = a × b`.
fn skew(a: &nalgebra::Vector3<f64>) -> nalgebra::Matrix3<f64> {
    nalgebra::Matrix3::new(0.0, -a[2], a[1], a[2], 0.0, -a[0], -a[1], a[0], 0.0)
}

/// S5: a 15-state INS error-state filter aided by GPS position and velocity.
///
/// State: position, velocity and attitude errors, then accelerometer and gyro biases (3 each).
/// The vehicle is level and aligned with the navigation frame under gravity only, and the
/// continuous model `δṗ = δv`, `δv̇ = -[f×] δψ - δb_a`, `δψ̇ = -δb_g`, with random-walk biases,
/// is discretized to first order at 100 Hz.
pub fn s5(steps: usize, seed: u64) -> Scenario<15, 6> {
    use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
    let dt = 0.01;
    let i3 = Matrix3::identity();
    let specific_force = Vector3::new(0.0, 0.0, -9.81);

    let mut a = SMatrix::<f64, 15, 15>::zeros();
    a.fixed_view_mut::<3, 3>(0, 3).copy_from(&i3);
    a.fixed_view_mut::<3, 3>(3, 6)
        .copy_from(&-skew(&specific_force));
    a.fixed_view_mut::<3, 3>(3, 9).copy_from(&-i3);
    a.fixed_view_mut::<3, 3>(6, 12).copy_from(&-i3);
    let f = SMatrix::<f64, 15, 15>::identity() + a * dt;

    let mut h = SMatrix::<f64, 6, 15>::zeros();
    h.fixed_view_mut::<6, 6>(0, 0)
        .copy_from(&SMatrix::<f64, 6, 6>::identity());

    // Noise densities: position (regularization only), accelerometer, gyro, and bias walks.
    let densities = [1e-12, 1e-4, 1e-8, 1e-10, 1e-12];
    let q = SMatrix::<f64, 15, 15>::from_diagonal(&SVector::from_fn(|i, _| densities[i / 3] * dt));
    let r = SMatrix::<f64, 6, 6>::from_diagonal(&SVector::from_column_slice(&[
        1.0, 1.0, 1.0, 0.01, 0.01, 0.01,
    ]));
    let prior = [10.0, 1.0, 1e-4, 1e-4, 1e-8];
    let p0 = SMatrix::<f64, 15, 15>::from_diagonal(&SVector::from_fn(|i, _| prior[i / 3]));
    Scenario::simulate(f, h, q, r, SVector::zeros(), p0, steps, &mut Rng::new(seed))
}
