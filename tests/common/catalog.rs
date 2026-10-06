//! The five frozen benchmark scenarios: exactly what `examples/generate_vectors.rs` writes to
//! `tests/vectors/`, and what `tests/vectors.rs` checks the files against.
//!
//! Changing anything here changes the scenario files, which invalidates every result collected
//! from them, in all four implementations. Don't, once results exist.

use super::rng::Rng;
use super::scenarios::{self, Scenario};
use super::vectors::Spec;

/// S1: 1D constant velocity, 10,000 steps.
pub fn s1() -> (Spec<'static>, Vec<Scenario<2, 1>>) {
    let spec = Spec {
        id: "S1",
        description: "1D constant velocity [x, vx], position measured; overhead on tiny problems",
        measurement_model: "linear",
        dt: 0.1,
        seeds: vec![1],
        measurement_dtype: "float64",
        with_truth: true,
    };
    (spec, vec![scenarios::s1(10_000, 1)])
}

/// S2: 2D constant velocity, 10,000 steps.
pub fn s2() -> (Spec<'static>, Vec<Scenario<4, 2>>) {
    let spec = Spec {
        id: "S2",
        description: "2D constant velocity [x, y, vx, vy], position measured; typical tracking",
        measurement_model: "linear",
        dt: 0.1,
        seeds: vec![2],
        measurement_dtype: "float64",
        with_truth: true,
    };
    (spec, vec![scenarios::s2(10_000, 2)])
}

/// S3: range-bearing tracking, 500 steps × 200 seeds.
pub fn s3() -> (Spec<'static>, Vec<Scenario<4, 2>>) {
    let seeds: Vec<u64> = (3000..3200).collect();
    let runs = seeds.iter().map(|&seed| scenarios::s3(500, seed)).collect();
    let spec = Spec {
        id: "S3",
        description: "2D constant velocity [x, y, vx, vy] tracked by range and bearing from the \
                      origin; nonlinear accuracy. h is unused; z = [hypot(x, y), atan2(y, x)]",
        measurement_model: "range_bearing",
        dt: 1.0,
        seeds,
        measurement_dtype: "float64",
        with_truth: true,
    };
    (spec, runs)
}

/// The number of S4 steps.
pub const S4_STEPS: usize = 1_000_000;

/// S4: ill-conditioned problem, 1,000,000 single-precision steps, measurements only.
pub fn s4() -> (Spec<'static>, Vec<Scenario<4, 2>>) {
    let mut sc = scenarios::s4();
    let mut rng = Rng::new(4);
    let mut sim = sc.simulator(&mut rng);
    sc.zs = (0..S4_STEPS).map(|_| sim.next_step(&mut rng).1).collect();
    let spec = Spec {
        id: "S4",
        description: "2D constant velocity with a vague prior and precise measurements; \
                      numerical stability in 32-bit floats. Truth isn't stored",
        measurement_model: "linear",
        dt: 0.01,
        seeds: vec![4],
        measurement_dtype: "float32",
        with_truth: false,
    };
    (spec, vec![sc])
}

/// S5: 15-state INS error-state filter, 10,000 steps.
pub fn s5() -> (Spec<'static>, Vec<Scenario<15, 6>>) {
    let spec = Spec {
        id: "S5",
        description: "INS error state [dp, dv, dpsi, b_a, b_g] (3 each), GPS position and \
                      velocity measured; speed on a larger state",
        measurement_model: "linear",
        dt: 0.01,
        seeds: vec![5],
        measurement_dtype: "float64",
        with_truth: true,
    };
    (spec, vec![scenarios::s5(10_000, 5)])
}
