//! Helpers shared by the integration tests and benchmarks.
//!
//! Each test binary uses a different subset, so unused items are expected.
#![allow(dead_code, unused_imports)]

pub mod baseline;
pub mod catalog;
pub mod naive;
pub mod rng;
pub mod scenarios;
pub mod vectors;

use scenarios::Scenario;
use typed_kalman::linear::LinearKf;

/// Builds a filter initialized from a scenario's prior.
pub trait FromScenario<const N: usize, const M: usize> {
    /// Returns a filter starting at the scenario's `x0` and `p0`.
    fn from_scenario(sc: &Scenario<N, M>) -> Self;
}

impl<const N: usize, const M: usize> FromScenario<N, M> for LinearKf<N> {
    fn from_scenario(sc: &Scenario<N, M>) -> Self {
        LinearKf::new(sc.x0, sc.p0)
    }
}
