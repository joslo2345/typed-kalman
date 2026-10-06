//! Adapter around adskalman, the baseline we compare against.
//!
//! This is the only file that calls adskalman, so an API change in it touches nothing else.
#![allow(non_snake_case)] // adskalman's trait methods are named F, Q, H, R.

use adskalman::{
    CovarianceUpdateMethod, KalmanFilterNoControl, ObservationModel, StateAndCovariance,
    TransitionModelLinearNoControl,
};
use nalgebra::{OMatrix, U2, U4};

use super::scenarios::Scenario;

pub use adskalman::CovarianceUpdateMethod as Method;

struct Motion {
    f: OMatrix<f64, U4, U4>,
    ft: OMatrix<f64, U4, U4>,
    q: OMatrix<f64, U4, U4>,
}

impl TransitionModelLinearNoControl<f64, U4> for Motion {
    fn F(&self) -> &OMatrix<f64, U4, U4> {
        &self.f
    }
    fn FT(&self) -> &OMatrix<f64, U4, U4> {
        &self.ft
    }
    fn Q(&self) -> &OMatrix<f64, U4, U4> {
        &self.q
    }
}

struct Observation {
    h: OMatrix<f64, U2, U4>,
    ht: OMatrix<f64, U4, U2>,
    r: OMatrix<f64, U2, U2>,
}

impl ObservationModel<f64, U4, U2> for Observation {
    fn H(&self) -> &OMatrix<f64, U2, U4> {
        &self.h
    }
    fn HT(&self) -> &OMatrix<f64, U4, U2> {
        &self.ht
    }
    fn R(&self) -> &OMatrix<f64, U2, U2> {
        &self.r
    }
}

/// adskalman's models for a scenario, built once so benchmarks don't time the setup.
pub struct Models {
    motion: Motion,
    observation: Observation,
}

impl Models {
    /// Builds the models for `sc`.
    pub fn new(sc: &Scenario<4, 2>) -> Self {
        Self {
            motion: Motion {
                f: sc.f,
                ft: sc.f.transpose(),
                q: sc.q,
            },
            observation: Observation {
                h: sc.h,
                ht: sc.h.transpose(),
                r: sc.r,
            },
        }
    }

    /// Runs adskalman over `sc` with `method`, calling `on_step` with each estimate.
    ///
    /// Each step predicts and then updates, starting from the prior, the same order as
    /// our filters.
    pub fn run(
        &self,
        sc: &Scenario<4, 2>,
        method: CovarianceUpdateMethod,
        mut on_step: impl FnMut(&StateAndCovariance<f64, U4>),
    ) {
        let kf = KalmanFilterNoControl::new(&self.motion, &self.observation);
        let mut estimate = StateAndCovariance::new(sc.x0, sc.p0);
        for z in &sc.zs {
            estimate = kf
                .step_with_options(&estimate, z, method)
                .expect("adskalman step failed");
            on_step(&estimate);
        }
    }

    /// Runs adskalman's RTS smoother over `sc`.
    pub fn smooth(&self, sc: &Scenario<4, 2>) -> Vec<StateAndCovariance<f64, U4>> {
        let kf = KalmanFilterNoControl::new(&self.motion, &self.observation);
        kf.smooth(&StateAndCovariance::new(sc.x0, sc.p0), &sc.zs)
            .expect("adskalman smoother failed")
    }
}

/// Returns adskalman's Joseph-form state estimates for every step of `sc`.
pub fn adskalman_filter(sc: &Scenario<4, 2>) -> Vec<StateAndCovariance<f64, U4>> {
    let mut out = Vec::with_capacity(sc.zs.len());
    Models::new(sc).run(sc, CovarianceUpdateMethod::JosephForm, |e| {
        out.push(e.clone())
    });
    out
}
