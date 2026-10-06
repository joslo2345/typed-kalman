//! S2 with adskalman (Joseph form, the like-for-like comparison), f32.
#![no_std]
#![no_main]
#![allow(non_snake_case)] // adskalman's trait methods are named F, Q, H, R.

use adskalman::{
    CovarianceUpdateMethod, KalmanFilterNoControl, ObservationModel, StateAndCovariance,
    TransitionModelLinearNoControl,
};
use cortex_m_rt::entry;
use firmware::{matrix, measurement, s2, time_and_halt};
use nalgebra::{OMatrix, SVector, U2, U4};
use panic_halt as _;

struct Motion {
    f: OMatrix<f32, U4, U4>,
    ft: OMatrix<f32, U4, U4>,
    q: OMatrix<f32, U4, U4>,
}

impl TransitionModelLinearNoControl<f32, U4> for Motion {
    fn F(&self) -> &OMatrix<f32, U4, U4> {
        &self.f
    }
    fn FT(&self) -> &OMatrix<f32, U4, U4> {
        &self.ft
    }
    fn Q(&self) -> &OMatrix<f32, U4, U4> {
        &self.q
    }
}

struct Observation {
    h: OMatrix<f32, U2, U4>,
    ht: OMatrix<f32, U4, U2>,
    r: OMatrix<f32, U2, U2>,
}

impl ObservationModel<f32, U4, U2> for Observation {
    fn H(&self) -> &OMatrix<f32, U2, U4> {
        &self.h
    }
    fn HT(&self) -> &OMatrix<f32, U4, U2> {
        &self.ht
    }
    fn R(&self) -> &OMatrix<f32, U2, U2> {
        &self.r
    }
}

#[entry]
fn main() -> ! {
    let f = matrix(&s2::F);
    let h = matrix(&s2::H);
    let motion = Motion { f, ft: f.transpose(), q: matrix(&s2::Q) };
    let observation = Observation { h, ht: h.transpose(), r: matrix(&s2::R) };
    let kf = KalmanFilterNoControl::new(&motion, &observation);
    let mut estimate = StateAndCovariance::new(SVector::from(s2::X0), matrix(&s2::P0));
    time_and_halt(|| {
        for k in 0..s2::STEPS {
            if let Ok(next) =
                kf.step_with_options(&estimate, &measurement(k), CovarianceUpdateMethod::JosephForm)
            {
                estimate = next;
            }
        }
        *estimate.state()
    })
}
