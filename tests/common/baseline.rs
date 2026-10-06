//! Adapter around adskalman, the baseline we compare against.
//!
//! This is the only file that calls adskalman, so an API change in it touches nothing else.
#![allow(non_snake_case)] // adskalman's trait methods are named F, Q, H, R.

use adskalman::{
    CovarianceUpdateMethod, KalmanFilterNoControl, ObservationModel, StateAndCovariance,
    TransitionModelLinearNoControl,
};
use nalgebra::{Const, DimMin, OMatrix, OVector, RealField, U2, U4};

use super::scenarios::{self, Scenario};

pub use adskalman::CovarianceUpdateMethod as Method;

/// Every covariance update adskalman offers, with the library names used in results.
pub const METHODS: [(&str, Method); 3] = [
    ("adskalman-joseph", Method::JosephForm),
    ("adskalman-optimal", Method::OptimalKalman),
    (
        "adskalman-optimal-symmetric",
        Method::OptimalKalmanForcedSymmetric,
    ),
];

struct Motion<const N: usize, R: RealField + Copy> {
    f: OMatrix<R, Const<N>, Const<N>>,
    ft: OMatrix<R, Const<N>, Const<N>>,
    q: OMatrix<R, Const<N>, Const<N>>,
}

impl<const N: usize, R: RealField + Copy> TransitionModelLinearNoControl<R, Const<N>>
    for Motion<N, R>
{
    fn F(&self) -> &OMatrix<R, Const<N>, Const<N>> {
        &self.f
    }
    fn FT(&self) -> &OMatrix<R, Const<N>, Const<N>> {
        &self.ft
    }
    fn Q(&self) -> &OMatrix<R, Const<N>, Const<N>> {
        &self.q
    }
}

struct Observation<const N: usize, const M: usize, R: RealField + Copy> {
    h: OMatrix<R, Const<M>, Const<N>>,
    ht: OMatrix<R, Const<N>, Const<M>>,
    r: OMatrix<R, Const<M>, Const<M>>,
}

impl<const N: usize, const M: usize, R: RealField + Copy> ObservationModel<R, Const<N>, Const<M>>
    for Observation<N, M, R>
where
    Const<M>: DimMin<Const<M>, Output = Const<M>>,
{
    fn H(&self) -> &OMatrix<R, Const<M>, Const<N>> {
        &self.h
    }
    fn HT(&self) -> &OMatrix<R, Const<N>, Const<M>> {
        &self.ht
    }
    fn R(&self) -> &OMatrix<R, Const<M>, Const<M>> {
        &self.r
    }
}

/// adskalman's models for a linear scenario in precision `R`, built once so benchmarks don't
/// time the setup.
pub struct Models<const N: usize, const M: usize, R: RealField + Copy = f64> {
    motion: Motion<N, R>,
    observation: Observation<N, M, R>,
    x0: OVector<R, Const<N>>,
    p0: OMatrix<R, Const<N>, Const<N>>,
}

impl<const N: usize, const M: usize, R: RealField + Copy> Models<N, M, R>
where
    Const<M>: DimMin<Const<M>, Output = Const<M>>,
{
    /// Builds the models for `sc`, converted to precision `R`.
    pub fn new(sc: &Scenario<N, M>) -> Self {
        let f = sc.f.cast::<R>();
        let h = sc.h.cast::<R>();
        Self {
            motion: Motion {
                f,
                ft: f.transpose(),
                q: sc.q.cast(),
            },
            observation: Observation {
                h,
                ht: h.transpose(),
                r: sc.r.cast(),
            },
            x0: sc.x0.cast(),
            p0: sc.p0.cast(),
        }
    }

    /// Runs adskalman over `zs` with `method`, calling `on_step` with each estimate.
    ///
    /// Each step predicts and then updates, starting from the prior, the same order as
    /// our filters. Returns the first error, if any.
    pub fn try_run(
        &self,
        zs: &[OVector<R, Const<M>>],
        method: CovarianceUpdateMethod,
        mut on_step: impl FnMut(&StateAndCovariance<R, Const<N>>),
    ) -> Result<(), adskalman::Error> {
        let kf = KalmanFilterNoControl::new(&self.motion, &self.observation);
        let mut estimate = StateAndCovariance::new(self.x0, self.p0);
        for z in zs {
            estimate = kf.step_with_options(&estimate, z, method)?;
            on_step(&estimate);
        }
        Ok(())
    }

    /// Runs adskalman over `zs`, panicking on error.
    pub fn run(
        &self,
        zs: &[OVector<R, Const<M>>],
        method: CovarianceUpdateMethod,
        on_step: impl FnMut(&StateAndCovariance<R, Const<N>>),
    ) {
        self.try_run(zs, method, on_step)
            .expect("adskalman step failed");
    }

    /// Runs adskalman's RTS smoother over `zs`.
    pub fn smooth(&self, zs: &[OVector<R, Const<M>>]) -> Vec<StateAndCovariance<R, Const<N>>> {
        let kf = KalmanFilterNoControl::new(&self.motion, &self.observation);
        kf.smooth(&StateAndCovariance::new(self.x0, self.p0), zs)
            .expect("adskalman smoother failed")
    }
}

/// Returns adskalman's Joseph-form estimates for every step of `sc`.
pub fn adskalman_filter<const N: usize, const M: usize>(
    sc: &Scenario<N, M>,
) -> Vec<StateAndCovariance<f64, Const<N>>>
where
    Const<M>: DimMin<Const<M>, Output = Const<M>>,
{
    let mut out = Vec::with_capacity(sc.zs.len());
    Models::<N, M>::new(sc).run(&sc.zs, Method::JosephForm, |e| out.push(e.clone()));
    out
}

/// A range-bearing observation linearized at one state, for adskalman's EKF-style use: its
/// `H` is the Jacobian there, and `predict_observation` is the nonlinear model.
struct RangeBearing {
    h: OMatrix<f64, U2, U4>,
    ht: OMatrix<f64, U4, U2>,
    r: OMatrix<f64, U2, U2>,
}

impl RangeBearing {
    fn linearized_at(x: &OVector<f64, U4>, r: OMatrix<f64, U2, U2>) -> Self {
        let r2 = x[0] * x[0] + x[1] * x[1];
        let range = r2.sqrt();
        #[rustfmt::skip]
        let h = OMatrix::<f64, U2, U4>::new(
            x[0] / range, x[1] / range, 0.0, 0.0,
            -x[1] / r2,   x[0] / r2,    0.0, 0.0,
        );
        Self {
            h,
            ht: h.transpose(),
            r,
        }
    }
}

impl ObservationModel<f64, U4, U2> for RangeBearing {
    fn predict_observation(&self, state: &OVector<f64, U4>) -> OVector<f64, U2> {
        scenarios::range_bearing(state)
    }
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

/// Runs adskalman as an EKF on the range-bearing scenario S3: its linear transition model
/// predicts, then an observation model linearized at the prediction updates (Joseph form).
pub fn adskalman_ekf_s3(
    sc: &Scenario<4, 2>,
    mut on_step: impl FnMut(&StateAndCovariance<f64, U4>),
) -> Result<(), adskalman::Error> {
    let motion = Motion {
        f: sc.f,
        ft: sc.f.transpose(),
        q: sc.q,
    };
    let mut estimate = StateAndCovariance::new(sc.x0, sc.p0);
    for z in &sc.zs {
        let prior = motion.predict(&estimate);
        let observation = RangeBearing::linearized_at(prior.state(), sc.r);
        estimate = observation.update(&prior, z, Method::JosephForm)?;
        on_step(&estimate);
    }
    Ok(())
}
