//! Accuracy, consistency, stability and allocation numbers for the comparison table.
//!
//! Prints CSV rows in the `results/results.csv` schema. Usage:
//! `cargo run --release --example accuracy -- "commit,cpu,os,toolchain,date" >> results/results.csv`
//!
//! Every library reads the same frozen scenario files. A metric a library can't produce (such
//! as a UKF from adskalman) is left out, which the table shows as "n/a".

#[path = "../tests/common/mod.rs"]
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::baseline::{self, Models, METHODS};
use common::naive::NaiveKf;
use common::scenarios::{range_bearing, Scenario};
use common::vectors::load;
use nalgebra::{Cholesky, Const, DimMin, Matrix2, Matrix2x4, Matrix4, SMatrix, SVector};
use nalgebra::{Vector2, Vector4};
use typed_kalman::diagnostics::nees;
use typed_kalman::{Ekf, LinearKf, MeasurementJacobian, MeasurementModel, ProcessJacobian};
use typed_kalman::{ProcessModel, SqrtKf, SqrtUkf, Ukf};

/// Counts heap allocations, for the `heap_allocations` metric.
struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Returns how many allocations `f` makes.
fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCS.load(Ordering::SeqCst);
    f();
    ALLOCS.load(Ordering::SeqCst) - before
}

/// Prints CSV rows with versions and the shared environment fields filled in.
struct Report {
    env: String,
    commit: String,
    adskalman_version: String,
}

impl Report {
    /// Prints one row. The arguments are the CSV schema's columns, in order.
    #[allow(clippy::too_many_arguments)]
    fn row(
        &self,
        library: &str,
        scenario: &str,
        filter: &str,
        precision: &str,
        metric: &str,
        value: f64,
        unit: &str,
    ) {
        let version = if library.starts_with("adskalman") {
            &self.adskalman_version
        } else {
            &self.commit
        };
        println!(
            "{library},{version},{scenario},{filter},{precision},{metric},{value},{unit},{}",
            self.env
        );
    }
}

/// Reads adskalman's resolved version from `Cargo.lock`.
fn adskalman_version() -> String {
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock")).unwrap();
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line == "name = \"adskalman\"" {
            let version = lines.next().unwrap();
            return version
                .trim_start_matches("version = ")
                .trim_matches('"')
                .to_string();
        }
    }
    panic!("adskalman isn't in Cargo.lock");
}

/// Accumulates squared error and NEES over many steps.
#[derive(Default)]
struct Errors {
    squared: f64,
    nees: f64,
    count: usize,
}

impl Errors {
    fn add<const N: usize>(
        &mut self,
        truth: &SVector<f64, N>,
        x: &SVector<f64, N>,
        p: &SMatrix<f64, N, N>,
    ) {
        self.squared += (truth - x).norm_squared();
        self.nees += nees(truth, x, p).unwrap();
        self.count += 1;
    }

    fn report(&self, r: &Report, library: &str, scenario: &str, filter: &str) {
        let n = self.count as f64;
        r.row(
            library,
            scenario,
            filter,
            "float64",
            "rmse",
            (self.squared / n).sqrt(),
            "state",
        );
        r.row(
            library,
            scenario,
            filter,
            "float64",
            "nees",
            self.nees / n,
            "-",
        );
    }
}

/// The factor of `m`, for the square-root filters.
fn sqrt<const D: usize>(m: &SMatrix<f64, D, D>) -> SMatrix<f64, D, D> {
    Cholesky::new(*m).unwrap().unpack()
}

/// RMSE, NEES, agreement and allocations for the linear scenarios S1, S2 and S5.
fn linear<const N: usize, const M: usize>(r: &Report, id: &str, max_diff: bool)
where
    Const<M>: DimMin<Const<M>, Output = Const<M>>,
{
    let (_, runs) = load::<N, M>(id);
    let sc = &runs[0];
    let (q_sqrt, r_sqrt) = (sqrt(&sc.q), sqrt(&sc.r));

    let mut ours = Vec::with_capacity(sc.zs.len());
    let mut errors = Errors::default();
    let mut kf = LinearKf::new(sc.x0, sc.p0);
    for (z, truth) in sc.zs.iter().zip(&sc.truth) {
        kf.predict(&sc.f, &sc.q);
        kf.update(&sc.h, z, &sc.r).unwrap();
        errors.add(truth, kf.state(), kf.covariance());
        ours.push(*kf.state());
    }
    errors.report(r, "typed-kalman", id, "KF");

    let mut errors = Errors::default();
    let mut skf = SqrtKf::new(sc.x0, sc.p0).unwrap();
    for (z, truth) in sc.zs.iter().zip(&sc.truth) {
        skf.predict(&sc.f, &q_sqrt).unwrap();
        skf.update(&sc.h, z, &r_sqrt).unwrap();
        errors.add(truth, skf.state(), &skf.covariance());
    }
    errors.report(r, "typed-kalman-sqrt", id, "KF");

    let models = Models::<N, M>::new(sc);
    for (name, method) in METHODS {
        let mut errors = Errors::default();
        let mut diff = 0.0f64;
        let mut k = 0;
        models.run(&sc.zs, method, |e| {
            errors.add(&sc.truth[k], e.state(), e.covariance());
            diff = diff.max((e.state() - ours[k]).abs().max());
            k += 1;
        });
        errors.report(r, name, id, "KF");
        if max_diff {
            r.row(name, id, "KF", "float64", "max_abs_diff", diff, "state");
        }
    }

    // Allocations during the step loop only; setup happens first.
    let mut kf = LinearKf::new(sc.x0, sc.p0);
    let ours_allocs = allocations(|| {
        for z in &sc.zs {
            kf.predict(&sc.f, &sc.q);
            kf.update(&sc.h, z, &sc.r).unwrap();
        }
    });
    r.row(
        "typed-kalman",
        id,
        "KF",
        "float64",
        "heap_allocations",
        ours_allocs as f64,
        "count",
    );
    let mut skf = SqrtKf::new(sc.x0, sc.p0).unwrap();
    let sqrt_allocs = allocations(|| {
        for z in &sc.zs {
            skf.predict(&sc.f, &q_sqrt).unwrap();
            skf.update(&sc.h, z, &r_sqrt).unwrap();
        }
    });
    r.row(
        "typed-kalman-sqrt",
        id,
        "KF",
        "float64",
        "heap_allocations",
        sqrt_allocs as f64,
        "count",
    );
    for (name, method) in METHODS {
        let theirs = allocations(|| {
            models.run(&sc.zs, method, |e| {
                std::hint::black_box(e);
            })
        });
        r.row(
            name,
            id,
            "KF",
            "float64",
            "heap_allocations",
            theirs as f64,
            "count",
        );
    }
}

/// S3's models: linear constant-velocity motion, range-bearing measurement.
struct RangeBearingModel {
    f: Matrix4<f64>,
}

impl ProcessModel<4> for RangeBearingModel {
    fn predict(&self, x: &Vector4<f64>, _dt: f64) -> Vector4<f64> {
        self.f * x
    }
}

impl ProcessJacobian<4> for RangeBearingModel {
    fn jacobian(&self, _x: &Vector4<f64>, _dt: f64) -> Matrix4<f64> {
        self.f
    }
}

impl MeasurementModel<4, 2> for RangeBearingModel {
    fn measure(&self, x: &Vector4<f64>) -> Vector2<f64> {
        range_bearing(x)
    }
}

impl MeasurementJacobian<4, 2> for RangeBearingModel {
    fn jacobian(&self, x: &Vector4<f64>) -> Matrix2x4<f64> {
        let r2 = x[0] * x[0] + x[1] * x[1];
        let range = r2.sqrt();
        #[rustfmt::skip]
        let h = Matrix2x4::new(
            x[0] / range, x[1] / range, 0.0, 0.0,
            -x[1] / r2,   x[0] / r2,    0.0, 0.0,
        );
        h
    }
}

/// RMSE and NEES on the nonlinear scenario S3, over all 200 runs.
fn s3(r: &Report) {
    let (model, runs) = load::<4, 2>("S3");
    let dt = model.dt;
    let mut ekf_err = Errors::default();
    let mut ukf_err = Errors::default();
    let mut sr_err = Errors::default();
    let mut ads_err = Errors::default();

    for sc in &runs {
        let m = RangeBearingModel { f: sc.f };
        let (q_sqrt, r_sqrt): (Matrix4<f64>, Matrix2<f64>) = (sqrt(&sc.q), sqrt(&sc.r));
        let mut ekf = Ekf::new(sc.x0, sc.p0);
        let mut ukf = Ukf::new(sc.x0, sc.p0);
        let mut sr = SqrtUkf::new(sc.x0, sc.p0).unwrap();
        for (z, truth) in sc.zs.iter().zip(&sc.truth) {
            ekf.predict(&m, &sc.q, dt);
            ekf.update(&m, z, &sc.r).unwrap();
            ekf_err.add(truth, ekf.state(), ekf.covariance());
            ukf.predict(&m, &sc.q, dt).unwrap();
            ukf.update(&m, z, &sc.r).unwrap();
            ukf_err.add(truth, ukf.state(), ukf.covariance());
            sr.predict(&m, &q_sqrt, dt).unwrap();
            sr.update(&m, z, &r_sqrt).unwrap();
            sr_err.add(truth, sr.state(), &sr.covariance());
        }
        let mut k = 0;
        baseline::adskalman_ekf_s3(sc, |e| {
            ads_err.add(&sc.truth[k], e.state(), e.covariance());
            k += 1;
        })
        .unwrap();
    }
    ekf_err.report(r, "typed-kalman", "S3", "EKF");
    ads_err.report(r, "adskalman-joseph", "S3", "EKF");
    ukf_err.report(r, "typed-kalman", "S3", "UKF");
    sr_err.report(r, "typed-kalman-sqrt", "S3", "UKF");
}

/// Whether `p` is finite, symmetric to within 1e-4 of its largest entry, and positive-definite.
fn is_spd(p: &Matrix4<f32>) -> bool {
    p.iter().all(|v| v.is_finite())
        && (p - p.transpose()).abs().max() <= 1e-4 * p.abs().max()
        && Cholesky::new(*p).is_some()
}

/// Returns the number of steps before `step` first fails, or the scenario length if it never
/// does.
fn steps_to_failure(steps: usize, mut step: impl FnMut(usize) -> bool) -> f64 {
    (0..steps).find(|&k| !step(k)).unwrap_or(steps) as f64
}

/// Stability on S4 in 32-bit floats: steps until the covariance stops being symmetric
/// positive-definite (or the factor degenerates, for the square-root filter).
fn s4(r: &Report) {
    let (_, runs) = load::<4, 2>("S4");
    let sc: &Scenario<4, 2> = &runs[0];
    let steps = sc.zs.len();
    let (f, h, q, rr) = (
        sc.f.cast::<f32>(),
        sc.h.cast::<f32>(),
        sc.q.cast::<f32>(),
        sc.r.cast::<f32>(),
    );
    let (x0, p0) = (sc.x0.cast::<f32>(), sc.p0.cast::<f32>());
    let zs: Vec<Vector2<f32>> = sc.zs.iter().map(|z| z.cast()).collect();
    let row = |library: &str, value: f64| {
        r.row(
            library,
            "S4",
            "KF",
            "float32",
            "steps_to_failure",
            value,
            "steps",
        )
    };

    let mut kf = LinearKf::new(x0, p0);
    row(
        "typed-kalman",
        steps_to_failure(steps, |k| {
            kf.predict(&f, &q);
            kf.update(&h, &zs[k], &rr).is_ok() && is_spd(kf.covariance())
        }),
    );

    let (q_sqrt, r_sqrt) = (sqrt(&sc.q).cast::<f32>(), sqrt(&sc.r).cast::<f32>());
    let mut skf = SqrtKf::new(x0, p0).unwrap();
    row(
        "typed-kalman-sqrt",
        steps_to_failure(steps, |k| {
            let ok = skf
                .predict(&f, &q_sqrt)
                .and_then(|_| skf.update(&h, &zs[k], &r_sqrt))
                .is_ok();
            let s = skf.sqrt_covariance();
            ok && s.iter().all(|v| v.is_finite()) && (0..4).all(|i| s[(i, i)] > 0.0)
        }),
    );

    let models = Models::<4, 2, f32>::new(sc);
    for (name, method) in METHODS {
        // Stops at the first failure: an adskalman error, or a covariance that isn't SPD.
        let mut survived = 0;
        let mut healthy = true;
        let _ = models.try_run(&zs, method, |e| {
            if healthy && is_spd(e.covariance()) {
                survived += 1;
            } else {
                healthy = false;
            }
        });
        row(name, survived as f64);
    }

    let mut naive = NaiveKf { x: x0, p: p0 };
    row(
        "naive",
        steps_to_failure(steps, |k| {
            naive.predict(&f, &q);
            naive.update(&h, &zs[k], &rr) && is_spd(&naive.p)
        }),
    );
}

fn main() {
    let env = std::env::args()
        .nth(1)
        .expect("usage: accuracy \"commit,cpu,os,toolchain,date\"");
    let commit = env.split(',').next().unwrap().to_string();
    let r = Report {
        env,
        commit,
        adskalman_version: adskalman_version(),
    };
    linear::<2, 1>(&r, "S1", true);
    linear::<4, 2>(&r, "S2", true);
    s3(&r);
    s4(&r);
    linear::<15, 6>(&r, "S5", false);
}
