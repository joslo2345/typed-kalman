//! The filter loops never allocate.
//!
//! The counting global allocator applies to this whole test binary, so this must stay the only
//! test in this file.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::{Cholesky, Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};
use typed_kalman::smoother::{smooth, Estimate, RtsStep};
use typed_kalman::{Ekf, LinearKf, MeasurementJacobian, MeasurementModel, ProcessJacobian};
use typed_kalman::{ProcessModel, SqrtKf, SqrtUkf, Ukf};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The scenario's linear model, for the filters that take model traits.
struct Linear {
    f: Matrix4<f64>,
    h: Matrix2x4<f64>,
}

impl ProcessModel<4> for Linear {
    fn predict(&self, x: &Vector4<f64>, _dt: f64) -> Vector4<f64> {
        self.f * x
    }
}

impl ProcessJacobian<4> for Linear {
    fn jacobian(&self, _x: &Vector4<f64>, _dt: f64) -> Matrix4<f64> {
        self.f
    }
}

impl MeasurementModel<4, 2> for Linear {
    fn measure(&self, x: &Vector4<f64>) -> Vector2<f64> {
        self.h * x
    }
}

impl MeasurementJacobian<4, 2> for Linear {
    fn jacobian(&self, _x: &Vector4<f64>) -> Matrix2x4<f64> {
        self.h
    }
}

#[test]
fn filter_loops_do_not_allocate() {
    // Everything that allocates happens before counting starts.
    const STEPS: usize = 100;
    let sc = common::scenarios::constant_velocity(STEPS, 1);
    let model = Linear { f: sc.f, h: sc.h };
    let q_sqrt = Cholesky::new(sc.q).unwrap().unpack();
    let r_sqrt: Matrix2<f64> = Cholesky::new(sc.r).unwrap().unpack();

    let mut kf = LinearKf::new(sc.x0, sc.p0);
    let mut ekf = Ekf::new(sc.x0, sc.p0);
    let mut ukf = Ukf::new(sc.x0, sc.p0);
    let mut sr = SqrtUkf::new(sc.x0, sc.p0).unwrap();
    let mut skf = SqrtKf::new(sc.x0, sc.p0).unwrap();
    let prior = Estimate { x: sc.x0, p: sc.p0 };
    let mut history = [RtsStep {
        f: sc.f,
        predicted: prior,
        filtered: prior,
    }; STEPS];
    let mut smoothed = [prior; STEPS];

    let before = ALLOCS.load(Ordering::SeqCst);
    for (z, step) in sc.zs.iter().zip(history.iter_mut()) {
        kf.predict(&sc.f, &sc.q);
        let predicted = Estimate {
            x: *kf.state(),
            p: *kf.covariance(),
        };
        kf.update(&sc.h, z, &sc.r).unwrap();
        *step = RtsStep {
            f: sc.f,
            predicted,
            filtered: Estimate {
                x: *kf.state(),
                p: *kf.covariance(),
            },
        };

        ekf.predict(&model, &sc.q, 0.1);
        ekf.update(&model, z, &sc.r).unwrap();
        ukf.predict(&model, &sc.q, 0.1).unwrap();
        ukf.update(&model, z, &sc.r).unwrap();
        sr.predict(&model, &q_sqrt, 0.1).unwrap();
        sr.update(&model, z, &r_sqrt).unwrap();
        skf.predict(&sc.f, &q_sqrt).unwrap();
        skf.update(&sc.h, z, &r_sqrt).unwrap();
    }
    smooth(&history, &mut smoothed).unwrap();
    let allocations = ALLOCS.load(Ordering::SeqCst) - before;

    assert_eq!(allocations, 0, "filter loops allocated {allocations} times");

    // Guard against a vacuous pass: the counter must see a real allocation.
    let counted = ALLOCS.load(Ordering::SeqCst);
    std::hint::black_box(Box::new(0u8));
    assert!(
        ALLOCS.load(Ordering::SeqCst) > counted,
        "allocation counter isn't counting"
    );
}
