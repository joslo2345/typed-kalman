//! Speed of predict + update against adskalman, on the same 1,000-step scenario.

#[path = "../tests/common/mod.rs"]
mod common;

use std::hint::black_box;

use common::baseline::{Method, Models};
use common::scenarios::{self, Scenario};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kalman_rs::linear::LinearKf;

/// Runs our filter over `sc`, returning the final state so the work can't be optimized away.
fn run_ours(sc: &Scenario<4, 2>) -> nalgebra::Vector4<f64> {
    let mut kf = LinearKf::new(sc.x0, sc.p0);
    for z in &sc.zs {
        kf.predict(&sc.f, &sc.q);
        kf.update(&sc.h, z, &sc.r).unwrap();
    }
    *kf.state()
}

fn compare(c: &mut Criterion) {
    let sc = scenarios::constant_velocity(1000, 0);
    let models = Models::new(&sc);
    let mut group = c.benchmark_group("constant_velocity_1000_steps");
    group.throughput(Throughput::Elements(sc.zs.len() as u64));

    group.bench_function("kalman-rs", |b| b.iter(|| run_ours(black_box(&sc))));
    // Joseph form is the like-for-like comparison; the other two trade stability for speed.
    for (name, method) in [
        ("adskalman-joseph", Method::JosephForm),
        ("adskalman-optimal", Method::OptimalKalman),
        (
            "adskalman-optimal-symmetric",
            Method::OptimalKalmanForcedSymmetric,
        ),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                models.run(black_box(&sc), method, |e| {
                    black_box(e);
                })
            })
        });
    }
    group.finish();
}

criterion_group!(benches, compare);
criterion_main!(benches);
