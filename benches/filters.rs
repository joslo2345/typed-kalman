//! Speed of predict + update against adskalman, on the same 1,000-step scenario.

#[path = "../tests/common/mod.rs"]
mod common;

use std::hint::black_box;

use common::baseline::{Method, Models};
use common::scenarios::{self, Scenario};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kalman_rs::linear::LinearKf;
use kalman_rs::SqrtKf;
use nalgebra::{Cholesky, Matrix2};

/// Runs our filter over `sc`, returning the final state so the work can't be optimized away.
fn run_ours(sc: &Scenario<4, 2>) -> nalgebra::Vector4<f64> {
    let mut kf = LinearKf::new(sc.x0, sc.p0);
    for z in &sc.zs {
        kf.predict(&sc.f, &sc.q);
        kf.update(&sc.h, z, &sc.r).unwrap();
    }
    *kf.state()
}

/// Runs our square-root filter over `sc`, with the noise factors precomputed.
fn run_ours_sqrt(
    sc: &Scenario<4, 2>,
    q_sqrt: &nalgebra::Matrix4<f64>,
    r_sqrt: &Matrix2<f64>,
) -> nalgebra::Vector4<f64> {
    let mut kf = SqrtKf::new(sc.x0, sc.p0).unwrap();
    for z in &sc.zs {
        kf.predict(&sc.f, q_sqrt).unwrap();
        kf.update(&sc.h, z, r_sqrt).unwrap();
    }
    *kf.state()
}

fn compare(c: &mut Criterion) {
    let sc = scenarios::constant_velocity(1000, 0);
    let models = Models::new(&sc);
    let mut group = c.benchmark_group("constant_velocity_1000_steps");
    group.throughput(Throughput::Elements(sc.zs.len() as u64));

    group.bench_function("kalman-rs", |b| b.iter(|| run_ours(black_box(&sc))));
    // Our square-root variant: slower, but stable far beyond the covariance form's range.
    let q_sqrt = Cholesky::new(sc.q).unwrap().unpack();
    let r_sqrt = Cholesky::new(sc.r).unwrap().unpack();
    group.bench_function("kalman-rs-sqrt", |b| {
        b.iter(|| run_ours_sqrt(black_box(&sc), &q_sqrt, &r_sqrt))
    });
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
