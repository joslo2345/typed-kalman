//! Time per predict + update step on the frozen scenarios, for the comparison table.
//!
//! Group names are `<Scenario>_<Filter>_<precision>` and function names are library names,
//! because `scripts/criterion_to_csv.py` reads them back from `target/criterion/`. Only the
//! filter loop is timed: loading, conversion and setup happen outside `iter`.

#[path = "../tests/common/mod.rs"]
mod common;

use std::hint::black_box;

use common::baseline::{Models, METHODS};
use common::scenarios::Scenario;
use common::vectors::load;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nalgebra::{Cholesky, Const, DimMin, SMatrix, SVector};
use typed_kalman::{Float, LinearKf, SqrtKf};

/// A scenario's inputs converted to precision `T` up front.
struct Inputs<const N: usize, const M: usize, T: Float> {
    f: SMatrix<T, N, N>,
    h: SMatrix<T, M, N>,
    q: SMatrix<T, N, N>,
    r: SMatrix<T, M, M>,
    q_sqrt: SMatrix<T, N, N>,
    r_sqrt: SMatrix<T, M, M>,
    x0: SVector<T, N>,
    p0: SMatrix<T, N, N>,
    zs: Vec<SVector<T, M>>,
}

impl<const N: usize, const M: usize, T: Float> Inputs<N, M, T> {
    fn new(sc: &Scenario<N, M>) -> Self {
        Self {
            f: sc.f.cast(),
            h: sc.h.cast(),
            q: sc.q.cast(),
            r: sc.r.cast(),
            q_sqrt: Cholesky::new(sc.q).unwrap().unpack().cast(),
            r_sqrt: Cholesky::new(sc.r).unwrap().unpack().cast(),
            x0: sc.x0.cast(),
            p0: sc.p0.cast(),
            zs: sc.zs.iter().map(|z| z.cast()).collect(),
        }
    }
}

fn run_linear<const N: usize, const M: usize, T: Float>(i: &Inputs<N, M, T>) -> SVector<T, N> {
    let mut kf = LinearKf::new(i.x0, i.p0);
    for z in &i.zs {
        kf.predict(&i.f, &i.q);
        kf.update(&i.h, z, &i.r).unwrap();
    }
    *kf.state()
}

fn run_sqrt<const N: usize, const M: usize, T: Float>(i: &Inputs<N, M, T>) -> SVector<T, N> {
    let mut kf = SqrtKf::new(i.x0, i.p0).unwrap();
    for z in &i.zs {
        kf.predict(&i.f, &i.q_sqrt).unwrap();
        kf.update(&i.h, z, &i.r_sqrt).unwrap();
    }
    *kf.state()
}

/// Benchmarks every linear filter on scenario `id` in precision `T`.
fn bench_linear<const N: usize, const M: usize, T: Float>(
    c: &mut Criterion,
    id: &str,
    precision: &str,
) where
    Const<M>: DimMin<Const<M>, Output = Const<M>>,
{
    let (_, runs) = load::<N, M>(id);
    let sc = &runs[0];
    let inputs = Inputs::<N, M, T>::new(sc);
    let baseline = Models::<N, M, T>::new(sc);

    let mut group = c.benchmark_group(format!("{id}_KF_{precision}"));
    group.throughput(Throughput::Elements(sc.zs.len() as u64));
    group.bench_function("typed-kalman", |b| {
        b.iter(|| run_linear(black_box(&inputs)))
    });
    group.bench_function("typed-kalman-sqrt", |b| {
        b.iter(|| run_sqrt(black_box(&inputs)))
    });
    for (name, method) in METHODS {
        group.bench_function(name, |b| {
            b.iter(|| {
                baseline.run(black_box(&inputs.zs), method, |e| {
                    black_box(e);
                })
            })
        });
    }
    group.finish();
}

fn s1(c: &mut Criterion) {
    bench_linear::<2, 1, f64>(c, "S1", "float64");
}

fn s2(c: &mut Criterion) {
    bench_linear::<4, 2, f64>(c, "S2", "float64");
    bench_linear::<4, 2, f32>(c, "S2", "float32");
}

fn s5(c: &mut Criterion) {
    bench_linear::<15, 6, f64>(c, "S5", "float64");
}

criterion_group!(benches, s1, s2, s5);
criterion_main!(benches);
