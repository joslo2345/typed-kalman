use criterion::{criterion_group, criterion_main, Criterion};

fn filters(_c: &mut Criterion) {}

criterion_group!(benches, filters);
criterion_main!(benches);
