//! The frozen scenario files still match the generator, bit for bit.
//!
//! If this fails after a deliberate change to a scenario, every collected result is invalid:
//! regenerate with `cargo run --release --example generate_vectors`, update
//! `tests/vectors/SHA256SUMS`, and re-collect results in all four implementations.

mod common;

use common::catalog;
use common::scenarios::Scenario;
use common::vectors::load;

fn assert_matches<const N: usize, const M: usize>(id: &str, expected: &[Scenario<N, M>]) {
    let (model, loaded) = load::<N, M>(id);
    assert_eq!(model.runs, expected.len(), "{id}: run count");
    for (run, (got, want)) in loaded.iter().zip(expected).enumerate() {
        assert_eq!(
            (got.f, got.h, got.q, got.r),
            (want.f, want.h, want.q, want.r),
            "{id} model"
        );
        assert_eq!((got.x0, got.p0), (want.x0, want.p0), "{id} prior");
        if model.truth.is_some() {
            assert!(got.truth == want.truth, "{id} run {run}: truth differs");
        }
        let same_zs = match model.measurements.dtype.as_str() {
            "float32" => got
                .zs
                .iter()
                .zip(&want.zs)
                .all(|(g, w)| g.cast::<f32>() == w.cast::<f32>()),
            _ => got.zs == want.zs,
        };
        assert!(
            same_zs && got.zs.len() == want.zs.len(),
            "{id} run {run}: measurements differ"
        );
    }
}

#[test]
fn s1_matches_generator() {
    assert_matches("S1", &catalog::s1().1);
}

#[test]
fn s2_matches_generator() {
    assert_matches("S2", &catalog::s2().1);
}

#[test]
fn s3_matches_generator() {
    assert_matches("S3", &catalog::s3().1);
}

#[test]
fn s4_matches_generator() {
    assert_matches("S4", &catalog::s4().1);
}

#[test]
fn s5_matches_generator() {
    assert_matches("S5", &catalog::s5().1);
}
