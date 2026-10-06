//! Writes the five benchmark scenarios to `tests/vectors/`.
//!
//! The files are frozen once results have been collected; `tests/vectors.rs` checks that they
//! still match this generator. Run with `cargo run --release --example generate_vectors`.

#[path = "../tests/common/mod.rs"]
mod common;

use common::catalog;
use common::vectors::write;

fn main() {
    let (spec, runs) = catalog::s1();
    write(&spec, &runs);
    let (spec, runs) = catalog::s2();
    write(&spec, &runs);
    let (spec, runs) = catalog::s3();
    write(&spec, &runs);
    let (spec, runs) = catalog::s4();
    write(&spec, &runs);
    let (spec, runs) = catalog::s5();
    write(&spec, &runs);
    println!("wrote S1-S5 to tests/vectors/");
}
