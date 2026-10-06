//! Dimension and precision mismatches are compile errors, not runtime failures.
//!
//! Each file in `tests/compile_fail/` must fail to compile with the error stored in the `.stderr`
//! file next to it. After a deliberate change to an error, regenerate them with
//! `TRYBUILD=overwrite cargo test --test compile_fail` and review the diff.

#[test]
fn mismatches_do_not_compile() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
