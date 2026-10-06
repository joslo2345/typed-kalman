//! Dimension and precision mismatches are compile errors, not runtime failures.
//!
//! Each file in `tests/compile_fail/` must fail to compile with the error stored in the `.stderr`
//! file next to it. After a deliberate change to an error, regenerate them with
//! `TRYBUILD=overwrite cargo test --test compile_fail` and review the diff.
//!
//! Compiler messages change between Rust versions, so the `.stderr` files match only the
//! toolchain that generated them (CI pins it in its `ui` job). Set `SKIP_UI_TESTS=1` to skip
//! this test on other toolchains, as CI's stable, beta and MSRV jobs do.

#[test]
fn mismatches_do_not_compile() {
    if std::env::var_os("SKIP_UI_TESTS").is_some_and(|v| v == "1") {
        eprintln!("skipping compile-fail tests (SKIP_UI_TESTS=1)");
        return;
    }
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
