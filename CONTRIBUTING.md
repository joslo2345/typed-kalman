# Contributing to typed-kalman

Thanks for helping! Bug reports, questions, benchmark numbers from real hardware, and pull
requests are all welcome. Please open an issue before a large change, so we can agree on the
design first.

## Ground rules

These hold for every filter and every change:

- **Never panic on data.** Bad input (NaN, infinity, a singular or indefinite covariance)
  returns a `KalmanError` and leaves the filter exactly as it was.
- **Never allocate** in a filter step. `tests/no_alloc.rs` checks this with a counting
  allocator.
- **Work without `std`, in `f32` and `f64`.** Call math through `nalgebra::ComplexField` /
  `RealField` (for example `ComplexField::sqrt(x)`), not `x.sqrt()`: inherent float methods don't
  exist without `std`.
- **Stay on Rust 1.89** (the MSRV, set by nalgebra 0.35).
- **Document every public item.** `#![deny(missing_docs)]` enforces it.

## Before you open a pull request

Run what CI runs:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --all-targets --no-default-features --features autodiff -- -D warnings
cargo test --all-features
cargo test --no-default-features
cargo build --target thumbv7em-none-eabihf --no-default-features --features autodiff
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

The bare-metal build needs `rustup target add thumbv7em-none-eabihf`. CI also runs the tests on
beta and Rust 1.89, and builds the firmware in `bench/firmware`.

## Things that need care

- **Compile-fail tests** (`tests/compile_fail/`) compare compiler messages with stored `.stderr`
  files, which only match the toolchain CI pins (see the `ui` job). Set `SKIP_UI_TESTS=1` on
  other toolchains. If you change an error on purpose, regenerate them with
  `TRYBUILD=overwrite cargo test --test compile_fail` on that toolchain, and review the diff.
- **Benchmark scenarios** in `tests/vectors/` are frozen: other implementations and published
  results depend on them. Don't regenerate them.
- **Performance changes:** compare variants side by side in one benchmark binary. Separate
  runs on a laptop drift by 5–15%, enough to reverse a conclusion. Pull requests get an
  automatic check that fails if one of our benchmarks becomes more than 10% slower.

## License

By contributing, you agree that your contributions are dual-licensed under MIT and Apache-2.0,
as the project is, without any additional terms or conditions.
