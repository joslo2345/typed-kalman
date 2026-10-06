# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

`kalman-rust-repo-guide.md` is the design spec and build plan for the crate `kalman-rs` (crate import name `kalman_rs`). Treat it as the source of truth, and read the relevant section before implementing a step. Steps 1–4 (scaffold, layout, `Cargo.toml`, `lib.rs`) are done. The crate lives at the repo root, not in a `kalman-rs/` subdirectory. Step 5 is done: the model traits are in `src/model.rs` (not listed in the guide's layout), `KalmanError` is in `src/error.rs`, and `Ekf<N>` is implemented in `src/ekf.rs`. Step 6 items 1–2 are done: `LinearKf<N>` in `src/linear.rs` takes the F/H/Q/R matrices directly. Both filters share the private `update::joseph` function, which computes the result without modifying the filter so that errors leave the state unchanged. `Ukf<N>` (with `UkfParams`) is in `src/ukf.rs`, and `SqrtUkf<N>` is in `src/sqrt_ukf.rs`, which completes Step 6 item 3. These types are re-exported from the crate root. `smoother` and `diagnostics` are still doc-only stubs, and work continues with Step 6 item 4 (RTS smoother).

Dependency versions are newer than the ones in the guide: `nalgebra` 0.35 (which matches what `adskalman` 0.18 depends on, so their matrix types are compatible) and `criterion` 0.8 (use `std::hint::black_box` instead of `criterion::black_box`).

## Goal

A type-safe, `no_std`, allocation-free Kalman filter crate built on `nalgebra` static matrices (`SMatrix`/`SVector`) with const-generic dimensions, so dimension mismatches fail at compile time. Filter family: linear KF, EKF, UKF, square-root UKF, RTS smoother, NIS/NEES diagnostics, optional dual-number autodiff Jacobians. Licensed `MIT OR Apache-2.0`.

On accuracy the target is to **match** existing crates (for linear Gaussian systems the KF is optimal), not to beat them. The claimed advantages are stability, compile-time safety, zero allocation, `no_std`, error handling, and features that `adskalman` lacks (UKF, autodiff, diagnostics).

## Planned commands

```bash
cargo build
cargo test                                   # all tests
cargo test --test equivalence                # one integration test file
cargo test --test stability covariance_stays # one test by name filter
cargo clippy -- -D warnings
cargo fmt --check
cargo doc
cargo run --example constant_velocity

# no_std check (must pass on every commit)
rustup target add thumbv7em-none-eabihf
cargo build --target thumbv7em-none-eabihf --no-default-features

cargo deny check                             # licenses and advisories

# Benchmark comparison pipeline (from the repo root)
ENV="$(git rev-parse --short HEAD),$(uname -m),$(uname -s),$(rustc --version | cut -d' ' -f2),$(date -I)"
cargo bench --bench compare
python scripts/criterion_to_csv.py "$ENV"
cargo run --release --example accuracy -- "$ENV" >> results/results.csv
python scripts/make_table.py results/results.csv kalman-rs
```

CI runs tests on stable, beta, and the MSRV, plus clippy, fmt, the thumbv7em `no_std` build, `cargo deny`, and `cargo doc`.

## Architecture and design rules

- **Features:** `std` is on by default and maps to `nalgebra/std`. `nalgebra` uses `default-features = false, features = ["libm"]`. `autodiff` gates the dual-number Jacobians. The RTS smoother goes behind `std` if it needs to store history. `lib.rs` uses `#![cfg_attr(not(feature = "std"), no_std)]` and `#![deny(missing_docs)]`, so every public item needs docs.
- **Model traits:** these differ from the guide. `ProcessModel<N>` (`predict`) and `MeasurementModel<N, M>` (`measure`) are function-only, and that's all the UKF needs. `ProcessJacobian<N>` and `MeasurementJacobian<N, M>` extend them with `jacobian`, and the EKF requires those. Filters hold `x: SVector<N>` and `p: SMatrix<N, N>`.
- **UKF sigma points:** stable Rust can't use `2N + 1` as a const generic, so `SigmaPoints<D, N>` stores a center point plus `plus`/`minus` `D × N` matrices. The default `UkfParams` are `alpha = 1, beta = 2, kappa = 0`, which keep every weight non-negative. UKF `predict` returns `Result` because drawing sigma points needs a Cholesky factorization of P.
- **Square-root UKF:** it stores the lower-triangular factor `S` (`P = S Sᵀ`), and `S` is always kept lower-triangular because the rank-one update and downdate functions in `sqrt_ukf.rs` depend on that. Instead of the textbook QR of a `D × (2N + K)` matrix, which can't be expressed with const generics, `weighted_factor` builds factors with Givens rank-one updates, starting from zero. Noise is passed as square roots (`q_sqrt: SMatrix<N, K>`, `r_sqrt: SMatrix<M, K>`, any `G` with `Q = G Gᵀ`), so singular or low-rank noise works. It reuses `Weights` and `SigmaPoints` from `ukf.rs`, which are `pub(crate)`.
- **`no_std` math:** `f64::sqrt` and similar functions don't exist in `core`. Call them through `nalgebra::ComplexField` (for example `ComplexField::sqrt(x)`), which uses `libm`.
- **Updates:** the KF and EKF use the Joseph-form covariance update. The UKF uses `P − K S Kᵀ` and rejects a result that isn't positive-definite. `update` returns `Result<f64, KalmanError>`, where the `f64` is the NIS. Failures such as NaN input or a singular innovation covariance return `Err` and leave the filter state unchanged. Never panic.
- **Filter loop never allocates.** This is checked by a counting global allocator.
- **Implementation order:** KF (Joseph), then EKF, UKF + sqrt-UKF, RTS smoother, NIS/NEES, autodiff, and finally a generic scalar for `f32` support.

## Testing layout and constraints

- `tests/common/` holds shared helpers. `scenarios.rs` has the problem generators with fixed seeds. `baseline.rs` is the **only** place that calls `adskalman`, so an API change touches one file. `naive.rs` is an unstabilized textbook filter used for comparison.
- `tests/no_alloc.rs` must stay in its own file, because its `#[global_allocator]` applies to the whole test binary.
- `tests/compile_fail.rs` uses `trybuild` over `tests/compile_fail/*.rs`, and the expected compiler errors live in `.stderr` files next to each case. Regenerate them deliberately (`TRYBUILD=overwrite`) and review the diff.
- Pass conditions: equivalence with `adskalman` within 1e-9; `P` stays symmetric and positive-definite under `proptest` ill-conditioned problems and over a 1M-step `f32` run; Monte Carlo NIS/NEES inside the 95% chi-squared bounds; speed equal to or better than `adskalman-joseph` (the like-for-like baseline), with no regression over 10%.
- Comparison crates (`adskalman`, `approx`, `proptest`, `criterion`, `trybuild`) are dev-dependencies only.

## Benchmarks

- Five frozen scenarios (S1–S5) are shared across sibling C, C++, Python, and Rust implementations through a test-vectors Git submodule at `tests/vectors/`. Don't change scenario files after results have been collected.
- Criterion group names must be `<Scenario>_<Filter>_<precision>` (e.g. `S2_KF_float64`) and function names must be the library name, because `scripts/criterion_to_csv.py` parses them from `target/criterion/`.
- Results are appended to `results/results.csv` using the schema `library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date`. `make_table.py` writes the README table between `<!-- BENCH:START -->` and `<!-- BENCH:END -->`.
- Benchmark `adskalman` with all three `CovarianceUpdateMethod`s. A feature a baseline lacks is reported as "n/a", never as a failure.
- Embedded numbers come from a separate firmware crate in `bench/firmware` (DWT cycle counter, `cargo size`), with an identical build profile for each library.

## Inconsistencies in the guide to resolve when scaffolding

- The `Cargo.toml` snippet declares `[[bench]] name = "filters"`, but the comparison pipeline runs `--bench compare` (`benches/compare.rs`). Each bench file needs its own `[[bench]]` entry with `harness = false`.
- The adskalman adapter is shown both at `tests/common/baseline.rs` and at `bench/baselines/adskalman_adapter.rs`. Pick one location that both tests and benches can share.
