# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**Start here.** `typed-kalman` **0.1.1** is published (crates.io, docs.rs) from
https://github.com/joslo2345/typed-kalman, where `master` is protected and CI passes. All 11
steps of the guide are done. The next work is **0.2.0**, planned in `ROADMAP.md` and tracked as
GitHub issues #1–#14 in the 0.2.0 milestone (#1, every `predict` returning `Result`, comes
first, since #3 depends on it). Issues #15–#18 are infrastructure. Waiting on the user: posting
the announcements (`docs/announcement-drafts.md`; they cover 0.1.0, so mention 0.1.1's
additions) and a board for cycle counts (#15; see `bench/firmware/README.md`).

`kalman-rust-repo-guide.md` is the design spec and build plan for the crate, which the guide calls `kalman-rs`. It's published as **`typed-kalman`** (import name `typed_kalman`), because crates.io already has `kalman_rs`, and it treats `-` and `_` as the same in names. Treat it as the source of truth, and read the relevant section before implementing a step. Steps 1–4 (scaffold, layout, `Cargo.toml`, `lib.rs`) are done. The crate lives at the repo root, not in a `kalman-rs/` subdirectory. Step 5 is done: the model traits are in `src/model.rs` (not listed in the guide's layout), `KalmanError` is in `src/error.rs`, and `Ekf<N>` is implemented in `src/ekf.rs`. Step 6 items 1–2 are done: `LinearKf<N>` in `src/linear.rs` takes the F/H/Q/R matrices directly. Both filters share the private `update::joseph` function, which computes the result without modifying the filter so that errors leave the state unchanged. `Ukf<N>` (with `UkfParams`) is in `src/ukf.rs`, and `SqrtUkf<N>` is in `src/sqrt_ukf.rs`, which completes Step 6 item 3. The RTS smoother is in `src/smoother.rs` (`smooth`, `RtsStep`, `Estimate`), which completes Step 6 item 4. These types are re-exported from the crate root. `src/diagnostics.rs` (Step 6 item 5) has `nees`, `chi_squared_quantile` and `chi_squared_bounds`, and these stay under `kalman_rs::diagnostics`. Step 6 item 6 is in `src/autodiff.rs`, behind the `autodiff` feature and not re-exported at the root. Step 6 item 7 (generic scalar) is done, which completes Step 6. Step 7 is done: the integration tests under `tests/`, `benches/filters.rs`, and a square-root linear KF, `SqrtKf` (`src/sqrt_kf.rs`, not in the guide), added so the guide's full-range stability test can pass. Step 8 is done except for the parts that need things this repo doesn't have: a shared test-vectors repo (the scenarios live in `tests/vectors/` for now) and a board for `cycles_per_step` (the firmware writes it to `CYCLES_PER_STEP` for a probe to read). Step 9 is done: `.github/workflows/ci.yml` and `deny.toml`. The repo has no remote yet, so the workflow has only been validated by running every job's commands locally. Step 10 is done: crate-level docs with a quick start and a filter-choice table in `src/lib.rs`, runnable `constant_velocity` and `imu_attitude` examples (the latter needs `--features autodiff`, declared with `required-features`), and the README.

Step 11 (release) is done; see "Releasing".

Dependency versions are newer than the ones in the guide: `nalgebra` 0.35 (which matches what `adskalman` 0.18 depends on, so their matrix types are compatible) and `criterion` 0.8 (use `std::hint::black_box` instead of `criterion::black_box`).

## Goal

A type-safe, `no_std`, allocation-free Kalman filter crate built on `nalgebra` static matrices (`SMatrix`/`SVector`) with const-generic dimensions, so dimension mismatches fail at compile time. Filter family: linear KF, EKF, UKF, square-root UKF, RTS smoother, NIS/NEES diagnostics, optional dual-number autodiff Jacobians. Licensed `MIT OR Apache-2.0`.

On accuracy the target is to **match** existing crates (for linear Gaussian systems the KF is optimal), not to beat them. The claimed advantages are stability, compile-time safety, zero allocation, `no_std`, error handling, and features that `adskalman` lacks (UKF, autodiff, diagnostics).

## Planned commands

```bash
cargo build
cargo test                                   # all tests
cargo test --no-default-features             # tests without std
cargo test --all-features                     # includes autodiff tests and doctest
cargo test --no-default-features --features autodiff
cargo test --lib sqrt_ukf                     # unit tests in one module
cargo test --test equivalence                # one integration test file
cargo test --test stability covariance_stays # one test by name filter
cargo clippy -- -D warnings
cargo fmt --check
cargo doc
cargo run --example constant_velocity

# no_std check (must pass on every commit)
rustup target add thumbv7em-none-eabihf
cargo build --target thumbv7em-none-eabihf --no-default-features
cargo build --target thumbv7em-none-eabihf --no-default-features --features autodiff

cargo deny check                             # advisories, bans, licenses, sources (deny.toml)
SKIP_UI_TESTS=1 cargo test                   # skip compile-fail snapshots (any toolchain but 1.95.0)
cargo +1.89 test --all-features              # MSRV; see the macOS SDK note below
python3 scripts/check_regression.py 0.10     # after cargo bench --bench compare -- --baseline <name>

# Comparison numbers: everything, about 4 minutes; prints the README table
scripts/run_comparison.sh
cargo bench --bench compare -- 'S2_KF_float32'   # one benchmark group
cargo run --release --example generate_vectors   # rewrite tests/vectors/ (frozen: don't, once results exist)
(cd bench/firmware && cargo build --release)     # firmware images; scripts/firmware_sizes.sh reports their sizes
```

CI (`.github/workflows/ci.yml`) runs:

- `test`: stable, beta and 1.89 (the MSRV), with all four feature combinations.
- `ui`: compile-fail snapshots on pinned 1.95.0.
- `lint`: fmt and clippy, including the firmware crate.
- `embedded`: the thumbv7em builds plus the firmware crate, which is the only check that instantiates the f32 filters for the target.
- `docs`: `RUSTDOCFLAGS=-D warnings`.
- `deny`: `cargo deny`.
- `vectors`: `sha256sum -c`.
- `report`: the accuracy, stability and firmware numbers as a `comparison-report` artifact and step summary.
- `bench` (PRs only): base and head measured back to back on one runner, failing if the low end of Criterion's 95% interval for any of our benchmarks is over +10%.

CI facts learned the hard way:

- **CI's `stable` moves ahead of the local toolchain**, and new clippy lints fail the lint job (the first public run hit Rust 1.99's `chunks_exact_to_as_chunks`). Before pushing, run clippy on the newest stable as a side toolchain: `rustup toolchain install <ver> --profile minimal --component clippy`, then `cargo +<ver> clippy …`. Install it alongside rather than updating the user's default `stable`.
- **MSRV is 1.89** because nalgebra 0.35 declares it (`rust-version` in `Cargo.toml`); everything else needs 1.89 or less. If you raise a dependency, check its `rust-version`.
- **`trybuild` `.stderr` snapshots only match the compiler that wrote them.** On any other toolchain, set `SKIP_UI_TESTS=1`. After upgrading the `ui` job's pinned toolchain, regenerate them.
- **Old toolchains on this Mac:** rustc 1.89 can't link against the macOS 27 SDK (`tapi error: ... unknown architecture`). Use `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk cargo +1.89 test`. This only affects the local machine; CI runs on Linux.
- **A `RUSTFLAGS` environment variable replaces `.cargo/config.toml` rustflags.** That's why the firmware passes `-Tlink.x` from `build.rs` (`cargo:rustc-link-arg-bins`). Without it, the linker silently produced a 211-byte image with no code, and `firmware_sizes.sh` now fails on an image with no `.text`.
- **Without `std`, adskalman's generic constructors infer the wrong dimension.** That's why `baseline::Models::filter`/`prior` spell out `<R, Const<N>, Const<M>>`. Run `cargo test --no-default-features` after touching `tests/common/`, because the default build doesn't catch it.
- **Criterion leaves `change/estimates.json` files from earlier runs**, and the CI cache restores them. Clear `target/criterion` before a baseline run, as the `bench` job does.

## Architecture and design rules

- **Features:** `std` is on by default and maps to `nalgebra/std`. `nalgebra` uses `default-features = false, features = ["libm"]`. `autodiff` gates the dual-number Jacobians. Unlike the guide's plan, the RTS smoother isn't behind `std`: it works on caller-provided slices (`&[RtsStep<N>]` in, `&mut [Estimate<N>]` out), so the history can be a fixed-size array. `lib.rs` uses `#![cfg_attr(not(feature = "std"), no_std)]` and `#![deny(missing_docs)]`, so every public item needs docs.
- **Model traits:** these differ from the guide. `ProcessModel<N>` (`predict`) and `MeasurementModel<N, M>` (`measure`) are function-only, and that's all the UKF needs. `ProcessJacobian<N>` and `MeasurementJacobian<N, M>` extend them with `jacobian`, and the EKF requires those. Filters hold `x: SVector<N>` and `p: SMatrix<N, N>`.
- **UKF sigma points:** stable Rust can't use `2N + 1` as a const generic, so `SigmaPoints<D, N>` stores a center point plus `plus`/`minus` `D × N` matrices. The default `UkfParams` are `alpha = 1, beta = 2, kappa = 0`, which keep every weight non-negative. UKF `predict` returns `Result` because drawing sigma points needs a Cholesky factorization of P.
- **Square-root KF (`SqrtKf`):** it stores the lower-triangular factor `S` and takes noise as square roots, like `SqrtUkf`. Predict builds `S⁻` from the columns of `[F S, Q½]`. Update builds `S_z` from `[H S, R½]`, then `S⁺` from `[(I − K H) S, K R½]` (Joseph form). Every step is a positive Givens update from zero (`cholupdate::update_columns`), so nothing cancels, and `S` has the square root of P's condition number. It survives 1M f32 steps where `LinearKf` breaks at step 2, and the full `log_r ∈ [−12, 2]` f64 range. It costs about 3.5× `LinearKf` per step. Valid-factor check: finite, with a strictly positive diagonal (`cholupdate::is_singular` rejects zeros).
- **`cholupdate` module:** `update` (Givens, works from a zero factor), `downdate` (hyperbolic, `None` if the result would be indefinite), `update_columns` and `is_singular`, shared by both square-root filters.
- **Square-root UKF:** it stores the lower-triangular factor `S` (`P = S Sᵀ`), and `S` is always kept lower-triangular because the rank-one update and downdate functions in `sqrt_ukf.rs` depend on that. Instead of the textbook QR of a `D × (2N + K)` matrix, which can't be expressed with const generics, `weighted_factor` builds factors with Givens rank-one updates, starting from zero. Noise is passed as square roots (`q_sqrt: SMatrix<N, K>`, `r_sqrt: SMatrix<M, K>`, any `G` with `Q = G Gᵀ`), so singular or low-rank noise works. A zero on the diagonal of `S` is rejected as `CovarianceNotPositiveDefinite`. In f32 this happens when `|x|` grows until the sigma-point spread is below its resolution and the points collapse onto the mean, a limit of every sigma-point filter in low precision. It reuses `Weights` and `SigmaPoints` from `ukf.rs`, which are `pub(crate)`.
- **Scalar type:** every filter, model trait, `Estimate`/`RtsStep`, `smooth` and `nees` are generic over `T: Float` (`src/scalar.rs`: `RealField + Copy`, blanket-implemented, so `f32` and `f64`). Types and traits default to `T = f64` after the dimensions (`LinearKf<N, T = f64>`), so `LinearKf<4>` means f64. Functions can't have default type parameters, so a call where nothing determines `T`, such as `smooth::<2, f64>(&[], &mut [])`, needs an annotation. In generic code, write constants with `scalar::lit::<T>(0.5)`, and use `T::zero()` / `T::one()`. `UkfParams` and the chi-squared bounds stay `f64`; UKF `Weights` are computed in f64 and then converted.
- **ARM build caveat:** `cargo build --target thumbv7em-none-eabihf` only type-checks generic code and doesn't instantiate the f32 filters. To check that the f32 `libm` path actually compiles for the target, build a small `no_std` staticlib that uses the filters with `f32` (planned as `bench/firmware` in Step 8).
- **`no_std` tests:** `cargo test --no-default-features` must also pass. Unit tests that need `Vec` or `DMatrix` (such as the smoother's batch-posterior reference) are gated with `#[cfg(feature = "std")]`.
- **Chi-squared bounds:** the consistency tests should use `diagnostics::chi_squared_bounds(dof, runs, 0.95)`, not `statrs` as the guide suggests. The quantile is computed in the crate (regularized incomplete gamma plus bisection, checked against published tables), so it also works without `std`. `diagnostics::tests` includes a deterministic xorshift and Box-Muller Gaussian generator for Monte Carlo tests that need no dependencies.
- **Smoothers:** `smooth` (RTS, from `RtsStep { f, … }`) and `smooth_with_cross` (unscented RTS, from `CrossStep { cross, … }`) share one private backward pass; `smooth` passes `P_filtered Fᵀ` as the cross-covariance. `Ukf::predict` and `predict_with_cross` share `predict_impl::<P, CROSS>`, so a plain predict doesn't compute the cross-covariance. The SR-UKF has no `predict_with_cross` yet, and smoother differences don't go through `state_residual` (wrapped states aren't smoothed correctly near ±π).
- **Control inputs:** `LinearKf`/`SqrtKf::predict_with_input(f, b, u, q)` computes `x = F x + B u` and treats the input as exact, so the covariance is unchanged. Model-based filters take `ControlledProcess`/`ControlledJacobian` through the `WithInput::new(&model, u)` adapter (in `model.rs`), which implements `ProcessModel`/`ProcessJacobian` and forwards `state_residual`, so it works with every filter without per-filter methods. `autodiff` models still carry inputs as struct fields (see `examples/imu_attitude.rs`).
- **Wrapped measurements:** `MeasurementModel::residual(a, b)` (default `a - b`) is used for every innovation. The UKFs compute the predicted measurement as `ζ₀ + Σ wᵢ residual(ζᵢ, ζ₀)` and take all deviations through it (`SigmaPoints::residual_moments`); that equals the plain weighted mean when nothing wraps, but stays correct across ±π. `LinearKf` and `SqrtKf` take matrices, not models, so they have no residual hook. Wrapped state components (a heading) use `ProcessModel::state_residual`, applied the same way in both UKFs' predict; the estimate itself is only kept in range if the model's `predict` wraps it. `tests/angle_wrapping.rs` crosses the seam, with a control case proving it does.
- **Autodiff:** users write models once, generic over `autodiff::Real` (implemented for `f64` and `Dual<N>`), through `AutoProcess` / `AutoMeasurement`. The `AutoDiff(model)` wrapper implements all four model traits. It's a wrapper, not a blanket impl, because a blanket impl that exists only when a feature is on can cause coherence conflicts downstream. `Dual<N>` carries the full N-dimensional gradient, so one model evaluation gives the whole Jacobian. Only the state is differentiated. The scalar is a defaulted parameter everywhere: `Real<S = f64>`, `Dual<N, S = f64>`, `AutoProcess<N, S = f64>` and `AutoMeasurement<N, M, S = f64>`. Constants of type `S` go on the right (`T * S`), so an f32 model writes `T * 0.5f32`. `Real` is implemented for `f32` and `f64` by the `impl_real_for_scalar!` macro and for `Dual<N, S>` generically; `Dual`'s chain rules call `ComplexField::` functions on `S` so they work without std. A model with no constants can be generic over `S` (`impl<S: Float> AutoMeasurement<4, 2, S>`).
- **`no_std` math:** `f64::sqrt` and similar functions don't exist in `core`. Call them through `nalgebra::ComplexField` (for example `ComplexField::sqrt(x)`), which uses `libm`. Always use the qualified form: with `std` on, `x.sqrt()` compiles because the inherent method wins, but without `std` it's ambiguous whenever two traits with that method are in scope (such as `Real` and `ComplexField` in `autodiff.rs`).
- **Updates:** every filter uses a Joseph-form covariance update. The KF and EKF share `update::joseph`. Both UKFs use the sigma-point form, `Σ W (δχ − K δζ)(…)ᵀ + K R Kᵀ` (`SigmaPoints::corrected_deviations`), which equals `P − K S Kᵀ` but is a sum of positive terms, so it can't cancel to an indefinite result. The textbook `P − K S Kᵀ`, and the SR-UKF's rank-one downdates of it, failed in f32 at step 1. `symmetrize` mirrors the lower triangle with `fill_upper_triangle_with_lower_triangle`; averaging with the transpose cost about 9% of the KF step time, and an indexed loop was slower still. The trade-off: past f64's representable range, mirroring loses positive-definiteness sooner than averaging did (step 266 vs 578 on the proptest regression case, about where adskalman fails). `SqrtKf` is the answer for that regime. `update` returns `Result<f64, KalmanError>`, where the `f64` is the NIS. Failures such as NaN input or a singular innovation covariance return `Err` and leave the filter state unchanged. Never panic.
- **Filter loop never allocates.** This is checked by a counting global allocator.
- **Docs:** `src/lib.rs` compiles the README's Rust blocks as doctests (`#[cfg(doctest)] #[doc = include_str!("../README.md")]`), so README code must compile as written. It can't use hidden `# ` lines, because GitHub would show them. Don't intra-doc-link to `autodiff` from always-built docs: the link breaks when the feature is off. docs.rs builds with `all-features = true`.
- **README claims about adskalman** must be checked against its source (`~/.cargo/registry/src/*/adskalman-0.18.0/src`), and numbers must come from our own measurements. The comparison table lists only verified facts.
- **Implementation order:** KF (Joseph), then EKF, UKF + sqrt-UKF, RTS smoother, NIS/NEES, autodiff, and finally a generic scalar for `f32` support.

## Testing layout and constraints

- `tests/common/` holds shared helpers, and `benches/` reuses it through `#[path = "../tests/common/mod.rs"]`. `scenarios.rs` has the problem generators with fixed seeds (`Scenario::simulator` streams very long runs without storing them). `baseline.rs` is the **only** place that calls `adskalman`, so an API change touches one file. `naive.rs` is an unstabilized textbook filter used for comparison. `rng.rs` is a deterministic Gaussian generator. `FromScenario` provides `LinearKf::from_scenario(&sc)`.
- `adskalman`'s `step_with_options` predicts and then updates, the same order as our loops, so both start from `(x0, P0)` with no offset. It treats a NaN observation as missing, while ours returns `Err`. In debug builds it returns `Err` when P's asymmetry exceeds 1e-5, so its non-Joseph methods can fail in tests on hard problems.
- Tests run with `[profile.test] opt-level = 3`, because the 1M-step f32 test takes about 150 s unoptimized. Debug assertions stay on.
- Stability tests only make sense while the covariance is representable. In f64, P's condition number must stay well below 1/ε ≈ 4.5e15; past that, Cholesky fails on every covariance-form filter (ours, naive and adskalman alike). That's why `LinearKf`'s proptest uses `log_r ≥ -8`, while `SqrtKf`'s proptest covers the guide's full -12 to 2 range and checks the factor, not `S Sᵀ` (forming P squares the condition number). In f32, the threshold is about 1e7. Before tightening a stability test, check whether adskalman fails the same input (`baseline::Models::run`). If it does, the problem exceeds the format, not the filter.
- S4 (`scenarios::ill_conditioned`) must break the naive filter, and `f32_million_steps` asserts that it does, so the scenario keeps testing something. Report lines are printed with a `report:` prefix (`cargo test -- --nocapture`) for the Step 8/9 comparison report.
- `tests/no_alloc.rs` must stay in its own file, because its `#[global_allocator]` applies to the whole test binary.
- `tests/compile_fail.rs` uses `trybuild` over `tests/compile_fail/*.rs`, and the expected compiler errors live in `.stderr` files next to each case. Regenerate them deliberately (`TRYBUILD=overwrite`) and review the diff.
- Pass conditions: equivalence with `adskalman` within 1e-9; `P` stays symmetric and positive-definite under `proptest` ill-conditioned problems and over a 1M-step `f32` run; Monte Carlo NIS/NEES inside the 95% chi-squared bounds; speed equal to or better than `adskalman-joseph` (the like-for-like baseline), with no regression over 10%. On the development laptop the KF is at parity with `adskalman-joseph` (about 87 µs per 1,000 steps), but run-to-run noise there is ±5% or more, so only a dedicated machine can settle "equal or faster". `benches/filters.rs` also runs `typed-kalman-sqrt` (`SqrtKf`); `make_table.py` treats `typed-kalman-*` names as our own variants.
- Comparison crates (`adskalman`, `approx`, `proptest`, `criterion`, `trybuild`) are dev-dependencies only.

## Benchmarks

- Five frozen scenarios (S1–S5) live in `tests/vectors/` (format in `tests/vectors/README.md`: `model.json` plus raw little-endian arrays). The guide wants them in a shared test-vectors repo, added as a Git submodule at `tests/vectors/`; that repo doesn't exist yet, so for now this repo generates and holds them. Don't change scenario files after results have been collected.
- `tests/common/catalog.rs` defines exactly what each file contains (seeds, runs, dtype). `examples/generate_vectors.rs` writes the files from it, and `tests/vectors.rs` checks them bit for bit. If you change the catalog deliberately, regenerate the files and `SHA256SUMS` (`cd tests/vectors && shasum -a 256 S*/* > SHA256SUMS`). Load scenarios with `common::vectors::load::<N, M>("S2")`.
- **The generator must be bit-reproducible across platforms.** `tests/vectors.rs` compares bit for bit, and the system math library's `ln`, `cos`, `pow`, `hypot` and `atan2` differ in the last bit between macOS and Linux. The first CI run failed on exactly that. So `rng.rs` and `scenarios.rs` use the pure-Rust `libm` crate for transcendental functions and plain multiplication instead of `powi`; only `sqrt` (correctly rounded everywhere) comes from std. Keep any new generator math on `libm`.
- S3 must stay away from the bearing wrap at ±π, because the filters don't wrap residuals. The first S3 draft started at 2 km and random-walked to x < 0, which is why it now starts at 10 km.
- Criterion group names must be `<Scenario>_<Filter>_<precision>` (e.g. `S2_KF_float64`) and function names must be the library name, because `scripts/criterion_to_csv.py` parses them from `target/criterion/`.
- Results are appended to `results/results.csv` using the schema `library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date`. `make_table.py` writes the README table between `<!-- BENCH:START -->` and `<!-- BENCH:END -->`.
- Benchmark `adskalman` with all three `CovarianceUpdateMethod`s. A feature a baseline lacks is reported as "n/a", never as a failure.
- Embedded numbers come from a separate firmware crate in `bench/firmware` (DWT cycle counter, `cargo size`), with an identical build profile for each library.

- `examples/accuracy.rs` prints the rmse, nees, max_abs_diff, steps_to_failure and heap_allocations rows. It has its own counting global allocator, which is fine because an example is its own binary. adskalman runs S3 as an EKF through `baseline::adskalman_ekf_s3`: its transition model predicts, then an observation model linearized at the prediction updates.
- S5's NEES is about 9.9 against 15 degrees of freedom for every library. That's a single-run artifact, not inconsistency: yaw and some bias combinations are unobservable, so each run carries a few fixed χ²(1) draws from the prior. Over 100 seeds, per-run NEES ranges from 7.4 to 29.2, with a mean of 14.0. Use the 500-run tests in `tests/consistency.rs` for consistency claims.
- `bench/firmware` is a standalone crate (its own `[workspace]`), built for an STM32F446RE (`memory.x`). Its `build.rs` turns S2's `model.json` and first 1,000 measurements into f32 constants, so both images carry identical data. `ram_bytes` (`.data` + `.bss`) counts only static RAM, which is 24 bytes for both; the filter state lives on the stack, which this metric doesn't measure.
- **Results on the development laptop**, against `adskalman-joseph` in the same benchmark run: the KF is **faster on every scenario** (S2 f32 725–733 vs 738 µs, S2 f64 737–739 vs 841 µs, S5 23.9 vs 24.5 ms), and **flash is 14% smaller** (13,988 vs 16,276 bytes).
- **Keep `#[inline]` on the generic step paths** (`update::joseph`, `symmetrize`, `all_finite`, the filters' `predict`/`update`, `SigmaPoints` methods, `cholupdate`). Without the hints LLVM didn't inline `update::joseph` and its `Corrected` return, so every step copied matrices through memory. That cost 11% on S2 f32 and 39% for `SqrtKf`, which was first misattributed to transposing F and H. A side-by-side variant that transposed every step was as fast as one with precomputed transposes, which ruled that out. Firmware size is unaffected: LTO already inlined there. Accuracy, NEES and allocations are identical; on S4 both survive 1M steps. "Ours vs best other" compares with `adskalman-optimal`, which is fastest but fails S4 at step 0.
- Measure performance changes **side by side in one benchmark binary**, with the variants as const-generic flags of a single function. Sequential A/B runs on the laptop drift by ±5–15%, and twice gave the opposite conclusion (inverting S looked 2% slower sequentially but was 8% faster side by side).

## Roadmap

`ROADMAP.md` holds the plan for 0.2.0: breaking API cleanups (every `predict` returning `Result`, one `covariance()` return type, a common filter trait), additive gaps, examples, and infrastructure. Bundle all breaking changes into 0.2.0. Update the file as items are done or user feedback reorders them.

## Releasing

Publishing to crates.io is public and permanent (versions can be yanked, never deleted), so it only happens on the user's explicit go-ahead.

1. Make sure CI is green on `main`, and `cargo publish --dry-run` passes. The package is about 55 files and 67 KiB compressed; `exclude` in `Cargo.toml` keeps the 14.5 MB of scenario files out.
2. Move the `CHANGELOG.md` entries from `[Unreleased]` into the new version. `cargo release <version>` (configured in `release.toml`) does this, bumps the version, commits and tags. It's a preview unless you pass `--execute`.
3. `cargo publish`. It needs a crates.io token (`cargo login`).
4. The guide says to announce on the Rust users forum, the Rust Embedded community and r/rust. Draft the posts for the user; never post them yourself.

Status: **0.1.1 is the latest release** (crates.io, 2026-10-06, tagged `v0.1.1` at `067cafd`); 0.1.0 was tagged `v0.1.0` at `11f7ad1`. Both have GitHub releases. A version bump must also update `bench/firmware/Cargo.lock` (`cargo update -p typed-kalman` there), because CI builds the firmware with `--locked`. Wait for CI to pass on the release commit before `cargo publish`. The repository is https://github.com/joslo2345/typed-kalman (public). Its secret scanning, push protection and Dependabot alerts are on. Announcement drafts are in `docs/announcement-drafts.md`; the user posts them. crates.io requires a verified email on the publishing account (the first attempt failed without one).

## Resolved inconsistencies in the guide

- The guide names the benchmark both `filters` (`Cargo.toml`) and `compare` (Step 8 pipeline). `benches/filters.rs` is the Step 7 speed comparison. Step 8's `benches/compare.rs` needs its own `[[bench]]` entry with `harness = false`.
- The adskalman adapter lives only in `tests/common/baseline.rs`, not in `bench/baselines/`. Benches include it with `#[path]`.
