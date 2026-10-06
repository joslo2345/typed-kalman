# Building a Kalman Filter Crate in Rust

A step-by-step guide to setting up the repository for a type-safe, embedded-friendly Rust Kalman filter crate.

## Project goals

The crate should fill the gaps in the current Rust ecosystem:

- **`no_std` support.** It runs on microcontrollers with no heap allocation.
- **Compile-time dimensions.** It uses const generics and `nalgebra` static matrices so dimension errors don't compile.
- **A full filter family.** It offers KF, EKF, UKF, square-root variants, and an RTS smoother.
- **Optional automatic Jacobians.** Dual numbers derive EKF Jacobians from the model.
- **Diagnostics and docs.** It includes NIS/NEES checks, thorough documentation, and runnable examples.

## Step 1: Create the repository

```bash
cargo new --lib kalman-rs
cd kalman-rs
```

`cargo new` initializes Git automatically. Dual-license under MIT and Apache-2.0, following the Rust ecosystem convention, by adding `LICENSE-MIT` and `LICENSE-APACHE`.

## Step 2: Set up the directory layout

```
kalman-rs/
├── src/
│   ├── lib.rs
│   ├── linear.rs
│   ├── ekf.rs
│   ├── ukf.rs
│   ├── sqrt_ukf.rs
│   ├── smoother.rs
│   ├── diagnostics.rs
│   └── error.rs
├── tests/
│   └── vectors/            # shared reference test data
├── examples/
│   ├── constant_velocity.rs
│   └── imu_attitude.rs
├── benches/
├── Cargo.toml
├── README.md
├── LICENSE-MIT
└── LICENSE-APACHE
```

## Step 3: Configure `Cargo.toml`

Make `std` a default feature that embedded users can turn off.

```toml
[package]
name = "kalman-rs"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "Type-safe, no_std Kalman filters: KF, EKF, UKF and smoothers"
categories = ["mathematics", "science::robotics", "embedded", "no-std"]

[features]
default = ["std"]
std = ["nalgebra/std"]
autodiff = []

[dependencies]
nalgebra = { version = "0.33", default-features = false, features = ["libm"] }

[dev-dependencies]
approx = "0.5"
proptest = "1"
criterion = "0.5"

[[bench]]
name = "filters"
harness = false
```

Check crates.io for the latest `nalgebra` version before starting.

## Step 4: Enable `no_std` in `lib.rs`

```rust
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

//! Type-safe Kalman filters for desktop and embedded targets.

pub mod linear;
pub mod ekf;
pub mod ukf;
pub mod smoother;
pub mod diagnostics;
pub mod error;
```

## Step 5: Design the API with traits and const generics

```rust
use nalgebra::{SMatrix, SVector};

pub trait ProcessModel<const N: usize> {
    fn predict(&self, x: &SVector<f64, N>, dt: f64) -> SVector<f64, N>;
    fn jacobian(&self, x: &SVector<f64, N>, dt: f64) -> SMatrix<f64, N, N>;
}

pub trait MeasurementModel<const N: usize, const M: usize> {
    fn measure(&self, x: &SVector<f64, N>) -> SVector<f64, M>;
    fn jacobian(&self, x: &SVector<f64, N>) -> SMatrix<f64, M, N>;
}

pub struct Ekf<const N: usize> {
    x: SVector<f64, N>,
    p: SMatrix<f64, N, N>,
}

impl<const N: usize> Ekf<N> {
    pub fn predict<P: ProcessModel<N>>(&mut self, model: &P, q: &SMatrix<f64, N, N>, dt: f64);
    pub fn update<H: MeasurementModel<N, M>, const M: usize>(
        &mut self, model: &H, z: &SVector<f64, M>, r: &SMatrix<f64, M, M>,
    ) -> Result<f64, KalmanError>; // returns NIS
}
```

Return `Result` with a custom error type for failures such as a non-invertible innovation covariance, rather than panicking.

## Step 6: Implement in this order

1. Implement the linear KF with the Joseph-form update.
2. Add the EKF with user-supplied Jacobians.
3. Add the UKF and a square-root UKF.
4. Add the RTS smoother (behind the `std` feature if it needs to store a history).
5. Add NIS/NEES diagnostics.
6. Add optional autodiff Jacobians using dual numbers behind the `autodiff` feature.
7. Add `f32` support through a generic scalar type, since many microcontrollers only have single-precision FPUs.

## Step 7: Write unit tests that prove the crate is better

Use the built-in test framework with `approx` for float comparisons, `proptest` for property-based tests, and `trybuild` for compile-fail tests. Add the comparison crates as development dependencies only:

```toml
[dev-dependencies]
approx = "0.5"
proptest = "1"
criterion = "0.5"
trybuild = "1"
adskalman = "0.18"   # latest release as of October 2026
```

### Define "better" before writing tests

For a linear system with Gaussian noise, the Kalman filter is the optimal estimator. Two correct implementations should produce the **same** answer, so on accuracy the goal is to *match* existing crates, not beat them. Measurable improvements come from safety, stability, embedded support, and features:

| Criterion | What the test does | Pass condition |
|---|---|---|
| Equivalence | Runs the same problem through our filter and `adskalman` | States match within 1e-9 |
| Numerical stability | Generates random ill-conditioned problems with `proptest`, plus a 1M-step `f32` run | Our covariance always stays symmetric and positive-definite |
| Compile-time safety | Tries to compile code with mismatched dimensions | Compilation fails with a clear error |
| No allocation | Counts heap allocations during the filter loop | Zero allocations |
| `no_std` | Builds for a bare-metal ARM target | Build succeeds |
| Error handling | Feeds NaN measurements and a singular innovation covariance | Returns `Err` and leaves the state unchanged, never panics |
| Consistency | Runs Monte Carlo trials and computes NIS/NEES | Falls inside the 95% chi-squared bounds |
| Speed | Benchmarks predict+update with `criterion` | Equal to or faster than `adskalman`, with no regressions over 10% |

### Organize the tests

```
tests/
├── common/
│   ├── mod.rs
│   ├── scenarios.rs      # shared problem generators with fixed seeds
│   ├── baseline.rs       # adapter around adskalman
│   └── naive.rs          # textbook filter with no stabilization
├── equivalence.rs
├── stability.rs
├── no_alloc.rs           # must stay in its own file (see below)
├── errors.rs
├── consistency.rs
├── compile_fail.rs
├── compile_fail/         # code that must NOT compile
└── vectors/              # shared reference test data
```

Put the `adskalman` calls inside `tests/common/baseline.rs` so that if its API changes, only one file needs updating.

### Example: equivalence with adskalman

```rust
mod common;
use common::{baseline, scenarios};
use kalman_rs::linear::LinearKf;

#[test]
fn matches_adskalman_on_constant_velocity() {
    let sc = scenarios::constant_velocity(1000, 42);
    let theirs = baseline::adskalman_filter(&sc); // Vec<SVector<f64, 4>>

    let mut ours = LinearKf::from_scenario(&sc);
    for (k, z) in sc.zs.iter().enumerate() {
        ours.predict(&sc.f, &sc.q);
        ours.update(&sc.h, z, &sc.r).unwrap();
        approx::assert_relative_eq!(ours.state(), &theirs[k], epsilon = 1e-9);
    }
}
```

### Example: stability across random ill-conditioned problems

```rust
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn covariance_stays_symmetric_positive_definite(
        seed in any::<u64>(),
        log_r in -12.0f64..2.0,
    ) {
        let sc = scenarios::random_linear(seed, 10f64.powf(log_r), 2000);
        let mut kf = LinearKf::from_scenario(&sc);

        for z in &sc.zs {
            kf.predict(&sc.f, &sc.q);
            kf.update(&sc.h, z, &sc.r).unwrap();

            let p = kf.covariance();
            prop_assert!((p - p.transpose()).abs().max() < 1e-10);
            prop_assert!(p.cholesky().is_some());
        }
    }
}
```

Add a second test that runs 1,000,000 steps in `f32` alongside the textbook filter in `common/naive.rs`. Require only that our filter stays stable, and print the step where the naive filter fails (if it does) for the comparison report.

### Example: dimension mismatches don't compile

```rust
// tests/compile_fail.rs
#[test]
fn dimension_mismatches_do_not_compile() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
```

```rust
// tests/compile_fail/wrong_measurement_size.rs
use kalman_rs::linear::LinearKf;
use nalgebra::SVector;

fn main() {
    let mut kf: LinearKf<4> = LinearKf::default();
    let z = SVector::<f64, 3>::zeros();
    kf.update(&H_2x4, &z, &R_2x2).unwrap(); // must not compile: z has 3 rows, H has 2
}
```

`trybuild` stores the expected compiler error in a `.stderr` file next to each case. That also guards against error messages becoming confusing over time.

### Example: the filter loop never allocates

A counting global allocator applies to the whole test binary, so this test must live alone in `tests/no_alloc.rs`:

```rust
mod common;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn filter_loop_does_not_allocate() {
    let sc = common::scenarios::constant_velocity(100, 1); // allocates, so do it first
    let mut kf = kalman_rs::linear::LinearKf::from_scenario(&sc);

    let before = ALLOCS.load(Ordering::SeqCst);
    for z in &sc.zs {
        kf.predict(&sc.f, &sc.q);
        kf.update(&sc.h, z, &sc.r).unwrap();
    }
    assert_eq!(ALLOCS.load(Ordering::SeqCst), before);
}
```

### Example: errors instead of panics

```rust
#[test]
fn nan_measurement_returns_error_and_keeps_state() {
    let sc = scenarios::constant_velocity(1, 0);
    let mut kf = LinearKf::from_scenario(&sc);
    let before = kf.clone();

    let z = nalgebra::Vector2::new(f64::NAN, 1.0);
    assert!(matches!(kf.update(&sc.h, &z, &sc.r), Err(KalmanError::InvalidInput)));
    assert_eq!(kf, before);
}
```

### Example: `no_std` build check

```bash
rustup target add thumbv7em-none-eabihf
cargo build --target thumbv7em-none-eabihf --no-default-features
```

Run this in CI on every commit, since a single accidental `std` import breaks embedded users.

### Consistency tests

Run 500 Monte Carlo trials where the true state is known. Check that average NIS and NEES fall inside the 95% chi-squared bounds. You can hard-code the bounds for your chosen dimensions or compute them with the `statrs` crate as a dev dependency.

### Speed comparison

Benchmark both crates side by side in `benches/filters.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn compare(c: &mut Criterion) {
    let sc = scenarios::constant_velocity(1000, 0);
    let mut group = c.benchmark_group("constant_velocity_1000_steps");
    group.bench_function("kalman-rs", |b| b.iter(|| run_ours(black_box(&sc))));
    group.bench_function("adskalman", |b| b.iter(|| baseline::adskalman_filter(black_box(&sc))));
    group.finish();
}

criterion_group!(benches, compare);
criterion_main!(benches);
```

Criterion saves results between runs and reports regressions automatically.

### Publish a comparison report

Have CI collect the printed report lines and Criterion results into a `comparison.md` table and upload it as a build artifact. Fail the build if any pass condition breaks or speed regresses by more than 10%. This report is the evidence behind any claim of being better than existing crates.

## Step 8: Produce comparison numbers against existing libraries

The unit tests in the previous step prove the library is *correct*. This step produces the *numbers* for the README: a table that shows, scenario by scenario, how this library compares to the existing ones. Everything runs from one command, writes raw data to a CSV file, and regenerates the table automatically, so anyone can reproduce the results.

### Benchmark scenarios

Use the same five scenarios in all four repos (C, C++, Python, Rust). Store them as data files (true states, measurements, F, H, Q, R, x0, P0) in the shared test-vectors repo and pull it in as a Git submodule. Because every library reads exactly the same inputs, the numbers are comparable against the external libraries *and* across our four implementations.

| ID | Scenario | State / measurement size | Filters | Length | What it shows |
|---|---|---|---|---|---|
| S1 | 1D constant velocity | 2 / 1 | KF | 10,000 steps | Overhead on tiny problems |
| S2 | 2D constant velocity | 4 / 2 | KF | 10,000 steps | A typical tracking workload |
| S3 | Range-bearing tracking | 4 / 2 | EKF, UKF | 500 steps × 200 seeds | Accuracy on a nonlinear problem |
| S4 | Ill-conditioned problem | 4 / 2 | KF | 1,000,000 steps, 32-bit floats | Numerical stability |
| S5 | INS error-state | 15 / 6 | KF / EKF | 10,000 steps | Speed on a larger state |

Freeze the scenario files before collecting results. Changing a scenario after seeing which library wins makes the numbers meaningless.

### Metrics

| Metric (CSV name) | Unit | How it's measured | Better is |
|---|---|---|---|
| `time_per_step` | ns | Median of 30 timed runs of predict+update, after a warm-up run | Lower |
| `rmse` | state units | Root-mean-square error against the true states in the scenario file | Lower |
| `max_abs_diff` | state units | Largest difference from each baseline's estimate (S1, S2) | Close to 0 |
| `nees` | – | Average normalized estimation error squared, compared with 95% chi-squared bounds | Inside bounds |
| `steps_to_failure` | steps | First step where the covariance is no longer symmetric positive-definite (S4) | Higher |
| `heap_allocations` | count | Allocations during the step loop, counted with the global allocator from Step 7 | Lower |
| `flash_bytes` | bytes | `.text` + `.rodata` + `.data` of a firmware image that only runs the filter on S2 | Lower |
| `ram_bytes` | bytes | `.data` + `.bss` of the same firmware image | Lower |
| `cycles_per_step` | CPU cycles | ARM DWT cycle counter on a real microcontroller | Lower |

### What to compare against

| Library | Scenarios | Why | How to include it |
|---|---|---|---|
| [adskalman](https://docs.rs/adskalman) | S1, S2, S4, S5 (and possibly S3) | The most established Kalman filter crate: linear KF, RTS smoother, `no_std`, compile-time dimensions | `adskalman = "0.18"` in `[dev-dependencies]`; the exact version is read from `Cargo.lock` |
| Naive textbook filter | S4 | Shows what an unstabilized hand-written filter does | `tests/common/naive.rs` from Step 7 |

adskalman is a strong baseline. It already has compile-time dimensions, `no_std`, an RTS smoother, missing-measurement handling, and a choice of covariance updates (`JosephForm`, `OptimalKalman`, `OptimalKalmanForcedSymmetric`). Expect the linear-filter numbers to be close. The clearer differences will come from features it doesn't have, such as a UKF, automatic Jacobians, and NIS/NEES diagnostics. Benchmark it with each covariance update method so the comparison uses its best settings, not its weakest.

In S3 the motion model is linear and only the range-bearing measurement is nonlinear. adskalman's `ObservationModel` trait is documented as potentially non-linear, so check its current docs: if it supports this case, include it as an EKF baseline for S3; otherwise mark it "n/a".

### The adskalman adapter (real API)

This follows the API used in adskalman's own integration tests: a transition model and an observation model implemented as traits, `KalmanFilterNoControl::new`, and `step_with_options` from one `StateAndCovariance` to the next.

```rust
// bench/baselines/adskalman_adapter.rs
#![allow(non_snake_case)] // adskalman's trait methods are named F, Q, H, R

use adskalman::{
    CovarianceUpdateMethod, KalmanFilterNoControl, ObservationModel, StateAndCovariance,
    TransitionModelLinearNoControl,
};
use nalgebra::{OMatrix, OVector, U2, U4};

use crate::scenarios::Scenario;

pub struct MotionS2 {
    f: OMatrix<f64, U4, U4>,
    ft: OMatrix<f64, U4, U4>,
    q: OMatrix<f64, U4, U4>,
}

impl TransitionModelLinearNoControl<f64, U4> for MotionS2 {
    fn F(&self) -> &OMatrix<f64, U4, U4> { &self.f }
    fn FT(&self) -> &OMatrix<f64, U4, U4> { &self.ft }
    fn Q(&self) -> &OMatrix<f64, U4, U4> { &self.q }
}

pub struct ObservationS2 {
    h: OMatrix<f64, U2, U4>,
    ht: OMatrix<f64, U4, U2>,
    r: OMatrix<f64, U2, U2>,
}

impl ObservationModel<f64, U4, U2> for ObservationS2 {
    fn H(&self) -> &OMatrix<f64, U2, U4> { &self.h }
    fn HT(&self) -> &OMatrix<f64, U4, U2> { &self.ht }
    fn R(&self) -> &OMatrix<f64, U2, U2> { &self.r }
}

/// Runs adskalman over scenario S2 and calls `on_step` with each state estimate.
pub fn adskalman_s2(
    sc: &Scenario<4, 2>,
    method: CovarianceUpdateMethod,
    mut on_step: impl FnMut(&OVector<f64, U4>),
) {
    let motion = MotionS2 { f: sc.f, ft: sc.f.transpose(), q: sc.q };
    let observation = ObservationS2 { h: sc.h, ht: sc.h.transpose(), r: sc.r };
    let kf = KalmanFilterNoControl::new(&motion, &observation);

    let mut estimate = StateAndCovariance::new(sc.x0, sc.p0);
    for z in &sc.zs {
        estimate = kf.step_with_options(&estimate, z, method).unwrap();
        on_step(estimate.state());
    }
}
```

Write the same adapter for S1 (`U2`, `U1`) and S5 (`U15`, `U6`), or generate all three with a small macro. The `on_step` callback lets the benchmark pass an empty closure, so no time is spent storing results, while the accuracy run collects every estimate to compute `rmse` and `max_abs_diff`.

Before trusting any numbers, run the equivalence test from Step 7 through this adapter. If the states don't match, the most likely cause is a difference in when the first prediction happens relative to the first measurement; align the initial state so both filters start from the same prior.

### The harness

Use `criterion` with one benchmark group per scenario. The group name holds the scenario, filter type and precision, and the function name holds the library, so the converter can read them back:

```rust
// benches/compare.rs
use adskalman::CovarianceUpdateMethod;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn s2(c: &mut Criterion) {
    let sc = scenarios::load::<4, 2>("S2");
    let mut group = c.benchmark_group("S2_KF_float64");
    group.throughput(Throughput::Elements(sc.zs.len() as u64));

    group.bench_function("kalman-rs", |b| {
        b.iter(|| run_ours(black_box(&sc), |_| {}))
    });
    for (name, method) in [
        ("adskalman-joseph", CovarianceUpdateMethod::JosephForm),
        ("adskalman-optimal", CovarianceUpdateMethod::OptimalKalman),
        ("adskalman-optimal-symmetric", CovarianceUpdateMethod::OptimalKalmanForcedSymmetric),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| baseline::adskalman_s2(black_box(&sc), method, |_| {}))
        });
    }
    group.finish();
}

criterion_group!(benches, s2 /* , s1, s5 */);
criterion_main!(benches);
```

Our filter uses the Joseph form by default, so `adskalman-joseph` is the like-for-like comparison. The other two rows show the speed adskalman gains by giving up some stability, which is useful context next to the S4 results.

Criterion stores each result in `target/criterion/<group>/<function>/new/estimates.json`. Convert the medians into the shared CSV format with `scripts/criterion_to_csv.py`:

```python
"""Usage: python scripts/criterion_to_csv.py "commit,cpu,os,toolchain,date" """
import csv
import json
import sys
import tomllib
from pathlib import Path

STEPS = {"S1": 10_000, "S2": 10_000, "S5": 10_000}
env = sys.argv[1].split(",")
lock = tomllib.load(open("Cargo.lock", "rb"))
adskalman_version = next(p["version"] for p in lock["package"] if p["name"] == "adskalman")

with open("results/results.csv", "a", newline="") as f:
    out = csv.writer(f)
    for path in Path("target/criterion").glob("*/*/new/estimates.json"):
        group, library = path.parts[-4], path.parts[-3]
        scenario, filt, precision = group.split("_")
        ns_per_step = json.load(open(path))["median"]["point_estimate"] / STEPS[scenario]
        version = env[0] if library == "kalman-rs" else adskalman_version
        out.writerow([library, version, scenario, filt, precision,
                      "time_per_step", f"{ns_per_step:.2f}", "ns", *env])
```

Accuracy and stability (`rmse`, `nees`, `max_abs_diff`, `steps_to_failure`) come from a separate `cargo run --release --example accuracy` that runs each library once per scenario (200 seeds for S3) and prints CSV rows.

### Embedded numbers

Create a separate firmware crate in `bench/firmware` with one binary per library, each running S2 for 1,000 steps on a real board (for example an STM32 Nucleo with a Cortex-M4F). Time it with the DWT cycle counter from the `cortex-m` crate:

```rust
use cortex_m::peripheral::{Peripherals, DWT};

let mut cp = Peripherals::take().unwrap();
cp.DCB.enable_trace();
cp.DWT.enable_cycle_counter();

let start = DWT::cycle_count();
run_filter_s2();
let cycles_per_step = DWT::cycle_count().wrapping_sub(start) / 1_000;
```

Send the result over a serial port or RTT in the CSV format. Measure flash and RAM with [cargo-binutils](https://github.com/rust-embedded/cargo-binutils):

```bash
cargo size --release --target thumbv7em-none-eabihf --bin fw_kalman_rs -- -A
cargo size --release --target thumbv7em-none-eabihf --bin fw_adskalman -- -A
```

Build both binaries with the same profile (`opt-level`, `lto`, `codegen-units = 1`) and identical code apart from the filter, so the difference is only the library.

### One command to produce everything

```bash
ENV="$(git rev-parse --short HEAD),$(uname -m),$(uname -s),$(rustc --version | cut -d' ' -f2),$(date -I)"
cargo bench --bench compare
python scripts/criterion_to_csv.py "$ENV"
cargo run --release --example accuracy -- "$ENV" >> results/results.csv
python scripts/make_table.py results/results.csv kalman-rs
```

### Rules for a fair comparison

- Every library gets the same scenario file, random seed, initial state, and noise matrices.
- Every library uses the same floating-point precision. If a baseline doesn't support the precision a scenario needs, its cell is marked "n/a".
- Only predict+update is timed. Setup, file loading, data conversion, and printing are excluded.
- Every result records the library version (or commit hash), CPU, OS, and toolchain.
- A feature a baseline doesn't have (for example a UKF) is reported as "n/a", never as a failure.
- Publish the raw CSV and the scripts, so the authors of the other libraries can check the setup and reproduce the numbers.

### Output format

Every benchmark run appends rows to `results/results.csv` with this schema:

```
library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date
```

The script below turns the CSV into the README table and calculates how our library compares with the best other library for each row (above 1.00x means ours is better). Save it as `scripts/make_table.py`. It's Python in every repo, since it's only tooling.

```python
"""Usage: python scripts/make_table.py results/results.csv <our-library-name>"""
import csv
import sys
from collections import defaultdict

LOWER_IS_BETTER = {"time_per_step", "cycles_per_step", "rmse", "heap_allocations",
                   "peak_memory", "flash_bytes", "ram_bytes"}
HIGHER_IS_BETTER = {"steps_to_failure"}

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
ours = sys.argv[2]
libs = [ours] + sorted({r["library"] for r in rows} - {ours})

cells = defaultdict(dict)
for r in rows:
    key = (r["scenario"], r["filter"], r["precision"], r["metric"], r["unit"])
    cells[key][r["library"]] = float(r["value"])


def ratio(metric, vals):
    # Libraries named "<ours>-something" are our own variants (e.g. another backend)
    others = [v for lib, v in vals.items() if not lib.startswith(ours)]
    if ours not in vals or not others:
        return "n/a"
    if metric in LOWER_IS_BETTER:
        best = min(others)
        return "–" if vals[ours] == 0 else f"{best / vals[ours]:.2f}x"
    if metric in HIGHER_IS_BETTER:
        best = max(others)
        return "–" if best == 0 else f"{vals[ours] / best:.2f}x"
    return "–"  # nees, max_abs_diff: read the values directly


print("| Scenario | Filter | Precision | Metric | " + " | ".join(libs) + " | Ours vs best other |")
print("|" + "---|" * (len(libs) + 5))
for (scen, filt, prec, metric, unit), vals in sorted(cells.items()):
    values = [f"{vals[lib]:.4g}" if lib in vals else "n/a" for lib in libs]
    print(f"| {scen} | {filt} | {prec} | {metric} ({unit}) | " + " | ".join(values)
          + f" | {ratio(metric, vals)} |")
```

Paste the output into the README between `<!-- BENCH:START -->` and `<!-- BENCH:END -->` markers, or have CI do it.

### Where to run the numbers

GitHub-hosted CI runners share hardware, so their timings can vary by 10–20% between runs. Use them to catch regressions, but produce the published numbers on one dedicated machine, with nothing else running and a fixed CPU frequency (on Linux, set the `performance` governor). State that machine's specs above the table.

### What the README table will look like

The values below are placeholders until the benchmarks run.

| Scenario | Filter | Precision | Metric | kalman-rs | adskalman-joseph | adskalman-optimal | naive | Ours vs best other |
|---|---|---|---|---|---|---|---|---|
| S2 | KF | float64 | time_per_step (ns) | – | – | – | n/a | – |
| S2 | KF | float32 | cycles_per_step (cycles) | – | – | – | n/a | – |
| S2 | KF | float32 | flash_bytes (bytes) | – | – | – | n/a | – |
| S3 | EKF | float64 | rmse (state) | – | check | check | n/a | – |
| S3 | UKF | float64 | rmse (state) | – | n/a | n/a | n/a | n/a |
| S4 | KF | float32 | steps_to_failure (steps) | – | – | – | – | – |
| S5 | KF | float64 | time_per_step (ns) | – | – | – | n/a | – |

## Step 9: Add continuous integration

Create a GitHub Actions workflow that:

- Runs `cargo test` on stable, beta, and the minimum supported Rust version (MSRV).
- Runs `cargo clippy -- -D warnings` and `cargo fmt --check`.
- Builds for `thumbv7em-none-eabihf` with `--no-default-features`.
- Runs `cargo deny check` for licenses and advisories.
- Builds docs with `cargo doc` to catch broken links.

## Step 10: Write documentation

- Put a quick-start example at the top of `lib.rs` so it appears on docs.rs and is compiled as a doctest.
- Document every public item (enforced by `#![deny(missing_docs)]`).
- Provide runnable examples with `cargo run --example constant_velocity`.
- In the README, include a feature comparison with existing crates and an embedded usage example.

## Step 11: Release

- Publish with `cargo publish` after a `cargo publish --dry-run`.
- Use semantic versioning and keep a `CHANGELOG.md`; consider `cargo-release` to automate it.
- Announce on the Rust users forum, the Rust Embedded community, and r/rust to gather early feedback.
