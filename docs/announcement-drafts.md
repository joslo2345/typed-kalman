# Announcement drafts for typed-kalman 0.1.0

Drafts for the three places the plan names. Post them only after `cargo publish` succeeds and
the repository is public. Replace `<REPO_URL>` with the repository's address first. Every
number below was measured in this repo; keep them in sync if anything changes before release.

---

## 1. Rust users forum (users.rust-lang.org, category "announcements")

**Title:** typed-kalman 0.1: type-safe, no_std Kalman filters (KF, EKF, UKF, square-root, smoother)

Hi all! I've just released **typed-kalman**, a Kalman filter crate built on `nalgebra` static
matrices.

- crates.io: https://crates.io/crates/typed-kalman
- docs: https://docs.rs/typed-kalman
- repo: <REPO_URL>

**What it does**

- The whole filter family: a linear KF, an EKF, a UKF, square-root versions of the KF and UKF,
  and a Rauch-Tung-Striebel smoother.
- Dimensions are const generics, so passing a 3-element measurement to a model that expects 2
  is a compile error. `trybuild` tests keep those errors readable.
- `no_std`, no allocation, and every filter runs in `f32` or `f64`.
- Bad input never panics: a NaN reading or a singular covariance returns an error and leaves
  the filter unchanged, so you can skip the measurement and carry on.
- Diagnostics: every `update` returns the NIS, and there are NEES and chi-squared bounds for
  consistency checks.
- Optional automatic Jacobians (`autodiff` feature): write a nonlinear model once, generic over
  a scalar trait, and the EKF gets exact Jacobians from dual numbers. The UKFs use the same
  model unchanged.

**How it compares with adskalman**

adskalman is the established crate, and a good one. typed-kalman's linear filter and smoother
agree with it to within 1e-9 on the same problems, which is what you'd expect: for a linear
Gaussian system two correct Kalman filters must give the same answer. The differences are in
scope, not accuracy:

- Nonlinear filters: EKF, UKF and square-root UKF, plus automatic Jacobians.
- A square-root KF. On the test suite's hardest single-precision problem, the covariance-form
  filter loses positive-definiteness within two steps, while the square-root one runs a
  million.
- NIS/NEES diagnostics.
- A smaller embedded footprint. A complete Cortex-M4F firmware image running a 4-state filter
  is 13,988 bytes of flash, against 16,276 for the same image with adskalman (Rust 1.95, about
  8 KB of each is benchmark data).

On speed, the two are about on par on my development machine. The repo has a reproducible
benchmark pipeline with five shared scenarios, and I'll publish numbers once they've been run
on a dedicated machine rather than a laptop.

**How it's tested**

Equivalence against adskalman, Monte Carlo consistency (500 runs, NEES and NIS within the 95%
bounds), property tests on random ill-conditioned problems, million-step `f32` stability runs,
a counting allocator that proves the filter loops never allocate, and CI on stable, beta and
the MSRV (1.89) with and without `std`, plus a bare-metal ARM build.

**Feedback I'd love**

- The model traits: is passing per-step inputs (like gyro readings) through the model struct
  comfortable enough, or would you want an explicit control input?
- Angle wrapping for residuals (bearings near ±π) is next. How do you prefer to express it?
- Would you use `autodiff` in `f32`?

Thanks for reading, and thanks to the adskalman authors, whose crate was the benchmark for all
of this.

---

## 2. Rust Embedded community (Matrix: #rust-embedded:matrix.org, or the rust-embedded discussions)

**typed-kalman 0.1: no_std Kalman filters that fit on a Cortex-M**

I've released a Kalman filter crate designed for microcontrollers from the start:

- `no_std`, no heap, and everything works in `f32`. Turn off the default `std` feature and
  nothing else changes.
- Dimension mismatches fail at compile time, through const generics on `nalgebra` static
  matrices.
- It never panics on bad sensor data: NaN or degenerate input returns an error and leaves the
  filter state untouched, so firmware can drop the reading and keep running.
- KF, EKF, UKF, square-root KF and UKF, an RTS smoother over a fixed-size array, and optional
  dual-number Jacobians.

Footprint: a complete STM32F446RE image running a 4-state, 2-measurement filter for 1,000
steps is 13,988 bytes of flash and 24 bytes of static RAM (Rust 1.95, `opt-level = 3`, LTO;
about 8 KB of the flash is the benchmark data). The same image with adskalman is 16,276 bytes.

Single precision tip: if your sensor is very precise compared with your prior, use `SqrtKf`.
It propagates a Cholesky factor, which needs half the precision. In the test suite's hardest
`f32` problem it ran a million steps where the covariance-form filter failed at step two.

The firmware crate in the repo already reads the DWT cycle counter, but I haven't run it on a
board yet. **If anyone has a Nucleo-F446RE (or another M4F) and a probe, I'd love cycle counts**:
`bench/firmware` has the two images, and the result lands in a `CYCLES_PER_STEP` static.

crates.io: https://crates.io/crates/typed-kalman · repo: <REPO_URL>

---

## 3. r/rust

**Title:** typed-kalman 0.1: no_std Kalman filters with compile-time dimensions, a UKF, and automatic Jacobians

**Body:**

typed-kalman is a Kalman filter crate on `nalgebra` static matrices:

- **KF, EKF, UKF**, square-root KF and UKF, and an RTS smoother
- **Compile-time dimensions**: a wrong-sized measurement doesn't compile
- **`no_std`, zero allocation, `f32` or `f64`**
- **Never panics on bad data**: errors leave the filter unchanged
- **Automatic Jacobians** from dual numbers (optional `autodiff` feature)
- **NIS/NEES diagnostics** with chi-squared bounds

It agrees with adskalman, the established crate, to within 1e-9 on linear problems, and adds
the nonlinear filters, a square-root KF for ill-conditioned problems and `f32`, diagnostics, and
autodiff. Its firmware footprint is also smaller (13,988 vs 16,276 bytes of flash for the same
Cortex-M4F image). Speed is about on par on my machine; proper benchmark numbers will follow
from a dedicated machine.

A few things I learned building it, in case they're useful:

- Stable Rust can't write `2N + 1` as a const generic, so the UKF stores its sigma points as a
  center point plus two N×N matrices.
- The textbook UKF update `P - K S Kᵀ` failed in `f32` at step one. Rewriting it in Joseph
  form over the sigma points (a sum of positive terms) fixed it.
- In `f64`, some "stability tests" from the literature can't pass for *any* filter that stores
  the covariance: P's condition number passes 1e16. A square-root filter handles them, because
  its factor has the square root of that condition number.

Docs: https://docs.rs/typed-kalman · Repo: <REPO_URL>

Feedback very welcome, especially on the model-trait API and how you'd like angle wrapping for
residuals to work.
