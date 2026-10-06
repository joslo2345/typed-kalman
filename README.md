<p align="center">
  <img src="docs/assets/logo.svg" alt="typed-kalman logo" width="140" />
</p>

<h1 align="center">typed-kalman</h1>

<p align="center">
  <strong>Type-safe Kalman filters for Rust, from desktops to microcontrollers</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/typed-kalman"><img src="https://img.shields.io/crates/v/typed-kalman?color=blue" alt="crates.io version" /></a>
  <a href="https://docs.rs/typed-kalman"><img src="https://img.shields.io/docsrs/typed-kalman" alt="docs.rs" /></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT_OR_Apache--2.0-green" alt="MIT OR Apache-2.0 license" /></a>
  <img src="https://img.shields.io/badge/MSRV-1.89-orange" alt="Minimum supported Rust version 1.89" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/🧮_Dimensions-Checked_at_Compile_Time-blue" alt="Dimensions checked at compile time" />
  <img src="https://img.shields.io/badge/🔩_no__std-Zero_Allocation-orange" alt="no_std, zero allocation" />
  <img src="https://img.shields.io/badge/🛡️_Bad_Input-Never_Panics-brightgreen" alt="Never panics on bad input" />
  <img src="https://img.shields.io/badge/📐_Precision-f32_%7C_f64-blueviolet" alt="f32 or f64" />
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="https://docs.rs/typed-kalman"><strong>Documentation</strong></a> ·
  <a href="#examples"><strong>Examples</strong></a> ·
  <a href="#compared-with-adskalman"><strong>Comparison</strong></a> ·
  <a href="#benchmarks"><strong>Benchmarks</strong></a> ·
  <a href="CHANGELOG.md"><strong>Changelog</strong></a>
</p>

---

A linear KF, EKF, UKF, square-root KF and UKF, and an RTS smoother, with NIS/NEES consistency
diagnostics and optional automatic Jacobians. Built on `nalgebra` static matrices.

<a id="highlights"></a>
## ✨ Highlights

<table align="center">
  <tr align="center" valign="top">
    <td width="33%">
      <strong>🧮 Dimension Errors Don't Compile</strong><br/><br/>
      States and measurements are sized by const generics, so a 3-element measurement where
      the model expects 2 is a type error, not a runtime surprise.<br/><br/>
      <a href="#quick-start">Quick start →</a>
    </td>
    <td width="33%">
      <strong>🔩 Bare Metal, No Heap</strong><br/><br/>
      Every filter works without <code>std</code> and never allocates, in <code>f32</code>
      or <code>f64</code>. A full Cortex-M4F image is under 14 KB of flash.<br/><br/>
      <a href="#embedded-use">Embedded use →</a>
    </td>
    <td width="33%">
      <strong>🛡️ Never Panics on Bad Data</strong><br/><br/>
      A NaN reading or a singular covariance returns an error and leaves the filter exactly
      as it was, so you skip that measurement and carry on.<br/><br/>
      <a href="https://docs.rs/typed-kalman">Error handling →</a>
    </td>
  </tr>
  <tr align="center" valign="top">
    <td width="33%">
      <strong>📐 Numerically Careful</strong><br/><br/>
      Every covariance update uses the Joseph form. The square-root filters keep working
      where any filter that stores the covariance breaks down.<br/><br/>
      <a href="#choosing-a-filter">Choosing a filter →</a>
    </td>
    <td width="33%">
      <strong>🎯 The Whole Filter Family</strong><br/><br/>
      KF, EKF, UKF, square-root KF and UKF, and an RTS smoother, with one consistent API and
      error type.<br/><br/>
      <a href="#examples">Examples →</a>
    </td>
    <td width="33%">
      <strong>🤖 Automatic Jacobians</strong><br/><br/>
      Write a nonlinear model once; dual numbers give the EKF exact Jacobians, and the UKFs
      use the same model unchanged.<br/><br/>
      <a href="#examples">IMU example →</a>
    </td>
  </tr>
</table>

<a id="quick-start"></a>
## 🚀 Quick Start

```toml
[dependencies]
typed-kalman = "0.1"
nalgebra = "0.35"
```

```rust
use typed_kalman::LinearKf;
use nalgebra::{Matrix1, Matrix1x2, Matrix2, Vector1, Vector2};

// State [position, velocity]; we measure position.
let dt: f64 = 0.1;
let f = Matrix2::new(1.0, dt, 0.0, 1.0);
let q = Matrix2::new(1e-4, 0.0, 0.0, 1e-2);
let h = Matrix1x2::new(1.0, 0.0);
let r = Matrix1::new(0.25);

let mut kf = LinearKf::new(Vector2::new(0.0, 0.0), Matrix2::identity());
for z in [0.21, 0.38, 0.62, 0.79] {
    kf.predict(&f, &q);
    // Returns the normalized innovation squared, or an error on a bad measurement.
    let nis = kf.update(&h, &Vector1::new(z), &r).expect("valid measurement");
    assert!(nis.is_finite());
}
println!("position {:.2}, velocity {:.2}", kf.state()[0], kf.state()[1]);
```

<a id="choosing-a-filter"></a>
## 🧭 Choosing a Filter

| Filter | You provide | Use it when |
|---|---|---|
| **`LinearKf`** | matrices `F`, `H` | the system is linear; the fastest choice |
| **`SqrtKf`** | matrices, noise as square roots | the covariance is too ill-conditioned for `LinearKf` (very precise sensors, `f32`); about 3.5× slower |
| **`Ekf`** | a model and its Jacobians | the system is mildly nonlinear; the `autodiff` feature can compute the Jacobians |
| **`Ukf`** | a model only | the system is nonlinear and Jacobians are awkward |
| **`SqrtUkf`** | a model, noise as square roots | as the UKF, with a covariance factor that can't lose positive-definiteness |

`smoother::smooth` runs a Rauch-Tung-Striebel smoother over a recorded history, and
`diagnostics` checks consistency with NEES and chi-squared bounds.

<a id="embedded-use"></a>
## 🔩 Embedded Use

Turn off the default `std` feature. Nothing else changes, and every filter works in single
precision:

```toml
[dependencies]
typed-kalman = { version = "0.1", default-features = false }
nalgebra = { version = "0.35", default-features = false, features = ["libm"] }
```

```rust
use typed_kalman::LinearKf;
use nalgebra::{Matrix1, Matrix1x2, Matrix2, Vector1};

fn step(kf: &mut LinearKf<2, f32>, position: f32) -> Option<f32> {
    let f = Matrix2::new(1.0, 0.01, 0.0, 1.0);
    let q = Matrix2::new(1e-6, 0.0, 0.0, 1e-4);
    kf.predict(&f, &q);
    // On a bad reading the filter is unchanged, so just skip it.
    kf.update(&Matrix1x2::new(1.0, 0.0), &Vector1::new(position), &Matrix1::new(0.01)).ok()?;
    Some(kf.state()[1]) // estimated velocity
}
```

| Firmware image (STM32F446RE, Rust 1.95) | Size |
|---|---|
| Flash, including ~8 KB of benchmark data | **13,988 bytes** |
| Static RAM (the filter state lives on the stack) | **24 bytes** |

> [!TIP]
> If your measurements are very precise compared with your prior, use `SqrtKf` in `f32`. It
> propagates a Cholesky factor of the covariance, which needs half the precision. On the test
> suite's hardest single-precision problem, `LinearKf` loses positive-definiteness within two
> steps while `SqrtKf` runs a million.

<a id="examples"></a>
## 🧪 Examples

| Example | What it shows | Run it |
|---|---|---|
| 🚗 **Constant velocity** | Track a 2D target, check consistency with the NIS, and smooth the history: the position error drops from 0.73 (raw) to 0.32 (filtered) to 0.16 (smoothed). | `cargo run --example constant_velocity` |
| 🛩️ **IMU attitude** | Estimate roll, pitch and gyro biases from a gyro and an accelerometer with an EKF whose Jacobians are computed automatically, and a UKF on the same models: 0.12° RMS error. | `cargo run --example imu_attitude --features autodiff` |

<a id="features"></a>
## 🧩 Features

| Feature | Default | What it adds |
|---|---|---|
| `std` | ✅ | Links the standard library. Turn it off for bare-metal targets. |
| `autodiff` | – | Exact Jacobians from dual numbers: write a model once and get its EKF Jacobians. |

The minimum supported Rust version is **1.89**, set by `nalgebra` 0.35.

<a id="compared-with-adskalman"></a>
## ⚖️ Compared with adskalman

[adskalman](https://crates.io/crates/adskalman) is the established Kalman filter crate, and a
strong one: it also has compile-time dimensions, `no_std` support, any `nalgebra` scalar, and an
RTS smoother. The test suite checks that the two agree to within 1e-9 on the same problems.

| | adskalman 0.18 | typed-kalman |
|---|---|---|
| **Linear KF, RTS smoother** | ✅ | ✅ |
| **Covariance update** | Joseph form, or two faster optimal-gain forms | Joseph form |
| **Square-root (Cholesky factor) KF** | – | ✅ |
| **Nonlinear filters** | linearize the observation yourself each step | ✅ EKF, UKF, square-root UKF |
| **Automatic Jacobians** | – | ✅ `autodiff` feature |
| **NIS from each update, NEES, chi-squared bounds** | – | ✅ |
| **Missing measurement** | a NaN measurement skips the update | returns an error; skip the update |
| **Firmware flash, S2** | 16,276 bytes | **13,988 bytes** |

Accuracy is the same: for a linear system with Gaussian noise the Kalman filter is optimal, so
two correct implementations must agree.

<a id="benchmarks"></a>
## 📊 Benchmarks

Every number comes from `scripts/run_comparison.sh`, which runs each library on the same frozen
scenarios (`tests/vectors/`, S1–S5) and regenerates this table from `results/results.csv`.
"Ours vs best other" compares `typed-kalman` with the best other library on each row, above
1.00x meaning ours is better. Numbers are only published from a dedicated benchmark machine.

| ID | Scenario | State / measurement | What it shows |
|---|---|---|---|
| S1 | 1D constant velocity | 2 / 1 | Overhead on tiny problems |
| S2 | 2D constant velocity | 4 / 2 | A typical tracking workload |
| S3 | Range-bearing tracking | 4 / 2 | Accuracy on a nonlinear problem |
| S4 | Ill-conditioned, 1M steps in `f32` | 4 / 2 | Numerical stability |
| S5 | 15-state INS error state | 15 / 6 | Speed on a larger state |

<!-- BENCH:START -->
_Not yet measured on the benchmark machine._
<!-- BENCH:END -->

<a id="verified"></a>
## ✅ How It's Verified

- 🔁 **Equivalence:** filter and smoother agree with adskalman to within 1e-9, including on 20
  random ill-conditioned problems.
- 🎲 **Consistency:** over 500 Monte Carlo runs, every filter's average NEES and NIS fall
  inside their 95% chi-squared bounds.
- 🧱 **Stability:** property tests on random ill-conditioned problems, plus million-step `f32`
  runs that break unstabilized filters.
- 🚫 **No allocation:** a counting global allocator checks that filter loops never touch the
  heap.
- 🛡️ **Bad input:** NaN, infinite and singular inputs return errors and leave every filter
  unchanged.
- 🧩 **Compile-time safety:** `trybuild` checks that dimension and precision mismatches don't
  compile.
- ⚙️ **CI:** stable, beta and the MSRV, with and without `std`, plus a bare-metal ARM build and
  firmware images.

<a id="roadmap"></a>
## 🗺️ Roadmap

| Item | Status |
|---|---|
| Cycle counts on real hardware | The firmware already records them; it needs a board run |
| Shared scenario files with the C, C++ and Python implementations | The files exist in `tests/vectors/`; they'll move to a shared repository |
| Angle wrapping for residuals (bearings near ±π) | ✅ Done: override `MeasurementModel::residual`, with `wrap_angle` |
| `autodiff` in `f32` | Planned |

<a id="license"></a>
## 📄 License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

<p align="center">
  <sub>Built on <a href="https://nalgebra.org">nalgebra</a>. Benchmarked against
  <a href="https://crates.io/crates/adskalman">adskalman</a>, with thanks to its authors.</sub>
</p>
