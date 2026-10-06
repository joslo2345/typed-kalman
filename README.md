# typed-kalman

Type-safe Kalman filters for Rust, from desktops to microcontrollers: a linear KF, EKF, UKF,
square-root KF and UKF, and an RTS smoother, with NIS/NEES consistency diagnostics and optional
automatic Jacobians.

- **Dimension errors don't compile.** States and measurements are `nalgebra` static matrices
  sized by const generics, so passing a 3-element measurement where the model expects 2 is a
  type error.
- **Runs without `std` or a heap.** Every filter works on bare metal and never allocates, in
  `f32` or `f64`.
- **Never panics on bad data.** A NaN reading or a singular covariance returns an error and
  leaves the filter unchanged, so you can skip that measurement and carry on.
- **Numerically careful.** Every covariance update uses the Joseph form, and the square-root
  filters keep working on problems too ill-conditioned for any filter that stores the
  covariance itself.

## Quick start

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

The [crate documentation](https://docs.rs/typed-kalman) explains which filter to choose. The
examples show complete programs:

- `cargo run --example constant_velocity`: track a 2D target, check consistency with the NIS,
  and smooth the history.
- `cargo run --example imu_attitude --features autodiff`: estimate roll, pitch and gyro biases
  with an EKF whose Jacobians are computed automatically, and a UKF on the same models.

## Embedded use

Turn off the default `std` feature. Nothing else changes, and single precision works with every
filter:

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

`bench/firmware` builds a complete Cortex-M4F image (STM32F446RE) that runs this kind of
filter on 1,000 steps of benchmark scenario S2. With Rust 1.95 it takes 13,988 bytes of flash,
about 8 KB of which is the scenario data, and 24 bytes of static RAM; the filter state lives on
the stack.

If your measurements are very precise compared with your prior, use `SqrtKf` in `f32`: it
propagates a Cholesky factor of the covariance, which needs half the precision. On the test
suite's hardest single-precision problem, `LinearKf` loses positive-definiteness within two
steps while `SqrtKf` runs a million.

## Features

| Feature | Default | What it adds |
|---|---|---|
| `std` | yes | Links the standard library. Turn it off for bare-metal targets. |
| `autodiff` | no | Exact Jacobians from dual numbers: write a model once and get its EKF Jacobians. |

The minimum supported Rust version is 1.89, set by `nalgebra` 0.35.

## Compared with adskalman

[adskalman](https://crates.io/crates/adskalman) is the established Kalman filter crate, and a
strong one: it also has compile-time dimensions, `no_std` support, any `nalgebra` scalar, and
an RTS smoother. The test suite checks that the two agree to within 1e-9 on the same problems.

| | typed-kalman | adskalman 0.18 |
|---|---|---|
| Linear KF, RTS smoother | ✓ | ✓ |
| Covariance update | Joseph form | Joseph form, or two faster optimal-gain forms |
| Square-root (Cholesky factor) KF | ✓ | – |
| Nonlinear filters | EKF, UKF, square-root UKF | linearize the observation yourself each step |
| Automatic Jacobians | ✓ (`autodiff` feature) | – |
| NIS from each update, NEES and chi-squared bounds | ✓ | – |
| Missing measurement | returns an error; skip the update | a NaN measurement skips the update |
| Firmware flash, S2 (see above) | 13,988 bytes | 16,276 bytes |

Accuracy is the same: for a linear system with Gaussian noise the Kalman filter is optimal, so
two correct implementations must agree.

## Benchmarks

Every number comes from `scripts/run_comparison.sh`, which runs each library on the same frozen
scenarios (`tests/vectors/`, S1–S5) and regenerates this table from `results/results.csv`.
"Ours vs best other" compares `typed-kalman` with the best other library on each row, above 1.00x
meaning ours is better. Numbers are only published from a dedicated benchmark machine.

<!-- BENCH:START -->
_Not yet measured on the benchmark machine._
<!-- BENCH:END -->

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
