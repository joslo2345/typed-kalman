# Benchmark scenarios S1–S5

Shared inputs for the C, C++, Python and Rust Kalman filter implementations. Every library
reads exactly these files, so their results are comparable. These files are **frozen**:
changing one after results exist invalidates every result collected from it.

| ID | Problem | State / measurement | Runs × steps | Measurements |
|----|---------|---------------------|--------------|--------------|
| S1 | 1D constant velocity | 2 / 1 | 1 × 10,000 | float64 |
| S2 | 2D constant velocity | 4 / 2 | 1 × 10,000 | float64 |
| S3 | Range-bearing tracking (nonlinear) | 4 / 2 | 200 × 500 | float64 |
| S4 | Ill-conditioned, for 32-bit stability | 4 / 2 | 1 × 1,000,000 | float32, no truth |
| S5 | 15-state INS error state, GPS-aided | 15 / 6 | 1 × 10,000 | float64 |

## Format

Each `S<n>/` directory holds:

- `model.json`: the problem. Matrices `f`, `h`, `q`, `r`, `p0` are **row-major** nested
  arrays (`f[i][j]` is row `i`, column `j`); `x0` is a flat array. `dt` is the time step,
  `seeds` the generator seed of each run.
- `measurements.bin`, and `truth.bin` unless `truth` is `null`: raw **little-endian** arrays
  of the `dtype` and `shape` given in `model.json`, laid out row-major as
  `[runs, steps, dim]`. There's no header.

Reading them:

```python
import json, numpy as np
m = json.load(open("S2/model.json"))
zs = np.fromfile("S2/measurements.bin", dtype="<f8").reshape(m["measurements"]["shape"])
```

```c
/* S2: 1 run x 10000 steps x 2 doubles */
double zs[10000][2];
fread(zs, sizeof(double), 10000 * 2, fopen("S2/measurements.bin", "rb"));
```

## Conventions every implementation must follow

- Start each run from `x0`, `p0`. Each step **predicts with `f`, `q`, then updates with that
  step's measurement**. `truth[k]` is the true state after step `k`'s transition, the same
  instant as `measurements[k]`.
- S3's measurement is `z = [hypot(x, y), atan2(y, x)]` with the sensor at the origin (`h` is
  zero and unused). Targets stay at `x > 6800`, so bearings stay within ±0.36 rad and need no
  angle wrapping. Linearize at the predicted state for an EKF.
- S4 runs in 32-bit floats: convert every matrix and measurement to `float32` before
  filtering. Only stability is measured (steps until the covariance stops being symmetric
  positive-definite), so truth isn't stored.
- `SHA256SUMS` lists the checksum of every file. Verify it with `shasum -a 256 -c SHA256SUMS`.

The Rust repo generates the files (`cargo run --release --example generate_vectors`) and checks
them against the generator in `tests/vectors.rs`.
