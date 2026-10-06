# Roadmap

What's planned after 0.1.1. Feedback from users comes first: if an issue or the announcement
threads ask for something not listed here, weigh it against this plan before starting.

## 0.2.0

0.2.0 is allowed to break the API, so it's the release to fix inconsistencies that 0.1 had
to keep. Bundle every breaking change into it, so users migrate once.

### Breaking: make the API consistent

1. **Every `predict` returns `Result` and checks its inputs.** Today `LinearKf::predict` and
   `Ekf::predict` return `()` and don't check for NaN or infinity, so a bad `F`, `Q` or `dt`
   silently corrupts the state. The UKFs and square-root filters already return `Result`.
   Make them all validate and return `KalmanError::InvalidInput`, leaving the filter unchanged,
   like every `update` does.
2. **One return type for `covariance()`.** `LinearKf`, `Ekf` and `Ukf` return `&SMatrix`,
   while `SqrtKf` and `SqrtUkf` compute `S Sᵀ` and return it by value. Pick one (by value
   everywhere is simplest), so generic code over filters works.
3. **A common filter trait.** With 1 and 2 done, a `Filter<N>` trait (`state`, `covariance`,
   `predict`, `update`) would let users swap filters and let the tests and benchmarks drop
   their per-filter boilerplate. Decide whether the model-based and matrix-based filters can
   share it, or need two traits.
4. **Review naming before it's locked in by more users**, for example `predict_with_cross` vs
   a more descriptive name, and `CrossStep` vs `UkfStep`.

### Additive: close the known gaps

5. **Smoothing for the square-root UKF:** add `SqrtUkf::predict_with_cross`, so
   `smooth_with_cross` works with it too.
6. **Wrapped states in the smoothers:** the smoothers subtract states directly, so a heading
   near ±π isn't smoothed correctly. Give `smooth_with_cross` a variant that takes the model's
   `state_residual`.
7. **A residual hook for the matrix filters:** `LinearKf` and `SqrtKf` take matrices, not
   models, so they can't wrap a measurement residual. Add an `update` variant that takes a
   residual function.
8. **Control inputs with automatic Jacobians:** an `AutoControlledProcess` trait, so
   `autodiff` models can take `u` explicitly instead of carrying it in a struct field.
9. **Missing measurements:** adskalman treats a NaN measurement as "skip the update", which is
   convenient for sensors that drop samples. Decide whether to offer that explicitly (for
   example an `update_optional(Option<&z>)`) while keeping NaN an error by default.
10. **An optional faster covariance update:** adskalman's optimal-gain update is about 10%
    faster than the Joseph form but less stable. Consider offering it as an explicit opt-in
    for well-conditioned problems, with the trade-off documented.
11. **`UkfParams` in the filter's scalar type,** instead of always `f64`.

### Documentation and examples

12. **A sensor-fusion example** that uses the 0.1.1 features together: a vehicle with a
    heading (state wrapping), commanded acceleration (control input), GPS and a compass
    (wrapped measurement), and smoothing afterwards.
13. **An embedded example crate** that prints results over RTT, building on `bench/firmware`.
14. **Guidance for `f32` sigma-point filters:** in `f32`, once the state grows large, the
    sigma-point spread falls below its resolution and the UKFs report
    `CovarianceNotPositiveDefinite`. Document the remedy (an error-state formulation, or
    `SqrtKf` where the problem is linear) with an example.

## Benchmarks and infrastructure (not tied to a version)

15. **Cycle counts on real hardware:** run `bench/firmware` on an STM32F446RE (or another
    Cortex-M4F) with a probe, and read `CYCLES_PER_STEP`.
16. **Published speed numbers:** run `scripts/run_comparison.sh` on a dedicated machine with
    a fixed CPU frequency, and paste the table between the README's `BENCH` markers.
17. **Shared scenario files:** move `tests/vectors/` into a shared repository, added here as a
    Git submodule, once the C, C++ and Python implementations exist.
18. **Release automation:** run `cargo-release` from a GitHub Actions workflow on tag push,
    so releases don't depend on one machine's setup.

## Done in 0.1.1

Angle wrapping for measurements and states, control inputs, the unscented RTS smoother, and
the inlining that made the linear KF faster than adskalman's Joseph form on every scenario.
See `CHANGELOG.md`.
