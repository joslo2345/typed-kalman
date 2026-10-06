# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-10-06

### Changed

- Faster filter steps from inlining the generic update paths: the linear KF is 5–10% faster
  and now ahead of adskalman's Joseph form on every benchmark scenario (S2 `f32` 725 vs 738
  µs, S2 `f64` 738 vs 841 µs, S5 23.9 vs 24.5 ms per 10,000 steps); `SqrtKf` is up to 39%
  faster. Firmware size is unchanged.

### Added

- Unscented RTS smoother: `smoother::smooth_with_cross` over `CrossStep`s, with
  `Ukf::predict_with_cross` reporting the cross-covariance it needs. It equals the linear RTS
  smoother on linear problems; on a pendulum it halves the UKF's error.
- Control inputs: `LinearKf::predict_with_input` and `SqrtKf::predict_with_input`
  (`x = F x + B u`), and `ControlledProcess` / `ControlledJacobian` with the `WithInput` adapter,
  which works with every model-based filter.
- `ProcessModel::state_residual` (and `AutoProcess::state_residual`): states with a component
  that wraps, such as a heading, work across ±π. The UKF and SR-UKF use it to average and
  spread their propagated sigma points.

## [0.1.0] - 2026-10-06

First release.

### Added

- Filters, all generic over `f32` and `f64`, with dimensions as const generics:
  - `LinearKf`: linear Kalman filter with the Joseph-form covariance update.
  - `SqrtKf`: square-root linear filter that propagates a Cholesky factor of the covariance,
    for problems too ill-conditioned for `LinearKf`.
  - `Ekf`: extended Kalman filter, with Jacobians from `ProcessJacobian` and
    `MeasurementJacobian`.
  - `Ukf` and `SqrtUkf`: unscented and square-root unscented filters, configured by
    `UkfParams`.
- `smoother::smooth`: Rauch-Tung-Striebel smoother over a caller-provided history slice,
  without `std`.
- `diagnostics`: `nees`, `chi_squared_quantile` and `chi_squared_bounds` for consistency
  checks; every `update` returns the NIS.
- `autodiff` feature: exact Jacobians from dual numbers, through `AutoProcess`,
  `AutoMeasurement` and the `AutoDiff` wrapper, in `f64` or `f32`.
- `MeasurementModel::residual` and `wrap_angle`: wrapped measurements such as bearings work
  across ±π. The EKF uses the residual for its innovation; the UKFs also use it to average and
  spread their sigma points.
- `no_std` support with no heap allocation. Bad inputs return a `KalmanError` and leave the
  filter unchanged instead of panicking.
- Examples: `constant_velocity` and `imu_attitude` (needs the `autodiff` feature).

The minimum supported Rust version is 1.89.

[Unreleased]: https://github.com/joslo2345/typed-kalman/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/joslo2345/typed-kalman/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/joslo2345/typed-kalman/releases/tag/v0.1.0
