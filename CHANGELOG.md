# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  `AutoMeasurement` and the `AutoDiff` wrapper.
- `no_std` support with no heap allocation. Bad inputs return a `KalmanError` and leave the
  filter unchanged instead of panicking.
- Examples: `constant_velocity` and `imu_attitude` (needs the `autodiff` feature).

The minimum supported Rust version is 1.89.
