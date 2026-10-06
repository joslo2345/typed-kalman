# kalman-rs

Type-safe, `no_std` Kalman filters for Rust: KF, EKF, UKF, square-root UKF and an RTS smoother, built on `nalgebra` static matrices with compile-time dimensions.

## Comparison with other libraries

Every number comes from `scripts/run_comparison.sh`, which runs each library on the same frozen
scenarios (`tests/vectors/`, S1–S5) and regenerates this table from `results/results.csv`.
"Ours vs best other" compares `kalman-rs` with the best other library on each row, above 1.00x
meaning ours is better. Numbers are only published from a dedicated benchmark machine.

<!-- BENCH:START -->
_Not yet measured on the benchmark machine._
<!-- BENCH:END -->

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
