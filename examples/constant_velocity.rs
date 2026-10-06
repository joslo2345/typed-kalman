//! Tracks an object moving at constant velocity in 2D from noisy position measurements.
//!
//! Shows the core workflow: filter with `LinearKf`, check that the filter is consistent with
//! the NIS, and smooth the recorded history with the RTS smoother.
//!
//! Run with `cargo run --example constant_velocity`.

use kalman_rs::diagnostics::chi_squared_bounds;
use kalman_rs::smoother::{smooth, Estimate, RtsStep};
use kalman_rs::LinearKf;
use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};

const STEPS: usize = 500;

/// A small deterministic Gaussian noise source, so the example needs no dependencies.
struct Noise(u64);

impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard normal sample (Box-Muller).
    fn gaussian(&mut self) -> f64 {
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn main() -> Result<(), kalman_rs::KalmanError> {
    // State [x, y, vx, vy]; we measure [x, y].
    let dt = 0.1;
    #[rustfmt::skip]
    let f = Matrix4::new(
        1.0, 0.0, dt,  0.0,
        0.0, 1.0, 0.0, dt,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    #[rustfmt::skip]
    let h = Matrix2x4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    );
    let velocity_noise: f64 = 0.01; // variance added to each velocity per step
    let q = Matrix4::from_diagonal(&Vector4::new(0.0, 0.0, velocity_noise, velocity_noise));
    let measurement_sd: f64 = 0.5;
    let r = Matrix2::identity() * measurement_sd * measurement_sd;

    // Simulate a target, then filter its measurements, recording what the smoother needs.
    let mut noise = Noise(0x2545_f491_4f6c_dd1d);
    let mut truth = Vector4::new(0.0, 0.0, 1.0, 0.5);
    let mut kf = LinearKf::new(Vector4::zeros(), Matrix4::identity() * 10.0);
    let mut truths = Vec::with_capacity(STEPS);
    let mut measurements = Vec::with_capacity(STEPS);
    let mut history = Vec::with_capacity(STEPS);
    let mut nis_sum = 0.0;

    for _ in 0..STEPS {
        truth = f * truth;
        truth[2] += velocity_noise.sqrt() * noise.gaussian();
        truth[3] += velocity_noise.sqrt() * noise.gaussian();
        let z = h * truth + Vector2::new(noise.gaussian(), noise.gaussian()) * measurement_sd;

        kf.predict(&f, &q);
        let predicted = Estimate {
            x: *kf.state(),
            p: *kf.covariance(),
        };
        nis_sum += kf.update(&h, &z, &r)?;
        history.push(RtsStep {
            f,
            predicted,
            filtered: Estimate {
                x: *kf.state(),
                p: *kf.covariance(),
            },
        });
        truths.push(truth);
        measurements.push(z);
    }

    let mut smoothed = vec![history[0].filtered; STEPS];
    smooth(&history, &mut smoothed)?;

    // Position error of each source, against the truth.
    let rmse = |position: &dyn Fn(usize) -> Vector2<f64>| {
        let sum: f64 = (0..STEPS)
            .map(|k| (position(k) - truths[k].fixed_rows::<2>(0)).norm_squared())
            .sum();
        (sum / STEPS as f64).sqrt()
    };
    println!("position RMSE over {STEPS} steps:");
    println!("  raw measurements  {:.3}", rmse(&|k| measurements[k]));
    println!(
        "  filtered          {:.3}",
        rmse(&|k| history[k].filtered.x.fixed_rows::<2>(0).into())
    );
    println!(
        "  smoothed          {:.3}",
        rmse(&|k| smoothed[k].x.fixed_rows::<2>(0).into())
    );

    // A consistent filter's average NIS falls inside these bounds (2-D measurements).
    let average_nis = nis_sum / STEPS as f64;
    let bounds = chi_squared_bounds(2, STEPS, 0.95)?;
    println!(
        "average NIS {average_nis:.2}, 95% bounds [{:.2}, {:.2}]: {}",
        bounds.lower,
        bounds.upper,
        if bounds.contains(average_nis) {
            "consistent"
        } else {
            "inconsistent"
        },
    );
    let v = kf.state();
    println!(
        "final velocity estimate ({:.2}, {:.2}), true ({:.2}, {:.2})",
        v[2], v[3], truth[2], truth[3]
    );
    Ok(())
}
