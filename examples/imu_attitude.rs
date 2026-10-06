//! Estimates roll, pitch and gyro biases from a gyroscope and an accelerometer.
//!
//! The process (Euler-angle kinematics driven by the gyro) and the measurement (gravity seen by
//! the accelerometer) are both nonlinear. Each is written once, generic over `Real`, and
//! `AutoDiff` gives the EKF exact Jacobians from it; the same models drive the UKF unchanged.
//!
//! Run with `cargo run --example imu_attitude --features autodiff`.

use kalman_rs::autodiff::{AutoDiff, AutoMeasurement, AutoProcess, Real};
use kalman_rs::{Ekf, Ukf};
use nalgebra::{Matrix3, Matrix4, SVector, Vector3, Vector4};

const G: f64 = 9.81;
const DT: f64 = 0.01;
const STEPS: usize = 3000; // 30 seconds at 100 Hz

/// State `[roll, pitch, roll-rate bias, pitch-rate bias]` propagated with one gyro reading
/// `[p, q]` (body roll and pitch rates, rad/s). Yaw rate is assumed zero.
struct Kinematics {
    gyro: [f64; 2],
}

impl AutoProcess<4> for Kinematics {
    fn predict<T: Real>(&self, x: &SVector<T, 4>, dt: f64) -> SVector<T, 4> {
        let (roll, pitch) = (x[0], x[1]);
        // Bias-corrected rates. Constants go on the right of `T` arithmetic.
        let p = -(x[2] - self.gyro[0]);
        let q = -(x[3] - self.gyro[1]);
        let roll_rate = p + q * roll.sin() * pitch.tan();
        let pitch_rate = q * roll.cos();
        Vector4::new(roll + roll_rate * dt, pitch + pitch_rate * dt, x[2], x[3])
    }
}

/// The accelerometer at rest measures the specific force opposing gravity, in body axes.
struct Accelerometer;

impl AutoMeasurement<4, 3> for Accelerometer {
    fn measure<T: Real>(&self, x: &SVector<T, 4>) -> SVector<T, 3> {
        let (roll, pitch) = (x[0], x[1]);
        Vector3::new(
            pitch.sin() * G,
            -(roll.sin() * pitch.cos()) * G,
            -(roll.cos() * pitch.cos()) * G,
        )
    }
}

/// A small deterministic Gaussian noise source, so the example needs no dependencies.
struct Noise(u64);

impl Noise {
    fn gaussian(&mut self) -> f64 {
        let mut uniform = || {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64).max(1e-300)
        };
        let (u1, u2) = (uniform(), uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn main() -> Result<(), kalman_rs::KalmanError> {
    let true_bias = [0.02, -0.015]; // rad/s
    let (gyro_sd, accel_sd) = (0.01, 0.2);

    // Noise: gyro noise integrates into the angles; the biases drift very slowly.
    let angle_var = (gyro_sd * DT) * (gyro_sd * DT);
    let q = Matrix4::from_diagonal(&Vector4::new(angle_var, angle_var, 1e-12, 1e-12));
    let r = Matrix3::identity() * accel_sd * accel_sd;
    let (x0, p0) = (
        Vector4::zeros(),
        Matrix4::from_diagonal(&Vector4::new(0.1, 0.1, 1e-3, 1e-3)),
    );

    let mut ekf = Ekf::new(x0, p0);
    let mut ukf = Ukf::new(x0, p0);
    let accelerometer = AutoDiff(Accelerometer);
    let mut noise = Noise(0x9e37_79b9_7f4a_7c15);
    let (mut ekf_sq, mut ukf_sq) = (0.0, 0.0);

    for k in 1..=STEPS {
        // A vehicle rocking in roll and pitch; its body rates follow from the Euler-angle rates.
        let t = k as f64 * DT;
        let (roll, pitch) = (0.3 * (0.5 * t).sin(), 0.2 * (0.3 * t).sin());
        let (roll_rate, pitch_rate) = (0.15 * (0.5 * t).cos(), 0.06 * (0.3 * t).cos());
        let q_body = pitch_rate / roll.cos();
        let p_body = roll_rate - q_body * roll.sin() * pitch.tan();

        let gyro = [
            p_body + true_bias[0] + gyro_sd * noise.gaussian(),
            q_body + true_bias[1] + gyro_sd * noise.gaussian(),
        ];
        let truth = Vector4::new(roll, pitch, 0.0, 0.0);
        let accel = Accelerometer.measure(&truth)
            + Vector3::new(noise.gaussian(), noise.gaussian(), noise.gaussian()) * accel_sd;

        // The gyro reading drives the process model at this step.
        let process = AutoDiff(Kinematics { gyro });
        ekf.predict(&process, &q, DT);
        ekf.update(&accelerometer, &accel, &r)?;
        ukf.predict(&process, &q, DT)?;
        ukf.update(&accelerometer, &accel, &r)?;

        // Score the angles over the last two thirds, once the biases have settled.
        if k > STEPS / 3 {
            let angle_error = |x: &Vector4<f64>| (x[0] - roll).powi(2) + (x[1] - pitch).powi(2);
            ekf_sq += angle_error(ekf.state());
            ukf_sq += angle_error(ukf.state());
        }
    }

    let scored = (STEPS - STEPS / 3) as f64;
    let degrees = |sum_sq: f64| (sum_sq / scored).sqrt().to_degrees();
    println!("angle RMS error after convergence (roll and pitch):");
    println!("  EKF with automatic Jacobians  {:.3}°", degrees(ekf_sq));
    println!("  UKF, same models              {:.3}°", degrees(ukf_sq));
    for (name, x) in [("EKF", ekf.state()), ("UKF", ukf.state())] {
        println!(
            "{name} gyro bias estimate ({:+.4}, {:+.4}) rad/s, true ({:+.4}, {:+.4})",
            x[2], x[3], true_bias[0], true_bias[1]
        );
    }
    Ok(())
}
