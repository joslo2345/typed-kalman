use typed_kalman::linear::LinearKf;
use nalgebra::{Matrix2x4, Matrix3, Vector2};

fn main() {
    let mut kf: LinearKf<4> = LinearKf::default();
    let h = Matrix2x4::<f64>::zeros();
    let r = Matrix3::<f64>::identity();
    let z = Vector2::<f64>::zeros();
    // A 2-entry measurement needs a 2x2 noise covariance, not 3x3.
    kf.update(&h, &z, &r).unwrap();
}
