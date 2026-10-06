use kalman_rs::linear::LinearKf;
use nalgebra::{Matrix2, Matrix2x4, Vector3};

fn main() {
    let mut kf: LinearKf<4> = LinearKf::default();
    let h = Matrix2x4::<f64>::zeros();
    let r = Matrix2::<f64>::identity();
    let z = Vector3::<f64>::zeros();
    // H has 2 rows, so the measurement must have 2 entries, not 3.
    kf.update(&h, &z, &r).unwrap();
}
