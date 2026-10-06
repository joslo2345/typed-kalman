use typed_kalman::linear::LinearKf;
use nalgebra::{Matrix2, Matrix2x4, Vector2};

fn main() {
    let mut kf: LinearKf<4> = LinearKf::default();
    let h = Matrix2x4::<f64>::zeros();
    let r = Matrix2::<f64>::identity();
    let z = Vector2::<f32>::zeros();
    // The filter is f64; an f32 measurement must be converted explicitly.
    kf.update(&h, &z, &r).unwrap();
}
