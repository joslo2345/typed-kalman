use typed_kalman::{Ekf, ProcessJacobian, ProcessModel};
use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};

/// A process model for a 3-dimensional state.
struct ThreeState;

impl ProcessModel<3> for ThreeState {
    fn predict(&self, x: &Vector3<f64>, _dt: f64) -> Vector3<f64> {
        *x
    }
}

impl ProcessJacobian<3> for ThreeState {
    fn jacobian(&self, _x: &Vector3<f64>, _dt: f64) -> Matrix3<f64> {
        Matrix3::identity()
    }
}

fn main() {
    let mut ekf = Ekf::new(Vector4::<f64>::zeros(), Matrix4::identity());
    // The filter has a 4-dimensional state; the model describes a 3-dimensional one.
    ekf.predict(&ThreeState, &Matrix4::identity(), 0.1);
}
