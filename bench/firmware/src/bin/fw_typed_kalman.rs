//! S2 with typed-kalman's LinearKf (Joseph form), f32.
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use firmware::{matrix, measurement, s2, time_and_halt};
use nalgebra::SVector;
use panic_halt as _;
use typed_kalman::LinearKf;

#[entry]
fn main() -> ! {
    let (f, h, q, r) = (
        matrix(&s2::F),
        matrix(&s2::H),
        matrix(&s2::Q),
        matrix(&s2::R),
    );
    let mut kf = LinearKf::new(SVector::from(s2::X0), matrix(&s2::P0));
    time_and_halt(|| {
        for k in 0..s2::STEPS {
            kf.predict(&f, &q);
            let _ = kf.update(&h, &measurement(k), &r);
        }
        *kf.state()
    })
}
