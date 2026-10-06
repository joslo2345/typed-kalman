//! A small deterministic Gaussian generator (xorshift64* and Box-Muller).
//!
//! Every scenario is reproducible from its seed, bit for bit on every platform: the
//! transcendental functions come from the pure-Rust `libm` crate, not the system math library,
//! whose last-bit results differ between macOS and Linux.

use nalgebra::{SMatrix, SVector};

/// Generates standard normal samples from a 64-bit seed.
pub struct Rng(u64);

impl Rng {
    /// Creates a generator. Any seed works, including zero.
    pub fn new(seed: u64) -> Self {
        // xorshift has a fixed point at zero, so mix the seed first (SplitMix64).
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Self((z ^ (z >> 31)) | 1)
    }

    /// Returns a uniform sample in (0, 1).
    pub fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Returns a uniform sample in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Returns `10^u` for `u` uniform in `[lo_exp, hi_exp)`.
    pub fn log_uniform(&mut self, lo_exp: f64, hi_exp: f64) -> f64 {
        libm::pow(10.0, self.range(lo_exp, hi_exp))
    }

    /// Returns a standard normal sample.
    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        // sqrt is correctly rounded everywhere; ln and cos are not, so they come from libm.
        (-2.0 * libm::log(u1)).sqrt() * libm::cos(2.0 * std::f64::consts::PI * u2)
    }

    /// Returns a vector of standard normal samples.
    pub fn normal_vector<const D: usize>(&mut self) -> SVector<f64, D> {
        SVector::from_fn(|_, _| self.normal())
    }

    /// Returns a sample from N(0, L Lᵀ).
    pub fn correlated<const D: usize>(&mut self, l: &SMatrix<f64, D, D>) -> SVector<f64, D> {
        l * self.normal_vector::<D>()
    }
}
