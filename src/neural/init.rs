//! Deterministic initializer factories for neural building blocks.
//!
//! Initialization is caller-owned: [`Layer`](super::Layer) and
//! [`Mlp`](super::Mlp) record whatever payloads they are given and take a
//! shape-to-payload closure at construction. This module manufactures
//! such closures. Every factory takes an explicit `seed` and each
//! returned initializer owns its generator state, so runs are
//! bit-identical forever and concurrent initializers never share state:
//! there is no global generator and no clock.
//!
//! The generator is a splitmix64 — statistical quality suited to
//! initialization, not cryptography. It is carried here in a few lines
//! instead of a `rand` dependency because reproducibility is a feature:
//! a seeded example must not change output when a dependency upgrades,
//! and `rand`'s standard generator is documented as unstable across its
//! versions.

use crate::{Shape, Tensor};

/// Advances `state` and returns the next splitmix64 output.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

/// Returns the next value, uniformly distributed in `[0, 1)`.
fn unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Returns the next value from the standard normal distribution.
///
/// It is one half of a Box-Muller pair; the sine partner is discarded
/// for simplicity.
fn standard_normal(state: &mut u64) -> f64 {
    let radius = (-2.0 * (1.0 - unit(state)).ln()).sqrt();
    let angle = std::f64::consts::TAU * unit(state);
    radius * angle.cos()
}

/// Builds a tensor of `shape` with every element drawn from `draw`.
fn drawn(shape: &Shape, state: &mut u64, mut draw: impl FnMut(&mut u64) -> f64) -> Tensor<f64> {
    let elements: Vec<f64> = (0..shape.volume()).map(|_| draw(state)).collect();
    Tensor::new(shape.axes().iter().copied(), elements)
}

/// Returns an initializer filling every requested shape with values
/// uniformly distributed in `[-scale, scale)`.
pub fn uniform(seed: u64, scale: f64) -> impl FnMut(&Shape) -> Tensor<f64> {
    let mut state = seed;
    move |shape| drawn(shape, &mut state, |state| (unit(state) * 2.0 - 1.0) * scale)
}

/// Returns an initializer filling every requested shape with values
/// normally distributed around zero with the given standard `deviation`.
pub fn normal(seed: u64, deviation: f64) -> impl FnMut(&Shape) -> Tensor<f64> {
    let mut state = seed;
    move |shape| {
        drawn(shape, &mut state, |state| {
            standard_normal(state) * deviation
        })
    }
}

/// Returns the Glorot (Xavier) initializer: rank-2 `[inputs, outputs]`
/// weights are uniform within `±sqrt(6 / (inputs + outputs))`, keeping
/// activation variance steady in both directions through `tanh`-like
/// layers, and rank-1 shapes are zero — a bias identifies itself
/// structurally by its rank.
///
/// # Panics
/// The returned initializer panics on a shape that is neither rank 2 nor
/// rank 1.
///
/// # See also
/// - X. Glorot and Y. Bengio, "Understanding the difficulty of training
///   deep feedforward neural networks" (2010).
pub fn xavier(seed: u64) -> impl FnMut(&Shape) -> Tensor<f64> {
    let mut state = seed;
    move |shape| match shape.rank() {
        1 => Tensor::filled(shape.axes().iter().copied(), 0.0),
        2 => {
            let fan_total = (shape.axes()[0] + shape.axes()[1]) as f64;
            let bound = (6.0 / fan_total).sqrt();
            drawn(shape, &mut state, |state| (unit(state) * 2.0 - 1.0) * bound)
        }
        _ => panic!("xavier initialization expects rank-2 weights or rank-1 biases, got {shape}"),
    }
}

/// Returns the Kaiming (He) initializer: rank-2 `[inputs, outputs]`
/// weights are normal with deviation `sqrt(2 / inputs)`, compensating
/// the variance a ReLU halves, and rank-1 shapes are zero — a bias
/// identifies itself structurally by its rank.
///
/// # Panics
/// The returned initializer panics on a shape that is neither rank 2 nor
/// rank 1.
///
/// # See also
/// - K. He et al., "Delving Deep into Rectifiers" (2015).
pub fn kaiming(seed: u64) -> impl FnMut(&Shape) -> Tensor<f64> {
    let mut state = seed;
    move |shape| match shape.rank() {
        1 => Tensor::filled(shape.axes().iter().copied(), 0.0),
        2 => {
            let deviation = (2.0 / shape.axes()[0] as f64).sqrt();
            drawn(shape, &mut state, |state| {
                standard_normal(state) * deviation
            })
        }
        _ => panic!("kaiming initialization expects rank-2 weights or rank-1 biases, got {shape}"),
    }
}

#[cfg(test)]
#[path = "tests/init_tests.rs"]
mod tests;
