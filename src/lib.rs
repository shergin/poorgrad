//! `poorgrad` is a tiny autograd engine for the GPU-poor.
//!
//! Expressions record a static computation graph onto a shared
//! `Network`; `forward` materializes every value, `backward`
//! differentiates one scalar target, and `update` produces the next
//! network generation from a gradient step:
//!
//! ```
//! use poorgrad::Network;
//!
//! let network = Network::new();
//! let w = network.parameter(0.0_f64);
//! let x = network.input(0.0);
//! let y = network.input(0.0);
//!
//! // Operators record the graph; values are `Copy` and never consumed.
//! let error = w * x - y;
//! let loss = error * error;
//!
//! let w_symbol = w.symbol();
//! let x_symbol = x.symbol();
//! let y_symbol = y.symbol();
//! let loss_symbol = loss.symbol();
//!
//! // The graph is recorded once; every step feeds one sample of the line
//! // `y = 2 * x` and steps to the next generation, which shares the recorded
//! // graph while replacing the parameter payloads.
//! let samples = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
//! let mut network = network;
//! for step in 0..100 {
//!     let (sample_x, sample_y) = samples[step % samples.len()];
//!     let loss = network.resolve(loss_symbol);
//!     let run = network.forward_with([(x_symbol, sample_x), (y_symbol, sample_y)]);
//!     let gradients = run.backward(loss);
//!     network = network.update(&gradients, |w, g| w - 0.02 * g);
//! }
//!
//! let learned = network.resolve(w_symbol).payload().unwrap();
//! assert!((learned - 2.0).abs() < 1e-6);
//! ```
// The default build forbids `unsafe` outright. A backend feature
// drops `forbid` but keeps the crate-wide `deny`, so `unsafe`
// outside a scope-allowed backend module stays a compile error.
#![cfg_attr(
    not(any(
        feature = "accelerate",
        feature = "metal",
        feature = "simd",
        feature = "cuda"
    )),
    forbid(unsafe_code)
)]
#![deny(unsafe_code)]

mod backend;
mod emission;
mod engine;
mod neural;
#[cfg(feature = "evcxr")]
mod notebook;
mod payload;

pub use backend::{Backend, BackendUnavailable};
pub use emission::{EmitError, Emittable};
pub use engine::{
    Field, Gradients, Network, Plan, Retention, Run, Symbol, Value, ValueRef, concat, stack,
};
pub use neural::{
    Activation, Adam, AdamW, AveragePool, BatchNorm, BatchNormInference, Conv2d, Flatten,
    LayerNorm, Linear, MaxPool, Mlp, Module, Neuron, Normalization, Optimizer, Path, Reshape,
    Residual, RmsNorm, Segment, Sequential, Sgd, Visitor, average_pool, checkpoint, conv2d,
    cross_entropy, init, max_pool, named_parameters, parameters,
};
pub use payload::{
    Bf16, Differentiable, Elementary, GemmTask, MapOperation, Shape, Tensor, Tensorial,
};
