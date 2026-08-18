//! `topos` is a tiny autograd engine for the GPU-poor.
//!
//! Expressions record a static computation graph onto a [`Tape`];
//! sealing it yields an immutable [`Network`] (the spec) and a
//! caller-owned [`Parameters`] value (the state). `forward`
//! materializes every value, `backward` differentiates one scalar
//! target, and `step` is a pure data transformation of the
//! parameters — training never touches the graph:
//!
//! ```
//! use topos::Tape;
//!
//! let tape = Tape::new();
//! let w = tape.parameter(0.0_f64);
//! let x = tape.input(0.0);
//! let y = tape.input(0.0);
//!
//! // Operators record the graph; values are `Copy` and never consumed.
//! let error = w * x - y;
//! let loss = error * error;
//!
//! // Symbols are the detached names every later phase speaks.
//! let (w, x, y, loss) = (w.symbol(), x.symbol(), y.symbol(), loss.symbol());
//! let network = tape.into_network();
//! let mut parameters = network.parameters();
//!
//! // The graph is recorded once; every step feeds one sample of the line
//! // `y = 2 * x` and steps the parameters, leaving the network untouched.
//! let samples = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
//! for step in 0..100 {
//!     let (sample_x, sample_y) = samples[step % samples.len()];
//!     let run = network.forward(&parameters, [(x, sample_x), (y, sample_y)]);
//!     let gradients = run.backward(loss);
//!     parameters = parameters.step(&gradients, |w, g| w - 0.02 * g);
//! }
//!
//! let learned = *parameters.of(w);
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
    Compile, Field, Gradients, Network, Parameters, Plan, Run, Symbol, Tape, Value, concat, stack,
};
pub use neural::{
    Activation, Adam, AdamW, AveragePool, BatchNorm, BatchNormInference, Conv2d, Dropout, Flatten,
    LayerNorm, Linear, MaxPool, Mlp, Module, Neuron, Normalization, Optimizer, Path, Reshape,
    Residual, RmsNorm, Segment, Sequential, Sgd, Visitor, average_pool, checkpoint, conv2d,
    cross_entropy, init, max_pool, named_parameters, parameters,
};
pub use payload::{
    Bf16, Differentiable, Elementary, GemmTask, MapOperation, Shape, Tensor, Tensorial,
};
