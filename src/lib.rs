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
//!     let evaluation = network.forward_with([(x_symbol, sample_x), (y_symbol, sample_y)]);
//!     let gradients = evaluation.backward(loss);
//!     network = network.update(&gradients, |w, g| w - 0.02 * g);
//! }
//!
//! let learned = network.resolve(w_symbol).payload().unwrap();
//! assert!((learned - 2.0).abs() < 1e-6);
//! ```
#![forbid(unsafe_code)]

mod backend;
mod engine;
mod neural;
mod payload;

pub use engine::{Evaluation, Field, Gradients, Network, Symbol, Value};
pub use neural::{Activation, Layer, Mlp, Neuron, cross_entropy, init};
pub use payload::{Differentiable, Elementary, GemmTask, Shape, Tensor, Tensorial};
