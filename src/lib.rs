//! `poorgrad` is a tiny autograd engine for the GPU-poor.
//!
//! Expressions record a static computation graph onto a shared
//! `Network`; `forward` materializes every value, `backward`
//! differentiates one scalar target, and `updated` produces the next
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
//! // `y = 2 * x` and steps to the next generation, which shares everything
//! // but the parameters with the one before it.
//! let samples = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
//! let mut network = network;
//! for step in 0..100 {
//!     let (sample_x, sample_y) = samples[step % samples.len()];
//!     let loss = network.resolve(loss_symbol);
//!     let evaluation = network.forward_with([(x_symbol, sample_x), (y_symbol, sample_y)]);
//!     let gradients = evaluation.backward(loss);
//!     network = network.updated(gradients.as_field(), |w, g| w - 0.02 * g);
//! }
//!
//! let learned = network.resolve(w_symbol).data().unwrap();
//! assert!((learned - 2.0).abs() < 1e-6);
//! ```
#![forbid(unsafe_code)]

mod differentiable;
mod elementary;
mod evaluation;
mod field;
mod function;
mod gradients;
mod layer;
mod mlp;
mod network;
mod neuron;
mod shape;
mod symbol;
mod tape;
mod tensor;
mod tensorial;
mod value;

pub use differentiable::Differentiable;
pub use elementary::Elementary;
pub use evaluation::Evaluation;
pub use field::Field;
pub use gradients::Gradients;
pub use layer::Layer;
pub use mlp::Mlp;
pub use network::Network;
pub use neuron::{Activation, Neuron};
pub use shape::Shape;
pub use symbol::Symbol;
pub use tensor::Tensor;
pub use tensorial::Tensorial;
pub use value::Value;

pub(crate) use function::Function;
pub(crate) use tape::{Branch, Lineage, Segment, SlotId, Tape, chains_agree};
pub(crate) use value::ValueId;
