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
//! let x = network.leaf(3.0);
//! let y = network.leaf(15.0);
//!
//! // Operators record the graph; values are `Copy` and never consumed.
//! let error = w * x - y;
//! let loss = error * error;
//!
//! let w_symbol = w.symbol();
//! let loss_symbol = loss.symbol();
//!
//! // A training step is a state transition: each generation shares
//! // everything but the parameters with the one before it.
//! let mut network = network;
//! for _ in 0..100 {
//!     let loss = network.resolve(loss_symbol);
//!     let evaluation = network.forward();
//!     let gradients = evaluation.backward(loss);
//!     network = network.updated(gradients.as_field(), |w, g| w - 0.01 * g);
//! }
//!
//! let learned = network.resolve(w_symbol).data().unwrap();
//! assert!((learned - 5.0).abs() < 1e-6);
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
