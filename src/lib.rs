//! `poorgrad` is a tiny scalar autograd engine for the GPU-poor.

mod differentiable;
mod elementary;
mod evaluation;
mod function;
mod gradients;
mod network;
mod neuron;
mod symbol;
mod tape;
mod value;

pub use differentiable::Differentiable;
pub use elementary::Elementary;
pub use evaluation::Evaluation;
pub use gradients::Gradients;
pub use network::Network;
pub use neuron::Neuron;
pub use symbol::Symbol;
pub use value::Value;

pub(crate) use function::{Function, Operation};
pub(crate) use tape::Tape;
pub(crate) use value::ValueId;
