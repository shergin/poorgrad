//! `poorgrad` is a tiny scalar autograd engine for the GPU-poor.
#![forbid(unsafe_code)]

mod differentiable;
mod elementary;
mod evaluation;
mod field;
mod function;
mod gradients;
mod layer;
mod network;
mod neuron;
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
pub use network::Network;
pub use neuron::{Activation, Neuron};
pub use symbol::Symbol;
pub use tensor::Tensor;
pub use tensorial::Tensorial;
pub use value::Value;

pub(crate) use function::{Function, Operation};
pub(crate) use tape::{Lineage, Tape};
pub(crate) use value::ValueId;
