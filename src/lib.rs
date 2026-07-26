//! `poorgrad` is a tiny scalar autograd engine for the GPU-poor.

mod differentiable;
mod elementary;
mod function;
mod network;
mod neuron;
mod tape;
mod value;
mod value_inner;

pub use differentiable::Differentiable;
pub use elementary::Elementary;
pub use network::Network;
pub use neuron::Neuron;
pub use value::Value;

pub(crate) use function::Function;
pub(crate) use tape::Tape;
pub(crate) use value::ValueId;
pub(crate) use value_inner::ValueInner;
