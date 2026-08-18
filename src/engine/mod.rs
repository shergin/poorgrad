mod compile;
mod field;
mod function;
mod network;
mod plan;
mod run;

pub use compile::Compile;
pub use field::{Field, Gradients};
pub use network::{Network, Parameters, Symbol, Tape, Value, concat, stack};
pub use plan::Plan;
pub(crate) use plan::WindowProduct;
pub use run::Run;

pub(crate) use function::Function;
pub(crate) use network::{Operands, Origin, SlotId, SlotStore, Structure, ValueId};
pub(crate) use run::Posture;
