mod compile;
mod composite;
mod field;
mod function;
mod literal;
mod network;
mod plan;
mod run;
mod trace;

pub use compile::{Compile, Memory};
pub use composite::{concat, stack};
pub use field::{Field, Gradients};
pub use network::{Network, Symbol, Value, ValueRef};
pub use plan::Plan;
pub(crate) use plan::WindowProduct;
pub use run::Run;

pub(crate) use function::Function;
pub(crate) use network::{Designation, Misbinding, Operands, SlotId, Structure, ValueId, Witness};
pub(crate) use run::Posture;
pub(crate) use trace::Trace;
