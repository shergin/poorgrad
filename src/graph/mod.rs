//! Graph ownership across the two phases — the recording [`Tape`],
//! the sealed [`Network`], the caller-owned [`Parameters`], and the
//! node handles ([`Value`], [`Symbol`]) — together with the recording
//! surface built over them (the opcode mnemonics on `Value`, the
//! composite tier, the payload-literal operator sugar, and the
//! recording `Trace` payload behind `Tape::differentiate`) and the
//! value-aligned [`Field`] buffers that carry gradients and optimizer
//! state over the graph.

mod composite;
mod field;
mod literal;
// The module convention names each file after its main concept, and this
// module's main concept is the `Network` itself; the inception is
// deliberate.
#[allow(clippy::module_inception)]
mod network;
mod operands;
mod origin;
mod parameters;
mod slot_store;
mod structure;
mod symbol;
mod tape;
mod trace;
mod value;

pub use composite::{concat, stack};
pub use field::{Field, Gradients};
pub use network::Network;
pub use parameters::Parameters;
pub use symbol::Symbol;
pub use tape::Tape;
pub use value::Value;

pub(crate) use operands::Operands;
pub(crate) use origin::Origin;
pub(crate) use slot_store::SlotStore;
pub(crate) use structure::Structure;
pub(crate) use value::ValueId;
