//! Graph ownership across the two phases: the recording [`Tape`], the
//! sealed [`Network`], the caller-owned [`Parameters`], and the node
//! handles ([`Value`], [`Symbol`]).

// The module convention names each file after its main concept, and this
// module's main concept is the `Network` itself; the inception is
// deliberate.
#[allow(clippy::module_inception)]
mod network;
mod operands;
mod origin;
mod parameters;
mod slot;
mod slot_store;
mod structure;
mod symbol;
mod tape;
mod value;

pub use network::Network;
pub use parameters::Parameters;
pub use symbol::Symbol;
pub use tape::Tape;
pub use value::Value;

pub(crate) use operands::Operands;
pub(crate) use origin::Origin;
pub(crate) use slot::SlotId;
pub(crate) use slot_store::SlotStore;
pub(crate) use structure::Structure;
pub(crate) use value::ValueId;
