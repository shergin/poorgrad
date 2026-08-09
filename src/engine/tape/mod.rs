mod identity;
mod operands;
mod parameter_store;
mod slot;
// The module convention names each file after its main concept, and this
// module's main concept is the `Tape` itself; the inception is deliberate.
#[allow(clippy::module_inception)]
mod tape;

use identity::Tip;
pub(crate) use identity::{Branch, Lineage, Segment, chain_attributes, chains_agree};
pub(crate) use operands::Operands;
pub(super) use parameter_store::ParameterStore;
pub(crate) use slot::SlotId;
pub(crate) use tape::Tape;
