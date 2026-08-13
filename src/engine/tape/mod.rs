mod identity;
mod kinship;
mod operands;
mod parameter_store;
mod slot;
// The module convention names each file after its main concept, and this
// module's main concept is the `Tape` itself; the inception is deliberate.
#[allow(clippy::module_inception)]
mod tape;
mod tip;

pub(crate) use identity::{Branch, Lineage, Misbinding, Segment, chain_probe, chains_agree};
pub(crate) use kinship::Kinship;
pub(crate) use operands::Operands;
pub(super) use parameter_store::ParameterStore;
pub(crate) use slot::SlotId;
pub(crate) use tape::Tape;
use tip::Tip;
