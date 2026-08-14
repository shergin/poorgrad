// Identity concerns are scoped in `identity.rs` (Origin, Branch,
// Segment, Tip, Misbinding, live Identity). Witness is the detached
// export; chain agreement is pure chain math.
mod chain;
mod identity;
mod operands;
mod slot;
mod slot_store;
mod structure;
// The module convention names each file after its main concept, and this
// module's main concept is the `Tape` itself; the inception is deliberate.
#[allow(clippy::module_inception)]
mod tape;
mod witness;

pub(crate) use chain::chains_agree;
pub(crate) use identity::{Branch, Identity, Misbinding, Origin, Segment, chain_probe};
pub(crate) use operands::Operands;
pub(crate) use slot::SlotId;
pub(super) use slot_store::SlotStore;
pub(crate) use structure::Structure;
pub(crate) use tape::Tape;
pub(crate) use witness::Witness;

#[cfg(test)]
#[path = "tests/identity_tests.rs"]
mod identity_tests;
