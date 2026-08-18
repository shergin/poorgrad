use super::SlotId;

/// A learnable parameter: a leaf whose payload `Network::update`
/// replaces on each training step.
///
/// The node holds only its slot; the payload lives in the generation's
/// parameter slot store, which is what lets a gradient step swap state
/// without touching the recorded structure. It behaves exactly like
/// `Leaf` during runs: supplied rather than computed, with no gradients
/// routed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Parameter(pub(crate) SlotId);
