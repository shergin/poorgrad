use crate::engine::{SlotId, ValueId};

/// A learnable parameter: a leaf whose payload `Network::updated`
/// replaces on each training step.
///
/// The node holds only its slot; the payload lives in the generation's
/// `ParameterStore`, which is what lets a gradient step swap state
/// without touching the recorded structure. It behaves exactly like
/// `Leaf` during runs: supplied rather than computed, with no gradients
/// routed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Parameter(pub(crate) SlotId);

impl Parameter {
    /// Calls `visitor` with each operand link; a parameter has none.
    pub(crate) fn visit_operands(&self, _visitor: impl FnMut(ValueId)) {}
}
