use crate::engine::{SlotId, ValueId};

/// A declared per-run input: a leaf whose payload is supplied when a
/// run starts, falling back to a recorded default.
///
/// The node holds only its slot; the default payload lives in the
/// tape's input store, and `Network::forward_with` overlays fed
/// payloads for one run without touching the graph. It behaves exactly
/// like `Leaf` during runs: supplied rather than computed, with no
/// gradients routed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Input(pub(crate) SlotId);

impl Input {
    /// Calls `visitor` with each operand link; an input has none.
    pub(crate) fn visit_operands(&self, _visitor: impl FnMut(ValueId)) {}
}
