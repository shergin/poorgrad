use crate::engine::ValueId;

/// The payloads of one generation's parameters, indexed by slot.
///
/// It is the mutable state of a network, split from the immutable tape
/// columns: structure is recorded once and shared forever, while this
/// store turns over per generation. Forks share it through an `Arc`;
/// `updated` builds a fresh store because a gradient step rewrites every
/// slot. Beside each payload the store keeps the tape position of the
/// parameter node, mapping node-indexed gradients to slots and keeping
/// `updated` at O(parameters).
#[derive(Debug, Clone)]
pub(crate) struct ParameterStore<Data> {
    pub(super) payloads: Vec<Data>,
    pub(super) nodes: Vec<ValueId>,
}

impl<Data> ParameterStore<Data> {
    pub(super) fn new() -> Self {
        Self {
            payloads: Vec::new(),
            nodes: Vec::new(),
        }
    }

    /// Returns the payloads in slot order.
    pub(crate) fn payloads(&self) -> &[Data] {
        &self.payloads
    }
}
