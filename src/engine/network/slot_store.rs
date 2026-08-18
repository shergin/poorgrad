use crate::engine::ValueId;

use super::SlotId;

/// A dense, slot-indexed table of payloads with their tape nodes.
///
/// Parameters and inputs share this layout: each slot holds one payload
/// and the [`ValueId`] of the structure node that names that slot.
/// [`SlotId`] is the row index — assigned only by this type on
/// [`install`](Self::install) — so loads stay O(1) and bulk steps
/// (generation `update`, future input bulk APIs) stay O(slots) against
/// node-indexed buffers via the `nodes` column.
///
/// Structure is recorded once; these tables turn over independently
/// (parameter payloads per training step in the caller's
/// [`Parameters`](crate::Parameters), input defaults when a run
/// overlays feeds).
#[derive(Debug, Clone)]
pub(crate) struct SlotStore<Data> {
    payloads: Vec<Data>,
    nodes: Vec<ValueId>,
}

impl<Data> SlotStore<Data> {
    /// Creates an empty store.
    pub(crate) fn new() -> Self {
        Self {
            payloads: Vec::new(),
            nodes: Vec::new(),
        }
    }

    /// Returns the number of allocated slots.
    pub(crate) fn len(&self) -> usize {
        self.payloads.len()
    }

    /// Returns the payloads in slot order.
    pub(crate) fn payloads(&self) -> &[Data] {
        &self.payloads
    }

    /// Allocates a slot for `data`, records its structure node via
    /// `record`, and stores the returned node link.
    ///
    /// `record` receives the new [`SlotId`] so it can embed the slot in
    /// a `Function` and push a structure row; it must return that row's
    /// [`ValueId`]. The store is never left half-open: payload and node
    /// are committed together when `record` returns.
    ///
    /// Callers use disjoint field borrows so `record` may mutate the
    /// structure column while this store is mutably borrowed.
    pub(crate) fn install(
        &mut self,
        data: Data,
        record: impl FnOnce(SlotId) -> ValueId,
    ) -> ValueId {
        debug_assert_eq!(self.payloads.len(), self.nodes.len());
        let slot = SlotId::new(self.payloads.len());
        self.payloads.push(data);
        let node = record(slot);
        self.nodes.push(node);
        node
    }

    /// Replaces the payload at `slot`.
    ///
    /// Used when a run overlays fed input values onto a clone of the
    /// default store.
    ///
    /// # Panics
    /// Panics if `slot` is out of range.
    pub(crate) fn set(&mut self, slot: SlotId, data: Data) {
        self.payloads[slot.index()] = data;
    }

    /// Returns the node behind the highest slot, or `None` for an
    /// empty store.
    pub(crate) fn last_node(&self) -> Option<ValueId> {
        self.nodes.last().copied()
    }

    /// Returns the slot whose structure node is `node`, or `None` if
    /// no slot links to it.
    ///
    /// Slots are installed in recording order, so the node column is
    /// strictly increasing and the lookup is a binary search.
    pub(crate) fn slot_of(&self, node: ValueId) -> Option<SlotId> {
        self.nodes
            .binary_search_by_key(&node.index(), |linked| linked.index())
            .ok()
            .map(SlotId::new)
    }

    /// Returns a store with the same node links and `payloads` replaced.
    ///
    /// The training-step transition for parameters: structure and slot
    /// identity stay fixed; only the tensors turn over.
    ///
    /// # Panics
    /// Panics if `payloads` does not match the number of slots.
    pub(crate) fn with_payloads(&self, payloads: Vec<Data>) -> Self {
        assert_eq!(
            payloads.len(),
            self.nodes.len(),
            "payload count must match the store's slots"
        );
        Self {
            payloads,
            nodes: self.nodes.clone(),
        }
    }

    /// Iterates `(node, payload)` in slot order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (ValueId, &Data)> {
        debug_assert_eq!(self.payloads.len(), self.nodes.len());
        self.nodes.iter().copied().zip(self.payloads.iter())
    }
}
