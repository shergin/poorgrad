/// A lightweight handle to a payload's slot in a dense state store.
///
/// Slots are assigned in allocation order and never move: the
/// store-side mirror of `ValueId` used by parameters and inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SlotId(usize);

impl SlotId {
    /// Creates a slot for `index` in its store.
    pub(super) fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the position of the slot in its store.
    pub(crate) fn index(self) -> usize {
        self.0
    }
}
