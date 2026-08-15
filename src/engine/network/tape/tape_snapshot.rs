use std::sync::Arc;

use super::{SlotStore, Structure, Witness};

/// An atomically taken freeze of one tape: structure, generation
/// parameter payloads, input defaults, and identity witness.
///
/// All parts share their backing storage with the tape, so capture is
/// O(1); replaying never requires the tape lock. Detached from live
/// tip protocol — only a [`Witness`], not an [`super::Identity`].
#[derive(Debug)]
pub(crate) struct TapeSnapshot<Data> {
    pub(crate) structure: Structure<Data>,
    pub(crate) parameters: Arc<SlotStore<Data>>,
    pub(crate) inputs: Arc<SlotStore<Data>>,
    pub(crate) witness: Witness,
}
