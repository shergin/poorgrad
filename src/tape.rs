use std::sync::{Mutex, MutexGuard};

use cow_vec::CowVec;

use super::{Differentiable, ValueId, ValueInner};

/// The shared, append-only record of every node of one computation graph.
///
/// It is the engine's take on the classic autograd tape (a Wengert list):
/// expressions record nodes onto it as they are built, and
/// `Network::forward`/`backward` replay it in allocation order. One tape is
/// shared by a `Network` and all of its `Value` proxies, and it is the only
/// synchronization point in the crate: a single `Mutex` guards the `CowVec`
/// of nodes and is taken briefly to record or read. Recorded nodes are never
/// mutated or removed, so a `ValueId` stays valid for the tape's lifetime,
/// while `fork` takes an O(1) copy-on-write snapshot that is isolated from
/// later recordings.
#[derive(Debug)]
pub(crate) struct Tape<Data> {
    nodes: Mutex<CowVec<ValueInner<Data>>>,
}

impl<Data: Differentiable> Tape<Data> {
    /// Creates an empty `Tape`.
    pub(crate) fn new() -> Self {
        Self {
            nodes: Mutex::new(CowVec::new()),
        }
    }

    /// Records `inner` and returns its handle.
    pub(crate) fn record(&self, inner: ValueInner<Data>) -> ValueId {
        let mut nodes = self.lock();
        nodes.push(inner);
        ValueId(nodes.len() - 1)
    }

    /// Runs `reader` over the node behind `id` while holding the tape lock.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    pub(crate) fn with_node<Output>(
        &self,
        id: ValueId,
        reader: impl FnOnce(&ValueInner<Data>) -> Output,
    ) -> Output {
        let nodes = self.lock();
        let inner = nodes
            .get(id.0)
            .expect("`ValueId` is out of bounds for its tape");
        reader(inner)
    }

    /// Creates an independent copy of the tape in O(1).
    ///
    /// The copy shares the underlying arena but keeps its own node list, so
    /// later recordings on either tape never affect the other.
    pub(crate) fn fork(&self) -> Self {
        Self {
            nodes: Mutex::new(self.lock().clone()),
        }
    }

    /// Returns the number of recorded nodes.
    pub(crate) fn len(&self) -> usize {
        self.lock().len()
    }

    fn lock(&self) -> MutexGuard<'_, CowVec<ValueInner<Data>>> {
        self.nodes.lock().expect("tape lock is poisoned")
    }
}
