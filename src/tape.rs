use std::sync::{Arc, Mutex, MutexGuard};

use cow_vec::CowVec;

use static_assertions::assert_impl_all;

use super::{Differentiable, Function, ValueId};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`. The tape is the root every other guarantee rests on.
assert_impl_all!(Tape<f64>: Send, Sync);

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`. The token is what lets fields cross threads detached
// from any network, so its thread-safety is load-bearing on its own.
assert_impl_all!(Lineage: Send, Sync);

/// An opaque token identifying a family of related tapes.
///
/// Every fork and update clones the token, so two tapes share a lineage
/// exactly when they descend from a common origin; kinship is pointer
/// identity of the token. Positions are stable within a lineage, which is
/// what lets symbols resolve and fields combine across generations.
#[derive(Debug, Clone)]
pub(crate) struct Lineage(Arc<()>);

impl Lineage {
    fn new() -> Self {
        Self(Arc::new(()))
    }

    /// Returns `true` if `self` and `other` identify the same lineage.
    pub(crate) fn is_same(&self, other: &Lineage) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// The shared, append-only record of every node of one computation graph.
///
/// It is the engine's take on the classic autograd tape (a Wengert list):
/// expressions record `Function` nodes onto it as they are built, and
/// `Network::forward`/`backward` replay it in allocation order. One tape is
/// shared by a `Network` and all of its `Value` proxies, and it is the only
/// synchronization point in the crate: a single `Mutex` guards the `CowVec`
/// of nodes and is taken briefly to record or read. Recorded nodes are
/// never mutated or removed, so a `ValueId` stays valid for the tape's
/// lifetime, while `fork` takes an O(1) copy-on-write snapshot that is
/// isolated from later recordings.
#[derive(Debug)]
pub(crate) struct Tape<Data> {
    nodes: Mutex<CowVec<Function<Data>>>,
    lineage: Lineage,
}

impl<Data: Differentiable> Tape<Data> {
    /// Creates an empty `Tape`.
    pub(crate) fn new() -> Self {
        Self {
            nodes: Mutex::new(CowVec::new()),
            lineage: Lineage::new(),
        }
    }

    /// Returns the token of the lineage this tape belongs to.
    pub(crate) fn lineage(&self) -> &Lineage {
        &self.lineage
    }

    /// Records `function` and returns its handle.
    ///
    /// # Panics
    /// Panics if `function` references an operand that is not recorded on
    /// this tape.
    pub(crate) fn record(&self, function: Function<Data>) -> ValueId {
        let mut nodes = self.lock();
        function.visit_operands(|operand| {
            assert!(
                operand.index() < nodes.len(),
                "operand is out of bounds for its tape"
            );
        });
        nodes.push(function);
        ValueId(nodes.len() - 1)
    }

    /// Runs `reader` over the node behind `id` while holding the tape lock.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    pub(crate) fn with_node<Output>(
        &self,
        id: ValueId,
        reader: impl FnOnce(&Function<Data>) -> Output,
    ) -> Output {
        let nodes = self.lock();
        let function = nodes
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape");
        reader(function)
    }

    /// Returns an O(1) copy-on-write snapshot of the recorded nodes.
    ///
    /// The snapshot shares the underlying arena but is isolated from later
    /// recordings, so it can be replayed without holding the tape lock.
    pub(crate) fn snapshot(&self) -> CowVec<Function<Data>> {
        self.lock().clone()
    }

    /// Creates an independent copy of the tape in O(1).
    ///
    /// The copy shares the underlying arena but keeps its own node list, so
    /// later recordings on either tape never affect the other.
    pub(crate) fn fork(&self) -> Self {
        Self {
            nodes: Mutex::new(self.snapshot()),
            lineage: self.lineage.clone(),
        }
    }

    /// Returns a new tape with every parameter's payload replaced by
    /// `update(current, gradient)`.
    ///
    /// The new tape shares every node except the parameters, which are
    /// re-recorded in the shared arena; positions are preserved, so
    /// symbols keep resolving across the transition.
    ///
    /// # Panics
    /// Panics if `gradients` does not cover the whole tape.
    pub(crate) fn updated(
        &self,
        gradients: &[Data],
        update: impl Fn(&Data, &Data) -> Data,
    ) -> Self {
        let mut nodes = self.snapshot();
        assert_eq!(
            nodes.len(),
            gradients.len(),
            "field is stale: the network has grown since it was produced"
        );
        for index in 0..nodes.len() {
            let payload = match nodes
                .get(index)
                .expect("index is in bounds")
                .parameter_data()
            {
                Some(current) => update(current, &gradients[index]),
                None => continue,
            };
            nodes.set(index, Function::parameter(payload));
        }
        Self {
            nodes: Mutex::new(nodes),
            lineage: self.lineage.clone(),
        }
    }

    /// Returns the number of recorded nodes.
    pub(crate) fn len(&self) -> usize {
        self.lock().len()
    }

    fn lock(&self) -> MutexGuard<'_, CowVec<Function<Data>>> {
        self.nodes.lock().expect("tape lock is poisoned")
    }
}
