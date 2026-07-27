use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use cow_vec::CowVec;

use static_assertions::assert_impl_all;

use super::{Differentiable, Function, Shape, ValueId};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`. The tape is the root every other guarantee rests on.
assert_impl_all!(Tape<f64>: Send, Sync);

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`. The token is what lets symbols and fields cross
// threads detached from any network, so its thread-safety and `Copy` are
// load-bearing on their own.
assert_impl_all!(Lineage: Send, Sync, Copy);

/// An opaque token identifying a family of related tapes.
///
/// Every tape mints its identity from a process-global counter at
/// creation, and forks and updates carry it forward, so two tapes share a
/// lineage exactly when they descend from a common origin; kinship is
/// plain equality. Being a `Copy` integer rather than a reference-counted
/// token, it rides inside every `Symbol` without costing `Copy`, and
/// creating fields and evaluations never touches an atomic counter.
/// Positions are stable within a lineage, which is what lets symbols
/// resolve and fields combine across generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Lineage(u64);

impl Lineage {
    /// Mints a fresh lineage identity.
    ///
    /// `Relaxed` suffices: only uniqueness matters, and the identity
    /// reaches other threads through the tape it identifies.
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// The tape's position-indexed columns, guarded together by one lock.
///
/// The layout is data-oriented: functions are the hot column replayed by
/// every run, shapes the cold column read at record time. Shapes are
/// lineage-invariant — `updated` replaces functions but never touches
/// shapes, so one shape column serves every generation of a family. The
/// columns always have equal lengths.
#[derive(Debug)]
struct TapeInner<Data> {
    functions: CowVec<Function<Data>>,
    shapes: CowVec<Shape>,
}

/// The shared, append-only record of every node of one computation graph.
///
/// It is the engine's take on the classic autograd tape (a Wengert list):
/// expressions record `Function` nodes onto it as they are built — each
/// with its `Shape`, inferred and validated at record time — and
/// `Network::forward`/`backward` replay it in allocation order. One tape is
/// shared by a `Network` and all of its `Value` proxies, and it is the only
/// synchronization point in the crate: a single `Mutex` guards the columns
/// and is taken briefly to record or read. Recorded nodes are never
/// mutated or removed, so a `ValueId` stays valid for the tape's lifetime,
/// while `fork` takes an O(1) copy-on-write snapshot that is isolated from
/// later recordings.
#[derive(Debug)]
pub(crate) struct Tape<Data> {
    inner: Mutex<TapeInner<Data>>,
    lineage: Lineage,
}

impl<Data: Differentiable> Tape<Data> {
    /// Creates an empty `Tape`.
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(TapeInner {
                functions: CowVec::new(),
                shapes: CowVec::new(),
            }),
            lineage: Lineage::new(),
        }
    }

    /// Returns the token of the lineage this tape belongs to.
    pub(crate) fn lineage(&self) -> Lineage {
        self.lineage
    }

    /// Records `function` and returns its handle.
    ///
    /// It infers and stores the result's shape on the way in, so shape
    /// mismatches panic at the expression that records them, before
    /// anything runs.
    ///
    /// # Panics
    /// Panics if `function` references an operand that is not recorded on
    /// this tape, or if the operands' shapes are incompatible.
    pub(crate) fn record(&self, function: Function<Data>) -> ValueId {
        let mut inner = self.lock();
        function.visit_operands(|operand| {
            assert!(
                operand.index() < inner.functions.len(),
                "operand is out of bounds for its tape"
            );
        });
        let shape = {
            let shapes = &inner.shapes;
            function.inferred_shape(|id| {
                shapes
                    .get(id.index())
                    .expect("operand shape is recorded")
                    .clone()
            })
        };
        inner.functions.push(function);
        inner.shapes.push(shape);
        debug_assert_eq!(inner.functions.len(), inner.shapes.len());
        ValueId(inner.functions.len() - 1)
    }

    /// Returns the shape inferred for `id` when it was recorded.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    pub(crate) fn shape(&self, id: ValueId) -> Shape {
        self.lock()
            .shapes
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape")
            .clone()
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
        let inner = self.lock();
        let function = inner
            .functions
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape");
        reader(function)
    }

    /// Returns an O(1) copy-on-write snapshot of the recorded nodes.
    ///
    /// The snapshot shares the underlying arena but is isolated from later
    /// recordings, so it can be replayed without holding the tape lock.
    pub(crate) fn snapshot(&self) -> CowVec<Function<Data>> {
        self.lock().functions.clone()
    }

    /// Creates an independent copy of the tape in O(1).
    ///
    /// The copy shares the underlying arena but keeps its own node list, so
    /// later recordings on either tape never affect the other.
    pub(crate) fn fork(&self) -> Self {
        let inner = self.lock();
        Self {
            inner: Mutex::new(TapeInner {
                functions: inner.functions.clone(),
                shapes: inner.shapes.clone(),
            }),
            lineage: self.lineage,
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
    /// Panics if `gradients` does not cover the whole tape, or if
    /// `update` returns a payload whose shape differs from the
    /// parameter's recorded shape.
    pub(crate) fn updated(
        &self,
        gradients: &[Data],
        update: impl Fn(&Data, &Data) -> Data,
    ) -> Self {
        let (mut functions, shapes) = {
            let inner = self.lock();
            (inner.functions.clone(), inner.shapes.clone())
        };
        assert_eq!(
            functions.len(),
            gradients.len(),
            "field is stale: the network has grown since it was produced"
        );
        for (index, gradient) in gradients.iter().enumerate() {
            let payload = match functions
                .get(index)
                .expect("index is in bounds")
                .parameter_data()
            {
                Some(current) => update(current, gradient),
                None => continue,
            };
            let declared = shapes.get(index).expect("shapes cover the tape");
            assert_eq!(
                &payload.shape(),
                declared,
                "update must preserve the parameter's shape"
            );
            functions.set(index, Function::parameter(payload));
        }
        // Shapes are lineage-invariant: the update replaces payloads, never
        // shapes, so the shape column is shared as is.
        Self {
            inner: Mutex::new(TapeInner { functions, shapes }),
            lineage: self.lineage,
        }
    }

    /// Returns the number of recorded nodes.
    pub(crate) fn len(&self) -> usize {
        self.lock().functions.len()
    }

    /// Locks the tape's columns.
    ///
    /// A poisoned lock stays fatal on purpose: it means a recording
    /// panicked on this tape earlier, the panic was caught, and the
    /// program kept going — a state this crate's panics-mean-bugs
    /// contract does not support. The message names that cause so the
    /// debugging trail leads to the original panic.
    fn lock(&self) -> MutexGuard<'_, TapeInner<Data>> {
        self.inner
            .lock()
            .expect("tape is poisoned: a recording panicked earlier on this network")
    }
}
