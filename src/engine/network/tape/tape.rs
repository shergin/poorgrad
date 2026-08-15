use std::sync::{Arc, Mutex, MutexGuard};

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::engine::{Function, Symbol, ValueId};
use crate::{Differentiable, Shape};

use super::{
    Branch, Identity, Misbinding, Operands, SlotId, SlotStore, Structure, TapeSnapshot, Witness,
};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`. The tape is the root every other guarantee rests on.
assert_impl_all!(Tape<f64>: Send, Sync);

/// The tape's structure, generation payloads, and live identity, guarded
/// together by one lock.
///
/// Structure is origin-invariant — `update` replaces the parameter
/// [`SlotStore`] but never touches columns, so one set of columns serves
/// every generation of a family. Parameters and inputs share the same
/// store type; they stay separate fields because their lifecycles differ
/// (generation vs run). Structural identity (origin, branch chain, tip)
/// lives in [`Identity`].
#[derive(Debug)]
struct TapeInner<Data> {
    structure: Structure<Data>,
    parameters: Arc<SlotStore<Data>>,
    inputs: Arc<SlotStore<Data>>,
    identity: Identity,
}

impl<Data> TapeInner<Data> {
    /// Secures the right to record at the current tip before a push.
    fn claim_tip(&mut self) {
        let length = self.structure.len();
        self.identity.claim(length);
    }
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
}

impl<Data: Differentiable> Tape<Data> {
    /// Creates an empty `Tape`.
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(TapeInner {
                structure: Structure::new(),
                parameters: Arc::new(SlotStore::new()),
                inputs: Arc::new(SlotStore::new()),
                identity: Identity::new(),
            }),
        }
    }

    /// Returns the origin token of this tape's family.
    pub(crate) fn origin(&self) -> super::Origin {
        self.lock().identity.origin()
    }

    /// Records `function` with its positional `operands` and returns its
    /// handle.
    ///
    /// It infers and stores the result's shape on the way in, so shape
    /// mismatches panic at the expression that records them, before
    /// anything runs.
    ///
    /// # Panics
    /// Panics if `operands` does not match the function's arity or
    /// references a node that is not recorded on this tape, or if the
    /// operands' shapes are incompatible.
    pub(crate) fn record(&self, function: Function<Data>, operands: &[ValueId]) -> ValueId {
        assert_eq!(
            operands.len(),
            function.arity(),
            "operand count must match the operation's arity"
        );
        let mut inner = self.lock();
        for operand in operands {
            assert!(
                operand.index() < inner.structure.len(),
                "operand is out of bounds for its tape"
            );
        }
        let shape = {
            let shapes = &inner.structure.shapes;
            let operand_shapes: SmallVec<[Shape; 2]> = operands
                .iter()
                .map(|operand| {
                    shapes
                        .get(operand.index())
                        .expect("operand shape is recorded")
                        .clone()
                })
                .collect();
            function.infer_shape(&operand_shapes)
        };
        inner.claim_tip();
        inner
            .structure
            .push(function, Operands::from_slice(operands), shape)
    }

    /// Records a parameter node and stores its payload, returning the
    /// node's handle.
    ///
    /// One lock section keeps the slot and the node consistent under
    /// concurrent recording. If the store is shared with a fork, the
    /// first post-fork parameter allocation copies it (O(parameters),
    /// once) so the branches stay independent.
    pub(crate) fn record_parameter(&self, data: Data) -> ValueId {
        let shape = data.shape();
        let mut inner = self.lock();
        inner.claim_tip();
        let inner = &mut *inner;
        // Disjoint fields: the store borrow and the structure push in
        // `install`'s closure are simultaneous without conflict.
        let structure = &mut inner.structure;
        Arc::make_mut(&mut inner.parameters).install(data, |slot| {
            structure.push(Function::parameter(slot), Operands::none(), shape)
        })
    }

    /// Records an input node and stores its default payload, returning
    /// the node's handle.
    ///
    /// One lock section keeps the slot and the node consistent under
    /// concurrent recording, exactly like `record_parameter`.
    pub(crate) fn record_input(&self, initial: Data) -> ValueId {
        let shape = initial.shape();
        let mut inner = self.lock();
        inner.claim_tip();
        let inner = &mut *inner;
        let structure = &mut inner.structure;
        Arc::make_mut(&mut inner.inputs).install(initial, |slot| {
            structure.push(Function::input(slot), Operands::none(), shape)
        })
    }

    /// Returns the input slot behind `id`, or `None` if the node is not
    /// an input.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    pub(crate) fn input_slot(&self, id: ValueId) -> Option<SlotId> {
        let inner = self.lock();
        match inner
            .structure
            .functions
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape")
        {
            Function::Input(input) => Some(input.0),
            _ => None,
        }
    }

    /// Returns the branch that owns position `id` on this tape.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    pub(crate) fn branch_of(&self, id: ValueId) -> Branch {
        let inner = self.lock();
        assert!(
            id.index() < inner.structure.len(),
            "`ValueId` is out of bounds for its tape"
        );
        inner.identity.branch_of(id.index())
    }

    /// Returns whether `witness` names this tape's family.
    pub(crate) fn same_origin(&self, witness: &Witness) -> bool {
        self.lock().identity.same_origin(witness)
    }

    /// Returns whether `witness` belongs to this tape's family and
    /// attributes `[0, length)` to the same branches: the tape-side
    /// twin of [`Witness::agrees_with`], answered against the live
    /// chain under the lock.
    pub(crate) fn agrees_with(&self, witness: &Witness, length: usize) -> bool {
        self.lock().identity.agrees_with(witness, length)
    }

    /// Probes for the node `symbol` names on this tape without
    /// panicking: the resolution behind
    /// [`Network::try_resolve`](crate::Network::try_resolve).
    pub(crate) fn probe(&self, symbol: Symbol) -> Result<ValueId, Misbinding> {
        let inner = self.lock();
        inner.identity.probe(symbol, inner.structure.len())
    }

    /// Locates `symbol` on this tape: the resolution behind
    /// [`Network::resolve`](crate::Network::resolve) and the named
    /// form of [`ValueRef`](crate::ValueRef) reads, with one set of
    /// checks and panic messages for both.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network lineage or a
    /// divergent fork, or no value with that name is allocated here.
    pub(crate) fn locate(&self, symbol: Symbol) -> ValueId {
        match self.probe(symbol) {
            Ok(id) => id,
            Err(Misbinding::ForeignOrigin) => {
                panic!("symbol belongs to a different network lineage")
            }
            Err(Misbinding::DivergentBranch) => {
                panic!("symbol belongs to a divergent fork of this network")
            }
            Err(Misbinding::OutOfCoverage) => panic!("symbol is not allocated in this network"),
        }
    }

    /// Returns a clone of the payload behind `id`: a leaf's embedded
    /// payload or a parameter's current store entry, or `None` for
    /// computed values.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    pub(crate) fn payload_of(&self, id: ValueId) -> Option<Data> {
        let inner = self.lock();
        let function = inner
            .structure
            .functions
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape");
        match function {
            Function::Leaf(leaf) => Some(leaf.0.clone()),
            Function::Parameter(parameter) => {
                Some(inner.parameters.payloads()[parameter.0.index()].clone())
            }
            Function::Input(input) => Some(inner.inputs.payloads()[input.0.index()].clone()),
            _ => None,
        }
    }

    /// Returns the shape inferred for `id` when it was recorded.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    pub(crate) fn shape(&self, id: ValueId) -> Shape {
        self.lock()
            .structure
            .shapes
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape")
            .clone()
    }

    /// Returns a clone of the operand links recorded for `id`.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    #[cfg(test)]
    pub(crate) fn operands_of(&self, id: ValueId) -> Operands {
        self.lock()
            .structure
            .operands
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape")
            .clone()
    }

    /// Runs `reader` over the node behind `id` while holding the tape lock.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    #[cfg(test)]
    pub(crate) fn with_node<Output>(
        &self,
        id: ValueId,
        reader: impl FnOnce(&Function<Data>) -> Output,
    ) -> Output {
        let inner = self.lock();
        let function = inner
            .structure
            .functions
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape");
        reader(function)
    }

    /// Returns an O(1) freeze of the recorded structure, the current
    /// parameter payloads, and the identity witness, taken atomically
    /// under one lock section.
    ///
    /// The snapshot shares the underlying arena and store but is
    /// isolated from later recordings and updates, so it can be replayed
    /// without holding the tape lock.
    pub(crate) fn snapshot(&self) -> TapeSnapshot<Data> {
        let inner = self.lock();
        TapeSnapshot {
            structure: inner.structure.clone(),
            parameters: Arc::clone(&inner.parameters),
            inputs: Arc::clone(&inner.inputs),
            witness: inner.identity.witness(),
        }
    }

    /// Creates an independent copy of the tape in O(1).
    ///
    /// The copy shares the underlying arena, the parameter store, and
    /// the branch chain, so later recordings on either tape never affect
    /// the other: the first side to record continues the tip branch and
    /// the other mints its own, which is what keeps their symbols from
    /// misbinding after they diverge.
    pub(crate) fn fork(&self) -> Self {
        let mut inner = self.lock();
        let identity = inner.identity.share();
        Self {
            inner: Mutex::new(TapeInner {
                structure: inner.structure.clone(),
                parameters: Arc::clone(&inner.parameters),
                inputs: Arc::clone(&inner.inputs),
                identity,
            }),
        }
    }

    /// Returns an origin-compatible tape whose column arenas hold only
    /// this tape's live nodes.
    ///
    /// A plain [`Tape::fork`] shares the append-only column arena, so
    /// nodes recorded on a sibling fork stay allocated until every
    /// sharer of that arena drops. Compaction rebuilds the function,
    /// operand, and shape columns into fresh arenas from the live
    /// entries alone, so dropping the compacted tape (or replacing a
    /// polluted parent with it) can release sibling garbage the parent
    /// no longer reaches. Parameter and input stores, the branch chain,
    /// and tip contention match [`Tape::fork`]: same generation state,
    /// same origin, same first-writer branch rule.
    ///
    /// Cost is O(live nodes) — cloning each live column entry into the
    /// new arenas — not O(1). Prefer [`Tape::fork`] for train-only
    /// what-ifs that never record after the clone.
    pub(crate) fn compacted(&self) -> Self {
        let mut inner = self.lock();
        let identity = inner.identity.share();
        Self {
            inner: Mutex::new(TapeInner {
                structure: inner.structure.compacted(),
                parameters: Arc::clone(&inner.parameters),
                inputs: Arc::clone(&inner.inputs),
                identity,
            }),
        }
    }

    /// Returns a new tape with every parameter's payload replaced by
    /// `rule(current, gradient)`.
    ///
    /// The new tape shares the structure columns untouched and builds a
    /// fresh parameter store: a gradient step rewrites every slot, so
    /// both the work and the allocations are O(parameters), and the
    /// previous store is reclaimed when its generation drops. Positions
    /// are preserved, so symbols keep resolving across the transition.
    ///
    /// # Panics
    /// Panics if `gradients` does not cover the whole tape, or if
    /// `rule` returns a payload whose shape differs from the
    /// parameter's recorded shape.
    pub(crate) fn update(
        &self,
        gradients: &[Data],
        mut rule: impl FnMut(ValueId, &Data, &Data) -> Data,
    ) -> Self {
        let (structure, parameters, inputs, identity) = {
            let mut inner = self.lock();
            let identity = inner.identity.share();
            (
                inner.structure.clone(),
                Arc::clone(&inner.parameters),
                Arc::clone(&inner.inputs),
                identity,
            )
        };
        assert_eq!(
            structure.len(),
            gradients.len(),
            "field is stale: the network has grown since it was produced"
        );
        let mut payloads = Vec::with_capacity(parameters.len());
        for (node, payload) in parameters.iter() {
            let next = rule(node, payload, &gradients[node.index()]);
            let declared = structure
                .shapes
                .get(node.index())
                .expect("shapes cover the tape");
            assert_eq!(
                &next.shape(),
                declared,
                "update must preserve the parameter's shape"
            );
            payloads.push(next);
        }
        Self {
            inner: Mutex::new(TapeInner {
                structure,
                parameters: Arc::new(parameters.with_payloads(payloads)),
                inputs,
                identity,
            }),
        }
    }

    /// Returns the number of recorded nodes.
    pub(crate) fn len(&self) -> usize {
        self.lock().structure.len()
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
