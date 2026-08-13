use std::sync::{Arc, Mutex, MutexGuard};

use cow_vec::CowVec;

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::engine::{Function, Symbol, ValueId};
use crate::{Differentiable, Shape};

use super::{
    Branch, Kinship, Lineage, Misbinding, Operands, ParameterStore, Segment, SlotId, Tip,
    chain_probe, chains_agree,
};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`. The tape is the root every other guarantee rests on.
assert_impl_all!(Tape<f64>: Send, Sync);

/// An atomically taken snapshot of one tape: the recorded nodes with
/// their operand links, the generation's parameter payloads, input
/// defaults, and kinship witness.
///
/// All parts share their backing storage with the tape, so taking
/// a snapshot is O(1); replaying it never requires the tape lock.
#[derive(Debug)]
pub(crate) struct Snapshot<Data> {
    pub(crate) functions: CowVec<Function<Data>>,
    pub(crate) operands: CowVec<Operands>,
    pub(crate) parameters: Arc<ParameterStore<Data>>,
    pub(crate) inputs: Arc<Vec<Data>>,
    pub(crate) kinship: Kinship,
}

/// The tape's position-indexed columns and the generation's parameter
/// store, guarded together by one lock.
///
/// The layout is data-oriented: functions and operands are the hot
/// columns replayed by every run — what each node computes and which
/// earlier nodes it reads — while shapes are the cold column read at
/// record time. All columns are lineage-invariant — `update` replaces
/// the parameter store but never touches columns, so one set of columns
/// serves every generation of a family. The columns always have equal
/// lengths.
#[derive(Debug)]
struct TapeInner<Data> {
    functions: CowVec<Function<Data>>,
    operands: CowVec<Operands>,
    shapes: CowVec<Shape>,
    parameters: Arc<ParameterStore<Data>>,
    // The default payloads of declared inputs, indexed by slot. Kept
    // separate from the parameter store because the lifecycles differ:
    // parameters turn over per generation, inputs per run.
    inputs: Arc<Vec<Data>>,
    chain: Arc<Vec<Segment>>,
    tip: Tip,
}

impl<Data> TapeInner<Data> {
    /// Secures the right to record at the current tip before a push,
    /// wiring this tape's chain and length into [`Tip::claim`].
    fn claim_tip(&mut self) {
        let length = self.functions.len();
        self.tip.claim(&mut self.chain, length);
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
    lineage: Lineage,
}

impl<Data: Differentiable> Tape<Data> {
    /// Creates an empty `Tape`.
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(TapeInner {
                functions: CowVec::new(),
                operands: CowVec::new(),
                shapes: CowVec::new(),
                parameters: Arc::new(ParameterStore::new()),
                inputs: Arc::new(Vec::new()),
                chain: Arc::new(vec![Segment {
                    branch: Branch::new(),
                    start: 0,
                }]),
                tip: Tip::Owned,
            }),
            lineage: Lineage::new(),
        }
    }

    /// Returns the token of the lineage this tape belongs to.
    pub(crate) fn lineage(&self) -> Lineage {
        self.lineage
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
                operand.index() < inner.functions.len(),
                "operand is out of bounds for its tape"
            );
        }
        let shape = {
            let shapes = &inner.shapes;
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
        inner.functions.push(function);
        inner.operands.push(Operands::from_slice(operands));
        inner.shapes.push(shape);
        debug_assert_eq!(inner.functions.len(), inner.operands.len());
        debug_assert_eq!(inner.functions.len(), inner.shapes.len());
        ValueId(inner.functions.len() - 1)
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
        let store = Arc::make_mut(&mut inner.parameters);
        let slot = SlotId::new(store.payloads.len());
        inner.functions.push(Function::parameter(slot));
        inner.operands.push(Operands::none());
        inner.shapes.push(shape);
        debug_assert_eq!(inner.functions.len(), inner.operands.len());
        debug_assert_eq!(inner.functions.len(), inner.shapes.len());
        let id = ValueId(inner.functions.len() - 1);
        store.payloads.push(data);
        store.nodes.push(id);
        id
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
        let defaults = Arc::make_mut(&mut inner.inputs);
        let slot = SlotId::new(defaults.len());
        inner.functions.push(Function::input(slot));
        inner.operands.push(Operands::none());
        inner.shapes.push(shape);
        debug_assert_eq!(inner.functions.len(), inner.operands.len());
        debug_assert_eq!(inner.functions.len(), inner.shapes.len());
        defaults.push(initial);
        ValueId(inner.functions.len() - 1)
    }

    /// Returns the input slot behind `id`, or `None` if the node is not
    /// an input.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    pub(crate) fn input_slot(&self, id: ValueId) -> Option<SlotId> {
        let inner = self.lock();
        match inner
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
            id.index() < inner.functions.len(),
            "`ValueId` is out of bounds for its tape"
        );
        inner
            .chain
            .iter()
            .rev()
            .find(|segment| segment.start <= id.index())
            .expect("the root segment starts at zero")
            .branch
    }

    /// Returns whether `kinship` names this tape's family.
    pub(crate) fn is_family(&self, kinship: &Kinship) -> bool {
        kinship.lineage() == self.lineage
    }

    /// Returns whether `kinship` belongs to this tape's family and
    /// attributes `[0, length)` to the same branches: the tape-side
    /// twin of [`Kinship::agrees_with`], answered against the live
    /// chain under the lock.
    pub(crate) fn agrees_with(&self, kinship: &Kinship, length: usize) -> bool {
        self.is_family(kinship) && chains_agree(&self.lock().chain, kinship.chain(), length)
    }

    /// Probes for the node `symbol` names on this tape without
    /// panicking: the resolution behind
    /// [`Network::try_resolve`](crate::Network::try_resolve).
    pub(crate) fn probe(&self, symbol: Symbol) -> Result<ValueId, Misbinding> {
        let inner = self.lock();
        chain_probe(self.lineage, &inner.chain, inner.functions.len(), symbol)
    }

    /// Locates `symbol` on this tape: the resolution behind
    /// [`Network::resolve`](crate::Network::resolve) and the named
    /// form of [`ValueRef`](crate::ValueRef) reads, with one set of
    /// checks and panic messages for both.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different lineage or a
    /// divergent fork, or no value with that name is allocated here.
    pub(crate) fn locate(&self, symbol: Symbol) -> ValueId {
        match self.probe(symbol) {
            Ok(id) => id,
            Err(Misbinding::ForeignLineage) => {
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
            .functions
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape");
        match function {
            Function::Leaf(leaf) => Some(leaf.0.clone()),
            Function::Parameter(parameter) => {
                Some(inner.parameters.payloads[parameter.0.index()].clone())
            }
            Function::Input(input) => Some(inner.inputs[input.0.index()].clone()),
            _ => None,
        }
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

    /// Returns a clone of the operand links recorded for `id`.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    #[cfg(test)]
    pub(crate) fn operands_of(&self, id: ValueId) -> Operands {
        self.lock()
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
            .functions
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape");
        reader(function)
    }

    /// Returns an O(1) snapshot of the recorded nodes, the current
    /// parameter payloads, and the kinship witness, taken atomically
    /// under one lock section.
    ///
    /// The snapshot shares the underlying arena and store but is
    /// isolated from later recordings and updates, so it can be replayed
    /// without holding the tape lock.
    pub(crate) fn snapshot(&self) -> Snapshot<Data> {
        let inner = self.lock();
        Snapshot {
            functions: inner.functions.clone(),
            operands: inner.operands.clone(),
            parameters: Arc::clone(&inner.parameters),
            inputs: Arc::clone(&inner.inputs),
            kinship: Kinship::new(self.lineage, Arc::clone(&inner.chain)),
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
        let tip = inner.tip.share();
        Self {
            inner: Mutex::new(TapeInner {
                functions: inner.functions.clone(),
                operands: inner.operands.clone(),
                shapes: inner.shapes.clone(),
                parameters: Arc::clone(&inner.parameters),
                inputs: Arc::clone(&inner.inputs),
                chain: Arc::clone(&inner.chain),
                tip,
            }),
            lineage: self.lineage,
        }
    }

    /// Returns a lineage-compatible tape whose column arenas hold only
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
    /// same lineage, same first-writer branch rule.
    ///
    /// Cost is O(live nodes) — cloning each live column entry into the
    /// new arenas — not O(1). Prefer [`Tape::fork`] for train-only
    /// what-ifs that never record after the clone.
    pub(crate) fn compacted(&self) -> Self {
        let mut inner = self.lock();
        let tip = inner.tip.share();
        // `From<Vec<_>>` builds a private arena sized to the live
        // column; `CowVec::clone` would keep sharing any polluted one.
        let functions: CowVec<Function<Data>> = inner.functions.to_vec().into();
        let operands: CowVec<Operands> = inner.operands.to_vec().into();
        let shapes: CowVec<Shape> = inner.shapes.to_vec().into();
        Self {
            inner: Mutex::new(TapeInner {
                functions,
                operands,
                shapes,
                parameters: Arc::clone(&inner.parameters),
                inputs: Arc::clone(&inner.inputs),
                chain: Arc::clone(&inner.chain),
                tip,
            }),
            lineage: self.lineage,
        }
    }

    /// Returns a new tape with every parameter's payload replaced by
    /// `rule(current, gradient)`.
    ///
    /// The new tape shares the function and shape columns untouched and
    /// builds a fresh parameter store: a gradient step rewrites every
    /// slot, so both the work and the allocations are O(parameters), and
    /// the previous store is reclaimed when its generation drops.
    /// Positions are preserved, so symbols keep resolving across the
    /// transition.
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
        let (functions, operands, shapes, parameters, inputs, chain, tip) = {
            let mut inner = self.lock();
            let tip = inner.tip.share();
            (
                inner.functions.clone(),
                inner.operands.clone(),
                inner.shapes.clone(),
                Arc::clone(&inner.parameters),
                Arc::clone(&inner.inputs),
                Arc::clone(&inner.chain),
                tip,
            )
        };
        assert_eq!(
            functions.len(),
            gradients.len(),
            "field is stale: the network has grown since it was produced"
        );
        let mut payloads = Vec::with_capacity(parameters.payloads.len());
        for (payload, &node) in parameters.payloads.iter().zip(&parameters.nodes) {
            let next = rule(node, payload, &gradients[node.index()]);
            let declared = shapes.get(node.index()).expect("shapes cover the tape");
            assert_eq!(
                &next.shape(),
                declared,
                "update must preserve the parameter's shape"
            );
            payloads.push(next);
        }
        Self {
            inner: Mutex::new(TapeInner {
                functions,
                operands,
                shapes,
                parameters: Arc::new(ParameterStore {
                    payloads,
                    nodes: parameters.nodes.clone(),
                }),
                inputs,
                chain,
                tip,
            }),
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
