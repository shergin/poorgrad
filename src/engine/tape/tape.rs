use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use cow_vec::CowVec;

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::engine::{Function, ValueId};
use crate::{Differentiable, Shape};

use super::{Branch, Lineage, Operands, ParameterStore, Segment, SlotId, Tip, chains_agree};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`. The tape is the root every other guarantee rests on.
assert_impl_all!(Tape<f64>: Send, Sync);

/// An atomically taken snapshot of one tape: the recorded nodes with
/// their operand links, the generation's parameter payloads, input
/// defaults, and branch chain.
///
/// All parts share their backing storage with the tape, so taking
/// a snapshot is O(1); replaying it never requires the tape lock.
#[derive(Debug)]
pub(crate) struct Snapshot<Data> {
    pub(crate) functions: CowVec<Function<Data>>,
    pub(crate) operands: CowVec<Operands>,
    pub(crate) parameters: Arc<ParameterStore<Data>>,
    pub(crate) inputs: Arc<Vec<Data>>,
    pub(crate) chain: Arc<Vec<Segment>>,
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
    /// Secures the right to record at the current tip before a push.
    ///
    /// An owned tip records freely. A contended tip races its siblings
    /// on the shared token: the winner continues the tip branch, a
    /// loser mints a fresh branch starting at its own length. Either
    /// way this tape owns its tip afterwards. `AcqRel` documents the
    /// token as a synchronization point between sibling tapes; the data
    /// it guards is only the branch continuation decision.
    fn claim_tip(&mut self) {
        let Tip::Contended(token) = &self.tip else {
            return;
        };
        let won = token
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
        if !won {
            Arc::make_mut(&mut self.chain).push(Segment {
                branch: Branch::new(),
                start: self.functions.len(),
            });
        }
        self.tip = Tip::Owned;
    }

    /// Prepares the tip for duplication and returns the copy's tip.
    ///
    /// Both sides must re-win the right to extend the tip branch, so an
    /// owned tip becomes contended on a fresh token shared with the
    /// copy. An already contended tip hands the copy the same token:
    /// every tape sharing an unextended tip contends on one token, so
    /// exactly one of them ever continues the branch.
    fn share_tip(&mut self) -> Tip {
        match &self.tip {
            Tip::Contended(token) => Tip::Contended(Arc::clone(token)),
            Tip::Owned => {
                let token = Arc::new(AtomicBool::new(false));
                self.tip = Tip::Contended(Arc::clone(&token));
                Tip::Contended(token)
            }
        }
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

    /// Returns the index range `branch` owns on this tape, or `None` if
    /// the branch does not appear in this tape's chain.
    pub(crate) fn segment_range(&self, branch: Branch) -> Option<Range<usize>> {
        let inner = self.lock();
        let chain = inner.chain.as_slice();
        chain
            .iter()
            .position(|segment| segment.branch == branch)
            .map(|position| {
                let end = chain
                    .get(position + 1)
                    .map_or(inner.functions.len(), |next| next.start);
                chain[position].start..end
            })
    }

    /// Returns whether this tape attributes `[0, length)` to the same
    /// branches as `chain`.
    pub(crate) fn agrees_with_chain(&self, chain: &Arc<Vec<Segment>>, length: usize) -> bool {
        chains_agree(&self.lock().chain, chain, length)
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
    /// parameter payloads, and the branch chain, taken atomically under
    /// one lock section.
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
            chain: Arc::clone(&inner.chain),
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
        let tip = inner.share_tip();
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
    pub(crate) fn update(&self, gradients: &[Data], rule: impl Fn(&Data, &Data) -> Data) -> Self {
        let (functions, operands, shapes, parameters, inputs, chain, tip) = {
            let mut inner = self.lock();
            let tip = inner.share_tip();
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
            let next = rule(payload, &gradients[node.index()]);
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
