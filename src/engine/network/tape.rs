use std::sync::{Mutex, MutexGuard};

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::engine::{Function, Trace};
use crate::{Differentiable, Shape, Tensorial};

use super::{Network, Operands, Origin, SlotStore, Structure, Symbol, Value, ValueId};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`. The tape is the root every other guarantee rests on.
assert_impl_all!(Tape<f64>: Send, Sync);

/// The recorded columns and stores, guarded together by one lock.
///
/// Parameters and inputs share the same store type; they stay separate
/// fields because their roles differ: parameter initials seed
/// [`Parameters`](crate::Parameters), input defaults are spec that
/// feeds overlay per run.
#[derive(Debug)]
struct TapeInner<Data> {
    structure: Structure<Data>,
    initials: SlotStore<Data>,
    inputs: SlotStore<Data>,
}

/// The construction phase of a network: an append-only record of every
/// node of one computation graph.
///
/// It is the engine's take on the classic autograd tape (a Wengert
/// list): expressions record `Function` nodes onto it as they are built
/// — each with its `Shape`, inferred and validated at record time — so
/// invalid expressions panic at the expression that records them,
/// before anything runs. Recording happens through [`Value`] proxies
/// and their operators; the tape is the only synchronization point in
/// the crate, a single `Mutex` taken briefly per recording, quarantined
/// to this phase.
///
/// [`Tape::into_network`] consumes the tape and seals the recording
/// into an immutable [`Network`]; [`Network::into_tape`] consumes the
/// network to reopen it. Both conversions are consuming, so one
/// origin's history is linear by ownership: two divergent futures of
/// one recording cannot be constructed. Recorded nodes are never
/// mutated or removed, and linear extension never moves one, so a
/// [`Symbol`] stays valid across every round trip.
#[derive(Debug)]
pub struct Tape<Data> {
    origin: Origin,
    inner: Mutex<TapeInner<Data>>,
}

impl<Data: Differentiable> Tape<Data> {
    /// Creates an empty `Tape`.
    pub fn new() -> Self {
        Self {
            origin: Origin::new(),
            inner: Mutex::new(TapeInner {
                structure: Structure::new(),
                initials: SlotStore::new(),
                inputs: SlotStore::new(),
            }),
        }
    }

    /// Reopens `network` for further recording: the inverse of
    /// [`Tape::into_network`], with the same origin, so every existing
    /// [`Symbol`] keeps naming its node.
    pub(super) fn reopen(origin: Origin, network: Network<Data>) -> Self {
        let (structure, initials, inputs) = network.into_stores();
        Self {
            origin,
            inner: Mutex::new(TapeInner {
                structure,
                initials,
                inputs,
            }),
        }
    }

    /// Seals the recording into an immutable [`Network`], consuming the
    /// tape.
    ///
    /// The conversion is infallible and moves the recorded columns and
    /// stores without copying. Take [`Value::symbol`] for every value
    /// the later phases will name first: proxies borrow the tape, so
    /// the borrow checker rejects one outliving this call — the phase
    /// boundary is a compile error, not a runtime check.
    pub fn into_network(self) -> Network<Data> {
        let inner = self
            .inner
            .into_inner()
            .expect("tape is poisoned: a recording panicked earlier on this tape");
        Network::seal(self.origin, inner.structure, inner.initials, inner.inputs)
    }

    /// Returns the origin token of this tape's family.
    pub(crate) fn origin(&self) -> Origin {
        self.origin
    }

    /// Allocates a constant leaf and returns a proxy to it.
    ///
    /// Constants are fixed at recording time; see `parameter` for
    /// trainable leaves and `input` for leaves fed per run.
    pub fn leaf(&self, data: Data) -> Value<'_, Data> {
        let id = self.record(Function::leaf(data), &[]);
        Value::bind(self, id)
    }

    /// Allocates a learnable parameter and returns a proxy to it.
    ///
    /// `data` is the parameter's record-site initial: the payload a
    /// fresh [`Network::parameters`](crate::Network::parameters)
    /// starts from. Live payloads are caller-owned
    /// [`Parameters`](crate::Parameters) state; training never touches
    /// the recorded node.
    pub fn parameter(&self, data: Data) -> Value<'_, Data> {
        let shape = data.shape();
        let id = {
            let mut guard = self.lock();
            let inner = &mut *guard;
            // Disjoint fields: the store borrow and the structure push
            // in `install`'s closure are simultaneous without conflict.
            let structure = &mut inner.structure;
            inner.initials.install(data, |slot| {
                structure.push(Function::parameter(slot), Operands::none(), shape)
            })
        };
        Value::bind(self, id)
    }

    /// Allocates a declared per-run input and returns a proxy to it.
    ///
    /// `initial` supplies the input's recorded shape and its default
    /// payload — part of the spec, so a network with its defaults is
    /// runnable standalone; feeds overlay the defaults per run.
    pub fn input(&self, initial: Data) -> Value<'_, Data> {
        let shape = initial.shape();
        let id = {
            let mut guard = self.lock();
            let inner = &mut *guard;
            let structure = &mut inner.structure;
            inner.inputs.install(initial, |slot| {
                structure.push(Function::input(slot), Operands::none(), shape)
            })
        };
        Value::bind(self, id)
    }

    /// Resolves `symbol` back into a proxy on this tape: the reopen
    /// flow's bridge from the eternal name to the recording phase.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or is not
    /// allocated on this tape.
    pub fn resolve(&self, symbol: Symbol) -> Value<'_, Data> {
        assert!(
            symbol.origin == self.origin,
            "symbol belongs to a different network"
        );
        assert!(
            symbol.id.index() < self.len(),
            "symbol is not allocated on this tape"
        );
        Value::bind(self, symbol.id)
    }

    /// Resolves `symbol`, or returns `None` if it belongs to a
    /// different network or is not allocated here: the probing form of
    /// `resolve`.
    pub fn try_resolve(&self, symbol: Symbol) -> Option<Value<'_, Data>> {
        if symbol.origin != self.origin || symbol.id.index() >= self.len() {
            return None;
        }
        Some(Value::bind(self, symbol.id))
    }

    /// Returns the number of recorded nodes.
    pub fn len(&self) -> usize {
        self.lock().structure.len()
    }

    /// Returns `true` if it holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Records `function` with its positional `operands` and returns
    /// its handle.
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
        inner
            .structure
            .push(function, Operands::from_slice(operands), shape)
    }

    /// Returns a clone of the payload behind `id`: a leaf's embedded
    /// payload, a parameter's record-site initial, or an input's
    /// default, or `None` for computed values.
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
                Some(inner.initials.payloads()[parameter.0.index()].clone())
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

    /// Returns an O(1) freeze of the recorded columns, so a scan can
    /// read them without holding the lock while new nodes record.
    fn structure_freeze(&self) -> Structure<Data> {
        self.lock().structure.clone()
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
            .expect("tape is poisoned: a recording panicked earlier on this tape")
    }
}

impl<Data: Tensorial> Tape<Data> {
    /// Records the reverse-mode gradient of `loss` with respect to each
    /// `wrt` entry as ordinary computed nodes on this tape, and returns
    /// their symbols in `wrt` order.
    ///
    /// It is `backward` as a tape-to-tape transform: the same reverse
    /// scan the engine runs over payload buffers runs here over
    /// recording [`Trace`] handles, applying the very same derivative
    /// rules — so the recorded gradient and the engine's are one body
    /// of knowledge, and a compiled plan over `[loss, gradients...]`
    /// reproduces [`Run::backward`](crate::Run::backward) bitwise
    /// (same seed, same accumulation order). Gradients become
    /// first-class values: compilable, emittable, readable, and
    /// differentiable again for higher-order derivatives.
    ///
    /// A `wrt` value that is not an ancestor of the loss answers a
    /// recorded zero of its own shape, exactly as
    /// [`Gradients`](crate::Gradients) would. The transform reads
    /// graph structure only, never payloads; recording appends to the
    /// tape and leaves every existing node untouched.
    ///
    /// # Panics
    /// Panics if `loss` is not a recorded scalar (reduce with `sum`
    /// first) or any symbol belongs to a different network.
    pub fn differentiate(
        &self,
        loss: impl Into<Symbol>,
        wrt: impl IntoIterator<Item = impl Into<Symbol>>,
    ) -> Vec<Symbol> {
        let loss_value = self.resolve(loss.into());
        assert_eq!(
            loss_value.shape().rank(),
            0,
            "differentiate requires a scalar loss; reduce it with `sum` first"
        );
        let output_index = loss_value.id().index();
        let structure = self.structure_freeze();
        let trace = |index: usize| Trace::of(Value::bind(self, ValueId(index)));

        // The scan mirrors `Run::backward` deliberately and
        // exactly — the ones seed, the ancestor marking through `Some`
        // cotangents, the zero-seeded accumulation in reverse scan
        // order — because the bitwise parity contract welds the two:
        // any change to either scan's arithmetic must reach both.
        let mut cotangents: Vec<Option<Trace<'_, Data>>> = vec![None; output_index + 1];
        cotangents[output_index] = Some(Trace::of(
            loss_value.literal(Data::counted(loss_value.shape(), 1)),
        ));
        let mut ancestors = vec![false; output_index + 1];
        ancestors[output_index] = true;
        for index in (0..=output_index).rev() {
            if !ancestors[index] {
                continue;
            }
            let links = structure
                .operands
                .get(index)
                .expect("the freeze cannot shrink")
                .as_slice();
            if links.is_empty() {
                // Sources: leaves, parameters, and inputs, where
                // gradients stop and get read out below.
                continue;
            }
            let function = structure
                .functions
                .get(index)
                .expect("the freeze cannot shrink");
            let operand_traces: SmallVec<[Trace<'_, Data>; 2]> =
                links.iter().map(|link| trace(link.index())).collect();
            let operands: SmallVec<[&Trace<'_, Data>; 2]> = operand_traces.iter().collect();
            let gradient = cotangents[index].expect("ancestors carry cotangents");
            let recorded = function.backward(&operands, &trace(index), &gradient);
            debug_assert_eq!(recorded.len(), links.len());
            for (&link, cotangent) in links.iter().zip(recorded) {
                if let Some(contribution) = cotangent {
                    let slot = link.index();
                    ancestors[slot] = true;
                    let seeded = match cotangents[slot] {
                        Some(existing) => existing,
                        None => trace(slot).zero_like(),
                    };
                    cotangents[slot] = Some(seeded + contribution);
                }
            }
        }

        wrt.into_iter()
            .map(|target| {
                let value = self.resolve(target.into());
                match cotangents.get(value.id().index()).copied().flatten() {
                    Some(gradient) => gradient.value().symbol(),
                    // A non-ancestor's gradient is a recorded zero of
                    // its own shape, the tape twin of the zeros a
                    // gradient field holds there.
                    None => value.literal(Data::counted(value.shape(), 0)).symbol(),
                }
            })
            .collect()
    }
}

impl<Data: Differentiable> Default for Tape<Data> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/tape_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/differentiate_tests.rs"]
mod differentiate_tests;
