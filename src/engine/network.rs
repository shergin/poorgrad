use std::ptr;
use std::sync::Arc;

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Tensorial};

use super::{
    Designation, Evaluation, Field, Function, Symbol, Tape, Trace, Value, ValueId, ValueRef,
};

// Compile-time thread-safety contract. `Differentiable` already requires
// `Data: Send + Sync`, so only a structural change (an `Rc`, a `RefCell`, a
// raw pointer) could break sharing across threads; a single concrete anchor
// is enough to catch that.
assert_impl_all!(Network<f64>: Send, Sync);

/// An append-only computation graph and its generation-specific payload state.
///
/// A network owns its recorded nodes, parameter payloads, and default input
/// payloads. Recording methods and [`Value`] operators append nodes through
/// shared interior synchronization. The returned values are `Copy` handles
/// that borrow the network and therefore cannot outlive it.
///
/// A network can be shared for concurrent recording and evaluation. Cloning
/// creates an O(1) fork, while [`Network::update`] creates a new generation
/// with a freshly computed parameter store. Both operations share the
/// existing graph storage; subsequent recordings on separate networks remain
/// isolated from one another.
#[derive(Debug)]
pub struct Network<Data> {
    tape: Tape<Data>,
}

impl<Data: Differentiable> Network<Data> {
    /// Creates an empty `Network`.
    pub fn new() -> Self {
        Self { tape: Tape::new() }
    }

    /// Returns the tape, for the engine's sibling modules (plans
    /// validate kinship and read columns through it).
    pub(crate) fn tape(&self) -> &Tape<Data> {
        &self.tape
    }

    /// Allocates a constant leaf and returns a proxy to it.
    ///
    /// Constants are fixed at recording time; see `parameter` for
    /// trainable leaves and `input` for leaves fed per run.
    pub fn leaf(&self, data: Data) -> Value<'_, Data> {
        let id = self.tape.record(Function::leaf(data), &[]);
        Value::bind(&self.tape, id)
    }

    /// Allocates a learnable parameter and returns a proxy to it.
    ///
    /// Parameters behave like leaves during runs. [`Network::update`] computes
    /// their payloads for the next generation without replacing their recorded
    /// nodes; the nodes live on the graph and the payloads live in the
    /// generation's parameter store.
    pub fn parameter(&self, data: Data) -> Value<'_, Data> {
        let id = self.tape.record_parameter(data);
        Value::bind(&self.tape, id)
    }

    /// Allocates a declared per-run input and returns a proxy to it.
    ///
    /// `initial` supplies the input's recorded shape and its default
    /// payload: a plain `forward` uses the default, while
    /// `forward_with` binds a fed payload for one run. Inputs behave
    /// like leaves during runs and are never touched by [`Network::update`].
    pub fn input(&self, initial: Data) -> Value<'_, Data> {
        let id = self.tape.record_input(initial);
        Value::bind(&self.tape, id)
    }

    /// Resolves `symbol` in this generation, returning its `Value`.
    ///
    /// Proxies borrow the generation that created them, so a proxy taken
    /// before a fork or an update belongs to the old generation; `resolve`
    /// produces the equivalent proxy for this one. The symbol carries its
    /// lineage and branch, so kinship is verified before the positional
    /// lookup: an unrelated network and a fork that diverged before the
    /// symbol was minted are both rejected. A failed resolution is a
    /// programmer error, like every other positional misuse;
    /// `try_resolve` is the probing form.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network lineage or to a
    /// divergent fork, or is not allocated in this generation.
    pub fn resolve(&self, symbol: Symbol) -> Value<'_, Data> {
        Value::bind(&self.tape, self.tape.locate(symbol))
    }

    /// Locates `reference` on this network: a bound value proves
    /// identity by tape pointer, a symbol resolves with the checks of
    /// [`Network::resolve`].
    pub(crate) fn locate(&self, reference: impl ValueRef<Data>) -> ValueId {
        match reference.designation() {
            Designation::Bound { tape, id } => {
                assert!(
                    ptr::eq(&self.tape, tape),
                    "value belongs to a different network"
                );
                id
            }
            Designation::Named(symbol) => self.tape.locate(symbol),
        }
    }

    /// Returns the detached name of `reference`: a symbol passes
    /// through and is validated where it is used, a bound value must
    /// belong to this network.
    pub(crate) fn named(&self, reference: impl ValueRef<Data>) -> Symbol {
        match reference.designation() {
            Designation::Bound { tape, id } => {
                assert!(
                    ptr::eq(&self.tape, tape),
                    "value belongs to a different network"
                );
                Value::bind(&self.tape, id).symbol()
            }
            Designation::Named(symbol) => symbol,
        }
    }

    /// Resolves `symbol` in this generation, or returns `None` if the
    /// symbol belongs to a different lineage or a divergent fork, or no
    /// value with that name is allocated here: the probing form of
    /// `resolve`.
    pub fn try_resolve(&self, symbol: Symbol) -> Option<Value<'_, Data>> {
        let id = self.tape.probe(symbol).ok()?;
        Some(Value::bind(&self.tape, id))
    }

    /// Returns the number of allocated values.
    pub fn len(&self) -> usize {
        self.tape.len()
    }

    /// Returns `true` if it holds no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a new network generation with every parameter's payload
    /// replaced by `rule(current, direction)`.
    ///
    /// It is the training-step state transition, and `direction` is any
    /// field over this network's lineage: the [`Gradients`](super::Gradients) of
    /// a backward run, or a derived update direction such as a momentum
    /// velocity. The new generation shares the
    /// complete recorded graph and rebuilds only the parameter store, so node
    /// positions remain stable and compatible [`Symbol`]s keep resolving. The
    /// old generation remains fully usable with its own proxies and runs. The
    /// update performs O(parameters) work and allocations; replaced payloads
    /// are reclaimed when the old generation is dropped.
    ///
    /// # Panics
    /// Panics if `direction` belongs to a different network lineage or a
    /// divergent fork, is stale, or if `rule` returns a payload whose
    /// shape differs from the parameter's recorded shape.
    pub fn update(
        &self,
        direction: &Field<Data>,
        mut rule: impl FnMut(&Data, &Data) -> Data,
    ) -> Self {
        self.update_each(direction, move |_, current, direction| {
            rule(current, direction)
        })
    }

    /// Returns a new network generation like [`Network::update`], with
    /// the parameter's own [`Value`] passed to the rule: the
    /// identity-aware form, for per-parameter policy — an optimizer's
    /// selective weight decay, per-parameter clipping, or logging —
    /// decided from the parameter's symbol, shape, or rank at the call
    /// site.
    ///
    /// The rule runs once per parameter, in parameter-store order (the
    /// order the parameters were allocated); an `FnMut` rule may
    /// observe that order, and it is part of the method's contract.
    ///
    /// # Panics
    /// Panics as [`Network::update`] panics.
    pub fn update_each(
        &self,
        direction: &Field<Data>,
        mut rule: impl FnMut(Value<'_, Data>, &Data, &Data) -> Data,
    ) -> Self {
        assert!(
            direction.kinship().lineage() == self.tape.lineage(),
            "field belongs to a different network lineage"
        );
        assert!(
            self.tape
                .agrees_with_chain(direction.kinship().chain(), direction.as_slice().len()),
            "field belongs to a divergent fork of this network"
        );
        Self {
            tape: self
                .tape
                .update(direction.as_slice(), |node, current, direction| {
                    rule(Value::bind(&self.tape, node), current, direction)
                }),
        }
    }

    /// Returns a lineage-compatible network whose structural arenas hold
    /// only this network's live nodes.
    ///
    /// [`Network::clone`] is O(1) because it shares the append-only
    /// column arena. That is the right trade for train-only forks
    /// (`update` only): parameters live in the per-generation store and
    /// never touch the arena. It is the wrong trade when siblings
    /// *record* after the fork — those nodes stay allocated in the
    /// shared arena until every sharer drops, even after the recording
    /// forks themselves are gone.
    ///
    /// Compaction rebuilds the structure columns into private arenas
    /// from the live entries alone. Replacing a parent with its
    /// compacted form (`network = network.compacted()`) after
    /// experimental forks have dropped lets the process release their
    /// structural garbage. The result stays in the same lineage:
    /// symbols resolve, parameter payloads are the current generation's,
    /// and the first side to record continues the tip branch as after a
    /// plain fork.
    ///
    /// Cost is O(live nodes), not O(1). Prefer [`Network::clone`] when
    /// no post-fork recording is involved.
    pub fn compacted(&self) -> Self {
        Self {
            tape: self.tape.compacted(),
        }
    }
}

// Running the tape requires the full payload contract (`Tensorial`, for
// the transcendental and tensor-native operations), while building and
// updating the graph needs only arithmetic.
impl<Data: Tensorial> Network<Data> {
    /// Evaluates every node in allocation order with every input at its
    /// default payload; see `forward_with`.
    pub fn forward(&self) -> Evaluation<'_, Data> {
        self.forward_with(std::iter::empty())
    }

    /// Evaluates every node in allocation order, materializing the payload
    /// of each value into a fresh `Evaluation`, with `feeds` bound to
    /// declared inputs for this run only.
    ///
    /// Feeds are run-local state: they overlay the input defaults
    /// without touching the graph, so any number of threads can forward
    /// one shared network on different batches concurrently. Unfed
    /// inputs use their defaults. The replay works off an O(1) snapshot
    /// of the tape, the parameter store, and the input defaults, taken
    /// atomically, so the network is never locked during the run and
    /// concurrent recordings and updates do not disturb it. Allocation
    /// order is dependency order by construction, which is what makes
    /// the single forward scan sufficient. The snapshot travels with the
    /// returned evaluation, whose `backward` replays it in reverse.
    ///
    /// # Panics
    /// Panics if a fed symbol does not resolve in this generation, names
    /// a node that is not an input, or carries a payload whose shape
    /// differs from the input's recorded shape.
    pub fn forward_with(
        &self,
        feeds: impl IntoIterator<Item = (Symbol, Data)>,
    ) -> Evaluation<'_, Data> {
        self.run(None, feeds)
    }

    /// Evaluates only the ancestors of `targets` — the target-sliced
    /// run — with `feeds` bound to declared inputs for this run only.
    ///
    /// It is `forward_with` restricted to what the caller will read:
    /// reachability over the operand links selects the targets'
    /// ancestor closure, and every node outside it is skipped, its slot
    /// holding an O(1) zero placeholder of the recorded shape. The
    /// placeholders keep [`Network::update`] sound — a parameter
    /// outside the closure receives its true gradient, zero — while
    /// reads stay loud: [`Evaluation::of`] and [`Evaluation::backward`]
    /// panic on a skipped value instead of answering with a
    /// placeholder.
    ///
    /// With several expressions recorded on one tape (the training and
    /// evaluation twins of the examples), slicing to one expression's
    /// targets skips the other expression entirely — the first rung of
    /// the plan-lowering ladder, applied without any plan object.
    ///
    /// # Panics
    /// Panics if a target does not resolve in this generation, or as
    /// `forward_with` panics for `feeds`.
    pub fn forward_for(
        &self,
        targets: impl IntoIterator<Item = impl ValueRef<Data>>,
        feeds: impl IntoIterator<Item = (Symbol, Data)>,
    ) -> Evaluation<'_, Data> {
        let targets: Vec<ValueId> = targets
            .into_iter()
            .map(|target| self.locate(target))
            .collect();
        self.run(Some(targets), feeds)
    }

    /// Runs the tape over an atomic snapshot: the shared body of
    /// `forward_with` (every node) and `forward_for` (the targets'
    /// ancestor closure).
    fn run(
        &self,
        targets: Option<Vec<ValueId>>,
        feeds: impl IntoIterator<Item = (Symbol, Data)>,
    ) -> Evaluation<'_, Data> {
        let mut bindings = Vec::new();
        for (symbol, payload) in feeds {
            let value = self.resolve(symbol);
            let slot = self
                .tape
                .input_slot(value.id())
                .expect("only inputs can be fed");
            assert_eq!(
                payload.shape(),
                value.shape(),
                "fed payload must match the input's recorded shape"
            );
            bindings.push((slot, payload));
        }

        let snapshot = self.tape.snapshot();
        let inputs = if bindings.is_empty() {
            snapshot.inputs
        } else {
            let mut overlaid = snapshot.inputs.as_ref().clone();
            for (slot, payload) in bindings {
                overlaid[slot.index()] = payload;
            }
            Arc::new(overlaid)
        };
        // Reachability doubles the backward scan's trick in reverse:
        // operands live below their consumers, so one descending sweep
        // marks the whole ancestor closure.
        let evaluated = targets.map(|targets| {
            let mut wanted = vec![false; snapshot.functions.len()];
            for target in targets {
                wanted[target.index()] = true;
            }
            for index in (0..wanted.len()).rev() {
                if !wanted[index] {
                    continue;
                }
                let links = snapshot
                    .operands
                    .get(index)
                    .expect("snapshot cannot shrink");
                for link in links.as_slice() {
                    wanted[link.index()] = true;
                }
            }
            wanted
        });
        let mut values = Vec::with_capacity(snapshot.functions.len());
        for (index, (function, links)) in snapshot
            .functions
            .iter()
            .zip(snapshot.operands.iter())
            .enumerate()
        {
            let skipped = matches!(&evaluated, Some(wanted) if !wanted[index]);
            let value = if skipped {
                // A shape-correct, non-allocating zero: never read back
                // (`of` checks the evaluated set), but shaped so that
                // gradient buffers and `update` stay coherent.
                Data::counted(self.tape.shape(ValueId(index)), 0)
            } else {
                let operands: SmallVec<[&Data; 2]> = links
                    .as_slice()
                    .iter()
                    .map(|link| &values[link.index()])
                    .collect();
                let value = function.forward(&operands, snapshot.parameters.payloads(), &inputs);
                // The recorded shape is the type of this node; a payload
                // whose rule answers a different shape has broken the
                // operation contract at exactly this producing node.
                debug_assert_eq!(
                    value.shape(),
                    self.tape.shape(ValueId(index)),
                    "operation output shape disagrees with the recorded shape at node {index}"
                );
                value
            };
            values.push(value);
        }
        Evaluation::new(
            &self.tape,
            snapshot.functions,
            snapshot.operands,
            snapshot.kinship,
            values,
            evaluated,
            true,
            None,
            None,
        )
    }
}

impl<Data: Tensorial> Network<Data> {
    /// Records the reverse-mode gradient of `loss` with respect to each
    /// `wrt` entry as ordinary computed nodes on this network, and
    /// returns their symbols in `wrt` order.
    ///
    /// It is `backward` as a tape-to-tape transform: the same reverse
    /// scan the engine runs over payload buffers runs here over
    /// recording [`Trace`] handles, applying the very same derivative
    /// rules — so the recorded gradient and the engine's are one body
    /// of knowledge, and a compiled plan over `[loss, gradients...]`
    /// reproduces [`Evaluation::backward`] bitwise (same seed, same
    /// accumulation order). Gradients become first-class values:
    /// compilable, emittable, readable, and differentiable again for
    /// higher-order derivatives.
    ///
    /// A `wrt` value that is not an ancestor of the loss answers a
    /// recorded zero of its own shape, exactly as
    /// [`Gradients`](super::Gradients) would. The transform reads graph
    /// structure only, never payloads, so it is generation-independent
    /// the way plans are; recording appends to the current tape and
    /// leaves every existing node untouched.
    ///
    /// # Panics
    /// Panics if `loss` is not a recorded scalar (reduce with `sum`
    /// first) or any symbol belongs to a different lineage or a
    /// divergent fork.
    pub fn differentiate(
        &self,
        loss: impl ValueRef<Data>,
        wrt: impl IntoIterator<Item = impl ValueRef<Data>>,
    ) -> Vec<Symbol> {
        let loss_value = Value::bind(&self.tape, self.locate(loss));
        assert_eq!(
            loss_value.shape().rank(),
            0,
            "differentiate requires a scalar loss; reduce it with `sum` first"
        );
        let output_index = loss_value.id().index();
        let snapshot = self.tape.snapshot();
        let trace = |index: usize| Trace::of(Value::bind(&self.tape, ValueId(index)));

        // The scan mirrors `Evaluation::backward` deliberately and
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
            let links = snapshot
                .operands
                .get(index)
                .expect("snapshot cannot shrink")
                .as_slice();
            if links.is_empty() {
                // Sources: leaves, parameters, and inputs, where
                // gradients stop and get read out below.
                continue;
            }
            let function = snapshot
                .functions
                .get(index)
                .expect("snapshot cannot shrink");
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
                let value = Value::bind(&self.tape, self.locate(target));
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

impl<Data: Differentiable> Clone for Network<Data> {
    /// Forks the network in O(1).
    ///
    /// The fork shares the underlying arena but keeps an independent tape:
    /// later allocations on either network never affect the other, while
    /// every node allocated before the fork stays reachable in both.
    /// Parameter payloads live outside the arena (the parameter store);
    /// structural nodes recorded on a sibling after the fork can keep
    /// arena memory alive until every sharer drops — see
    /// [`Network::compacted`].
    fn clone(&self) -> Self {
        Self {
            tape: self.tape.fork(),
        }
    }
}

impl<Data: Differentiable> Default for Network<Data> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/network_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/differentiate_tests.rs"]
mod differentiate_tests;
