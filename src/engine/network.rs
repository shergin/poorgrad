use std::sync::Arc;

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Tensorial};

use super::{Evaluation, Field, Function, Symbol, Tape, Value, ValueId};

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
        assert!(
            symbol.lineage == self.tape.lineage(),
            "symbol belongs to a different network lineage"
        );
        let range = self
            .tape
            .segment_range(symbol.branch)
            .expect("symbol belongs to a divergent fork of this network");
        assert!(
            range.contains(&symbol.id.index()),
            "symbol is not allocated in this network"
        );
        Value::bind(&self.tape, symbol.id)
    }

    /// Resolves `symbol` in this generation, or returns `None` if the
    /// symbol belongs to a different lineage or a divergent fork, or no
    /// value with that name is allocated here: the probing form of
    /// `resolve`.
    pub fn try_resolve(&self, symbol: Symbol) -> Option<Value<'_, Data>> {
        if symbol.lineage != self.tape.lineage() {
            return None;
        }
        let range = self.tape.segment_range(symbol.branch)?;
        if !range.contains(&symbol.id.index()) {
            return None;
        }
        Some(Value::bind(&self.tape, symbol.id))
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
    pub fn update(&self, direction: &Field<Data>, rule: impl Fn(&Data, &Data) -> Data) -> Self {
        assert!(
            direction.lineage() == self.tape.lineage(),
            "field belongs to a different network lineage"
        );
        assert!(
            self.tape
                .agrees_with_chain(direction.chain(), direction.as_slice().len()),
            "field belongs to a divergent fork of this network"
        );
        Self {
            tape: self.tape.update(direction.as_slice(), rule),
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
        targets: impl IntoIterator<Item = Symbol>,
        feeds: impl IntoIterator<Item = (Symbol, Data)>,
    ) -> Evaluation<'_, Data> {
        let targets: Vec<ValueId> = targets
            .into_iter()
            .map(|symbol| self.resolve(symbol).id())
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
            snapshot.chain,
            values,
            evaluated,
            true,
            None,
            None,
        )
    }
}

impl<Data: Differentiable> Clone for Network<Data> {
    /// Forks the network in O(1).
    ///
    /// The fork shares the underlying arena but keeps an independent tape:
    /// later allocations on either network never affect the other, while
    /// every node allocated before the fork stays reachable in both.
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
