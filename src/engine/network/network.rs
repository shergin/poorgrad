use std::sync::Arc;

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::engine::{Function, Posture, Run};
use crate::{Differentiable, Tensorial};

use super::{Origin, Parameters, SlotId, SlotStore, Structure, Symbol, Tape, ValueId};

// Compile-time thread-safety contract. `Differentiable` already requires
// `Data: Send + Sync`, so only a structural change (an `Rc`, a `RefCell`, a
// raw pointer) could break sharing across threads; a single concrete anchor
// is enough to catch that.
assert_impl_all!(Network<f64>: Send, Sync);

/// The sealed phase of a recording: an immutable computation-graph
/// spec.
///
/// A network holds structure, shapes, parameter initials, and input
/// defaults — the whole spec, runnable standalone — and no live state:
/// parameter payloads are the caller's [`Parameters`], fed inputs are
/// per-run overlays. Nothing mutates a network, so it is `Send + Sync`
/// with no lock, and any number of threads can run one shared network
/// concurrently through `&Network` or `Arc<Network>`.
///
/// A network is only ever born from a tape ([`Tape::into_network`]),
/// and [`Network::into_tape`] consumes it to reopen recording — the
/// consuming pair keeps one origin's history linear by ownership, so
/// symbols and plans stay valid across every round trip. It is
/// deliberately not `Clone`: a second sealed copy could be reopened
/// into a divergent future, which is exactly what the ownership rule
/// exists to make unrepresentable.
#[derive(Debug)]
pub struct Network<Data> {
    origin: Origin,
    structure: Structure<Data>,
    initials: SlotStore<Data>,
    inputs: Arc<SlotStore<Data>>,
}

impl<Data: Differentiable> Network<Data> {
    /// Seals the recorded columns and stores under `origin`: the body
    /// of [`Tape::into_network`].
    pub(super) fn seal(
        origin: Origin,
        structure: Structure<Data>,
        initials: SlotStore<Data>,
        inputs: SlotStore<Data>,
    ) -> Self {
        Self {
            origin,
            structure,
            initials,
            inputs: Arc::new(inputs),
        }
    }

    /// Reopens the network for further recording, consuming it: the
    /// inverse of [`Tape::into_network`].
    ///
    /// The tape keeps the same origin, so every existing [`Symbol`]
    /// keeps naming its node, and extension is linear: a consumed
    /// network cannot also stay sealed, which is what makes divergent
    /// histories unconstructible. State carried in a
    /// [`Parameters`] value survives the round trip through
    /// [`Parameters::carried`].
    pub fn into_tape(self) -> Tape<Data> {
        Tape::reopen(self.origin, self)
    }

    /// Hands the stores back for [`Tape::reopen`], unsharing the input
    /// defaults if a plan still holds them.
    pub(super) fn into_stores(self) -> (Structure<Data>, SlotStore<Data>, SlotStore<Data>) {
        (
            self.structure,
            self.initials,
            Arc::unwrap_or_clone(self.inputs),
        )
    }

    /// Materializes the record-site initials into a fresh caller-owned
    /// [`Parameters`] value.
    ///
    /// Every call answers a new value, so initialization stays visible
    /// at the record site and what-if states are independent from
    /// birth.
    pub fn parameters(&self) -> Parameters<Data> {
        Parameters::new(self.origin, self.initials.clone())
    }

    /// Returns the number of recorded nodes.
    pub fn len(&self) -> usize {
        self.structure.len()
    }

    /// Returns `true` if it holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the origin token of this network's family.
    pub(crate) fn origin(&self) -> Origin {
        self.origin
    }

    /// Returns the recorded node columns.
    pub(crate) fn structure(&self) -> &Structure<Data> {
        &self.structure
    }

    /// Returns the input-default store, shared for plan freezes.
    pub(crate) fn inputs(&self) -> &Arc<SlotStore<Data>> {
        &self.inputs
    }

    /// Returns the number of recorded parameter slots.
    pub(crate) fn parameters_len(&self) -> usize {
        self.initials.len()
    }

    /// Locates the node `symbol` names on this network.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or is not
    /// allocated in it.
    pub(crate) fn locate(&self, symbol: Symbol) -> ValueId {
        assert!(
            symbol.origin == self.origin,
            "symbol belongs to a different network"
        );
        assert!(
            symbol.id.index() < self.structure.len(),
            "symbol is not allocated in this network"
        );
        symbol.id
    }

    /// Panics unless `parameters` was born from this network's exact
    /// extent: the run-side kinship check.
    fn assert_covering(&self, parameters: &Parameters<Data>) {
        assert!(
            parameters.origin() == self.origin,
            "parameters belong to a different network"
        );
        assert_eq!(
            parameters.len(),
            self.initials.len(),
            "parameters do not cover this network's parameter slots; \
             carry them across a reopen with `Parameters::carried`"
        );
    }
}

// Running the graph requires the full payload contract (`Tensorial`,
// for the transcendental and tensor-native operations), while sealing
// and reopening it needs only arithmetic.
impl<Data: Tensorial> Network<Data> {
    /// Evaluates every node in allocation order, materializing the
    /// payload of each value into a fresh [`Run`], reading parameter
    /// payloads from `parameters` and binding `feeds` to declared
    /// inputs for this run only.
    ///
    /// Feeds are run-local state: they overlay the input defaults
    /// without touching the spec, so any number of threads can forward
    /// one shared network on different batches — or different
    /// [`Parameters`] — concurrently. Unfed inputs use their defaults.
    /// Allocation order is dependency order by construction, which is
    /// what makes the single forward scan sufficient. The returned run
    /// owns its values, so [`Run::backward`] needs no network borrow.
    ///
    /// # Panics
    /// Panics if `parameters` belongs to a different network or does
    /// not cover this one, if a fed symbol does not resolve here or
    /// names a node that is not an input, or if a fed payload's shape
    /// differs from the input's recorded shape.
    pub fn forward(
        &self,
        parameters: &Parameters<Data>,
        feeds: impl IntoIterator<Item = (Symbol, Data)>,
    ) -> Run<Data> {
        self.run(parameters, None, feeds)
    }

    /// Evaluates only the ancestors of `targets` — the target-sliced
    /// run — with `feeds` bound to declared inputs for this run only.
    ///
    /// It is `forward` restricted to what the caller will read:
    /// reachability over the operand links selects the targets'
    /// ancestor closure, and every node outside it is skipped, its slot
    /// holding an O(1) zero placeholder of the recorded shape. Reads
    /// stay loud: [`Run::of`] and [`Run::backward`] panic on a skipped
    /// value instead of answering with a placeholder.
    ///
    /// With several expressions recorded on one tape (the training and
    /// evaluation twins of the examples), slicing to one expression's
    /// targets skips the other expression entirely — the first rung of
    /// the plan-lowering ladder, applied without any plan object.
    ///
    /// # Panics
    /// Panics if a target does not resolve in this network, or as
    /// [`Network::forward`] panics.
    pub fn forward_for(
        &self,
        parameters: &Parameters<Data>,
        targets: impl IntoIterator<Item = Symbol>,
        feeds: impl IntoIterator<Item = (Symbol, Data)>,
    ) -> Run<Data> {
        let targets: Vec<ValueId> = targets
            .into_iter()
            .map(|target| self.locate(target))
            .collect();
        self.run(parameters, Some(targets), feeds)
    }

    /// Returns the input slot behind `id`, or `None` if the node is
    /// not an input.
    fn input_slot(&self, id: ValueId) -> Option<SlotId> {
        match self
            .structure
            .functions
            .get(id.index())
            .expect("`ValueId` is in bounds for its network")
        {
            Function::Input(input) => Some(input.0),
            _ => None,
        }
    }

    /// Replays the recording: the shared body of `forward` (every
    /// node) and `forward_for` (the targets' ancestor closure).
    fn run(
        &self,
        parameters: &Parameters<Data>,
        targets: Option<Vec<ValueId>>,
        feeds: impl IntoIterator<Item = (Symbol, Data)>,
    ) -> Run<Data> {
        self.assert_covering(parameters);
        let mut bindings = Vec::new();
        for (symbol, payload) in feeds {
            let id = self.locate(symbol);
            let slot = self.input_slot(id).expect("only inputs can be fed");
            let declared = self
                .structure
                .shapes
                .get(id.index())
                .expect("shapes cover the network");
            assert_eq!(
                &payload.shape(),
                declared,
                "fed payload must match the input's recorded shape"
            );
            bindings.push((slot, payload));
        }
        let inputs = if bindings.is_empty() {
            Arc::clone(&self.inputs)
        } else {
            let mut overlaid = self.inputs.as_ref().clone();
            for (slot, payload) in bindings {
                overlaid.set(slot, payload);
            }
            Arc::new(overlaid)
        };

        let structure = &self.structure;
        // Reachability doubles the backward scan's trick in reverse:
        // operands live below their consumers, so one descending sweep
        // marks the whole ancestor closure.
        let computed = targets.map(|targets| {
            let mut wanted = vec![false; structure.len()];
            for target in targets {
                wanted[target.index()] = true;
            }
            for index in (0..wanted.len()).rev() {
                if !wanted[index] {
                    continue;
                }
                let links = structure
                    .operands
                    .get(index)
                    .expect("operands cover the network");
                for link in links.as_slice() {
                    wanted[link.index()] = true;
                }
            }
            wanted
        });
        let mut values = Vec::with_capacity(structure.len());
        for (index, (function, links)) in structure
            .functions
            .iter()
            .zip(structure.operands.iter())
            .enumerate()
        {
            let skipped = matches!(&computed, Some(wanted) if !wanted[index]);
            let value = if skipped {
                // A shape-correct, non-allocating zero: never read back
                // (`of` checks the computed set), but shaped so that
                // gradient buffers stay coherent.
                let shape = structure
                    .shapes
                    .get(index)
                    .expect("shapes cover the network")
                    .clone();
                Data::counted(shape, 0)
            } else {
                let operands: SmallVec<[&Data; 2]> = links
                    .as_slice()
                    .iter()
                    .map(|link| &values[link.index()])
                    .collect();
                let value = function.forward(&operands, parameters.payloads(), inputs.payloads());
                // The recorded shape is the type of this node; a payload
                // whose rule answers a different shape has broken the
                // operation contract at exactly this producing node.
                debug_assert_eq!(
                    value.shape(),
                    *structure
                        .shapes
                        .get(index)
                        .expect("shapes cover the network"),
                    "operation output shape disagrees with the recorded shape at node {index}"
                );
                value
            };
            values.push(value);
        }
        let posture = match computed {
            Some(computed) => Posture::Sliced { computed },
            None => Posture::Complete,
        };
        Run::new(structure.clone(), self.origin, values, posture)
    }
}

#[cfg(test)]
#[path = "tests/network_tests.rs"]
mod tests;
