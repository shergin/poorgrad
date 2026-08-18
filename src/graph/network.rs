use std::sync::Arc;

use static_assertions::assert_impl_all;

use crate::Differentiable;

use super::{Origin, Parameters, SlotStore, Structure, Symbol, Tape, ValueId};

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
}

#[cfg(test)]
#[path = "tests/network_tests.rs"]
mod tests;
