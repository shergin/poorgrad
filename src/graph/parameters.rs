use static_assertions::assert_impl_all;

use crate::Differentiable;

use super::Field;

use super::{Network, Origin, SlotStore, Symbol};

// Request-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Parameters<f64>: Send, Sync);

/// The live parameter payloads of one network, as a caller-owned value.
///
/// Where the [`Network`](crate::Network) is the immutable spec, this is
/// the state: born from the record-site initials
/// ([`Network::parameters`](crate::Network::parameters)) or a
/// checkpoint, passed by reference into every run and plan, and stepped
/// as pure data — no run mutates it, and training mints no new network.
/// `Clone` is honest and O(parameters), which is the whole cost of a
/// what-if: one spec, any number of states.
///
/// Optimizer state (moments, velocities) is [`Field`](crate::Field)
/// algebra held next to a `Parameters` value in the caller's structs;
/// nothing hides in the graph.
#[derive(Debug, Clone)]
pub struct Parameters<Data> {
    origin: Origin,
    store: SlotStore<Data>,
}

impl<Data: Differentiable> Parameters<Data> {
    /// Wraps `store` as the parameter state of the `origin` family.
    pub(super) fn new(origin: Origin, store: SlotStore<Data>) -> Self {
        Self { origin, store }
    }

    /// Returns the origin token of the network family this state
    /// steps.
    pub(crate) fn origin(&self) -> Origin {
        self.origin
    }

    /// Returns the number of parameter slots.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` if it carries no parameters.
    pub fn is_empty(&self) -> bool {
        self.store.len() == 0
    }

    /// Returns the payloads in slot order, for the engine's node
    /// evaluation.
    pub(crate) fn payloads(&self) -> &[Data] {
        self.store.payloads()
    }

    /// Returns the payload of the parameter named by `symbol`.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or does not
    /// name a parameter these parameters carry.
    pub fn of(&self, symbol: Symbol) -> &Data {
        assert!(
            symbol.origin == self.origin,
            "symbol belongs to a different network"
        );
        let Some(slot) = self.store.slot_of(symbol.id) else {
            panic!("symbol does not name a parameter these parameters carry");
        };
        &self.store.payloads()[slot.index()]
    }

    /// Returns the state with every payload replaced by
    /// `rule(current, direction)`: the training-step transition.
    ///
    /// `direction` is any field over this network family: the
    /// [`Gradients`](crate::Gradients) of a backward run, or a derived
    /// update direction such as a momentum velocity. The step is pure
    /// data — O(parameters) work and allocations, no new network, no
    /// lock — and slot order is preserved, so symbols keep naming
    /// their parameters.
    ///
    /// # Panics
    /// Panics if `direction` belongs to a different network or is
    /// stale, or if `rule` returns a payload whose shape differs from
    /// the parameter's.
    pub fn step(
        &self,
        direction: &Field<Data>,
        mut rule: impl FnMut(&Data, &Data) -> Data,
    ) -> Self {
        self.step_each(direction, move |_, current, direction| {
            rule(current, direction)
        })
    }

    /// Returns the state stepped like [`Parameters::step`], with the
    /// parameter's [`Symbol`] passed to the rule: the identity-aware
    /// form, for per-parameter policy — an optimizer's selective
    /// weight decay, per-parameter clipping, or logging — decided from
    /// the parameter's symbol or the payload's own shape at the call
    /// site.
    ///
    /// The rule runs once per parameter, in slot order (the order the
    /// parameters were recorded); an `FnMut` rule may observe that
    /// order, and it is part of the method's contract.
    ///
    /// # Panics
    /// Panics as [`Parameters::step`] panics.
    pub fn step_each(
        &self,
        direction: &Field<Data>,
        mut rule: impl FnMut(Symbol, &Data, &Data) -> Data,
    ) -> Self {
        assert!(
            direction.origin() == self.origin,
            "field belongs to a different network"
        );
        if let Some(last) = self.store.last_node() {
            assert!(
                last.index() < direction.len(),
                "field is stale: it does not cover every parameter"
            );
        }
        let mut payloads = Vec::with_capacity(self.store.len());
        for (node, current) in self.store.iter() {
            let symbol = Symbol {
                origin: self.origin,
                id: node,
            };
            let next = rule(symbol, current, &direction.payloads()[node.index()]);
            assert_eq!(
                next.shape(),
                current.shape(),
                "step must preserve the parameter's shape"
            );
            payloads.push(next);
        }
        Self {
            origin: self.origin,
            store: self.store.with_payloads(payloads),
        }
    }

    /// Returns the state carried across an
    /// [`Network::into_tape`](crate::Network::into_tape) round trip:
    /// existing slots keep these payloads, slots recorded since take
    /// their record-site initials.
    ///
    /// # Panics
    /// Panics if `network` belongs to a different family or records
    /// fewer parameters than this state carries.
    pub fn carried(&self, network: &Network<Data>) -> Self {
        assert!(
            network.origin() == self.origin,
            "parameters belong to a different network"
        );
        let fresh = network.parameters();
        assert!(
            self.len() <= fresh.len(),
            "parameters cover more slots than the network records"
        );
        let mut payloads: Vec<Data> = Vec::with_capacity(fresh.len());
        payloads.extend(self.store.payloads().iter().cloned());
        payloads.extend(fresh.store.payloads()[self.len()..].iter().cloned());
        Self {
            origin: self.origin,
            store: fresh.store.with_payloads(payloads),
        }
    }

    /// Returns the state with the named parameters' payloads replaced:
    /// the installation route for checkpoints and foreign weights.
    ///
    /// Every other slot keeps its payload; replacing the same symbol
    /// twice keeps the last entry.
    ///
    /// # Panics
    /// Panics if a symbol belongs to a different network or does not
    /// name a parameter, or if a replacement's shape differs from the
    /// parameter's.
    pub fn with_payloads(&self, replacements: impl IntoIterator<Item = (Symbol, Data)>) -> Self {
        let mut payloads = self.store.payloads().to_vec();
        for (symbol, payload) in replacements {
            assert!(
                symbol.origin == self.origin,
                "symbol belongs to a different network"
            );
            let Some(slot) = self.store.slot_of(symbol.id) else {
                panic!("symbol does not name a parameter these parameters carry");
            };
            assert_eq!(
                payload.shape(),
                payloads[slot.index()].shape(),
                "a replacement must preserve the parameter's shape"
            );
            payloads[slot.index()] = payload;
        }
        Self {
            origin: self.origin,
            store: self.store.with_payloads(payloads),
        }
    }
}

#[cfg(test)]
#[path = "tests/parameters_tests.rs"]
mod tests;
