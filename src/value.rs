use std::fmt;
use std::ops::{Add, Mul, Neg};
use std::ptr;

use static_assertions::assert_impl_all;

use super::{Differentiable, Elementary, Function, Symbol, Tape};

// Compile-time contract: proxies stay thread-safe and `Copy`; the anchor
// rationale is documented in `network.rs`.
assert_impl_all!(Value<'static, f64>: Send, Sync, Copy);

/// A lightweight, `Copy` handle to a value allocated in a `Network`.
///
/// It is an index into the network's tape rather than a pointer, so handles
/// are cheap to copy and carry no ownership of network memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ValueId(pub(crate) usize);

impl ValueId {
    /// Returns the position of the value on its tape.
    pub(crate) fn index(self) -> usize {
        self.0
    }
}

/// A `Copy` proxy to a value allocated in a `Network`.
///
/// It pairs a borrow of the network's `Tape` with the position of its node:
/// the network is the single owner of all value state, and a `Value` is a
/// view into it that cannot outlive it. Being `Copy`, proxies are never
/// consumed: arithmetic operators build the graph (`let x = v1 + v2;`
/// records a new computed node on the same network) while the operands stay
/// usable for further expressions. Accessors such as `data` briefly take
/// the tape lock and clone the payload out.
pub struct Value<'network, Data> {
    tape: &'network Tape<Data>,
    id: ValueId,
}

impl<'network, Data: Differentiable> Value<'network, Data> {
    /// Binds a proxy to the node `id` recorded on `tape`.
    pub(crate) fn bind(tape: &'network Tape<Data>, id: ValueId) -> Self {
        Self { tape, id }
    }

    /// Returns the handle of the node this proxy points to.
    pub(crate) fn id(&self) -> ValueId {
        self.id
    }

    /// Returns the tape this proxy points into.
    pub(crate) fn tape(&self) -> &'network Tape<Data> {
        self.tape
    }

    /// Returns the detached name of this value: its identity across
    /// network generations, resolved back into a proxy by
    /// `Network::resolve`.
    pub fn symbol(&self) -> Symbol {
        Symbol(self.id)
    }

    /// Returns a clone of the `Function` that produced this value.
    #[cfg(test)]
    pub(crate) fn function(&self) -> Function<Data> {
        self.tape.with_node(self.id, |function| function.clone())
    }

    /// Returns a clone of the leaf payload, or `None` for computed values.
    pub fn data(&self) -> Option<Data> {
        self.tape
            .with_node(self.id, |function| function.data().cloned())
    }

    /// Records a computed node produced by `function` on the same network
    /// and returns a proxy to it.
    fn apply(&self, function: Function<Data>) -> Self {
        let id = self.tape.record(function);
        Self::bind(self.tape, id)
    }

    /// Panics if `other` belongs to a different network.
    fn assert_same_network(&self, other: &Self) {
        assert!(
            ptr::eq(self.tape, other.tape),
            "values belong to different networks"
        );
    }
}

impl<'network, Data: Elementary> Value<'network, Data> {
    /// Records the hyperbolic tangent of this value on the same network
    /// and returns a proxy to it.
    pub fn tanh(self) -> Self {
        self.apply(Function::tanh(self.id))
    }
}

// Manual implementations avoid the `Data: Clone`/`Data: Copy` bounds a
// derive would add: the proxy copies a borrow and an index, never `Data`.
impl<Data> Clone for Value<'_, Data> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Data> Copy for Value<'_, Data> {}

/// It prints only the node position to avoid dumping the whole network.
impl<Data> fmt::Debug for Value<'_, Data> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Value")
            .field("id", &self.id)
            .finish()
    }
}

impl<'network, Data: Differentiable> Add for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn add(self, rhs: Self) -> Self::Output {
        self.assert_same_network(&rhs);
        self.apply(Function::add(self.id, rhs.id))
    }
}

impl<'network, Data: Differentiable> Mul for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.assert_same_network(&rhs);
        self.apply(Function::mul(self.id, rhs.id))
    }
}

impl<'network, Data: Differentiable> Neg for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn neg(self) -> Self::Output {
        self.apply(Function::neg(self.id))
    }
}
