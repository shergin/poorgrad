use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::ptr;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Elementary, Shape, Tensorial};

use super::{Function, Symbol, Tape};

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
/// A value stores its node position together with a borrow of the network, so
/// it cannot outlive the graph it refers to. Arithmetic and tensor operations
/// append computed nodes to that graph without consuming their operands.
/// Payload literals can be mixed directly into expressions, in either operand
/// order; every literal occurrence records a new leaf.
///
/// Operations validate network identity and shape compatibility when they are
/// recorded, so invalid expressions panic before a forward run begins.
///
/// [`Value::shape`] returns the shape inferred when the node was recorded.
/// [`Value::payload`] clones the stored payload of a leaf, parameter, or input;
/// computed values are read from an [`Evaluation`](super::Evaluation).
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

    /// Returns the detached name of this value: its identity across compatible
    /// network generations, resolved back into a proxy by
    /// [`Network::resolve`](crate::Network::resolve).
    pub fn symbol(&self) -> Symbol {
        Symbol {
            lineage: self.tape.lineage(),
            branch: self.tape.branch_of(self.id),
            id: self.id,
        }
    }

    /// Returns a clone of the `Function` that produced this value.
    #[cfg(test)]
    pub(crate) fn function(&self) -> Function<Data> {
        self.tape.with_node(self.id, |function| function.clone())
    }

    /// Returns the shape of this value, inferred when it was recorded.
    pub fn shape(&self) -> Shape {
        self.tape.shape(self.id)
    }

    /// Returns a clone of this node's stored payload, or `None` for a computed
    /// value.
    ///
    /// Leaves return their recorded payload, parameters return the current
    /// generation's payload, and inputs return their recorded default rather
    /// than a run-local feed. Use [`Evaluation::of`](super::Evaluation::of) to
    /// read the result of a particular forward run.
    pub fn payload(&self) -> Option<Data> {
        self.tape.payload_of(self.id)
    }

    /// Records a computed node produced by `function` on the same network
    /// and returns a proxy to it.
    fn apply(&self, function: Function<Data>) -> Self {
        let id = self.tape.record(function);
        Self::bind(self.tape, id)
    }

    /// Records `data` as a fresh leaf on the same network and returns a
    /// proxy to it.
    ///
    /// It backs the payload-literal operator sugar: every literal
    /// appearance records its own leaf.
    pub(crate) fn literal(&self, data: Data) -> Self {
        Self::bind(self.tape, self.tape.record(Function::leaf(data)))
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

    /// Records the exponential of this value on the same network and
    /// returns a proxy to it.
    pub fn exp(self) -> Self {
        self.apply(Function::exp(self.id))
    }

    /// Records the natural logarithm of this value on the same network
    /// and returns a proxy to it.
    pub fn ln(self) -> Self {
        self.apply(Function::ln(self.id))
    }
}

impl<'network, Data: Tensorial> Value<'network, Data> {
    /// Records the matrix product of this value and `rhs` on the same
    /// network and returns a proxy to it.
    ///
    /// # Panics
    /// Panics if the operands belong to different networks, either operand is
    /// not rank 2, or their inner dimensions differ.
    pub fn matmul(self, rhs: Self) -> Self {
        self.assert_same_network(&rhs);
        self.apply(Function::matmul(self.id, rhs.id))
    }

    /// Records the transposition of this value on the same network and
    /// returns a proxy to it.
    ///
    /// # Panics
    /// Panics if this value's rank exceeds 2.
    pub fn transposed(self) -> Self {
        self.apply(Function::transpose(self.id))
    }

    /// Records the sum of every value in this payload on the same network
    /// and returns a proxy to it.
    pub fn sum(self) -> Self {
        self.apply(Function::sum(self.id))
    }

    /// Records the sum of this value along `axis` on the same network
    /// and returns a proxy to it.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    pub fn sum_along(self, axis: usize) -> Self {
        self.apply(Function::sum_along(self.id, axis))
    }

    /// Records the explicit broadcast of this single-value payload across
    /// `reference`'s shape on the same network and returns a proxy to it.
    ///
    /// # Panics
    /// Panics if the values belong to different networks or this value's
    /// shape does not contain exactly one element.
    pub fn broadcast_like(self, reference: Self) -> Self {
        self.assert_same_network(&reference);
        self.apply(Function::broadcast(self.id, reference.id))
    }

    /// Records the explicit repetition of this value along `axis` of
    /// `reference`'s shape on the same network and returns a proxy to
    /// it; this value's shape must equal `reference`'s with that axis
    /// removed.
    ///
    /// # Panics
    /// Panics if the values belong to different networks, `axis` is out of
    /// `reference`'s rank, or the remaining shapes differ.
    pub fn broadcast_along(self, axis: usize, reference: Self) -> Self {
        self.assert_same_network(&reference);
        self.apply(Function::broadcast_along(self.id, reference.id, axis))
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

impl<'network, Data: Differentiable> Sub for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.assert_same_network(&rhs);
        self.apply(Function::sub(self.id, rhs.id))
    }
}

impl<'network, Data: Differentiable> Mul for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.assert_same_network(&rhs);
        self.apply(Function::mul(self.id, rhs.id))
    }
}

impl<'network, Data: Differentiable> Div for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn div(self, rhs: Self) -> Self::Output {
        self.assert_same_network(&rhs);
        self.apply(Function::div(self.id, rhs.id))
    }
}

impl<'network, Data: Differentiable> Neg for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn neg(self) -> Self::Output {
        self.apply(Function::neg(self.id))
    }
}

impl<'network, Data: Differentiable> Add<Data> for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn add(self, rhs: Data) -> Self::Output {
        let literal = self.literal(rhs);
        self + literal
    }
}

impl<'network, Data: Differentiable> Sub<Data> for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn sub(self, rhs: Data) -> Self::Output {
        let literal = self.literal(rhs);
        self - literal
    }
}

impl<'network, Data: Differentiable> Mul<Data> for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn mul(self, rhs: Data) -> Self::Output {
        let literal = self.literal(rhs);
        self * literal
    }
}

impl<'network, Data: Differentiable> Div<Data> for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn div(self, rhs: Data) -> Self::Output {
        let literal = self.literal(rhs);
        self / literal
    }
}

#[cfg(test)]
#[path = "tests/value_tests.rs"]
mod tests;
