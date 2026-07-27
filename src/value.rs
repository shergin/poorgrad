use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::ptr;

use static_assertions::assert_impl_all;

use super::{Differentiable, Elementary, Function, Shape, Symbol, Tape, Tensorial};

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
/// usable for further expressions. Payload literals mix directly into
/// expressions (`x * 2.0` on scalar networks, tensor literals on tensor
/// networks, in either order); each literal appearance records its own
/// fresh leaf on the same network. Accessors such as `data` briefly take
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

    /// Returns a clone of the leaf or parameter payload, or `None` for
    /// computed values.
    ///
    /// For a parameter it reads the current generation's store, so the
    /// same symbol resolved in different generations reads different
    /// payloads.
    pub fn data(&self) -> Option<Data> {
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
    pub fn matmul(self, rhs: Self) -> Self {
        self.assert_same_network(&rhs);
        self.apply(Function::matmul(self.id, rhs.id))
    }

    /// Records the transposition of this value on the same network and
    /// returns a proxy to it.
    pub fn transposed(self) -> Self {
        self.apply(Function::transpose(self.id))
    }

    /// Records the sum of every value in this payload on the same network
    /// and returns a proxy to it.
    pub fn sum(self) -> Self {
        self.apply(Function::sum(self.id))
    }

    /// Records the explicit broadcast of this single-value payload across
    /// `reference`'s shape on the same network and returns a proxy to it.
    pub fn broadcast_like(self, reference: Self) -> Self {
        self.assert_same_network(&reference);
        self.apply(Function::broadcast(self.id, reference.id))
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

// Coherence forbids the generic reverse (`impl Mul<Value<Data>> for Data`
// leaves the `Data` parameter uncovered), so the foreign scalar payloads
// get concrete implementations instead; `Tensor`, being a local type,
// gets its own generic ones in `tensor.rs`.
macro_rules! literal_operand_for {
    ($($payload:ty),*) => {$(
        impl<'network> Add<Value<'network, $payload>> for $payload {
            type Output = Value<'network, $payload>;

            fn add(self, rhs: Value<'network, $payload>) -> Self::Output {
                rhs.literal(self) + rhs
            }
        }

        impl<'network> Sub<Value<'network, $payload>> for $payload {
            type Output = Value<'network, $payload>;

            fn sub(self, rhs: Value<'network, $payload>) -> Self::Output {
                rhs.literal(self) - rhs
            }
        }

        impl<'network> Mul<Value<'network, $payload>> for $payload {
            type Output = Value<'network, $payload>;

            fn mul(self, rhs: Value<'network, $payload>) -> Self::Output {
                rhs.literal(self) * rhs
            }
        }

        impl<'network> Div<Value<'network, $payload>> for $payload {
            type Output = Value<'network, $payload>;

            fn div(self, rhs: Value<'network, $payload>) -> Self::Output {
                rhs.literal(self) / rhs
            }
        }
    )*};
}

literal_operand_for!(f32, f64);
