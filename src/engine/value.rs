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
/// The methods in this file are opcode mnemonics: each records exactly one
/// computed node, one per `Function` variant (payload literals additionally
/// record a leaf, which is data injection rather than computation). Methods
/// that expand to several computed nodes are composites and live in the
/// composition tier of `composite.rs`.
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

    /// Returns the operand links of this value's node.
    #[cfg(test)]
    pub(crate) fn operands(&self) -> Vec<ValueId> {
        self.tape.operands_of(self.id).as_slice().to_vec()
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

    /// Records a computed node produced by `function` over the positional
    /// `operands` on the same network and returns a proxy to it.
    fn apply(&self, function: Function<Data>, operands: &[ValueId]) -> Self {
        let id = self.tape.record(function, operands);
        Self::bind(self.tape, id)
    }

    /// Records `data` as a fresh leaf on the same network and returns a
    /// proxy to it.
    ///
    /// It backs the payload-literal operator sugar: every literal
    /// appearance records its own leaf.
    pub(crate) fn literal(&self, data: Data) -> Self {
        Self::bind(self.tape, self.tape.record(Function::leaf(data), &[]))
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
        self.apply(Function::tanh(), &[self.id])
    }

    /// Records the exponential of this value on the same network and
    /// returns a proxy to it.
    pub fn exp(self) -> Self {
        self.apply(Function::exp(), &[self.id])
    }

    /// Records the natural logarithm of this value on the same network
    /// and returns a proxy to it.
    pub fn ln(self) -> Self {
        self.apply(Function::ln(), &[self.id])
    }

    /// Records the square root of this value on the same network and
    /// returns a proxy to it.
    pub fn sqrt(self) -> Self {
        self.apply(Function::sqrt(), &[self.id])
    }

    /// Records this value raised elementwise to the power of `exponent`
    /// on the same network and returns a proxy to it.
    ///
    /// The exponent-side gradient involves the logarithm of this value,
    /// so it is a number only where this value is positive.
    ///
    /// # Panics
    /// Panics if the operands belong to different networks or their
    /// shapes differ.
    pub fn powf(self, exponent: Self) -> Self {
        self.assert_same_network(&exponent);
        self.apply(Function::powf(), &[self.id, exponent.id])
    }

    /// Records the elementwise maximum of this value and `rhs` on the
    /// same network and returns a proxy to it; on a tie the gradient goes
    /// to this value, not `rhs`.
    ///
    /// # Panics
    /// Panics if the operands belong to different networks or their
    /// shapes differ.
    pub fn maximum(self, rhs: Self) -> Self {
        self.assert_same_network(&rhs);
        self.apply(Function::maximum(), &[self.id, rhs.id])
    }

    /// Records the rectified linear unit of this value — its elementwise
    /// maximum with zero — on the same network and returns a proxy to it;
    /// the subgradient at zero is one.
    pub fn relu(self) -> Self {
        self.apply(Function::relu(), &[self.id])
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
        self.apply(Function::matmul(), &[self.id, rhs.id])
    }

    /// Records the transposition of this value on the same network and
    /// returns a proxy to it.
    ///
    /// # Panics
    /// Panics if this value's rank exceeds 2.
    pub fn transpose(self) -> Self {
        self.apply(Function::transpose(), &[self.id])
    }

    /// Records the sum of every value in this payload on the same network
    /// and returns a proxy to it.
    pub fn sum(self) -> Self {
        self.apply(Function::sum(), &[self.id])
    }

    /// Records the sum of this value along `axis` on the same network
    /// and returns a proxy to it.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    pub fn sum_along(self, axis: usize) -> Self {
        self.apply(Function::sum_along(axis), &[self.id])
    }

    /// Records the explicit broadcast of this single-value payload across
    /// `reference`'s shape on the same network and returns a proxy to it.
    ///
    /// This is the narrowest expansion opcode: the operand must hold
    /// exactly one element, and the target shape always comes from a
    /// reference value, never from an alignment rule. For a source of any
    /// broadcastable shape, use the composite
    /// [`broadcast_to`](Self::broadcast_to), which applies the
    /// right-aligned NumPy rule over this opcode and
    /// [`broadcast_along`](Self::broadcast_along).
    ///
    /// # Panics
    /// Panics if the values belong to different networks or this value's
    /// shape does not contain exactly one element.
    pub fn broadcast_like(self, reference: Self) -> Self {
        self.assert_same_network(&reference);
        self.apply(Function::broadcast(), &[self.id, reference.id])
    }

    /// Records the explicit repetition of this value along `axis` of
    /// `reference`'s shape on the same network and returns a proxy to
    /// it; this value's shape must equal `reference`'s with that axis
    /// removed.
    ///
    /// This opcode widens exactly one named axis and never infers an
    /// alignment. To widen several axes at once, or to expand under the
    /// right-aligned NumPy rule, use the composite
    /// [`broadcast_to`](Self::broadcast_to).
    ///
    /// # Panics
    /// Panics if the values belong to different networks, `axis` is out of
    /// `reference`'s rank, or the remaining shapes differ.
    pub fn broadcast_along(self, axis: usize, reference: Self) -> Self {
        self.assert_same_network(&reference);
        self.apply(Function::broadcast_along(axis), &[self.id, reference.id])
    }

    /// Records a reshape of this value to `shape` on the same network and
    /// returns a proxy to it; the elements keep their logical row-major
    /// order.
    ///
    /// # Panics
    /// Panics if `shape`'s volume differs from this value's.
    pub fn reshape(self, shape: impl Into<Shape>) -> Self {
        self.apply(Function::reshape(shape.into()), &[self.id])
    }

    /// Records a permutation of this value's axes by `order` on the same
    /// network and returns a proxy to it; axis `i` of the result takes
    /// axis `order[i]` of this value.
    ///
    /// # Panics
    /// Panics if `order` is not a permutation of `0..rank`.
    pub fn permute(self, order: impl IntoIterator<Item = usize>) -> Self {
        self.apply(Function::permute(order), &[self.id])
    }

    /// Records this value with a new extent-1 axis inserted at `axis`: a
    /// reshape that leaves the elements unchanged.
    ///
    /// # Panics
    /// Panics if `axis` exceeds this value's rank.
    pub fn unsqueeze(self, axis: usize) -> Self {
        let mut axes: Vec<usize> = self.shape().axes().to_vec();
        assert!(axis <= axes.len(), "unsqueeze axis {axis} is out of rank");
        axes.insert(axis, 1);
        self.reshape(axes)
    }

    /// Records this value with the extent-1 axis at `axis` removed: a
    /// reshape that leaves the elements unchanged.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank or that axis is not extent 1.
    pub fn squeeze(self, axis: usize) -> Self {
        let mut axes: Vec<usize> = self.shape().axes().to_vec();
        assert!(axis < axes.len(), "squeeze axis {axis} is out of rank");
        assert_eq!(axes[axis], 1, "squeeze requires an extent-1 axis");
        axes.remove(axis);
        self.reshape(axes)
    }

    /// Records the window of `len` elements from `start` along `axis` on
    /// the same network and returns a proxy to it; the forward is an O(1)
    /// view and the gradient scatters back into the unselected positions
    /// as zeros.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank, `len` is zero (tensors cannot be
    /// empty), or `start + len` overflows or exceeds the axis extent.
    pub fn narrow(self, axis: usize, start: usize, len: usize) -> Self {
        self.apply(Function::narrow(axis, start, len), &[self.id])
    }

    /// Records this value placed at `start ..` along `axis` inside zeros
    /// whose `axis` has extent `full_extent`, on the same network, and
    /// returns a proxy to it: the adjoint of [`Value::narrow`], with
    /// `narrow` as its own gradient rule.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank or the window overflows or
    /// exceeds `full_extent`.
    pub fn pad(self, axis: usize, start: usize, full_extent: usize) -> Self {
        self.apply(Function::pad(axis, start, full_extent), &[self.id])
    }

    /// Records the sliding windows of this value along `axis` on the
    /// same network and returns a proxy to it: the axis becomes a
    /// `(count, size)` pair where window `w` starts at `w * step` and
    /// takes every `dilation`-th element. The forward is a strided view;
    /// the gradient folds every window contribution back onto its
    /// source position, so overlapping windows accumulate.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank, `size`, `step`, or `dilation` is
    /// zero, or the dilated window span `dilation * (size - 1) + 1`
    /// overflows or exceeds the axis extent.
    pub fn unfold(self, axis: usize, size: usize, step: usize, dilation: usize) -> Self {
        self.apply(Function::unfold(axis, size, step, dilation), &[self.id])
    }

    /// Records the row gather of this value (the table) by `selection`, a
    /// one-hot `[count, vocab]` whose vocabulary matches the table's first
    /// axis: `output[i]` is the table row `selection` names for position
    /// `i`. The gradient scatter-adds into the table only; the selection is
    /// data and receives no gradient.
    ///
    /// It is the embedding lookup: feed `selection` per run, so one graph
    /// serves any batch of indices.
    ///
    /// # Panics
    /// Panics if the values belong to different networks, `selection` is not
    /// rank 2, or its vocabulary does not match this value's first axis.
    pub fn gather(self, selection: Self) -> Self {
        self.assert_same_network(&selection);
        self.apply(Function::gather(), &[self.id, selection.id])
    }

    /// Records the log-softmax of this value along `axis` on the same
    /// network and returns a proxy to it: the logarithm of the softmax
    /// probabilities, computed stably in one fused node.
    ///
    /// Exponentiating the result recovers the probabilities themselves; the
    /// fused form exists because the stable computation shifts by the axis
    /// maximum, which no composition of recorded operations can express.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    pub fn log_softmax(self, axis: usize) -> Self {
        self.apply(Function::log_softmax(axis), &[self.id])
    }

    /// Records the log-sum-exp of this value along `axis` on the same
    /// network and returns a proxy to it: the softmax family's normalizer
    /// and a smooth maximum; like `sum_along`, the reduced axis is
    /// removed.
    ///
    /// It is a fused node for the same reason as
    /// [`log_softmax`](Value::log_softmax): the stable form shifts by the
    /// axis maximum, so the result is finite for every finite operand —
    /// where the former composition over `log_softmax` returned `inf`
    /// once finite logits differed by more than the representable range.
    /// The gradient is the softmax.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    pub fn logsumexp(self, axis: usize) -> Self {
        self.apply(Function::log_sum_exp(axis), &[self.id])
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
        self.apply(Function::add(), &[self.id, rhs.id])
    }
}

impl<'network, Data: Differentiable> Sub for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.assert_same_network(&rhs);
        self.apply(Function::sub(), &[self.id, rhs.id])
    }
}

impl<'network, Data: Differentiable> Mul for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.assert_same_network(&rhs);
        self.apply(Function::mul(), &[self.id, rhs.id])
    }
}

impl<'network, Data: Differentiable> Div for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn div(self, rhs: Self) -> Self::Output {
        self.assert_same_network(&rhs);
        self.apply(Function::div(), &[self.id, rhs.id])
    }
}

impl<'network, Data: Differentiable> Neg for Value<'network, Data> {
    type Output = Value<'network, Data>;

    fn neg(self) -> Self::Output {
        self.apply(Function::neg(), &[self.id])
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
