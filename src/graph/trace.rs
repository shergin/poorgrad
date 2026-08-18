use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{Differentiable, Elementary, Shape, Tensorial};

use super::Value;

/// A payload that records instead of computing: the second
/// interpretation of the derivative rules.
///
/// Every `Operation::backward` is written against the payload traits,
/// not against a concrete number type. `Trace` implements those traits
/// by appending the corresponding node to the tape and answering with
/// a handle, so running a rule with `Data = Trace` emits the rule's
/// computation as recorded graph — which is all
/// [`Tape::differentiate`](super::Tape::differentiate) does. The
/// rules cannot tell the difference, and that indistinguishability is
/// the design: derivative knowledge lives in exactly one place, and a
/// rule change reaches the engine's backward and the recorded gradient
/// alike, because both are the same code.
///
/// Trait members no derivative rule calls (`counted`, `max_along`)
/// panic with a named message; the per-variant closure tests run every
/// rule under `Trace`, so a future rule widening the vocabulary fails
/// its own test at introduction instead of hiding a latent trap.
pub(crate) struct Trace<'tape, Data> {
    value: Value<'tape, Data>,
}

impl<'tape, Data: Differentiable> Trace<'tape, Data> {
    /// Wraps a recorded value as a rule operand.
    pub(crate) fn of(value: Value<'tape, Data>) -> Self {
        Self { value }
    }

    /// Returns the recorded value this trace stands for.
    pub(crate) fn value(&self) -> Value<'tape, Data> {
        self.value
    }

    /// Records a literal leaf of `count` spread across this trace's
    /// recorded shape: the recording twin of `zero_like`/`one_like`.
    fn counted_like(&self, count: usize) -> Self {
        Self::of(self.value.literal(Data::counted(self.value.shape(), count)))
    }
}

// Manual implementations avoid the `Data: Clone`/`Data: Copy` bounds a
// derive would demand; a trace is an index pair like the value it wraps.
impl<Data> Clone for Trace<'_, Data> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Data> Copy for Trace<'_, Data> {}

impl<Data> fmt::Debug for Trace<'_, Data> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Trace").finish_non_exhaustive()
    }
}

impl<'tape, Data: Differentiable> Add for Trace<'tape, Data> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::of(self.value + rhs.value)
    }
}

impl<'tape, Data: Differentiable> Sub for Trace<'tape, Data> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::of(self.value - rhs.value)
    }
}

impl<'tape, Data: Differentiable> Mul for Trace<'tape, Data> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::of(self.value * rhs.value)
    }
}

impl<'tape, Data: Differentiable> Div for Trace<'tape, Data> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::of(self.value / rhs.value)
    }
}

impl<'tape, Data: Differentiable> Neg for Trace<'tape, Data> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::of(-self.value)
    }
}

impl<'tape, Data: Tensorial> Differentiable for Trace<'tape, Data> {
    /// A trace accumulates in itself: promotion would hide recorded
    /// arithmetic, and the underlying payload's accumulator already
    /// acts inside each recorded operation.
    type Accumulator = Self;

    fn promote(&self) -> Self {
        *self
    }

    fn demote(accumulated: Self) -> Self {
        accumulated
    }

    fn zero_like(&self) -> Self {
        self.counted_like(0)
    }

    fn one_like(&self) -> Self {
        self.counted_like(1)
    }

    fn counted(_shape: Shape, _count: usize) -> Self {
        // No derivative rule mints counted literals (verified by the
        // closure tests); a nullary constructor has no network to
        // record on, so this member cannot exist for a trace.
        panic!("`Trace` records derivative rules, which never call `counted`");
    }

    fn shape(&self) -> Shape {
        self.value.shape()
    }
}

impl<'tape, Data: Tensorial> Elementary for Trace<'tape, Data> {
    fn exp(&self) -> Self {
        Self::of(self.value.exp())
    }

    fn ln(&self) -> Self {
        Self::of(self.value.ln())
    }

    fn sqrt(&self) -> Self {
        Self::of(self.value.sqrt())
    }

    fn tanh(&self) -> Self {
        Self::of(self.value.tanh())
    }

    fn powf(&self, exponent: Self) -> Self {
        Self::of(self.value.powf(exponent.value))
    }

    fn maximum(&self, other: &Self) -> Self {
        Self::of(self.value.maximum(other.value))
    }

    fn step(&self, threshold: &Self) -> Self {
        Self::of(self.value.step(threshold.value))
    }
}

impl<'tape, Data: Tensorial> Tensorial for Trace<'tape, Data> {
    fn matmul(&self, rhs: &Self) -> Self {
        Self::of(self.value.matmul(rhs.value))
    }

    fn transpose(&self) -> Self {
        Self::of(self.value.transpose())
    }

    fn sum(&self) -> Self {
        Self::of(self.value.sum())
    }

    fn sum_along(&self, axis: usize) -> Self {
        Self::of(self.value.sum_along(axis))
    }

    fn max_along(&self, _axis: usize) -> Self {
        // The differentiable rules never reduce by maximum (the fused
        // log-domain rules recover probabilities from their outputs
        // instead), and no recorded operation exists for it.
        panic!("`Trace` records derivative rules, which never call `max_along`");
    }

    fn broadcast_like(&self, reference: &Self) -> Self {
        Self::of(self.value.broadcast_like(reference.value))
    }

    fn broadcast_along(&self, axis: usize, reference: &Self) -> Self {
        Self::of(self.value.broadcast_along(axis, reference.value))
    }

    fn reshape(&self, shape: Shape) -> Self {
        Self::of(self.value.reshape(shape))
    }

    fn permute(&self, order: &[usize]) -> Self {
        Self::of(self.value.permute(order.iter().copied()))
    }

    fn narrow(&self, axis: usize, start: usize, len: usize) -> Self {
        Self::of(self.value.narrow(axis, start, len))
    }

    fn pad(&self, axis: usize, start: usize, full_extent: usize) -> Self {
        Self::of(self.value.pad(axis, start, full_extent))
    }

    fn unfold(&self, axis: usize, size: usize, step: usize, dilation: usize) -> Self {
        Self::of(self.value.unfold(axis, size, step, dilation))
    }

    fn fold(&self, axis: usize, size: usize, step: usize, dilation: usize, extent: usize) -> Self {
        Self::of(self.value.fold(axis, size, step, dilation, extent))
    }

    fn gather(&self, selection: &Self) -> Self {
        Self::of(self.value.gather(selection.value))
    }

    fn scatter(&self, selection: &Self, rows: usize) -> Self {
        Self::of(self.value.scatter(selection.value, rows))
    }
}
