//! Composite expressions over values: the second tier of the operation
//! surface.
//!
//! The first tier is `value.rs`, where every method is an opcode mnemonic
//! recording exactly one computed node. Each method here expands to a
//! formula over those opcodes — several computed nodes whose gradient the
//! chain rule pays with no dedicated backward rule. Everything in this
//! file compiles against the public operation surface alone: composites
//! need no privileged access to the engine, and once recorded they are
//! indistinguishable from hand-written primitives, so the tape stays a
//! uniform IR. The third tier is named formulas whose operands play
//! distinct roles (a loss's logits and targets); those are free functions
//! in domain modules such as the loss module.
//!
//! A formula belongs here only while composition expresses it faithfully;
//! it earns a `Function` variant the moment floating point breaks the
//! composed form, the way `log_softmax` did.

use crate::{Elementary, Shape, Tensorial};

use super::Value;

impl<'network, Data: Elementary> Value<'network, Data> {
    /// Records the absolute value of this value as the composition
    /// `self.maximum(-self)` and returns a proxy to it; the subgradient
    /// at zero is one, by `maximum`'s left-biased tie rule.
    pub fn abs(self) -> Self {
        self.maximum(-self)
    }
}

impl<'network, Data: Tensorial> Value<'network, Data> {
    /// Records the softmax probabilities of this value along `axis` as
    /// the composition `self.log_softmax(axis).exp()` and returns a proxy
    /// to it.
    ///
    /// Stability is inherited from the fused core: log-probabilities are
    /// at most zero, so the exponential cannot overflow — which is why
    /// softmax needs no fused form of its own.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    pub fn softmax(self, axis: usize) -> Self {
        self.log_softmax(axis).exp()
    }

    /// Records the log-sum-exp of this value along `axis` — the softmax
    /// family's normalizer and a smooth maximum — and returns a proxy to
    /// it; like `sum_along`, the reduced axis is removed.
    ///
    /// It is composed as `self - self.log_softmax(axis)`, which equals
    /// the normalizer at every position along the axis, narrowed to one
    /// lane. The composed gradient works out to exactly the softmax, the
    /// known derivative of log-sum-exp.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    pub fn logsumexp(self, axis: usize) -> Self {
        (self - self.log_softmax(axis))
            .narrow(axis, 0, 1)
            .squeeze(axis)
    }

    /// Records the mean of this value along `axis` as the composition
    /// `self.sum_along(axis) / extent`, where the reduced axis's extent
    /// enters the graph as a [`counted`](crate::Differentiable::counted)
    /// literal; like `sum_along`, the reduced axis is removed.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    pub fn mean_along(self, axis: usize) -> Self {
        let shape = self.shape();
        assert!(axis < shape.rank(), "mean_along axis {axis} is out of rank");
        let extent = shape.axes()[axis];
        self.sum_along(axis) / Data::counted(shape.without_axis(axis), extent)
    }

    /// Records this value broadcast to `shape` under the right-aligned
    /// NumPy and TensorFlow rule, and returns a proxy to it.
    ///
    /// The two shapes align from the trailing axis: the target's rank must
    /// be at least this value's, and each source axis must either match its
    /// aligned target axis or have extent one, in which case it is repeated
    /// to the target extent. It composes the shape-changing primitives -- a
    /// right-aligning `reshape` that prepends the missing leading axes, then
    /// one `broadcast_along` per repeated axis, or a single `broadcast_like`
    /// when the source holds one element -- so the gradient is the chain
    /// rule over their adjoints: the incoming gradient summed back over
    /// every repeated axis.
    ///
    /// # Panics
    /// Panics if `shape`'s rank is smaller than this value's, or a source
    /// axis neither matches its aligned target axis nor has extent one.
    pub fn broadcast_to(self, shape: impl Into<Shape>) -> Self {
        let target = shape.into();
        let source = self.shape();
        if source == target {
            return self;
        }
        assert!(
            target.rank() >= source.rank(),
            "broadcast to {target} from {source} lowers the rank"
        );
        let offset = target.rank() - source.rank();
        for (axis, &extent) in source.axes().iter().enumerate() {
            let aligned = target.axes()[offset + axis];
            assert!(
                extent == aligned || extent == 1,
                "broadcast to {target} from {source} cannot align source axis \
                 {axis} of extent {extent} to extent {aligned}"
            );
        }
        // A single-element source reaches any shape in one node; the
        // reference operand carries only the target shape.
        if source.volume() == 1 {
            let reference = self.literal(Data::counted(target, 0));
            return self.broadcast_like(reference);
        }
        // Right-align the source under the target by prepending unit axes, so
        // every axis is then either already matched or an extent-one axis to
        // repeat.
        let mut current = if offset == 0 {
            self
        } else {
            let mut axes = vec![1; offset];
            axes.extend_from_slice(source.axes());
            self.reshape(axes)
        };
        for axis in 0..target.rank() {
            let aligned = target.axes()[axis];
            if current.shape().axes()[axis] == aligned {
                continue;
            }
            // The only remaining mismatch is an extent-one axis; drop it and
            // repeat it to the target extent through the axis-wise adjoint,
            // whose reference is the current shape with this axis widened.
            let mut axes = current.shape().axes().to_vec();
            axes[axis] = aligned;
            let reference = self.literal(Data::counted(Shape::new(axes), 0));
            current = current.squeeze(axis).broadcast_along(axis, reference);
        }
        current
    }

    /// Records both values broadcast to their common shape under the
    /// right-aligned NumPy and TensorFlow rule, and returns the two proxies
    /// in the operand order.
    ///
    /// The common shape takes the larger extent on every axis after trailing
    /// alignment; a value already at that shape is returned unchanged. It is
    /// the ergonomic entry for elementwise operations over unequal shapes:
    /// `let (left, right) = left.broadcast_pair(right)` yields operands that
    /// `add`, `mul`, and the other strict elementwise ops accept directly.
    ///
    /// # Panics
    /// Panics if the values belong to different networks or their shapes do
    /// not broadcast against each other.
    pub fn broadcast_pair(self, other: Self) -> (Self, Self) {
        let common = broadcasted_shape(&self.shape(), &other.shape());
        (
            self.broadcast_to(common.clone()),
            other.broadcast_to(common),
        )
    }
}

/// Records the concatenation of `values` along `axis` and returns a proxy
/// to it: each value is padded with zeros to the combined extent at its
/// running offset, and the pads are summed.
///
/// This is the designed route for sequence stacking and head
/// concatenation; a dedicated variadic opcode earns its node only if the
/// zero-padded intermediates ever measure. The gradient of each operand
/// is the incoming gradient narrowed back to its own window, through
/// `pad`'s adjoint.
///
/// # Panics
/// Panics if `values` is empty, the values belong to different networks,
/// `axis` is out of rank, or the shapes disagree anywhere but `axis`.
pub fn concat<'network, Data: Tensorial>(
    values: &[Value<'network, Data>],
    axis: usize,
) -> Value<'network, Data> {
    let first = values.first().expect("concat requires at least one value");
    let reference = first.shape();
    assert!(
        axis < reference.rank(),
        "concat axis {axis} is out of rank for {reference}"
    );
    for value in &values[1..] {
        let shape = value.shape();
        assert_eq!(
            shape.without_axis(axis),
            reference.without_axis(axis),
            "concat along axis {axis} requires equal shapes off the axis, \
             got {shape} against {reference}"
        );
    }
    if values.len() == 1 {
        return *first;
    }
    let combined: usize = values.iter().map(|value| value.shape().axes()[axis]).sum();
    let mut offset = 0;
    let mut total: Option<Value<'network, Data>> = None;
    for &value in values {
        let padded = value.pad(axis, offset, combined);
        offset += value.shape().axes()[axis];
        total = Some(match total {
            Some(sum) => sum + padded,
            None => padded,
        });
    }
    total.expect("concat combines at least one value")
}

/// Records the stacking of `values` along a new axis at `axis` and returns
/// a proxy to it: each value gains an extent-1 axis there (`unsqueeze`)
/// and the lifted values concatenate.
///
/// # Panics
/// Panics if `values` is empty, the values belong to different networks,
/// `axis` exceeds the values' rank, or the shapes differ.
pub fn stack<'network, Data: Tensorial>(
    values: &[Value<'network, Data>],
    axis: usize,
) -> Value<'network, Data> {
    let lifted: Vec<Value<'network, Data>> =
        values.iter().map(|&value| value.unsqueeze(axis)).collect();
    concat(&lifted, axis)
}

/// Returns the shape two operands broadcast to under the right-aligned rule:
/// the larger extent on every axis after aligning both from the trailing
/// axis, where a missing leading axis counts as extent one.
///
/// # Panics
/// Panics if an aligned axis pair differs with neither extent one.
fn broadcasted_shape(left: &Shape, right: &Shape) -> Shape {
    let rank = left.rank().max(right.rank());
    let mut axes = Vec::with_capacity(rank);
    for offset in 0..rank {
        let left_extent = extent_from_end(left, offset);
        let right_extent = extent_from_end(right, offset);
        assert!(
            left_extent == right_extent || left_extent == 1 || right_extent == 1,
            "broadcast of {left} and {right} cannot align extents \
             {left_extent} and {right_extent}"
        );
        axes.push(left_extent.max(right_extent));
    }
    axes.reverse();
    Shape::new(axes)
}

/// Returns the extent `offset` axes in from the trailing axis of `shape`, or
/// one when the offset reaches past the leading axis.
fn extent_from_end(shape: &Shape, offset: usize) -> usize {
    let rank = shape.rank();
    if offset < rank {
        shape.axes()[rank - 1 - offset]
    } else {
        1
    }
}

#[cfg(test)]
#[path = "tests/composite_tests.rs"]
mod tests;
