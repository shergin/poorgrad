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

use crate::{Elementary, Tensorial};

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
}

#[cfg(test)]
#[path = "tests/composite_tests.rs"]
mod tests;
