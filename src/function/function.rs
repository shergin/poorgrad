use crate::{Differentiable, ValueId};

use static_assertions::assert_impl_all;

use super::{Add, Leaf, Mul, Neg, Operation, Parameter};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Function<f64>: Send, Sync);

/// The differentiable operation that produced a value, together with the
/// operation's parameters and operand links.
///
/// It is a statically sized closed set: each variant owns exactly its
/// operand links (`ValueId`s) and parameters (such as a leaf's payload) as
/// fixed fields, and implements `Operation`; the enum dispatches to the
/// variants with a plain `match`, avoiding boxing and vtables.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Function<Data> {
    Leaf(Leaf<Data>),
    Parameter(Parameter<Data>),
    Add(Add),
    Mul(Mul),
    Neg(Neg),
}

impl<Data> Function<Data> {
    /// Creates a leaf function holding `data`.
    pub(crate) fn leaf(data: Data) -> Self {
        Function::Leaf(Leaf(data))
    }

    /// Creates a parameter function holding `data`.
    pub(crate) fn parameter(data: Data) -> Self {
        Function::Parameter(Parameter(data))
    }

    /// Creates the sum of `left` and `right`.
    pub(crate) fn add(left: ValueId, right: ValueId) -> Self {
        Function::Add(Add { left, right })
    }

    /// Creates the product of `left` and `right`.
    pub(crate) fn mul(left: ValueId, right: ValueId) -> Self {
        Function::Mul(Mul { left, right })
    }

    /// Creates the negation of `operand`.
    pub(crate) fn neg(operand: ValueId) -> Self {
        Function::Neg(Neg { operand })
    }

    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, visitor: impl FnMut(ValueId)) {
        match self {
            Function::Leaf(leaf) => leaf.visit_operands(visitor),
            Function::Parameter(parameter) => parameter.visit_operands(visitor),
            Function::Add(add) => add.visit_operands(visitor),
            Function::Mul(mul) => mul.visit_operands(visitor),
            Function::Neg(neg) => neg.visit_operands(visitor),
        }
    }

    /// Returns the payload of a payload-carrying variant (a leaf or a
    /// parameter), or `None` for computed variants.
    pub(crate) fn data(&self) -> Option<&Data> {
        match self {
            Function::Leaf(leaf) => Some(&leaf.0),
            Function::Parameter(parameter) => Some(&parameter.0),
            _ => None,
        }
    }

    /// Returns the parameter payload, or `None` for any other variant.
    pub(crate) fn parameter_data(&self) -> Option<&Data> {
        match self {
            Function::Parameter(parameter) => Some(&parameter.0),
            _ => None,
        }
    }
}

/// It hand-rolls the delegation an enum-dispatch macro would generate: a
/// plain `match` per method. Exhaustiveness makes adding a variant a
/// compile error until every method handles it.
impl<Data: Differentiable> Operation<Data> for Function<Data> {
    fn forward(&self, values: &[Data]) -> Data {
        match self {
            Function::Leaf(leaf) => leaf.forward(values),
            Function::Parameter(parameter) => parameter.forward(values),
            Function::Add(add) => add.forward(values),
            Function::Mul(mul) => mul.forward(values),
            Function::Neg(neg) => neg.forward(values),
        }
    }

    fn backward(&self, values: &[Data], gradient: &Data, gradients: &mut [Data]) {
        match self {
            Function::Leaf(leaf) => leaf.backward(values, gradient, gradients),
            Function::Parameter(parameter) => parameter.backward(values, gradient, gradients),
            Function::Add(add) => add.backward(values, gradient, gradients),
            Function::Mul(mul) => mul.backward(values, gradient, gradients),
            Function::Neg(neg) => neg.backward(values, gradient, gradients),
        }
    }
}
