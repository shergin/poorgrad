use crate::{Differentiable, Shape, SlotId, Tensorial, ValueId};

use static_assertions::assert_impl_all;

use super::{
    Add, Broadcast, Div, Exp, Leaf, Ln, MatMul, Mul, Neg, Operation, Parameter, Sub, Sum, Tanh,
    Transpose,
};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Function<f64>: Send, Sync);

/// The differentiable operation that produced a value, together with the
/// operation's parameters and operand links.
///
/// It is a statically sized closed set: each variant owns exactly its
/// operand links (`ValueId`s) and parameters (a leaf's payload, a
/// parameter's slot) as fixed fields; the enum dispatches to the
/// variants with a plain `match`, avoiding boxing and vtables.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Function<Data> {
    Leaf(Leaf<Data>),
    Parameter(Parameter),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    Neg(Neg),
    Tanh(Tanh),
    Exp(Exp),
    Ln(Ln),
    MatMul(MatMul),
    Transpose(Transpose),
    Sum(Sum),
    Broadcast(Broadcast),
}

impl<Data> Function<Data> {
    /// Creates a leaf function holding `data`.
    pub(crate) fn leaf(data: Data) -> Self {
        Function::Leaf(Leaf(data))
    }

    /// Creates a parameter function referencing `slot`.
    pub(crate) fn parameter(slot: SlotId) -> Self {
        Function::Parameter(Parameter(slot))
    }

    /// Creates the sum of `left` and `right`.
    pub(crate) fn add(left: ValueId, right: ValueId) -> Self {
        Function::Add(Add { left, right })
    }

    /// Creates the difference of `left` and `right`.
    pub(crate) fn sub(left: ValueId, right: ValueId) -> Self {
        Function::Sub(Sub { left, right })
    }

    /// Creates the product of `left` and `right`.
    pub(crate) fn mul(left: ValueId, right: ValueId) -> Self {
        Function::Mul(Mul { left, right })
    }

    /// Creates the quotient of `left` and `right`.
    pub(crate) fn div(left: ValueId, right: ValueId) -> Self {
        Function::Div(Div { left, right })
    }

    /// Creates the negation of `operand`.
    pub(crate) fn neg(operand: ValueId) -> Self {
        Function::Neg(Neg { operand })
    }

    /// Creates the hyperbolic tangent of `operand`.
    pub(crate) fn tanh(operand: ValueId) -> Self {
        Function::Tanh(Tanh { operand })
    }

    /// Creates the exponential of `operand`.
    pub(crate) fn exp(operand: ValueId) -> Self {
        Function::Exp(Exp { operand })
    }

    /// Creates the natural logarithm of `operand`.
    pub(crate) fn ln(operand: ValueId) -> Self {
        Function::Ln(Ln { operand })
    }

    /// Creates the matrix product of `left` and `right`.
    pub(crate) fn matmul(left: ValueId, right: ValueId) -> Self {
        Function::MatMul(MatMul { left, right })
    }

    /// Creates the transposition of `operand`.
    pub(crate) fn transpose(operand: ValueId) -> Self {
        Function::Transpose(Transpose { operand })
    }

    /// Creates the sum of every value in `operand`.
    pub(crate) fn sum(operand: ValueId) -> Self {
        Function::Sum(Sum { operand })
    }

    /// Creates the explicit broadcast of `operand` across `like`'s shape.
    pub(crate) fn broadcast(operand: ValueId, like: ValueId) -> Self {
        Function::Broadcast(Broadcast { operand, like })
    }

    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, visitor: impl FnMut(ValueId)) {
        match self {
            Function::Leaf(leaf) => leaf.visit_operands(visitor),
            Function::Parameter(parameter) => parameter.visit_operands(visitor),
            Function::Add(add) => add.visit_operands(visitor),
            Function::Sub(sub) => sub.visit_operands(visitor),
            Function::Mul(mul) => mul.visit_operands(visitor),
            Function::Div(div) => div.visit_operands(visitor),
            Function::Neg(neg) => neg.visit_operands(visitor),
            Function::Tanh(tanh) => tanh.visit_operands(visitor),
            Function::Exp(exp) => exp.visit_operands(visitor),
            Function::Ln(ln) => ln.visit_operands(visitor),
            Function::MatMul(matmul) => matmul.visit_operands(visitor),
            Function::Transpose(transpose) => transpose.visit_operands(visitor),
            Function::Sum(sum) => sum.visit_operands(visitor),
            Function::Broadcast(broadcast) => broadcast.visit_operands(visitor),
        }
    }

    /// Infers the shape of this function's result from its operands'
    /// shapes, panicking on incompatibility.
    ///
    /// It is the shape-level mirror of `forward`: the same fold over the
    /// tape, run once per node at record time instead of once per run.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape
    where
        Data: Differentiable,
    {
        match self {
            Function::Leaf(leaf) => leaf.inferred_shape(),
            Function::Parameter(_) => {
                unreachable!("parameter shapes are recorded by `record_parameter`")
            }
            Function::Add(add) => add.inferred_shape(shape_of),
            Function::Sub(sub) => sub.inferred_shape(shape_of),
            Function::Mul(mul) => mul.inferred_shape(shape_of),
            Function::Div(div) => div.inferred_shape(shape_of),
            Function::Neg(neg) => neg.inferred_shape(shape_of),
            Function::Tanh(tanh) => tanh.inferred_shape(shape_of),
            Function::Exp(exp) => exp.inferred_shape(shape_of),
            Function::Ln(ln) => ln.inferred_shape(shape_of),
            Function::MatMul(matmul) => matmul.inferred_shape(shape_of),
            Function::Transpose(transpose) => transpose.inferred_shape(shape_of),
            Function::Sum(sum) => sum.inferred_shape(shape_of),
            Function::Broadcast(broadcast) => broadcast.inferred_shape(shape_of),
        }
    }
}

/// It hand-rolls the delegation an enum-dispatch macro would generate: a
/// plain `match` per method. Exhaustiveness makes adding a variant a
/// compile error until every method handles it. Leaves and parameters
/// are supplied here rather than computed: they do not implement
/// `Operation`, whose contract is computing a payload from operands.
/// The bound is `Tensorial` rather than `Differentiable` because the
/// transcendental and tensor-native variants need it; building and
/// updating graphs stays arithmetic-only.
impl<Data: Tensorial> Function<Data> {
    /// Computes this node's payload from the values of earlier nodes, or
    /// supplies it: a leaf's embedded payload, or a parameter's entry in
    /// the run's `parameters` slots.
    pub(crate) fn forward(&self, values: &[Data], parameters: &[Data]) -> Data {
        match self {
            Function::Leaf(leaf) => leaf.0.clone(),
            Function::Parameter(parameter) => parameters[parameter.0.index()].clone(),
            Function::Add(add) => add.forward(values),
            Function::Sub(sub) => sub.forward(values),
            Function::Mul(mul) => mul.forward(values),
            Function::Div(div) => div.forward(values),
            Function::Neg(neg) => neg.forward(values),
            Function::Tanh(tanh) => tanh.forward(values),
            Function::Exp(exp) => exp.forward(values),
            Function::Ln(ln) => ln.forward(values),
            Function::MatMul(matmul) => matmul.forward(values),
            Function::Transpose(transpose) => transpose.forward(values),
            Function::Sum(sum) => sum.forward(values),
            Function::Broadcast(broadcast) => broadcast.forward(values),
        }
    }

    /// Accumulates operand gradients, given this node's computed
    /// `output` payload and its own `gradient`; a no-op for leaves and
    /// parameters, where gradients stop and get read out.
    pub(crate) fn backward(
        &self,
        values: &[Data],
        output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        match self {
            Function::Leaf(_) | Function::Parameter(_) => {}
            Function::Add(add) => add.backward(values, output, gradient, gradients),
            Function::Sub(sub) => sub.backward(values, output, gradient, gradients),
            Function::Mul(mul) => mul.backward(values, output, gradient, gradients),
            Function::Div(div) => div.backward(values, output, gradient, gradients),
            Function::Neg(neg) => neg.backward(values, output, gradient, gradients),
            Function::Tanh(tanh) => tanh.backward(values, output, gradient, gradients),
            Function::Exp(exp) => exp.backward(values, output, gradient, gradients),
            Function::Ln(ln) => ln.backward(values, output, gradient, gradients),
            Function::MatMul(matmul) => matmul.backward(values, output, gradient, gradients),
            Function::Transpose(transpose) => {
                transpose.backward(values, output, gradient, gradients)
            }
            Function::Sum(sum) => sum.backward(values, output, gradient, gradients),
            Function::Broadcast(broadcast) => {
                broadcast.backward(values, output, gradient, gradients)
            }
        }
    }
}
