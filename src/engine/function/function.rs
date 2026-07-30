use crate::engine::{SlotId, ValueId};
use crate::{Differentiable, Shape, Tensorial};

use static_assertions::assert_impl_all;

use super::{
    Add, Broadcast, BroadcastAlong, Div, Exp, Gather, Input, Leaf, Ln, MatMul, Mul, Narrow, Neg,
    Operation, Parameter, Permute, Reshape, Sub, Sum, SumAlong, Tanh, Transpose,
};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Function<f64>: Send, Sync);

/// The differentiable operation that produced a value, together with the
/// operation's parameters.
///
/// It is a statically sized closed set: each variant owns exactly its
/// parameters (a leaf's payload, a parameter's slot, a reduction's
/// axis), while the node's operand links live beside the node in the
/// tape's operands column and reach every method as a positional slice.
/// The enum dispatches to the variants with a plain `match`, avoiding
/// boxing and vtables.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Function<Data> {
    Leaf(Leaf<Data>),
    Parameter(Parameter),
    Input(Input),
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
    SumAlong(SumAlong),
    Broadcast(Broadcast),
    BroadcastAlong(BroadcastAlong),
    Reshape(Reshape),
    Permute(Permute),
    Narrow(Narrow),
    Gather(Gather),
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

    /// Creates an input function referencing `slot`.
    pub(crate) fn input(slot: SlotId) -> Self {
        Function::Input(Input(slot))
    }

    /// Creates the sum of the `[left, right]` operands.
    pub(crate) fn add() -> Self {
        Function::Add(Add)
    }

    /// Creates the difference of the `[left, right]` operands.
    pub(crate) fn sub() -> Self {
        Function::Sub(Sub)
    }

    /// Creates the product of the `[left, right]` operands.
    pub(crate) fn mul() -> Self {
        Function::Mul(Mul)
    }

    /// Creates the quotient of the `[left, right]` operands.
    pub(crate) fn div() -> Self {
        Function::Div(Div)
    }

    /// Creates the negation of the single operand.
    pub(crate) fn neg() -> Self {
        Function::Neg(Neg)
    }

    /// Creates the hyperbolic tangent of the single operand.
    pub(crate) fn tanh() -> Self {
        Function::Tanh(Tanh)
    }

    /// Creates the exponential of the single operand.
    pub(crate) fn exp() -> Self {
        Function::Exp(Exp)
    }

    /// Creates the natural logarithm of the single operand.
    pub(crate) fn ln() -> Self {
        Function::Ln(Ln)
    }

    /// Creates the matrix product of the `[left, right]` operands.
    pub(crate) fn matmul() -> Self {
        Function::MatMul(MatMul)
    }

    /// Creates the transposition of the single operand.
    pub(crate) fn transpose() -> Self {
        Function::Transpose(Transpose)
    }

    /// Creates the sum of every value in the single operand.
    pub(crate) fn sum() -> Self {
        Function::Sum(Sum)
    }

    /// Creates the sum of the single operand along `axis`.
    pub(crate) fn sum_along(axis: usize) -> Self {
        Function::SumAlong(SumAlong { axis })
    }

    /// Creates the explicit broadcast across the `[operand, like]`
    /// operands: the first spread across the second's shape.
    pub(crate) fn broadcast() -> Self {
        Function::Broadcast(Broadcast)
    }

    /// Creates the explicit repetition along `axis` for the
    /// `[operand, like]` operands: the first repeated along that axis of
    /// the second's shape.
    pub(crate) fn broadcast_along(axis: usize) -> Self {
        Function::BroadcastAlong(BroadcastAlong { axis })
    }

    /// Creates the reshape of the single operand to `shape`.
    pub(crate) fn reshape(shape: Shape) -> Self {
        Function::Reshape(Reshape { shape })
    }

    /// Creates the permutation of the single operand's axes by `order`.
    pub(crate) fn permute(order: impl IntoIterator<Item = usize>) -> Self {
        Function::Permute(Permute {
            order: order.into_iter().collect(),
        })
    }

    /// Creates the window of `len` elements from `start` along `axis` of
    /// the single operand.
    pub(crate) fn narrow(axis: usize, start: usize, len: usize) -> Self {
        Function::Narrow(Narrow { axis, start, len })
    }

    /// Creates the row gather over the `[table, selection]` operands: the
    /// table's rows picked by the one-hot selection.
    pub(crate) fn gather() -> Self {
        Function::Gather(Gather)
    }

    /// Infers the shape of this function's result from its `operands`'
    /// positional shapes, panicking on incompatibility.
    ///
    /// It is the shape-level mirror of `forward`: the same fold over the
    /// tape, run once per node at record time instead of once per run.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape
    where
        Data: Differentiable,
    {
        match self {
            Function::Leaf(leaf) => leaf.infer_shape(),
            Function::Parameter(_) => {
                unreachable!("parameter shapes are recorded by `record_parameter`")
            }
            Function::Input(_) => {
                unreachable!("input shapes are recorded by `record_input`")
            }
            Function::Add(add) => add.infer_shape(operands),
            Function::Sub(sub) => sub.infer_shape(operands),
            Function::Mul(mul) => mul.infer_shape(operands),
            Function::Div(div) => div.infer_shape(operands),
            Function::Neg(neg) => neg.infer_shape(operands),
            Function::Tanh(tanh) => tanh.infer_shape(operands),
            Function::Exp(exp) => exp.infer_shape(operands),
            Function::Ln(ln) => ln.infer_shape(operands),
            Function::MatMul(matmul) => matmul.infer_shape(operands),
            Function::Transpose(transpose) => transpose.infer_shape(operands),
            Function::Sum(sum) => sum.infer_shape(operands),
            Function::SumAlong(sum_along) => sum_along.infer_shape(operands),
            Function::Broadcast(broadcast) => broadcast.infer_shape(operands),
            Function::BroadcastAlong(broadcast_along) => broadcast_along.infer_shape(operands),
            Function::Reshape(reshape) => reshape.infer_shape(operands),
            Function::Permute(permute) => permute.infer_shape(operands),
            Function::Narrow(narrow) => narrow.infer_shape(operands),
            Function::Gather(gather) => gather.infer_shape(operands),
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
    /// Computes this node's payload from the values of the earlier nodes
    /// named by `operands`, or supplies it: a leaf's embedded payload, a
    /// parameter's entry in the run's `parameters` slots, or an input's
    /// entry in the run's `inputs` slots.
    pub(crate) fn forward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        parameters: &[Data],
        inputs: &[Data],
    ) -> Data {
        match self {
            Function::Leaf(leaf) => leaf.0.clone(),
            Function::Parameter(parameter) => parameters[parameter.0.index()].clone(),
            Function::Input(input) => inputs[input.0.index()].clone(),
            Function::Add(add) => add.forward(operands, values),
            Function::Sub(sub) => sub.forward(operands, values),
            Function::Mul(mul) => mul.forward(operands, values),
            Function::Div(div) => div.forward(operands, values),
            Function::Neg(neg) => neg.forward(operands, values),
            Function::Tanh(tanh) => tanh.forward(operands, values),
            Function::Exp(exp) => exp.forward(operands, values),
            Function::Ln(ln) => ln.forward(operands, values),
            Function::MatMul(matmul) => matmul.forward(operands, values),
            Function::Transpose(transpose) => transpose.forward(operands, values),
            Function::Sum(sum) => sum.forward(operands, values),
            Function::SumAlong(sum_along) => sum_along.forward(operands, values),
            Function::Broadcast(broadcast) => broadcast.forward(operands, values),
            Function::BroadcastAlong(broadcast_along) => broadcast_along.forward(operands, values),
            Function::Reshape(reshape) => reshape.forward(operands, values),
            Function::Permute(permute) => permute.forward(operands, values),
            Function::Narrow(narrow) => narrow.forward(operands, values),
            Function::Gather(gather) => gather.forward(operands, values),
        }
    }

    /// Accumulates gradients into the `operands`' slots, given this
    /// node's computed `output` payload and its own `gradient`; a no-op
    /// for leaves, parameters, and inputs, where gradients stop and get
    /// read out.
    pub(crate) fn backward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        match self {
            Function::Leaf(_) | Function::Parameter(_) | Function::Input(_) => {}
            Function::Add(add) => add.backward(operands, values, output, gradient, gradients),
            Function::Sub(sub) => sub.backward(operands, values, output, gradient, gradients),
            Function::Mul(mul) => mul.backward(operands, values, output, gradient, gradients),
            Function::Div(div) => div.backward(operands, values, output, gradient, gradients),
            Function::Neg(neg) => neg.backward(operands, values, output, gradient, gradients),
            Function::Tanh(tanh) => tanh.backward(operands, values, output, gradient, gradients),
            Function::Exp(exp) => exp.backward(operands, values, output, gradient, gradients),
            Function::Ln(ln) => ln.backward(operands, values, output, gradient, gradients),
            Function::MatMul(matmul) => {
                matmul.backward(operands, values, output, gradient, gradients)
            }
            Function::Transpose(transpose) => {
                transpose.backward(operands, values, output, gradient, gradients)
            }
            Function::Sum(sum) => sum.backward(operands, values, output, gradient, gradients),
            Function::SumAlong(sum_along) => {
                sum_along.backward(operands, values, output, gradient, gradients)
            }
            Function::Broadcast(broadcast) => {
                broadcast.backward(operands, values, output, gradient, gradients)
            }
            Function::BroadcastAlong(broadcast_along) => {
                broadcast_along.backward(operands, values, output, gradient, gradients)
            }
            Function::Reshape(reshape) => {
                reshape.backward(operands, values, output, gradient, gradients)
            }
            Function::Permute(permute) => {
                permute.backward(operands, values, output, gradient, gradients)
            }
            Function::Narrow(narrow) => {
                narrow.backward(operands, values, output, gradient, gradients)
            }
            Function::Gather(gather) => {
                gather.backward(operands, values, output, gradient, gradients)
            }
        }
    }
}
