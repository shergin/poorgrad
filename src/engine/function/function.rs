use crate::engine::SlotId;
use crate::{Differentiable, Shape, Tensorial};

use static_assertions::assert_impl_all;

use super::{
    Add, Broadcast, BroadcastAlong, Cotangents, Div, Exp, Fold, Gather, Input, Leaf, Ln,
    LogSoftmax, LogSumExp, MatMul, Maximum, Mul, Narrow, Neg, Operation, Pad, Parameter, Permute,
    Powf, Relu, Reshape, Retention, Scatter, Sqrt, Step, Sub, Sum, SumAlong, Tanh, Transpose,
    Unfold,
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
    Pad(Pad),
    Unfold(Unfold),
    Fold(Fold),
    Gather(Gather),
    Scatter(Scatter),
    LogSoftmax(LogSoftmax),
    LogSumExp(LogSumExp),
    Sqrt(Sqrt),
    Powf(Powf),
    Maximum(Maximum),
    Relu(Relu),
    Step(Step),
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

    /// Creates the single operand placed at `start ..` along `axis`
    /// inside zeros of `full_extent`.
    pub(crate) fn pad(axis: usize, start: usize, full_extent: usize) -> Self {
        Function::Pad(Pad {
            axis,
            start,
            full_extent,
        })
    }

    /// Creates the sliding windows of the single operand along `axis`:
    /// `size` elements every `dilation` steps, one window every `step`.
    pub(crate) fn unfold(axis: usize, size: usize, step: usize, dilation: usize) -> Self {
        Function::Unfold(Unfold {
            axis,
            size,
            step,
            dilation,
        })
    }

    /// Creates the row gather over the `[table, selection]` operands: the
    /// table's rows picked by the one-hot selection.
    pub(crate) fn gather() -> Self {
        Function::Gather(Gather)
    }

    /// Creates the log-softmax of the single operand along `axis`.
    pub(crate) fn log_softmax(axis: usize) -> Self {
        Function::LogSoftmax(LogSoftmax { axis })
    }

    /// Creates the fused log-sum-exp of the operand along `axis`.
    pub(crate) fn log_sum_exp(axis: usize) -> Self {
        Function::LogSumExp(LogSumExp { axis })
    }

    /// Creates the Heaviside step of `[operand, threshold]`.
    pub(crate) fn step() -> Self {
        Function::Step(Step)
    }

    /// Creates the fold of the operand's `(count, size)` pair at
    /// `axis` back onto an axis of `extent`.
    pub(crate) fn fold(
        axis: usize,
        size: usize,
        step: usize,
        dilation: usize,
        extent: usize,
    ) -> Self {
        Function::Fold(Fold {
            axis,
            size,
            step,
            dilation,
            extent,
        })
    }

    /// Creates the scatter-add of `[gradient, selection]` into `rows`
    /// rows.
    pub(crate) fn scatter(rows: usize) -> Self {
        Function::Scatter(Scatter { rows })
    }

    /// Creates the square root of the single operand.
    pub(crate) fn sqrt() -> Self {
        Function::Sqrt(Sqrt)
    }

    /// Creates the elementwise power of the `[base, exponent]` operands.
    pub(crate) fn powf() -> Self {
        Function::Powf(Powf)
    }

    /// Creates the elementwise maximum of the `[left, right]` operands.
    pub(crate) fn maximum() -> Self {
        Function::Maximum(Maximum)
    }

    /// Creates the rectified linear unit of the single operand.
    pub(crate) fn relu() -> Self {
        Function::Relu(Relu)
    }

    /// Returns the operation's display name, for plan introspection.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Function::Leaf(_) => "Leaf",
            Function::Parameter(_) => "Parameter",
            Function::Input(_) => "Input",
            Function::Add(_) => "Add",
            Function::Sub(_) => "Sub",
            Function::Mul(_) => "Mul",
            Function::Div(_) => "Div",
            Function::Neg(_) => "Neg",
            Function::Tanh(_) => "Tanh",
            Function::Exp(_) => "Exp",
            Function::Ln(_) => "Ln",
            Function::MatMul(_) => "MatMul",
            Function::Transpose(_) => "Transpose",
            Function::Sum(_) => "Sum",
            Function::SumAlong(_) => "SumAlong",
            Function::Broadcast(_) => "Broadcast",
            Function::BroadcastAlong(_) => "BroadcastAlong",
            Function::Reshape(_) => "Reshape",
            Function::Permute(_) => "Permute",
            Function::Narrow(_) => "Narrow",
            Function::Pad(_) => "Pad",
            Function::Unfold(_) => "Unfold",
            Function::Gather(_) => "Gather",
            Function::LogSoftmax(_) => "LogSoftmax",
            Function::LogSumExp(_) => "LogSumExp",
            Function::Step(_) => "Step",
            Function::Fold(_) => "Fold",
            Function::Scatter(_) => "Scatter",
            Function::Sqrt(_) => "Sqrt",
            Function::Powf(_) => "Powf",
            Function::Maximum(_) => "Maximum",
            Function::Relu(_) => "Relu",
        }
    }

    /// Returns whether this function is a source — a leaf, parameter,
    /// or input — supplied rather than computed, and therefore never a
    /// rematerialization candidate: recompute recursion bottoms out on
    /// sources.
    pub(crate) fn is_source(&self) -> bool {
        matches!(
            self,
            Function::Leaf(_) | Function::Parameter(_) | Function::Input(_)
        )
    }

    /// Returns which payload values this function's derivative rule
    /// reads: the retention contract behind training-plan liveness.
    /// Sources have no rule and retain nothing.
    pub(crate) fn retains(&self) -> Retention {
        match self {
            Function::Leaf(_) | Function::Parameter(_) | Function::Input(_) => Retention::NOTHING,
            Function::Add(add) => add.retains(),
            Function::Sub(sub) => sub.retains(),
            Function::Mul(mul) => mul.retains(),
            Function::Div(div) => div.retains(),
            Function::Neg(neg) => neg.retains(),
            Function::Tanh(tanh) => tanh.retains(),
            Function::Exp(exp) => exp.retains(),
            Function::Ln(ln) => ln.retains(),
            Function::MatMul(matmul) => matmul.retains(),
            Function::Transpose(transpose) => transpose.retains(),
            Function::Sum(sum) => sum.retains(),
            Function::SumAlong(sum_along) => sum_along.retains(),
            Function::Broadcast(broadcast) => broadcast.retains(),
            Function::BroadcastAlong(broadcast_along) => broadcast_along.retains(),
            Function::Reshape(reshape) => reshape.retains(),
            Function::Permute(permute) => permute.retains(),
            Function::Narrow(narrow) => narrow.retains(),
            Function::Pad(pad) => pad.retains(),
            Function::Unfold(unfold) => unfold.retains(),
            Function::Gather(gather) => gather.retains(),
            Function::LogSoftmax(log_softmax) => log_softmax.retains(),
            Function::LogSumExp(log_sum_exp) => log_sum_exp.retains(),
            Function::Step(step) => step.retains(),
            Function::Fold(fold) => fold.retains(),
            Function::Scatter(scatter) => scatter.retains(),
            Function::Sqrt(sqrt) => sqrt.retains(),
            Function::Powf(powf) => powf.retains(),
            Function::Maximum(maximum) => maximum.retains(),
            Function::Relu(relu) => relu.retains(),
        }
    }

    /// Returns the number of operand links this function expects.
    ///
    /// Sources have none; recording asserts every node's operand list
    /// against this count at the single append site.
    pub(crate) fn arity(&self) -> usize {
        match self {
            Function::Leaf(_) | Function::Parameter(_) | Function::Input(_) => 0,
            Function::Add(add) => add.arity(),
            Function::Sub(sub) => sub.arity(),
            Function::Mul(mul) => mul.arity(),
            Function::Div(div) => div.arity(),
            Function::Neg(neg) => neg.arity(),
            Function::Tanh(tanh) => tanh.arity(),
            Function::Exp(exp) => exp.arity(),
            Function::Ln(ln) => ln.arity(),
            Function::MatMul(matmul) => matmul.arity(),
            Function::Transpose(transpose) => transpose.arity(),
            Function::Sum(sum) => sum.arity(),
            Function::SumAlong(sum_along) => sum_along.arity(),
            Function::Broadcast(broadcast) => broadcast.arity(),
            Function::BroadcastAlong(broadcast_along) => broadcast_along.arity(),
            Function::Reshape(reshape) => reshape.arity(),
            Function::Permute(permute) => permute.arity(),
            Function::Narrow(narrow) => narrow.arity(),
            Function::Pad(pad) => pad.arity(),
            Function::Unfold(unfold) => unfold.arity(),
            Function::Gather(gather) => gather.arity(),
            Function::LogSoftmax(log_softmax) => log_softmax.arity(),
            Function::LogSumExp(log_sum_exp) => log_sum_exp.arity(),
            Function::Step(step) => step.arity(),
            Function::Fold(fold) => fold.arity(),
            Function::Scatter(scatter) => scatter.arity(),
            Function::Sqrt(sqrt) => sqrt.arity(),
            Function::Powf(powf) => powf.arity(),
            Function::Maximum(maximum) => maximum.arity(),
            Function::Relu(relu) => relu.arity(),
        }
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
            Function::Pad(pad) => pad.infer_shape(operands),
            Function::Unfold(unfold) => unfold.infer_shape(operands),
            Function::Gather(gather) => gather.infer_shape(operands),
            Function::LogSoftmax(log_softmax) => log_softmax.infer_shape(operands),
            Function::LogSumExp(log_sum_exp) => log_sum_exp.infer_shape(operands),
            Function::Step(step) => step.infer_shape(operands),
            Function::Fold(fold) => fold.infer_shape(operands),
            Function::Scatter(scatter) => scatter.infer_shape(operands),
            Function::Sqrt(sqrt) => sqrt.infer_shape(operands),
            Function::Powf(powf) => powf.infer_shape(operands),
            Function::Maximum(maximum) => maximum.infer_shape(operands),
            Function::Relu(relu) => relu.infer_shape(operands),
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
    /// Computes this node's payload from its `operands`' payloads
    /// (gathered positionally by the engine), or supplies it: a leaf's
    /// embedded payload, a parameter's entry in the run's `parameters`
    /// slots, or an input's entry in the run's `inputs` slots.
    pub(crate) fn forward(&self, operands: &[&Data], parameters: &[Data], inputs: &[Data]) -> Data {
        match self {
            Function::Leaf(leaf) => leaf.0.clone(),
            Function::Parameter(parameter) => parameters[parameter.0.index()].clone(),
            Function::Input(input) => inputs[input.0.index()].clone(),
            Function::Add(add) => add.forward(operands),
            Function::Sub(sub) => sub.forward(operands),
            Function::Mul(mul) => mul.forward(operands),
            Function::Div(div) => div.forward(operands),
            Function::Neg(neg) => neg.forward(operands),
            Function::Tanh(tanh) => tanh.forward(operands),
            Function::Exp(exp) => exp.forward(operands),
            Function::Ln(ln) => ln.forward(operands),
            Function::MatMul(matmul) => matmul.forward(operands),
            Function::Transpose(transpose) => transpose.forward(operands),
            Function::Sum(sum) => sum.forward(operands),
            Function::SumAlong(sum_along) => sum_along.forward(operands),
            Function::Broadcast(broadcast) => broadcast.forward(operands),
            Function::BroadcastAlong(broadcast_along) => broadcast_along.forward(operands),
            Function::Reshape(reshape) => reshape.forward(operands),
            Function::Permute(permute) => permute.forward(operands),
            Function::Narrow(narrow) => narrow.forward(operands),
            Function::Pad(pad) => pad.forward(operands),
            Function::Unfold(unfold) => unfold.forward(operands),
            Function::Gather(gather) => gather.forward(operands),
            Function::LogSoftmax(log_softmax) => log_softmax.forward(operands),
            Function::LogSumExp(log_sum_exp) => log_sum_exp.forward(operands),
            Function::Step(step) => step.forward(operands),
            Function::Fold(fold) => fold.forward(operands),
            Function::Scatter(scatter) => scatter.forward(operands),
            Function::Sqrt(sqrt) => sqrt.forward(operands),
            Function::Powf(powf) => powf.forward(operands),
            Function::Maximum(maximum) => maximum.forward(operands),
            Function::Relu(relu) => relu.forward(operands),
        }
    }

    /// Computes one cotangent per operand, given this node's computed
    /// `output` payload and its own `gradient`; empty for leaves,
    /// parameters, and inputs, where gradients stop and get read out.
    /// The engine accumulates the returned cotangents into its gradient
    /// buffer.
    ///
    /// The rule payload is a separate parameter from the tape's own
    /// `Data` because the rules are pure trait polymorphism: the engine
    /// applies them over payload buffers, and `differentiate` applies
    /// the very same rules over recording [`Trace`](super::Trace)
    /// handles — one source of derivative truth, two interpretations.
    pub(crate) fn backward<Rule: Tensorial>(
        &self,
        operands: &[&Rule],
        output: &Rule,
        gradient: &Rule,
    ) -> Cotangents<Rule> {
        match self {
            Function::Leaf(_) | Function::Parameter(_) | Function::Input(_) => Cotangents::new(),
            Function::Add(add) => add.backward(operands, output, gradient),
            Function::Sub(sub) => sub.backward(operands, output, gradient),
            Function::Mul(mul) => mul.backward(operands, output, gradient),
            Function::Div(div) => div.backward(operands, output, gradient),
            Function::Neg(neg) => neg.backward(operands, output, gradient),
            Function::Tanh(tanh) => tanh.backward(operands, output, gradient),
            Function::Exp(exp) => exp.backward(operands, output, gradient),
            Function::Ln(ln) => ln.backward(operands, output, gradient),
            Function::MatMul(matmul) => matmul.backward(operands, output, gradient),
            Function::Transpose(transpose) => transpose.backward(operands, output, gradient),
            Function::Sum(sum) => sum.backward(operands, output, gradient),
            Function::SumAlong(sum_along) => sum_along.backward(operands, output, gradient),
            Function::Broadcast(broadcast) => broadcast.backward(operands, output, gradient),
            Function::BroadcastAlong(broadcast_along) => {
                broadcast_along.backward(operands, output, gradient)
            }
            Function::Reshape(reshape) => reshape.backward(operands, output, gradient),
            Function::Permute(permute) => permute.backward(operands, output, gradient),
            Function::Narrow(narrow) => narrow.backward(operands, output, gradient),
            Function::Pad(pad) => pad.backward(operands, output, gradient),
            Function::Unfold(unfold) => unfold.backward(operands, output, gradient),
            Function::Gather(gather) => gather.backward(operands, output, gradient),
            Function::LogSoftmax(log_softmax) => log_softmax.backward(operands, output, gradient),
            Function::LogSumExp(log_sum_exp) => log_sum_exp.backward(operands, output, gradient),
            Function::Step(step) => step.backward(operands, output, gradient),
            Function::Fold(fold) => fold.backward(operands, output, gradient),
            Function::Scatter(scatter) => scatter.backward(operands, output, gradient),
            Function::Sqrt(sqrt) => sqrt.backward(operands, output, gradient),
            Function::Powf(powf) => powf.backward(operands, output, gradient),
            Function::Maximum(maximum) => maximum.backward(operands, output, gradient),
            Function::Relu(relu) => relu.backward(operands, output, gradient),
        }
    }
}
