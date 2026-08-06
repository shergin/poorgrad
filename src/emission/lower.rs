//! Lowering a compiled [`Plan`] to a textual StableHLO module.
//!
//! The unit of emission is the plan, not the tape: a plan is already a
//! closed, pure, statically shaped function whose parameters and inputs
//! are arguments and whose readable set is the result list, so emission
//! is writing that function down in the exchange dialect of the XLA
//! world. Every recorded operation lowers to primitive StableHLO ops in
//! plan order — near-1:1 for most of the op set, a short decomposition
//! for the fused `log_softmax`, and a `dot_general` for the one-hot
//! `gather` (the selection crosses the boundary as its dense one-hot
//! matrix, an ABI note rather than a semantic change). The interpreter
//! remains the semantic oracle: cross-boundary conformance is
//! envelope-based, never bitwise, because the target's reductions may
//! reassociate.

use std::error::Error;
use std::fmt::{self, Display, Write};

use crate::engine::Function;
use crate::{Plan, Tensor};

use super::builder::{
    Emittable, dense_index_literal, dense_literal, index_tensor_type, tensor_type,
};

/// Why a plan declined to emit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// The plan matched window-GEMM fusion groups, whose chains skip
    /// materialization; emitting them as the richer convolution op is
    /// the raising story, which has not landed.
    Fused {
        /// How many groups the plan matched.
        groups: usize,
    },
    /// A node's operation has no StableHLO lowering; reserved for
    /// future operations, since every current operation lowers.
    Unsupported {
        /// The node's plan index.
        node: usize,
        /// The operation's recorded name.
        operation: &'static str,
    },
}

impl Display for EmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmitError::Fused { groups } => write!(
                formatter,
                "the plan matched {groups} window-gemm fusion groups, which emission \
                 does not raise yet; compile the graph without a fusing posture"
            ),
            EmitError::Unsupported { node, operation } => write!(
                formatter,
                "node {node} records {operation}, which has no StableHLO lowering yet"
            ),
        }
    }
}

impl Error for EmitError {}

/// The running state of one emission: the SSA name of every lowered
/// node and the accumulated body text.
struct Emitter {
    names: Vec<Option<String>>,
    body: String,
}

impl Emitter {
    /// Returns the SSA name of operand `index`, which plan order
    /// guarantees was lowered before its consumer.
    fn name(&self, index: usize) -> &str {
        self.names[index]
            .as_deref()
            .expect("operands precede their consumers in plan order")
    }

    /// Writes one instruction line at function indentation.
    fn line(&mut self, rendered: String) {
        writeln!(self.body, "    {rendered}").expect("writing to a string cannot fail");
    }
}

impl<Element: Emittable> Plan<Tensor<Element>> {
    /// Serializes this plan as a textual StableHLO module: one
    /// `func.func @plan` whose arguments are the plan's parameters then
    /// its inputs, both in recording order, and whose results are the
    /// readable values in recording order. Leaves embed as constants.
    ///
    /// One-hot selections cross the boundary as their dense one-hot
    /// matrices — `gather` lowers to `dot_general` against the one-hot,
    /// so a fed selection input becomes a dense argument. The module is
    /// self-contained interchange text; parsing, bytecode serialization,
    /// and execution belong to toolchains outside the crate.
    ///
    /// # Errors
    /// Returns [`EmitError::Fused`] when the plan matched window-GEMM
    /// fusion groups; [`EmitError::Unsupported`] is reserved for future
    /// operations without lowerings.
    pub fn emit_stablehlo(&self) -> Result<String, EmitError> {
        let groups = self.fusion_groups();
        if groups > 0 {
            return Err(EmitError::Fused { groups });
        }

        let shapes = self.shapes();
        let wanted = self.wanted();
        let tensor = |index: usize| tensor_type::<Element>(&shapes[index]);

        // Arguments: parameters first, then inputs, in recording order.
        let mut emitter = Emitter {
            names: vec![None; self.len()],
            body: String::new(),
        };
        let mut arguments: Vec<String> = Vec::new();
        for pass in 0..2 {
            for (index, &wanted_node) in wanted.iter().enumerate() {
                if !wanted_node {
                    continue;
                }
                let function = self.functions().get(index).expect("plan columns are fixed");
                let argument = match (pass, function) {
                    (0, Function::Parameter(_)) | (1, Function::Input(_)) => {
                        format!("%arg{}", arguments.len())
                    }
                    _ => continue,
                };
                arguments.push(format!("{argument}: {}", tensor(index)));
                emitter.names[index] = Some(argument);
            }
        }

        for (index, &wanted_node) in wanted.iter().enumerate() {
            if !wanted_node || emitter.names[index].is_some() {
                continue;
            }
            self.lower(index, &mut emitter)?;
        }

        let mut results: Vec<(String, String)> = Vec::new();
        for index in 0..self.len() {
            if self.readable()[index] {
                results.push((emitter.name(index).to_string(), tensor(index)));
            }
        }
        let result_types: Vec<&str> = results.iter().map(|(_, kind)| kind.as_str()).collect();
        let result_names: Vec<&str> = results.iter().map(|(name, _)| name.as_str()).collect();

        let mut module = String::new();
        writeln!(module, "module @poorgrad {{").expect("writing to a string cannot fail");
        writeln!(
            module,
            "  func.func @plan({}) -> ({}) {{",
            arguments.join(", "),
            result_types.join(", "),
        )
        .expect("writing to a string cannot fail");
        module.push_str(&emitter.body);
        writeln!(
            module,
            "    return {} : {}",
            result_names.join(", "),
            result_types.join(", "),
        )
        .expect("writing to a string cannot fail");
        writeln!(module, "  }}").expect("writing to a string cannot fail");
        writeln!(module, "}}").expect("writing to a string cannot fail");
        Ok(module)
    }

    /// Lowers node `index` into `emitter`, naming its result.
    fn lower(&self, index: usize, emitter: &mut Emitter) -> Result<(), EmitError> {
        let shapes = self.shapes();
        let shape = &shapes[index];
        let result = format!("%v{index}");
        let result_type = tensor_type::<Element>(shape);
        let links = self.operands().get(index).expect("plan columns are fixed");
        let operand = |position: usize| links.as_slice()[position].index();
        let function = self.functions().get(index).expect("plan columns are fixed");

        // The elementwise families share one line shape each; everything
        // else renders its own syntax.
        let unary = |name: &str, emitter: &mut Emitter| {
            let source = emitter.name(operand(0)).to_string();
            emitter.line(format!(
                "{result} = stablehlo.{name} {source} : {result_type}"
            ));
        };
        let binary = |name: &str, emitter: &mut Emitter| {
            let left = emitter.name(operand(0)).to_string();
            let right = emitter.name(operand(1)).to_string();
            emitter.line(format!(
                "{result} = stablehlo.{name} {left}, {right} : {result_type}"
            ));
        };

        match function {
            Function::Leaf(leaf) => {
                let literal = dense_literal(shape, &leaf.0.to_vec());
                emitter.line(format!(
                    "{result} = stablehlo.constant {literal} : {result_type}"
                ));
            }
            Function::Add(_) => binary("add", emitter),
            Function::Sub(_) => binary("subtract", emitter),
            Function::Mul(_) => binary("multiply", emitter),
            Function::Div(_) => binary("divide", emitter),
            Function::Maximum(_) => binary("maximum", emitter),
            Function::Powf(_) => binary("power", emitter),
            Function::Neg(_) => unary("negate", emitter),
            Function::Tanh(_) => unary("tanh", emitter),
            Function::Exp(_) => unary("exponential", emitter),
            Function::Ln(_) => unary("log", emitter),
            Function::Sqrt(_) => unary("sqrt", emitter),
            Function::Relu(_) => {
                let zero = format!("%v{index}_zero");
                emitter.line(format!(
                    "{zero} = stablehlo.constant dense<{}> : {result_type}",
                    Element::ZERO
                ));
                let source = emitter.name(operand(0)).to_string();
                emitter.line(format!(
                    "{result} = stablehlo.maximum {source}, {zero} : {result_type}"
                ));
            }
            Function::MatMul(_) => {
                let left = operand(0);
                let right = operand(1);
                emitter.line(format!(
                    "{result} = stablehlo.dot_general {}, {}, contracting_dims = [1] x [0] \
                     : ({}, {}) -> {result_type}",
                    emitter.name(left),
                    emitter.name(right),
                    tensor_type::<Element>(&shapes[left]),
                    tensor_type::<Element>(&shapes[right]),
                ));
            }
            Function::Gather(_) => {
                // `output[i] = table[selection[i]]` over a one-hot
                // selection is exactly the one-hot times the table.
                let table = operand(0);
                let selection = operand(1);
                emitter.line(format!(
                    "{result} = stablehlo.dot_general {}, {}, contracting_dims = [1] x [0] \
                     : ({}, {}) -> {result_type}",
                    emitter.name(selection),
                    emitter.name(table),
                    tensor_type::<Element>(&shapes[selection]),
                    tensor_type::<Element>(&shapes[table]),
                ));
            }
            Function::Transpose(_) => {
                let source = operand(0);
                if shapes[source].rank() < 2 {
                    emitter.names[index] = Some(emitter.name(source).to_string());
                    return Ok(());
                }
                emitter.line(format!(
                    "{result} = stablehlo.transpose {}, dims = [1, 0] : ({}) -> {result_type}",
                    emitter.name(source),
                    tensor_type::<Element>(&shapes[source]),
                ));
            }
            Function::Permute(permute) => {
                let source = operand(0);
                emitter.line(format!(
                    "{result} = stablehlo.transpose {}, dims = {:?} : ({}) -> {result_type}",
                    emitter.name(source),
                    permute.order.as_slice(),
                    tensor_type::<Element>(&shapes[source]),
                ));
            }
            Function::Reshape(_) => {
                let source = operand(0);
                emitter.line(format!(
                    "{result} = stablehlo.reshape {} : ({}) -> {result_type}",
                    emitter.name(source),
                    tensor_type::<Element>(&shapes[source]),
                ));
            }
            Function::Sum(_) => {
                let source = operand(0);
                if shapes[source].rank() == 0 {
                    emitter.names[index] = Some(emitter.name(source).to_string());
                    return Ok(());
                }
                let axes: Vec<usize> = (0..shapes[source].rank()).collect();
                self.reduce(index, source, &axes, "add", Element::ZERO, emitter);
            }
            Function::SumAlong(along) => {
                self.reduce(
                    index,
                    operand(0),
                    &[along.axis],
                    "add",
                    Element::ZERO,
                    emitter,
                );
            }
            Function::Broadcast(_) => {
                // The reference operand contributes only its shape; the
                // single-element source flattens to a scalar and spreads.
                let source = operand(0);
                let mut spread = emitter.name(source).to_string();
                if shapes[source].rank() > 0 {
                    let flat = format!("%v{index}_scalar");
                    emitter.line(format!(
                        "{flat} = stablehlo.reshape {spread} : ({}) -> tensor<{}>",
                        tensor_type::<Element>(&shapes[source]),
                        Element::ELEMENT,
                    ));
                    spread = flat;
                }
                emitter.line(format!(
                    "{result} = stablehlo.broadcast_in_dim {spread}, dims = [] \
                     : (tensor<{}>) -> {result_type}",
                    Element::ELEMENT,
                ));
            }
            Function::BroadcastAlong(along) => {
                let source = operand(0);
                let dims: Vec<usize> = (0..shape.rank())
                    .filter(|&axis| axis != along.axis)
                    .collect();
                emitter.line(format!(
                    "{result} = stablehlo.broadcast_in_dim {}, dims = {dims:?} : ({}) -> {result_type}",
                    emitter.name(source),
                    tensor_type::<Element>(&shapes[source]),
                ));
            }
            Function::Narrow(narrow) => {
                let source = operand(0);
                let ranges: Vec<String> = shapes[source]
                    .axes()
                    .iter()
                    .enumerate()
                    .map(|(axis, &extent)| {
                        if axis == narrow.axis {
                            format!("{}:{}", narrow.start, narrow.start + narrow.len)
                        } else {
                            format!("0:{extent}")
                        }
                    })
                    .collect();
                emitter.line(format!(
                    "{result} = stablehlo.slice {} [{}] : ({}) -> {result_type}",
                    emitter.name(source),
                    ranges.join(", "),
                    tensor_type::<Element>(&shapes[source]),
                ));
            }
            Function::Pad(pad) => {
                let source = operand(0);
                let rank = shapes[source].rank();
                let mut low = vec![0usize; rank];
                let mut high = vec![0usize; rank];
                low[pad.axis] = pad.start;
                high[pad.axis] = pad.full_extent - pad.start - shapes[source].axes()[pad.axis];
                let zero = format!("%v{index}_zero");
                emitter.line(format!(
                    "{zero} = stablehlo.constant dense<{}> : tensor<{}>",
                    Element::ZERO,
                    Element::ELEMENT,
                ));
                emitter.line(format!(
                    "{result} = stablehlo.pad {}, {zero}, low = {low:?}, high = {high:?}, \
                     interior = {:?} : ({}, tensor<{}>) -> {result_type}",
                    emitter.name(source),
                    vec![0usize; rank],
                    tensor_type::<Element>(&shapes[source]),
                    Element::ELEMENT,
                ));
            }
            Function::LogSoftmax(softmax) => {
                self.lower_log_softmax(index, operand(0), softmax.axis, emitter);
            }
            Function::Unfold(unfold) => {
                // The completeness fallback the emission design names: the
                // windows' start coordinates bake into a constant and one
                // static gather reads them. Raising is the real path — a
                // canonical im2col or pooling chain should become
                // `convolution` or `reduce_window`, whose named kernels the
                // target holds — because this lowering materializes the
                // window view that fusion at home never materializes.
                // Emitted for closure of the op set, not for production.
                let source = operand(0);
                let source_shape = &shapes[source];
                let source_type = tensor_type::<Element>(source_shape);
                let source_name = emitter.name(source).to_string();
                let count = shape.axes()[unfold.axis];
                let size = shape.axes()[unfold.axis + 1];
                let coordinates: Vec<usize> = (0..count)
                    .flat_map(|window| {
                        (0..size)
                            .map(move |position| window * unfold.step + position * unfold.dilation)
                    })
                    .collect();
                let starts = format!("%v{index}_starts");
                let starts_type = index_tensor_type(&[count, size, 1]);
                emitter.line(format!(
                    "{starts} = stablehlo.constant {} : {starts_type}",
                    dense_index_literal(&[count, size, 1], &coordinates),
                ));
                // The two index batch dims land at the unfolded pair's
                // positions; every other output dim carries a slice dim in
                // order, with the gathered axis collapsed.
                let offset_dims: Vec<usize> = (0..source_shape.rank() + 1)
                    .filter(|&dim| dim != unfold.axis && dim != unfold.axis + 1)
                    .collect();
                let slice_sizes: Vec<String> = source_shape
                    .axes()
                    .iter()
                    .enumerate()
                    .map(|(dim, &extent)| {
                        if dim == unfold.axis {
                            "1".to_string()
                        } else {
                            extent.to_string()
                        }
                    })
                    .collect();
                emitter.line(format!(
                    "{result} = \"stablehlo.gather\"({source_name}, {starts}) \
                     <{{dimension_numbers = #stablehlo.gather<offset_dims = {offset_dims:?}, \
                     collapsed_slice_dims = [{axis}], start_index_map = [{axis}], \
                     index_vector_dim = 2>, indices_are_sorted = false, \
                     slice_sizes = array<i64: {sizes}>}}> \
                     : ({source_type}, {starts_type}) -> {result_type}",
                    axis = unfold.axis,
                    sizes = slice_sizes.join(", "),
                ));
            }
            Function::Parameter(_) | Function::Input(_) => {
                unreachable!("arguments are named before lowering")
            }
        }
        emitter.names[index] = Some(result);
        Ok(())
    }

    /// Writes the compact reduce of `source` over `axes` with the named
    /// reducer and its seed literal, producing node `index`'s value.
    fn reduce(
        &self,
        index: usize,
        source: usize,
        axes: &[usize],
        reducer: &str,
        seed: &str,
        emitter: &mut Emitter,
    ) {
        let seed_name = format!("%v{index}_seed");
        emitter.line(format!(
            "{seed_name} = stablehlo.constant dense<{seed}> : tensor<{}>",
            Element::ELEMENT,
        ));
        emitter.line(format!(
            "%v{index} = stablehlo.reduce({} init: {seed_name}) applies stablehlo.{reducer} \
             across dimensions = {axes:?} : ({}, tensor<{}>) -> {}",
            emitter.name(source),
            tensor_type::<Element>(&self.shapes()[source]),
            Element::ELEMENT,
            tensor_type::<Element>(&self.shapes()[index]),
        ));
    }

    /// Writes the fused `log_softmax` as its stable decomposition: shift
    /// by the axis maximum, exponentiate, normalize in the log domain.
    /// The target's rounding may differ from the fused interpreter rule,
    /// which conformance absorbs in its envelopes.
    fn lower_log_softmax(&self, index: usize, source: usize, axis: usize, emitter: &mut Emitter) {
        let shapes = self.shapes();
        let shape = &shapes[index];
        let reduced = shape.without_axis(axis);
        let reduced_type = tensor_type::<Element>(&reduced);
        let full_type = tensor_type::<Element>(shape);
        let dims: Vec<usize> = (0..shape.rank()).filter(|&a| a != axis).collect();
        let source_name = emitter.name(source).to_string();

        let seed = format!("%v{index}_low");
        emitter.line(format!(
            "{seed} = stablehlo.constant dense<{}> : tensor<{}>",
            Element::NEGATIVE_INFINITY,
            Element::ELEMENT,
        ));
        let peak = format!("%v{index}_peak");
        emitter.line(format!(
            "{peak} = stablehlo.reduce({source_name} init: {seed}) applies stablehlo.maximum \
             across dimensions = [{axis}] : ({full_type}, tensor<{}>) -> {reduced_type}",
            Element::ELEMENT,
        ));
        let spread_peak = format!("%v{index}_spread_peak");
        emitter.line(format!(
            "{spread_peak} = stablehlo.broadcast_in_dim {peak}, dims = {dims:?} \
             : ({reduced_type}) -> {full_type}",
        ));
        let centered = format!("%v{index}_centered");
        emitter.line(format!(
            "{centered} = stablehlo.subtract {source_name}, {spread_peak} : {full_type}"
        ));
        let exponentials = format!("%v{index}_exp");
        emitter.line(format!(
            "{exponentials} = stablehlo.exponential {centered} : {full_type}"
        ));
        let zero = format!("%v{index}_zero");
        emitter.line(format!(
            "{zero} = stablehlo.constant dense<{}> : tensor<{}>",
            Element::ZERO,
            Element::ELEMENT,
        ));
        let total = format!("%v{index}_total");
        emitter.line(format!(
            "{total} = stablehlo.reduce({exponentials} init: {zero}) applies stablehlo.add \
             across dimensions = [{axis}] : ({full_type}, tensor<{}>) -> {reduced_type}",
            Element::ELEMENT,
        ));
        let normalizer = format!("%v{index}_normalizer");
        emitter.line(format!(
            "{normalizer} = stablehlo.log {total} : {reduced_type}"
        ));
        let spread_normalizer = format!("%v{index}_spread_normalizer");
        emitter.line(format!(
            "{spread_normalizer} = stablehlo.broadcast_in_dim {normalizer}, dims = {dims:?} \
             : ({reduced_type}) -> {full_type}",
        ));
        emitter.line(format!(
            "%v{index} = stablehlo.subtract {centered}, {spread_normalizer} : {full_type}"
        ));
    }
}

#[cfg(test)]
#[path = "tests/lower_tests.rs"]
mod tests;
