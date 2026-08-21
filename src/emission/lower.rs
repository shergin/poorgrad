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

use crate::engine::{BatchNormalization, Catalog, Pattern, ReduceWindow, WindowProduct};
use crate::function::Function;
use crate::{Backend, Plan, Shape, Tensor};

use super::builder::{
    Emittable, dense_index_literal, dense_literal, index_tensor_type, named_tensor_type,
    pred_tensor_type, tensor_type,
};

/// Why a plan declined to emit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
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
    /// `func.func @main` whose arguments are the plan's parameters then
    /// its inputs, both in recording order, and whose results are the
    /// readable values in recording order. Leaves embed as constants.
    ///
    /// One-hot selections cross the boundary as their dense one-hot
    /// matrices — `gather` lowers to `dot_general` against the one-hot,
    /// so a fed selection input becomes a dense argument. The module is
    /// self-contained interchange text; parsing, bytecode serialization,
    /// and execution belong to toolchains outside the crate.
    ///
    /// Elected catalog groups raise: a window-GEMM group's matmul
    /// emits `stablehlo.convolution` from the source and kernel with
    /// the im2col chain never crossing the boundary, a max-pool
    /// window group emits `stablehlo.reduce_window` over its source,
    /// and a batch-normalization formula emits
    /// `stablehlo.batch_norm_training` / `batch_norm_inference` —
    /// the pattern library's second life, recovering the richer named
    /// operations the target holds library kernels for. Emission
    /// elects from the plan's candidate pool with a total repertoire,
    /// so raising happens on every memory posture: an engine-backward
    /// plan emits the same module as its forward twin.
    ///
    /// # Errors
    /// [`EmitError::Unsupported`] is reserved for future operations
    /// without lowerings; every current operation lowers.
    pub fn emit_stablehlo(&self) -> Result<String, EmitError> {
        let shapes = self.shapes();
        let wanted = self.wanted();
        let tensor = |index: usize| tensor_type::<Element>(&shapes[index]);

        // The emission consumer elects its catalog from the plan's
        // candidate pool by reading the `StableHlo` implementer's
        // coverage column, which is total — every pattern raises
        // here, on every memory posture — and stays total by test.
        let catalog = Catalog::elect(self.candidates(), |pattern| {
            Backend::StableHlo.coverage(pattern.formula()).serves()
        });

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
            // An elected entry's interior is replaced wholesale by the
            // operation raised at its group's root, exactly as runs
            // replace home interiors with the fused call.
            if !wanted_node || catalog.interior(index) || emitter.names[index].is_some() {
                continue;
            }
            self.lower(index, &catalog, &mut emitter)?;
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
        writeln!(module, "module @topos {{").expect("writing to a string cannot fail");
        writeln!(
            module,
            "  func.func @main({}) -> ({}) {{",
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

    /// Lowers node `index` into `emitter`, naming its result. A node
    /// carrying an elected catalog entry raises to its named operation
    /// instead of lowering primitives; the emitter never rematches.
    fn lower(
        &self,
        index: usize,
        catalog: &Catalog,
        emitter: &mut Emitter,
    ) -> Result<(), EmitError> {
        if let Some(pattern) = catalog.at(index) {
            match pattern {
                Pattern::WindowProduct(group) => {
                    self.raise_convolution(index, group, emitter);
                    return Ok(());
                }
                Pattern::ReduceWindow(group) => {
                    self.raise_reduce_window(index, group, emitter);
                    return Ok(());
                }
                Pattern::BatchNormTraining(group) => {
                    self.raise_batch_norm_training(index, group, emitter);
                    return Ok(());
                }
                Pattern::BatchNormInference(group) => {
                    self.raise_batch_norm_inference(index, group, emitter);
                    return Ok(());
                }
            }
        }
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
                // Rank two contracts plainly; the batched ranks name
                // every leading axis a batching dimension on both
                // sides — `dot_general` carries them natively.
                let rank = shapes[left].rank();
                let dims = if rank > 2 {
                    let batching: Vec<String> =
                        (0..rank - 2).map(|axis| axis.to_string()).collect();
                    format!(
                        "batching_dims = [{batching}] x [{batching}], \
                         contracting_dims = [{}] x [{}]",
                        rank - 1,
                        rank - 2,
                        batching = batching.join(", "),
                    )
                } else {
                    "contracting_dims = [1] x [0]".to_string()
                };
                // An element with a declared accumulation type states
                // it in the IR: the dot produces the wider result type
                // and converts back, exactly what the home `gemm` seam
                // computes.
                if let Some(accumulation) = Element::ACCUMULATION {
                    let accumulated = format!("%v{index}_accumulated");
                    let accumulated_type = named_tensor_type(shape, accumulation);
                    emitter.line(format!(
                        "{accumulated} = stablehlo.dot_general {}, {}, \
                         {dims} : ({}, {}) -> {accumulated_type}",
                        emitter.name(left),
                        emitter.name(right),
                        tensor_type::<Element>(&shapes[left]),
                        tensor_type::<Element>(&shapes[right]),
                    ));
                    emitter.line(format!(
                        "{result} = stablehlo.convert {accumulated} \
                         : ({accumulated_type}) -> {result_type}"
                    ));
                } else {
                    emitter.line(format!(
                        "{result} = stablehlo.dot_general {}, {}, {dims} \
                         : ({}, {}) -> {result_type}",
                        emitter.name(left),
                        emitter.name(right),
                        tensor_type::<Element>(&shapes[left]),
                        tensor_type::<Element>(&shapes[right]),
                    ));
                }
            }
            Function::Gather(_) => {
                // `output[i] = table[selection[i]]` over a one-hot
                // selection is exactly the one-hot times the table.
                // No accumulation form is needed: each output sums a
                // single nonzero term, exact in the element type.
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
            Function::LogSumExp(log_sum_exp) => {
                self.lower_log_sum_exp(index, operand(0), log_sum_exp.axis, emitter);
            }
            Function::Step(_) => {
                // No step exists in StableHLO; the indicator is a
                // `GE` compare selecting between splat ones and zeros.
                let source = operand(0);
                let threshold = operand(1);
                let full_type = tensor_type::<Element>(&shapes[source]);
                let mask_type = pred_tensor_type(&shapes[source]);
                let mask = format!("%v{index}_mask");
                emitter.line(format!(
                    "{mask} = stablehlo.compare GE, {}, {}, FLOAT \
                     : ({full_type}, {full_type}) -> {mask_type}",
                    emitter.name(source),
                    emitter.name(threshold),
                ));
                let ones = format!("%v{index}_ones");
                emitter.line(format!(
                    "{ones} = stablehlo.constant dense<{}> : {result_type}",
                    Element::counted(Shape::scalar(), 1).literal(),
                ));
                let zeros = format!("%v{index}_zeros");
                emitter.line(format!(
                    "{zeros} = stablehlo.constant dense<{}> : {result_type}",
                    Element::ZERO,
                ));
                emitter.line(format!(
                    "{result} = stablehlo.select {mask}, {ones}, {zeros} \
                     : {mask_type}, {result_type}",
                ));
            }
            Function::Scatter(_) => {
                // The adjoint of the one-hot gather: the one-hot with
                // its count axis contracted against the gradient's,
                // leaving `[vocab, ...]` — a scatter-add expressed as
                // the dense product, free abroad where the selection
                // already crosses the boundary as its dense matrix.
                let gradient = operand(0);
                let selection = operand(1);
                // Duplicate rows genuinely accumulate, so the
                // contraction honors the declared accumulation type,
                // like `MatMul`.
                if let Some(accumulation) = Element::ACCUMULATION {
                    let accumulated = format!("%v{index}_accumulated");
                    let accumulated_type = named_tensor_type(shape, accumulation);
                    emitter.line(format!(
                        "{accumulated} = stablehlo.dot_general {}, {}, \
                         contracting_dims = [0] x [0] : ({}, {}) -> {accumulated_type}",
                        emitter.name(selection),
                        emitter.name(gradient),
                        tensor_type::<Element>(&shapes[selection]),
                        tensor_type::<Element>(&shapes[gradient]),
                    ));
                    emitter.line(format!(
                        "{result} = stablehlo.convert {accumulated} \
                         : ({accumulated_type}) -> {result_type}"
                    ));
                } else {
                    emitter.line(format!(
                        "{result} = stablehlo.dot_general {}, {}, contracting_dims = [0] x [0] \
                         : ({}, {}) -> {result_type}",
                        emitter.name(selection),
                        emitter.name(gradient),
                        tensor_type::<Element>(&shapes[selection]),
                        tensor_type::<Element>(&shapes[gradient]),
                    ));
                }
            }
            Function::Fold(fold) => {
                // Fold is a linear map on the window pair, so it lowers
                // as one contraction against a constant 0/1 window
                // matrix `[count, size, extent]` marking which source
                // position each window element folds onto — the static
                // dual of `unfold`'s gather fallback — followed by a
                // transpose that returns the folded axis from the
                // contraction's trailing position to its own.
                let source = operand(0);
                let source_shape = &shapes[source];
                let count = source_shape.axes()[fold.axis];
                let weights_shape = Shape::new([count, fold.size, fold.extent]);
                let one = Element::counted(Shape::scalar(), 1);
                let zero = Element::counted(Shape::scalar(), 0);
                let mut weights = vec![zero; count * fold.size * fold.extent];
                for window in 0..count {
                    for position in 0..fold.size {
                        let target = window * fold.step + position * fold.dilation;
                        weights[(window * fold.size + position) * fold.extent + target] =
                            one.clone();
                    }
                }
                let weights_name = format!("%v{index}_weights");
                let weights_type = tensor_type::<Element>(&weights_shape);
                emitter.line(format!(
                    "{weights_name} = stablehlo.constant {} : {weights_type}",
                    dense_literal(&weights_shape, &weights),
                ));
                let joined_axes: Vec<usize> = source_shape
                    .axes()
                    .iter()
                    .enumerate()
                    .filter(|&(dim, _)| dim != fold.axis && dim != fold.axis + 1)
                    .map(|(_, &extent)| extent)
                    .chain(std::iter::once(fold.extent))
                    .collect();
                let joined_shape = Shape::new(joined_axes);
                let trailing = joined_shape.rank() - 1;
                let joined_name = if fold.axis == trailing {
                    format!("%v{index}")
                } else {
                    format!("%v{index}_joined")
                };
                // Overlapping windows genuinely accumulate, so the
                // contraction honors the declared accumulation type,
                // like `MatMul`.
                if let Some(accumulation) = Element::ACCUMULATION {
                    let accumulated = format!("%v{index}_accumulated");
                    let accumulated_type = named_tensor_type(&joined_shape, accumulation);
                    emitter.line(format!(
                        "{accumulated} = stablehlo.dot_general {}, {weights_name}, \
                         contracting_dims = [{}, {}] x [0, 1] \
                         : ({}, {weights_type}) -> {accumulated_type}",
                        emitter.name(source),
                        fold.axis,
                        fold.axis + 1,
                        tensor_type::<Element>(source_shape),
                    ));
                    emitter.line(format!(
                        "{joined_name} = stablehlo.convert {accumulated} \
                         : ({accumulated_type}) -> {}",
                        tensor_type::<Element>(&joined_shape),
                    ));
                } else {
                    emitter.line(format!(
                        "{joined_name} = stablehlo.dot_general {}, {weights_name}, \
                         contracting_dims = [{}, {}] x [0, 1] : ({}, {weights_type}) -> {}",
                        emitter.name(source),
                        fold.axis,
                        fold.axis + 1,
                        tensor_type::<Element>(source_shape),
                        tensor_type::<Element>(&joined_shape),
                    ));
                }
                if fold.axis != trailing {
                    let order: Vec<usize> = (0..joined_shape.rank())
                        .map(|dim| {
                            if dim < fold.axis {
                                dim
                            } else if dim == fold.axis {
                                trailing
                            } else {
                                dim - 1
                            }
                        })
                        .collect();
                    emitter.line(format!(
                        "{result} = stablehlo.transpose {joined_name}, dims = {order:?} \
                         : ({}) -> {result_type}",
                        tensor_type::<Element>(&joined_shape),
                    ));
                }
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
                     {{dimension_numbers = #stablehlo.gather<offset_dims = {offset_dims:?}, \
                     collapsed_slice_dims = [{axis}], start_index_map = [{axis}], \
                     index_vector_dim = 2>, indices_are_sorted = false, \
                     slice_sizes = array<i64: {sizes}>}} \
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

    /// Writes the raise of one window-GEMM fusion group: the flat GEMM
    /// kernel reshapes back to `[channels, height, width, filters]`,
    /// one `stablehlo.convolution` reads the rank-4 source directly
    /// (the folded symmetric pads ride as window padding), and the
    /// `[batch, out_h, out_w, filters]` result flattens to the group's
    /// matmul shape, so downstream nodes see exactly the recorded form.
    fn raise_convolution(&self, index: usize, group: &WindowProduct, emitter: &mut Emitter) {
        let shapes = self.shapes();
        let source_axes = shapes[group.source].axes();
        let (batch, channels, height, width) = (
            source_axes[0],
            source_axes[1],
            source_axes[2],
            source_axes[3],
        );
        let filters = shapes[group.kernel].axes()[1];
        let out_height = (height + 2 * group.padding - group.kernel_height) / group.stride + 1;
        let out_width = (width + 2 * group.padding - group.kernel_width) / group.stride + 1;
        assert_eq!(
            shapes[index].axes(),
            [batch * out_height * out_width, filters],
            "the fused matmul's shape disagrees with the group's geometry"
        );

        let kernel = format!("%v{index}_kernel");
        let kernel_type = tensor_type::<Element>(&Shape::new([
            channels,
            group.kernel_height,
            group.kernel_width,
            filters,
        ]));
        emitter.line(format!(
            "{kernel} = stablehlo.reshape {} : ({}) -> {kernel_type}",
            emitter.name(group.kernel),
            tensor_type::<Element>(&shapes[group.kernel]),
        ));
        let windows_shape = Shape::new([batch, out_height, out_width, filters]);
        let windows = format!("%v{index}_windows");
        let windows_type = tensor_type::<Element>(&windows_shape);
        // The convolution is the fused matmul, so it carries the same
        // declared accumulation type as `MatMul` and converts back —
        // the home fused executor computes through the same gemm seam.
        let convolved = match Element::ACCUMULATION {
            Some(_) => format!("%v{index}_accumulated"),
            None => windows.clone(),
        };
        let convolved_type = match Element::ACCUMULATION {
            Some(accumulation) => named_tensor_type(&windows_shape, accumulation),
            None => windows_type.clone(),
        };
        emitter.line(format!(
            "{convolved} = stablehlo.convolution({}, {kernel}) \
             dim_numbers = [b, f, 0, 1]x[i, 0, 1, o]->[b, 0, 1, f], \
             window = {{stride = [{stride}, {stride}], \
             pad = [[{pad}, {pad}], [{pad}, {pad}]]}} \
             {{batch_group_count = 1 : i64, feature_group_count = 1 : i64}} \
             : ({}, {kernel_type}) -> {convolved_type}",
            emitter.name(group.source),
            tensor_type::<Element>(&shapes[group.source]),
            stride = group.stride,
            pad = group.padding,
        ));
        if Element::ACCUMULATION.is_some() {
            emitter.line(format!(
                "{windows} = stablehlo.convert {convolved} \
                 : ({convolved_type}) -> {windows_type}"
            ));
        }
        emitter.line(format!(
            "%v{index} = stablehlo.reshape {windows} : ({windows_type}) -> {}",
            tensor_type::<Element>(&shapes[index]),
        ));
        emitter.names[index] = Some(format!("%v{index}"));
    }

    /// Writes the raise of one max-pool window group: a single
    /// `stablehlo.reduce_window` with a `maximum` region reads the
    /// rank-4 source directly, so the unfolded lanes and the recorded
    /// fold never cross the boundary. The window and strides ride the
    /// spatial axes only; v1 pools carry no padding. Tie-breaking is
    /// value-identical (`maximum` over a window is order-free on
    /// values), and gradients never cross: the raise serves forward
    /// plans, whose runs execute the recorded fold at home.
    fn raise_reduce_window(&self, index: usize, group: &ReduceWindow, emitter: &mut Emitter) {
        let shapes = self.shapes();
        let source_axes = shapes[group.source].axes();
        let out_height = (source_axes[2] - group.size) / group.stride + 1;
        let out_width = (source_axes[3] - group.size) / group.stride + 1;
        assert_eq!(
            shapes[index].axes(),
            [source_axes[0], source_axes[1], out_height, out_width],
            "the pooled root's shape disagrees with the group's geometry"
        );

        let seed = format!("%v{index}_seed");
        emitter.line(format!(
            "{seed} = stablehlo.constant dense<{}> : tensor<{}>",
            Element::NEGATIVE_INFINITY,
            Element::ELEMENT,
        ));
        let element = Element::ELEMENT;
        emitter.line(format!(
            "%v{index} = \"stablehlo.reduce_window\"({}, {seed}) ({{",
            emitter.name(group.source),
        ));
        emitter.line(format!(
            "^bb0(%v{index}_left: tensor<{element}>, %v{index}_right: tensor<{element}>):"
        ));
        emitter.line(format!(
            "  %v{index}_larger = stablehlo.maximum %v{index}_left, %v{index}_right \
             : tensor<{element}>"
        ));
        emitter.line(format!(
            "  stablehlo.return %v{index}_larger : tensor<{element}>"
        ));
        emitter.line(format!(
            "}}) {{window_dimensions = array<i64: 1, 1, {size}, {size}>, \
             window_strides = array<i64: 1, 1, {stride}, {stride}>}} \
             : ({}, tensor<{element}>) -> {}",
            tensor_type::<Element>(&shapes[group.source]),
            tensor_type::<Element>(&shapes[index]),
            size = group.size,
            stride = group.stride,
        ));
        emitter.names[index] = Some(format!("%v{index}"));
    }

    /// Renders the epsilon a batch-norm raise carries as its
    /// attribute, read from the matched single-value leaf. StableHLO
    /// types the attribute `f32` regardless of the module's element
    /// type, so a wider recorded epsilon rounds — within emission's
    /// envelope-based conformance contract, like the target's own
    /// reassociation.
    fn epsilon_literal(&self, index: usize) -> String {
        let Some(Function::Leaf(leaf)) = self.functions().get(index) else {
            unreachable!("the matcher requires a single-value leaf epsilon")
        };
        leaf.0.to_vec()[0].literal()
    }

    /// Writes the raise of one training-mode batch normalization: a
    /// single `stablehlo.batch_norm_training` over the input, scale,
    /// and shift, whose three results name the root, the mean, and
    /// the variance — the named results were emit-skipped, and this
    /// is the only name they receive, so an observed statistic lands
    /// in the result list directly from the raised operation.
    fn raise_batch_norm_training(
        &self,
        index: usize,
        group: &BatchNormalization,
        emitter: &mut Emitter,
    ) {
        let shapes = self.shapes();
        emitter.line(format!(
            "%v{index}:3 = \"stablehlo.batch_norm_training\"({}, {}, {}) \
             {{epsilon = {} : f32, feature_index = 1 : i64}} \
             : ({}, {}, {}) -> ({}, {stat}, {stat})",
            emitter.name(group.input),
            emitter.name(group.scale),
            emitter.name(group.shift),
            self.epsilon_literal(group.epsilon),
            tensor_type::<Element>(&shapes[group.input]),
            tensor_type::<Element>(&shapes[group.scale]),
            tensor_type::<Element>(&shapes[group.shift]),
            tensor_type::<Element>(&shapes[index]),
            stat = tensor_type::<Element>(&shapes[group.mean]),
        ));
        emitter.names[index] = Some(format!("%v{index}#0"));
        emitter.names[group.mean] = Some(format!("%v{index}#1"));
        emitter.names[group.variance] = Some(format!("%v{index}#2"));
    }

    /// Writes the raise of one inference-mode batch normalization: a
    /// single `stablehlo.batch_norm_inference` over the input, scale,
    /// shift, and the supplied statistics, which are ordinary
    /// already-named operands.
    fn raise_batch_norm_inference(
        &self,
        index: usize,
        group: &BatchNormalization,
        emitter: &mut Emitter,
    ) {
        let shapes = self.shapes();
        emitter.line(format!(
            "%v{index} = \"stablehlo.batch_norm_inference\"({}, {}, {}, {}, {}) \
             {{epsilon = {} : f32, feature_index = 1 : i64}} \
             : ({}, {}, {}, {}, {}) -> {}",
            emitter.name(group.input),
            emitter.name(group.scale),
            emitter.name(group.shift),
            emitter.name(group.mean),
            emitter.name(group.variance),
            self.epsilon_literal(group.epsilon),
            tensor_type::<Element>(&shapes[group.input]),
            tensor_type::<Element>(&shapes[group.scale]),
            tensor_type::<Element>(&shapes[group.shift]),
            tensor_type::<Element>(&shapes[group.mean]),
            tensor_type::<Element>(&shapes[group.variance]),
            tensor_type::<Element>(&shapes[index]),
        ));
        emitter.names[index] = Some(format!("%v{index}"));
    }

    /// Writes the compact reduce of `source` over `axes` with the named
    /// reducer and its seed literal, producing node `index`'s value.
    /// An add-reduce honors the element's declared accumulation type.
    fn reduce(
        &self,
        index: usize,
        source: usize,
        axes: &[usize],
        reducer: &str,
        seed: &str,
        emitter: &mut Emitter,
    ) {
        if reducer == "add" {
            let prefix = format!("%v{index}");
            let source_name = emitter.name(source).to_string();
            let source_shape = self.shapes()[source].clone();
            let result_shape = self.shapes()[index].clone();
            self.sum_reduce(
                &prefix,
                &source_name,
                &source_shape,
                &prefix,
                &result_shape,
                axes,
                emitter,
            );
            return;
        }
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

    /// Writes an add-reduce of `source_name` over `axes` into
    /// `result_name`, honoring the element's declared accumulation
    /// type: the operand converts up, the reduction runs there, and
    /// the total converts back once — exactly what the home
    /// `Accumulator` contract computes. Temporaries derive from
    /// `prefix`, which must be unique per call.
    #[allow(clippy::too_many_arguments)]
    fn sum_reduce(
        &self,
        prefix: &str,
        source_name: &str,
        source_shape: &Shape,
        result_name: &str,
        result_shape: &Shape,
        axes: &[usize],
        emitter: &mut Emitter,
    ) {
        let seed_name = format!("{prefix}_seed");
        if let Some(accumulation) = Element::ACCUMULATION {
            let promoted = format!("{prefix}_promoted");
            let promoted_type = named_tensor_type(source_shape, accumulation);
            emitter.line(format!(
                "{promoted} = stablehlo.convert {source_name} : ({}) -> {promoted_type}",
                tensor_type::<Element>(source_shape),
            ));
            emitter.line(format!(
                "{seed_name} = stablehlo.constant dense<0.0> : tensor<{accumulation}>"
            ));
            let accumulated = format!("{prefix}_accumulated");
            let accumulated_type = named_tensor_type(result_shape, accumulation);
            emitter.line(format!(
                "{accumulated} = stablehlo.reduce({promoted} init: {seed_name}) \
                 applies stablehlo.add across dimensions = {axes:?} \
                 : ({promoted_type}, tensor<{accumulation}>) -> {accumulated_type}",
            ));
            emitter.line(format!(
                "{result_name} = stablehlo.convert {accumulated} : ({accumulated_type}) -> {}",
                tensor_type::<Element>(result_shape),
            ));
            return;
        }
        emitter.line(format!(
            "{seed_name} = stablehlo.constant dense<{}> : tensor<{}>",
            Element::ZERO,
            Element::ELEMENT,
        ));
        emitter.line(format!(
            "{result_name} = stablehlo.reduce({source_name} init: {seed_name}) \
             applies stablehlo.add across dimensions = {axes:?} : ({}, tensor<{}>) -> {}",
            tensor_type::<Element>(source_shape),
            Element::ELEMENT,
            tensor_type::<Element>(result_shape),
        ));
    }

    /// Writes the fused `log_sum_exp` as its stable decomposition: shift
    /// by the axis maximum, exponentiate, reduce, and re-add the shift.
    /// The target's rounding may differ from the fused interpreter rule,
    /// which conformance absorbs in its envelopes.
    fn lower_log_sum_exp(&self, index: usize, source: usize, axis: usize, emitter: &mut Emitter) {
        let shapes = self.shapes();
        let source_shape = &shapes[source];
        let reduced_type = tensor_type::<Element>(&shapes[index]);
        let full_type = tensor_type::<Element>(source_shape);
        let dims: Vec<usize> = (0..source_shape.rank()).filter(|&a| a != axis).collect();
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
        let total = format!("%v{index}_total");
        self.sum_reduce(
            &total,
            &exponentials,
            source_shape,
            &total,
            &shapes[index],
            &[axis],
            emitter,
        );
        let normalizer = format!("%v{index}_normalizer");
        emitter.line(format!(
            "{normalizer} = stablehlo.log {total} : {reduced_type}"
        ));
        emitter.line(format!(
            "%v{index} = stablehlo.add {peak}, {normalizer} : {reduced_type}"
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
        let total = format!("%v{index}_total");
        self.sum_reduce(
            &total,
            &exponentials,
            shape,
            &total,
            &reduced,
            &[axis],
            emitter,
        );
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
