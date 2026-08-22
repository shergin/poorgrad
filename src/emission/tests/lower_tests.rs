use std::process::Command;

use crate::{Bf16, Differentiable, Request, Shape, Tape, Tensor, concat, cross_entropy};

/// One emitted module with the payloads and oracle results the
/// conformance tests replay: the arguments in the module's own order
/// (parameters then inputs, each in recording order) and the readable
/// values the plan's run produced.
///
/// Arguments and expectations are carried as `f32` regardless of the
/// module's element type — a narrower element expands exactly, and
/// the evaluator scripts read the argument dtype from the module's
/// own signature. `tolerance` is the case's relative error envelope,
/// scaled to the element type's epsilon.
struct Case {
    name: &'static str,
    tolerance: f64,
    module: String,
    arguments: Vec<Tensor<f32>>,
    expected: Vec<Vec<f32>>,
}

/// Builds the smallest interesting case: an input times a parameter,
/// rectified and summed.
fn small_case() -> Case {
    let tape = Tape::new();
    let weights = Tensor::new([2, 2], [1.0_f32, 2.0, 3.0, 4.0]);
    let weights_value = tape.parameter(weights.clone());
    let x = Tensor::new([2, 2], [0.5_f32, -1.0, 2.0, 3.0]);
    let x_value = tape.input(x.clone());
    let loss = x_value.matmul(weights_value).relu().sum().symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([loss]));
    let run = plan.forward(&network.parameters(), []);
    Case {
        name: "small",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![weights, x],
        expected: vec![run.of(loss).to_vec()],
    }
}

/// Builds a miniature of the transformer's sampling plan: embedding
/// gather, masked softmax over scaled scores, two heads joined by
/// concat.
fn attention_case() -> Case {
    let tape = Tape::new();
    let table = Tensor::new(
        [3, 4],
        (0..12)
            .map(|index| index as f32 / 10.0 - 0.5)
            .collect::<Vec<_>>(),
    );
    let table_value = tape.parameter(table.clone());
    let tokens = Tensor::selection(vec![0, 1], 3, 1.0_f32);
    let tokens_value = tape.input(tokens.clone());
    let mask = tape.leaf(Tensor::new([2, 2], [0.0_f32, f32::NEG_INFINITY, 0.0, 0.0]));
    let scale = tape.leaf(Tensor::filled([], 0.5_f32));

    let stream = table_value.gather(tokens_value);
    let heads: Vec<_> = (0..2)
        .map(|_| {
            let scores = stream.matmul(stream.transpose());
            let weights = (scores * scale.broadcast_like(scores) + mask).softmax(1);
            weights.matmul(stream)
        })
        .collect();
    let output = concat(&heads, 1).symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([output]));
    let run = plan.forward(&network.parameters(), []);
    // The one-hot selection crosses the boundary as its dense matrix.
    let dense_tokens = Tensor::new(Shape::new([2, 3]), tokens.to_vec());
    Case {
        name: "attention",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![table, dense_tokens],
        expected: vec![run.of(output).to_vec()],
    }
}

/// Builds a cross-entropy loss over the fused `log_sum_exp`: the
/// stable expanded form the loss composes, exercising the newest
/// lowering end to end.
fn cross_entropy_case() -> Case {
    let tape = Tape::new();
    let logits = Tensor::new(
        [2, 3],
        (0..6)
            .map(|index| index as f32 * 0.7 - 2.0)
            .collect::<Vec<_>>(),
    );
    let logits_value = tape.parameter(logits.clone());
    let targets = Tensor::selection(vec![0, 2], 3, 1.0_f32);
    let targets_value = tape.input(targets.clone());
    let loss = cross_entropy(logits_value, targets_value).symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([loss]));
    let run = plan.forward(&network.parameters(), []);
    // The one-hot selection crosses the boundary as its dense matrix.
    let dense_targets = Tensor::new(Shape::new([2, 3]), targets.to_vec());
    Case {
        name: "cross-entropy",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![logits, dense_targets],
        expected: vec![run.of(loss).to_vec()],
    }
}

/// Builds a differentiated module — the E2 shape: a loss over relu,
/// windows, and an embedding gather, with its recorded gradients in
/// the result list, exercising the `Step`, `Fold`, and `Scatter`
/// lowerings the derivative rules introduce.
fn gradient_case() -> Case {
    let tape = Tape::new();
    let signal = Tensor::new(
        [8],
        (0..8).map(|v| v as f32 * 0.6 - 2.1).collect::<Vec<_>>(),
    );
    let signal_value = tape.parameter(signal.clone());
    let mix = Tensor::new(
        [3, 3],
        (0..9).map(|v| v as f32 * 0.25 - 1.0).collect::<Vec<_>>(),
    );
    let mix_value = tape.parameter(mix.clone());
    let table = Tensor::new(
        [3, 2],
        (0..6).map(|v| v as f32 * 0.5 - 1.25).collect::<Vec<_>>(),
    );
    let table_value = tape.parameter(table.clone());
    let tokens = Tensor::selection(vec![0, 2, 0], 3, 1.0_f32);
    let tokens_value = tape.input(tokens.clone());

    let windows = (signal_value.unfold(0, 3, 2, 1) * mix_value).relu().sum();
    let lookup = table_value.gather(tokens_value).sum();
    let loss = windows + lookup;
    let gradients = tape.differentiate(loss, [signal_value, table_value]);

    // The module's result list follows recording order, so the
    // expected vectors must too.
    let mut readable: Vec<_> = std::iter::once(loss.into())
        .chain(gradients.iter().copied())
        .collect();
    readable.sort_by_key(|&symbol: &crate::Symbol| symbol.id.index());
    let network = tape.into_network();
    let plan = network.compile(Request::roots(readable.clone()));
    let run = plan.forward(&network.parameters(), []);
    let dense_tokens = Tensor::new(Shape::new([3, 3]), tokens.to_vec());
    Case {
        name: "gradient",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![signal, mix, table, dense_tokens],
        expected: readable
            .iter()
            .map(|&symbol| run.of(symbol).to_vec())
            .collect(),
    }
}

#[test]
fn differentiated_modules_carry_the_new_lowerings() {
    let module = gradient_case().module;
    assert!(module.contains("stablehlo.compare"), "{module}");
    assert!(module.contains("stablehlo.select"), "{module}");
    // Fold and scatter both lower to contractions; the fold carries
    // its window-matrix constant and trailing transpose.
    assert!(module.contains("_weights"), "{module}");
}

/// Builds a batched product with its recorded gradients: the batched
/// `dot_general` lowering (batching dims on both sides) and its
/// permute-closed adjoint.
fn batched_case() -> Case {
    let tape = Tape::new();
    let a = Tensor::new(
        [2, 2, 3],
        (0..12).map(|v| v as f32 / 6.0 - 1.0).collect::<Vec<_>>(),
    );
    let a_value = tape.parameter(a.clone());
    let b = Tensor::new(
        [2, 3, 2],
        (0..12).map(|v| v as f32 / 4.0 - 1.5).collect::<Vec<_>>(),
    );
    let b_value = tape.input(b.clone());
    let loss = a_value.matmul(b_value).sum();
    let gradients = tape.differentiate(loss, [a_value, b_value]);

    // The module's result list follows recording order, so the
    // expected vectors must too.
    let mut readable: Vec<_> = std::iter::once(loss.into())
        .chain(gradients.iter().copied())
        .collect();
    readable.sort_by_key(|&symbol: &crate::Symbol| symbol.id.index());
    let network = tape.into_network();
    let plan = network.compile(Request::roots(readable.clone()));
    let run = plan.forward(&network.parameters(), []);
    Case {
        name: "batched",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![a, b],
        expected: readable
            .iter()
            .map(|&symbol| run.of(symbol).to_vec())
            .collect(),
    }
}

#[test]
fn batched_products_lower_with_batching_dims() {
    let module = batched_case().module;
    assert!(module.contains("batching_dims = [0] x [0]"), "{module}");
    assert!(module.contains("contracting_dims = [2] x [1]"), "{module}");
}

/// Builds overlapping windows over a parameter: the static-gather
/// completeness fallback for `unfold`.
fn unfold_case() -> Case {
    let tape = Tape::new();
    let x = Tensor::new([8], (1..=8).map(|value| value as f32).collect::<Vec<_>>());
    let x_value = tape.parameter(x.clone());
    let windows = x_value.unfold(0, 3, 2, 1).symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([windows]));
    let run = plan.forward(&network.parameters(), []);
    Case {
        name: "unfold",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![x],
        expected: vec![run.of(windows).to_vec()],
    }
}

#[test]
fn a_small_plan_emits_the_golden_module() {
    let expected = "\
module @topos {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<f32>) {
    %v2 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %v3_zero = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %v3 = stablehlo.maximum %v2, %v3_zero : tensor<2x2xf32>
    %v4_seed = stablehlo.constant dense<0.0> : tensor<f32>
    %v4 = stablehlo.reduce(%v3 init: %v4_seed) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<f32>
    return %v4 : tensor<f32>
  }
}
";
    assert_eq!(small_case().module, expected);
}

#[test]
fn attention_shaped_plans_emit_their_composition() {
    let module = attention_case().module;
    for expected in [
        "stablehlo.dot_general",
        "stablehlo.transpose",
        "stablehlo.broadcast_in_dim",
        "stablehlo.reduce",
        "applies stablehlo.maximum",
        "stablehlo.exponential",
        "stablehlo.pad",
        "dense<0xFF800000>",
    ] {
        assert!(module.contains(expected), "missing {expected}:\n{module}");
    }
    // The one-hot selection input crosses the boundary as a dense
    // argument.
    assert!(module.contains("%arg1: tensor<2x3xf32>"));
}

#[test]
fn unfold_emits_a_static_gather() {
    let module = unfold_case().module;
    assert!(module.contains("\"stablehlo.gather\""), "{module}");
    // The window starts, spaced by the step, dilated within each row.
    assert!(
        module.contains("dense<[[[0], [1], [2]], [[2], [3], [4]], [[4], [5], [6]]]>"),
        "{module}"
    );
    assert!(module.contains("tensor<3x3x1xi64>"), "{module}");
}

/// Builds a padded strided convolution through the facade: the
/// forward-only plan fuses the im2col chain, and emission raises the
/// group to `stablehlo.convolution`.
fn convolution_case() -> Case {
    use crate::conv2d;

    let tape = Tape::new();
    let image = Tensor::new(
        [1, 2, 4, 4],
        (0..32)
            .map(|index| index as f32 / 8.0 - 2.0)
            .collect::<Vec<_>>(),
    );
    let image_value = tape.parameter(image.clone());
    let weights = Tensor::new(
        [2, 2, 2, 2],
        (0..16)
            .map(|index| index as f32 / 4.0 - 2.0)
            .collect::<Vec<_>>(),
    );
    let weights_value = tape.parameter(weights.clone());
    let bias = Tensor::new([2], [0.25_f32, -0.5]);
    let bias_value = tape.parameter(bias.clone());
    let convolved = conv2d(image_value, weights_value, bias_value, 2, 1).symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([convolved]));
    assert_eq!(plan.home().groups(), 1, "the forward plan fuses");
    let run = plan.forward(&network.parameters(), []);
    Case {
        name: "convolution",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![image, weights, bias],
        expected: vec![run.of(convolved).to_vec()],
    }
}

/// Builds a miniature of the mnist probe: convolution, relu, max
/// pooling, a dense head, and log-softmax scores — the whole conv
/// consumer family shape, emitted end to end.
fn probe_case() -> Case {
    use crate::{conv2d, max_pool};

    let tape = Tape::new();
    let image = Tensor::new(
        [1, 2, 6, 6],
        (0..72)
            .map(|index| (index % 13) as f32 / 6.0 - 1.0)
            .collect::<Vec<_>>(),
    );
    let image_value = tape.parameter(image.clone());
    let weights = Tensor::new(
        [3, 2, 3, 3],
        (0..54)
            .map(|index| (index % 11) as f32 / 5.0 - 1.0)
            .collect::<Vec<_>>(),
    );
    let weights_value = tape.parameter(weights.clone());
    let bias = Tensor::new([3], [0.1_f32, -0.2, 0.3]);
    let bias_value = tape.parameter(bias.clone());
    let dense = Tensor::new(
        [27, 5],
        (0..135)
            .map(|index| (index % 7) as f32 / 3.0 - 1.0)
            .collect::<Vec<_>>(),
    );
    let dense_value = tape.parameter(dense.clone());

    let features = conv2d(image_value, weights_value, bias_value, 1, 1).relu();
    let pooled = max_pool(features, 2, 2);
    let scores = pooled
        .reshape([1, 27])
        .matmul(dense_value)
        .log_softmax(1)
        .symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([scores]));
    assert_eq!(plan.home().groups(), 2, "the conv chain and the pool fuse");
    let run = plan.forward(&network.parameters(), []);
    Case {
        name: "probe",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![image, weights, bias, dense],
        expected: vec![run.of(scores).to_vec()],
    }
}

#[test]
fn probe_networks_emit_end_to_end() {
    let module = probe_case().module;
    // Both window chains raise: the conv to `convolution`, the pool
    // to `reduce_window` — no static-gather fallback remains.
    assert!(module.contains("stablehlo.convolution"), "{module}");
    assert!(module.contains("\"stablehlo.reduce_window\""), "{module}");
    assert!(!module.contains("stablehlo.gather"), "{module}");
}

/// Builds a plain max pool over a parameter: the raised
/// `reduce_window` path on its own.
fn pool_case() -> Case {
    use crate::max_pool;

    let tape = Tape::new();
    let image = Tensor::new(
        [1, 2, 4, 4],
        (0..32)
            .map(|index| (index % 9) as f32 / 4.0 - 1.0)
            .collect::<Vec<_>>(),
    );
    let image_value = tape.parameter(image.clone());
    let pooled = max_pool(image_value, 2, 2).symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([pooled]));
    let run = plan.forward(&network.parameters(), []);
    Case {
        name: "pool",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![image],
        expected: vec![run.of(pooled).to_vec()],
    }
}

#[test]
fn pooled_plans_raise_to_reduce_window() {
    let module = pool_case().module;
    assert!(module.contains("\"stablehlo.reduce_window\""), "{module}");
    assert!(
        module.contains("window_dimensions = array<i64: 1, 1, 2, 2>"),
        "{module}"
    );
    assert!(
        module.contains("window_strides = array<i64: 1, 1, 2, 2>"),
        "{module}"
    );
    // The negative-infinity seed and the maximum region.
    assert!(module.contains("dense<0xFF800000>"), "{module}");
    assert!(module.contains("stablehlo.return"), "{module}");
    // The lanes never cross the boundary: no gathered windows, no
    // lane slices.
    assert!(!module.contains("stablehlo.gather"), "{module}");
    assert!(!module.contains("stablehlo.slice"), "{module}");
}

#[test]
fn a_hand_rolled_pool_fold_raises_identically() {
    // Provenance-blind matching: the facade's exact spelling written
    // by hand raises to the same module.
    let tape = Tape::new();
    let image = Tensor::new(
        [1, 2, 4, 4],
        (0..32)
            .map(|index| (index % 9) as f32 / 4.0 - 1.0)
            .collect::<Vec<_>>(),
    );
    let image_value = tape.parameter(image);
    let lanes = image_value
        .unfold(2, 2, 2, 1)
        .unfold(4, 2, 2, 1)
        .permute([0, 1, 2, 4, 3, 5])
        .reshape([1, 2, 2, 2, 4]);
    let mut largest = lanes.narrow(4, 0, 1);
    for lane in 1..4 {
        largest = largest.maximum(lanes.narrow(4, lane, 1));
    }
    let pooled = largest.squeeze(4).symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([pooled]));
    let module = plan.emit_stablehlo().expect("the plan emits");
    assert_eq!(module, pool_case().module);
}

/// Builds a training-mode batch normalization with its statistics
/// observed: the three-result raise, whose mean and variance reach
/// the result list straight from the raised operation.
fn batch_norm_training_case() -> Case {
    use crate::BatchNorm;

    let tape = Tape::new();
    let scale = Tensor::new([2], [1.5_f32, 0.5]);
    let shift = Tensor::new([2], [0.25_f32, -0.25]);
    let layer = BatchNorm::new(
        &tape,
        scale.clone(),
        shift.clone(),
        Tensor::filled([], 1e-5_f32),
    );
    let input = Tensor::new(
        [3, 2],
        (0..6).map(|v| v as f32 * 0.7 - 2.0).collect::<Vec<_>>(),
    );
    let input_value = tape.input(input.clone());
    let normalization = layer.express(&tape, input_value);
    let output = normalization.output.symbol();
    let mean = normalization.mean.symbol();
    let variance = normalization.variance.symbol();
    // The module's result list follows recording order: the mean and
    // variance precede the trailing shift root.
    let mut readable: Vec<crate::Symbol> = vec![output, mean, variance];
    readable.sort_by_key(|&symbol: &crate::Symbol| symbol.id.index());
    let network = tape.into_network();
    let plan = network.compile(Request::roots([output]).observe([mean, variance]));
    let run = plan.forward(&network.parameters(), []);
    Case {
        name: "batch-norm-training",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![scale, shift, input],
        expected: readable
            .iter()
            .map(|&symbol| run.of(symbol).to_vec())
            .collect(),
    }
}

#[test]
fn training_batch_norms_raise_with_named_statistics() {
    let module = batch_norm_training_case().module;
    assert!(
        module.contains("\"stablehlo.batch_norm_training\""),
        "{module}"
    );
    assert!(module.contains("epsilon = 1.0e-5 : f32"), "{module}");
    assert!(module.contains("feature_index = 1 : i64"), "{module}");
    // The primitive decomposition never crosses the boundary: no
    // statistic reductions, no division, no square root.
    assert!(!module.contains("stablehlo.reduce"), "{module}");
    assert!(!module.contains("stablehlo.divide"), "{module}");
    assert!(!module.contains("stablehlo.sqrt"), "{module}");
    // The observed statistics return as the raise's own results.
    assert!(module.contains("#1"), "{module}");
    assert!(module.contains("#2"), "{module}");
}

/// Builds an inference-mode batch normalization over fed statistics:
/// the five-operand raise, statistics as ordinary arguments.
fn batch_norm_inference_case() -> Case {
    use crate::BatchNorm;

    let tape = Tape::new();
    let scale = Tensor::new([2], [1.5_f32, 0.5]);
    let shift = Tensor::new([2], [0.25_f32, -0.25]);
    let layer = BatchNorm::new(
        &tape,
        scale.clone(),
        shift.clone(),
        Tensor::filled([], 1e-5_f32),
    );
    let input = Tensor::new(
        [3, 2],
        (0..6).map(|v| v as f32 * 0.6 - 1.5).collect::<Vec<_>>(),
    );
    let input_value = tape.input(input.clone());
    let mean = Tensor::new([2], [0.1_f32, -0.2]);
    let mean_value = tape.input(mean.clone());
    let variance = Tensor::new([2], [1.25_f32, 0.75]);
    let variance_value = tape.input(variance.clone());
    let output = layer
        .express_with(&tape, input_value, mean_value, variance_value)
        .symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([output]));
    let run = plan.forward(&network.parameters(), []);
    Case {
        name: "batch-norm-inference",
        tolerance: 1e-4,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![scale, shift, input, mean, variance],
        expected: vec![run.of(output).to_vec()],
    }
}

#[test]
fn inference_batch_norms_raise_over_their_arguments() {
    let module = batch_norm_inference_case().module;
    assert!(
        module.contains("\"stablehlo.batch_norm_inference\""),
        "{module}"
    );
    assert!(!module.contains("stablehlo.divide"), "{module}");
    assert!(!module.contains("stablehlo.sqrt"), "{module}");
    // The fed statistics cross the boundary as ordinary arguments.
    assert!(module.contains("%arg3"), "{module}");
    assert!(module.contains("%arg4"), "{module}");
}

#[test]
fn engine_backward_plans_still_raise_the_pool() {
    // The pool pattern is raise-only, so its storage is not gated by
    // memory posture: an engine-backward plan executes the recorded
    // fold at home and still raises it abroad.
    use crate::max_pool;

    let tape = Tape::new();
    let image_value = tape.parameter(Tensor::new(
        [1, 2, 4, 4],
        (0..32)
            .map(|index| index as f32 / 8.0 - 2.0)
            .collect::<Vec<_>>(),
    ));
    let loss = max_pool(image_value, 2, 2).sum().symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([loss]).backward());
    let module = plan.emit_stablehlo().expect("the plan emits");
    assert!(module.contains("\"stablehlo.reduce_window\""), "{module}");
}

#[test]
fn engine_backward_plans_emit_the_same_module() {
    // Emission elects with a total repertoire on every memory
    // posture, so the backward plan of the same request emits
    // byte-identical text — including the raised convolution, which
    // the old posture gate wrongly kept primitive.
    use crate::conv2d;

    let tape = Tape::new();
    let image = tape.parameter(Tensor::new(
        [1, 1, 3, 3],
        (0..9)
            .map(|index| index as f32 / 3.0 - 1.0)
            .collect::<Vec<_>>(),
    ));
    let weights = tape.parameter(Tensor::new([1, 1, 2, 2], [0.5_f32, -0.5, 0.25, -0.25]));
    let bias = tape.parameter(Tensor::new([1], [0.1_f32]));
    let convolved = conv2d(image, weights, bias, 1, 0).symbol();
    let network = tape.into_network();

    let forward = network.compile(Request::roots([convolved]));
    let backward = network.compile(Request::roots([convolved]).backward());
    let forward_module = forward.emit_stablehlo().expect("the forward plan emits");
    let backward_module = backward.emit_stablehlo().expect("the backward plan emits");
    assert!(
        backward_module.contains("stablehlo.convolution"),
        "{backward_module}"
    );
    assert_eq!(forward_module, backward_module);
}

#[test]
fn a_convolution_plan_emits_the_golden_module() {
    // The full raised module, so catalog plumbing changes cannot
    // silently reshape the emitted text.
    let expected = r#"module @topos {
  func.func @main(%arg0: tensor<1x2x4x4xf32>, %arg1: tensor<2x2x2x2xf32>, %arg2: tensor<2xf32>) -> (tensor<1x2x3x3xf32>) {
    %v9 = stablehlo.transpose %arg1, dims = [1, 2, 3, 0] : (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
    %v10 = stablehlo.reshape %v9 : (tensor<2x2x2x2xf32>) -> tensor<8x2xf32>
    %v11_kernel = stablehlo.reshape %v10 : (tensor<8x2xf32>) -> tensor<2x2x2x2xf32>
    %v11_windows = stablehlo.convolution(%arg0, %v11_kernel) dim_numbers = [b, f, 0, 1]x[i, 0, 1, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x4x4xf32>, tensor<2x2x2x2xf32>) -> tensor<1x3x3x2xf32>
    %v11 = stablehlo.reshape %v11_windows : (tensor<1x3x3x2xf32>) -> tensor<9x2xf32>
    %v12 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2xf32>) -> tensor<9x2xf32>
    %v13 = stablehlo.add %v11, %v12 : tensor<9x2xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<9x2xf32>) -> tensor<1x3x3x2xf32>
    %v15 = stablehlo.transpose %v14, dims = [0, 3, 1, 2] : (tensor<1x3x3x2xf32>) -> tensor<1x2x3x3xf32>
    return %v15 : tensor<1x2x3x3xf32>
  }
}
"#;
    assert_eq!(convolution_case().module, expected);
}

#[test]
fn fused_plans_raise_to_convolution() {
    let module = convolution_case().module;
    assert!(
        module.contains("stablehlo.convolution"),
        "missing the raised convolution:\n{module}"
    );
    assert!(
        module.contains("dim_numbers = [b, f, 0, 1]x[i, 0, 1, o]->[b, 0, 1, f]"),
        "{module}"
    );
    assert!(
        module.contains("window = {stride = [2, 2], pad = [[1, 1], [1, 1]]}"),
        "{module}"
    );
    // The im2col chain never crosses the boundary: no gathered
    // windows, and the symmetric pads ride as window padding.
    assert!(!module.contains("stablehlo.gather"), "{module}");
    assert!(!module.contains("stablehlo.pad "), "{module}");
}

/// Returns an external command from `variable`, or the named binary
/// when it is on the path, or `None`: the conformance tests pass
/// vacuously without their toolchain.
fn toolchain(variable: &str, binary: &str) -> Option<Vec<String>> {
    if let Ok(command) = std::env::var(variable) {
        return Some(command.split_whitespace().map(str::to_string).collect());
    }
    let probe = Command::new(binary).arg("--version").output();
    if probe.is_ok_and(|output| output.status.success()) {
        return Some(vec![binary.to_string()]);
    }
    None
}

/// Writes `content` to a unique temp file and returns its path.
fn temp_file(name: &str, content: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("topos-{name}-{}", std::process::id()));
    std::fs::write(&path, content).expect("the temp file writes");
    path
}

/// Expands a bf16 tensor into the exact `f32` values it denotes, the
/// form the evaluator line protocol carries.
fn expanded(tensor: &Tensor<Bf16>) -> Tensor<f32> {
    let elements: Vec<f32> = tensor.iter().map(Bf16::to_f32).collect();
    Tensor::new(tensor.shape(), elements)
}

/// Builds the bf16 twin of the small case: an input times a
/// parameter, rectified and summed, with every value chosen exactly
/// representable so the case stays stable across accumulation
/// semantics.
fn bf16_case() -> Case {
    let tape = Tape::new();
    let weights_elements: Vec<Bf16> = [1.0_f32, 2.0, 3.0, 4.0].map(Bf16::from_f32).to_vec();
    let weights = Tensor::new([2, 2], weights_elements);
    let weights_value = tape.parameter(weights.clone());
    let x_elements: Vec<Bf16> = [0.5_f32, -1.0, 2.0, 3.0].map(Bf16::from_f32).to_vec();
    let x = Tensor::new([2, 2], x_elements);
    let x_value = tape.input(x.clone());
    let loss = x_value.matmul(weights_value).relu().sum().symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([loss]));
    let run = plan.forward(&network.parameters(), []);
    let expected: Vec<f32> = run.of(loss).iter().map(Bf16::to_f32).collect();
    Case {
        name: "bf16-small",
        // The envelope scales to the element: bf16's epsilon is 2^-8,
        // doubled for the two-deep accumulation this case performs.
        tolerance: 7.8125e-3,
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![expanded(&weights), expanded(&x)],
        expected: vec![expected],
    }
}

#[test]
fn bf16_matmuls_emit_the_accumulation_form() {
    // The declared accumulation type is IR semantics: the dot carries
    // an f32 result type and an explicit convert back, exactly what
    // the home gemm seam computes.
    let module = bf16_case().module;
    assert!(
        module.contains("-> tensor<2x2xf32>"),
        "the dot must produce the accumulation type:\n{module}"
    );
    assert!(
        module.contains("stablehlo.convert"),
        "the accumulated product must convert back to bf16:\n{module}"
    );
}

#[test]
fn emitted_modules_parse_through_the_toolchain() {
    // Tier-0 conformance: an external StableHLO parser must accept the
    // emitted text.
    let Some(command) = toolchain("TOPOS_STABLEHLO_VALIDATOR", "stablehlo-opt") else {
        eprintln!("no StableHLO validator available; skipping the round-trip");
        return;
    };
    for case in [
        small_case(),
        attention_case(),
        cross_entropy_case(),
        gradient_case(),
        batched_case(),
        unfold_case(),
        convolution_case(),
        probe_case(),
        pool_case(),
        batch_norm_training_case(),
        batch_norm_inference_case(),
        bf16_case(),
    ] {
        let path = temp_file(&format!("parse-{}.mlir", case.name), &case.module);
        let output = Command::new(&command[0])
            .args(&command[1..])
            .arg(&path)
            .output()
            .expect("the validator command runs");
        std::fs::remove_file(&path).expect("the temp module removes");
        assert!(
            output.status.success(),
            "the {} module failed to parse:\n{}\n{}",
            case.name,
            String::from_utf8_lossy(&output.stderr),
            case.module,
        );
    }
}

/// Renders one tensor as an evaluator line: the `x`-joined extents
/// (`-` for rank 0), then the elements in row-major order.
fn evaluator_line(tensor: &Tensor<f32>) -> String {
    let shape = tensor.shape();
    let dimensions: Vec<String> = shape
        .axes()
        .iter()
        .map(|extent| extent.to_string())
        .collect();
    let rendered = if dimensions.is_empty() {
        "-".to_string()
    } else {
        dimensions.join("x")
    };
    let values: Vec<String> = tensor
        .to_vec()
        .iter()
        .map(|value| format!("{value:?}"))
        .collect();
    format!("{rendered} {}", values.join(" "))
}

#[test]
fn emitted_modules_execute_within_the_oracle_envelope() {
    // Tier-1 conformance: the StableHLO reference interpreter — the
    // specification's executable semantics — must reproduce the plan's
    // own results. The envelope is a coarse relative tolerance for
    // now; deriving envelopes from an `f64` oracle run is the designed
    // refinement.
    let Some(command) = toolchain("TOPOS_STABLEHLO_EVALUATOR", "topos-stablehlo-eval") else {
        eprintln!("no StableHLO evaluator available; skipping the execution check");
        return;
    };
    for case in [
        small_case(),
        attention_case(),
        cross_entropy_case(),
        gradient_case(),
        batched_case(),
        unfold_case(),
        convolution_case(),
        probe_case(),
        pool_case(),
        // The batch-norm cases parse (tier 0) but stay out of this
        // list: the reference interpreter does not implement the
        // batch_norm operations yet (openxla/stablehlo#1571). The
        // XLA-backed `run-stablehlo-xla.py` evaluator executes them.
        bf16_case(),
    ] {
        let module_path = temp_file(&format!("eval-{}.mlir", case.name), &case.module);
        let lines: Vec<String> = case.arguments.iter().map(evaluator_line).collect();
        let arguments_path = temp_file(&format!("eval-{}-arguments", case.name), &lines.join("\n"));
        let output = Command::new(&command[0])
            .args(&command[1..])
            .arg(&module_path)
            .arg(&arguments_path)
            .output()
            .expect("the evaluator command runs");
        std::fs::remove_file(&module_path).expect("the temp module removes");
        std::fs::remove_file(&arguments_path).expect("the temp arguments remove");
        assert!(
            output.status.success(),
            "the {} module failed to execute:\n{}",
            case.name,
            String::from_utf8_lossy(&output.stderr),
        );
        let stdout = String::from_utf8(output.stdout).expect("the evaluator prints text");
        // Keep only protocol lines — a dimensions token then elements —
        // since some backends print banners to standard output.
        let results: Vec<Vec<f64>> = stdout
            .lines()
            .filter(|line| {
                let Some(first) = line.split_whitespace().next() else {
                    return false;
                };
                first == "-"
                    || first
                        .split('x')
                        .all(|extent| extent.parse::<usize>().is_ok())
            })
            .map(|line| {
                line.split_whitespace()
                    .skip(1)
                    .map(|value| value.parse().expect("the evaluator prints numbers"))
                    .collect()
            })
            .collect();
        assert_eq!(
            results.len(),
            case.expected.len(),
            "{}: result count",
            case.name
        );
        for (result, expected) in results.iter().zip(&case.expected) {
            assert_eq!(result.len(), expected.len(), "{}: element count", case.name);
            for (&actual, &expected) in result.iter().zip(expected) {
                let expected = expected as f64;
                let tolerance = case.tolerance * (1.0 + expected.abs());
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{}: {actual} differs from the oracle's {expected}",
                    case.name,
                );
            }
        }
    }
}
