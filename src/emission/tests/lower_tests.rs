use std::process::Command;

use crate::{Differentiable, Network, Shape, Tensor, concat};

use super::EmitError;

/// One emitted module with the payloads and oracle results the
/// conformance tests replay: the arguments in the module's own order
/// (parameters then inputs, each in recording order) and the readable
/// values the plan's run produced.
struct Case {
    name: &'static str,
    module: String,
    arguments: Vec<Tensor<f32>>,
    expected: Vec<Vec<f32>>,
}

/// Builds the smallest interesting case: an input times a parameter,
/// rectified and summed.
fn small_case() -> Case {
    let network = Network::new();
    let weights = Tensor::new([2, 2], [1.0_f32, 2.0, 3.0, 4.0]);
    let weights_value = network.parameter(weights.clone());
    let x = Tensor::new([2, 2], [0.5_f32, -1.0, 2.0, 3.0]);
    let x_value = network.input(x.clone());
    let loss = x_value.matmul(weights_value).relu().sum();
    let plan = network.compile([loss.symbol()], []);
    let evaluation = plan.forward(&network, []);
    Case {
        name: "small",
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![weights, x],
        expected: vec![evaluation.of(network.resolve(loss.symbol())).to_vec()],
    }
}

/// Builds a miniature of the transformer's sampling plan: embedding
/// gather, masked softmax over scaled scores, two heads joined by
/// concat.
fn attention_case() -> Case {
    let network = Network::new();
    let table = Tensor::new(
        [3, 4],
        (0..12)
            .map(|index| index as f32 / 10.0 - 0.5)
            .collect::<Vec<_>>(),
    );
    let table_value = network.parameter(table.clone());
    let tokens = Tensor::selection(vec![0, 1], 3, 1.0_f32);
    let tokens_value = network.input(tokens.clone());
    let mask = network.leaf(Tensor::new([2, 2], [0.0_f32, f32::NEG_INFINITY, 0.0, 0.0]));
    let scale = network.leaf(Tensor::filled([], 0.5_f32));

    let stream = table_value.gather(tokens_value);
    let heads: Vec<_> = (0..2)
        .map(|_| {
            let scores = stream.matmul(stream.transpose());
            let weights = (scores * scale.broadcast_like(scores) + mask).softmax(1);
            weights.matmul(stream)
        })
        .collect();
    let output = concat(&heads, 1);
    let plan = network.compile([output.symbol()], []);
    let evaluation = plan.forward(&network, []);
    // The one-hot selection crosses the boundary as its dense matrix.
    let dense_tokens = Tensor::new(Shape::new([2, 3]), tokens.to_vec());
    Case {
        name: "attention",
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![table, dense_tokens],
        expected: vec![evaluation.of(network.resolve(output.symbol())).to_vec()],
    }
}

/// Builds overlapping windows over a parameter: the static-gather
/// completeness fallback for `unfold`.
fn unfold_case() -> Case {
    let network = Network::new();
    let x = Tensor::new([8], (1..=8).map(|value| value as f32).collect::<Vec<_>>());
    let x_value = network.parameter(x.clone());
    let windows = x_value.unfold(0, 3, 2, 1);
    let plan = network.compile([windows.symbol()], []);
    let evaluation = plan.forward(&network, []);
    Case {
        name: "unfold",
        module: plan.emit_stablehlo().expect("the plan emits"),
        arguments: vec![x],
        expected: vec![evaluation.of(network.resolve(windows.symbol())).to_vec()],
    }
}

#[test]
fn a_small_plan_emits_the_golden_module() {
    let expected = "\
module @poorgrad {
  func.func @plan(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<f32>) {
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

#[test]
fn fused_plans_decline_with_the_group_count() {
    use crate::conv2d;

    let network = Network::new();
    let image = network.parameter(Tensor::new([1, 1, 4, 4], vec![1.0_f32; 16]));
    let kernel = network.parameter(Tensor::new([1, 1, 2, 2], vec![1.0_f32; 4]));
    let bias = network.parameter(Tensor::new([1], vec![0.0_f32]));
    let convolved = conv2d(image, kernel, bias, 1, 0).sum();
    // Forward-only plans always fuse the im2col chain, which emission
    // cannot raise yet.
    let plan = network.compile([convolved.symbol()], []);
    assert_eq!(plan.emit_stablehlo(), Err(EmitError::Fused { groups: 1 }));
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
    let path = std::env::temp_dir().join(format!("poorgrad-{name}-{}", std::process::id()));
    std::fs::write(&path, content).expect("the temp file writes");
    path
}

#[test]
fn emitted_modules_parse_through_the_toolchain() {
    // Tier-0 conformance: an external StableHLO parser must accept the
    // emitted text.
    let Some(command) = toolchain("POORGRAD_STABLEHLO_VALIDATOR", "stablehlo-opt") else {
        eprintln!("no StableHLO validator available; skipping the round-trip");
        return;
    };
    for case in [small_case(), attention_case(), unfold_case()] {
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
    let Some(command) = toolchain("POORGRAD_STABLEHLO_EVALUATOR", "poorgrad-stablehlo-eval") else {
        eprintln!("no StableHLO evaluator available; skipping the execution check");
        return;
    };
    for case in [small_case(), attention_case(), unfold_case()] {
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
        let results: Vec<Vec<f64>> = stdout
            .lines()
            .filter(|line| !line.trim().is_empty())
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
                let tolerance = 1e-4 * (1.0 + expected.abs());
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{}: {actual} differs from the oracle's {expected}",
                    case.name,
                );
            }
        }
    }
}
