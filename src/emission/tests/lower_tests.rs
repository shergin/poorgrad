use crate::{Network, Tensor, concat};

use super::EmitError;

#[test]
fn a_small_plan_emits_the_golden_module() {
    let network = Network::new();
    let weights = network.parameter(Tensor::new([2, 2], [1.0_f32, 2.0, 3.0, 4.0]));
    let x = network.input(Tensor::new([2, 2], [0.0_f32; 4]));
    let loss = x.matmul(weights).relu().sum();
    let plan = network.compile([loss.symbol()], []);

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
    assert_eq!(plan.emit_stablehlo().expect("the plan emits"), expected);
}

#[test]
fn attention_shaped_plans_emit_their_composition() {
    // A miniature of the transformer's sampling plan: embedding gather,
    // masked softmax over scaled scores, two heads joined by concat.
    let network = Network::new();
    let table = network.parameter(Tensor::new([3, 4], vec![0.1_f32; 12]));
    let tokens = network.input(Tensor::selection(vec![0, 1], 3, 1.0_f32));
    let mask = network.leaf(Tensor::new([2, 2], [0.0_f32, f32::NEG_INFINITY, 0.0, 0.0]));
    let scale = network.leaf(Tensor::filled([], 0.5_f32));

    let stream = table.gather(tokens);
    let heads: Vec<_> = (0..2)
        .map(|_| {
            let scores = stream.matmul(stream.transpose());
            let weights = (scores * scale.broadcast_like(scores) + mask).softmax(1);
            weights.matmul(stream)
        })
        .collect();
    let output = concat(&heads, 1);
    let plan = network.compile([output.symbol()], []);

    let module = plan.emit_stablehlo().expect("the plan emits");
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
fn unfold_declines_with_the_operation_name() {
    let network = Network::new();
    let x = network.parameter(Tensor::new([6], vec![1.0_f32; 6]));
    let windows = x.unfold(0, 2, 2, 1).sum();
    let plan = network.compile([windows.symbol()], []);
    assert_eq!(
        plan.emit_stablehlo(),
        Err(EmitError::Unsupported {
            node: 1,
            operation: "Unfold"
        })
    );
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
