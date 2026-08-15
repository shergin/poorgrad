use crate::{Network, Shape, Tensor};

use super::{Conv2d, conv2d};

#[test]
fn conv2d_matches_the_hand_computed_fixture() {
    let network = Network::new();
    let input = network.leaf(Tensor::new(
        [1, 1, 3, 3],
        (1..=9).map(|v| v as f64).collect::<Vec<_>>(),
    ));
    // The kernel picks the window's main diagonal: `[[1, 0], [0, 1]]`.
    let weights = network.leaf(Tensor::new([1, 1, 2, 2], [1.0, 0.0, 0.0, 1.0]));
    let bias = network.leaf(Tensor::new([1], [10.0]));

    let output = conv2d(input, weights, bias, 1, 0);
    assert_eq!(output.shape(), Shape::new([1, 1, 2, 2]));

    let run = network.forward();
    assert_eq!(run.of(output).to_vec(), &[16.0, 18.0, 22.0, 24.0]);
}

#[test]
fn conv2d_mixes_channels_into_filters() {
    let network = Network::new();
    // Two channels; 1x1 kernels make the channel mixing exact:
    // filter 0 sums the channels, filter 1 takes `2 * c0 + 3 * c1`.
    let input = network.leaf(Tensor::new(
        [1, 2, 2, 2],
        [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    ));
    let weights = network.leaf(Tensor::new([2, 2, 1, 1], [1.0, 1.0, 2.0, 3.0]));
    let bias = network.leaf(Tensor::filled([2], 0.0));

    let output = conv2d(input, weights, bias, 1, 0);
    assert_eq!(output.shape(), Shape::new([1, 2, 2, 2]));

    let run = network.forward();
    assert_eq!(
        run.of(output).to_vec(),
        &[11.0, 22.0, 33.0, 44.0, 32.0, 64.0, 96.0, 128.0]
    );
}

#[test]
fn conv2d_strides_the_windows() {
    let network = Network::new();
    let input = network.leaf(Tensor::new([1, 1, 1, 5], [1.0, 2.0, 3.0, 4.0, 5.0]));
    let weights = network.leaf(Tensor::new([1, 1, 1, 2], [1.0, 1.0]));
    let bias = network.leaf(Tensor::filled([1], 0.0));

    let output = conv2d(input, weights, bias, 2, 0);
    assert_eq!(output.shape(), Shape::new([1, 1, 1, 2]));

    let run = network.forward();
    assert_eq!(run.of(output).to_vec(), &[3.0, 7.0]);
}

#[test]
fn padding_surrounds_the_input_with_zeros() {
    let network = Network::new();
    let input = network.leaf(Tensor::new([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0]));
    // An all-ones 3x3 kernel over the zero-padded 2x2 input sums the
    // whole input at every output position.
    let weights = network.leaf(Tensor::filled([1, 1, 3, 3], 1.0));
    let bias = network.leaf(Tensor::filled([1], 0.0));

    let output = conv2d(input, weights, bias, 1, 1);
    assert_eq!(output.shape(), Shape::new([1, 1, 2, 2]));

    let run = network.forward();
    assert_eq!(run.of(output).to_vec(), &[10.0; 4]);
}

#[test]
fn conv2d_gradients_flow_to_every_operand() {
    let network = Network::new();
    let input = network.leaf(Tensor::new(
        [1, 1, 3, 3],
        (1..=9).map(|v| v as f64).collect::<Vec<_>>(),
    ));
    let weights = network.leaf(Tensor::filled([1, 1, 2, 2], 1.0));
    let bias = network.leaf(Tensor::filled([1], 0.0));

    let output = conv2d(input, weights, bias, 1, 0);
    let loss = output.sum();

    let run = network.forward();
    let gradients = run.backward(loss);
    // Each input position is graded by how many windows cover it.
    assert_eq!(
        gradients.of(input).to_vec(),
        &[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]
    );
    // Each kernel position sums the input values it was multiplied by.
    assert_eq!(gradients.of(weights).to_vec(), &[12.0, 16.0, 24.0, 28.0]);
    // The bias sees the seed once per output position.
    assert_eq!(gradients.of(bias).to_vec(), &[4.0]);
}

#[test]
#[should_panic(expected = "disagree on channels")]
fn conv2d_rejects_mismatched_channels() {
    let network = Network::new();
    let input = network.leaf(Tensor::filled([1, 2, 3, 3], 0.0));
    let weights = network.leaf(Tensor::filled([1, 3, 2, 2], 0.0));
    let bias = network.leaf(Tensor::filled([1], 0.0));
    conv2d(input, weights, bias, 1, 0);
}

#[test]
fn new_allocates_weights_and_bias() {
    let network = Network::new();
    let layer = Conv2d::new(
        &network,
        Tensor::filled([4, 2, 3, 3], 0.0_f64),
        Tensor::filled([4], 0.0),
        1,
        1,
    );
    // One kernel stack and one bias vector, regardless of size.
    assert_eq!(network.len(), 2);
    assert_eq!(layer.parameters().count(), 2);
}

#[test]
#[should_panic(expected = "must be rank 4")]
fn new_rejects_non_stack_weights() {
    let network = Network::new();
    Conv2d::new(
        &network,
        Tensor::filled([4, 2, 3], 0.0_f64),
        Tensor::filled([4], 0.0),
        1,
        0,
    );
}

#[test]
#[should_panic(expected = "disagree on filters")]
fn new_rejects_mismatched_bias() {
    let network = Network::new();
    Conv2d::new(
        &network,
        Tensor::filled([4, 2, 3, 3], 0.0_f64),
        Tensor::filled([3], 0.0),
        1,
        0,
    );
}

#[test]
#[should_panic(expected = "stride must be positive")]
fn new_rejects_a_zero_stride() {
    let network = Network::new();
    Conv2d::new(
        &network,
        Tensor::filled([4, 2, 3, 3], 0.0_f64),
        Tensor::filled([4], 0.0),
        0,
        0,
    );
}

#[test]
fn express_records_tensor_granularity() {
    let network = Network::new();
    let layer = Conv2d::new(
        &network,
        Tensor::new([1, 1, 2, 2], [1.0, 0.0, 0.0, 1.0]),
        Tensor::new([1], [10.0]),
        1,
        1,
    );
    let input = network.leaf(Tensor::new(
        [1, 1, 3, 3],
        (1..=9).map(|v| v as f64).collect::<Vec<_>>(),
    ));
    let nodes_before = network.len();

    let output = layer.express(&network, input);

    // Two pads, two unfolds, the patch permute + reshape, the weight
    // permute + reshape, the product, the bias broadcast and shift, and
    // the output reshape + permute: thirteen nodes, fixed regardless of
    // any size.
    assert_eq!(network.len(), nodes_before + 13);
    assert_eq!(output.shape(), Shape::new([1, 1, 4, 4]));
}
