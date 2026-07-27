use crate::{Differentiable, Network, Tensor, Value};

/// The half-width of the central difference.
const STEP: f64 = 1e-5;

/// The mixed absolute and relative tolerance of a comparison.
const TOLERANCE: f64 = 1e-6;

/// Asserts that the analytic gradients of `expression` match central
/// finite differences at `inputs`, for every input.
///
/// Each numeric probe rebuilds the graph on a fresh network with one
/// input nudged, so the check exercises recording, forward, and backward
/// exactly as a user would.
fn assert_gradients_match<const INPUTS: usize>(
    inputs: [f64; INPUTS],
    expression: impl for<'network> Fn([Value<'network, f64>; INPUTS]) -> Value<'network, f64>,
) {
    let evaluate = |point: [f64; INPUTS]| -> f64 {
        let network = Network::new();
        let target = expression(point.map(|value| network.leaf(value)));
        *network.forward().of(target)
    };

    let network = Network::new();
    let leaves = inputs.map(|value| network.leaf(value));
    let target = expression(leaves);
    let evaluation = network.forward();
    let gradients = evaluation.backward(target);

    for (index, leaf) in leaves.iter().enumerate() {
        let mut nudged_up = inputs;
        nudged_up[index] += STEP;
        let mut nudged_down = inputs;
        nudged_down[index] -= STEP;
        let numeric = (evaluate(nudged_up) - evaluate(nudged_down)) / (2.0 * STEP);
        let analytic = *gradients.of(*leaf);
        assert!(
            (analytic - numeric).abs() <= TOLERANCE * (1.0 + numeric.abs()),
            "gradient of input {index} diverges: analytic {analytic}, numeric {numeric}"
        );
    }
}

#[test]
fn arithmetic_gradients_match_finite_differences() {
    assert_gradients_match([2.0, 3.0], |[a, b]| a + b);
    assert_gradients_match([2.0, 3.0], |[a, b]| a - b);
    assert_gradients_match([2.0, 3.0], |[a, b]| a * b);
    assert_gradients_match([2.0, 3.0], |[a, b]| a / b);
    assert_gradients_match([2.0], |[a]| -a);
    assert_gradients_match([2.0, 3.0], |[a, b]| a * b + a - b / a);
}

#[test]
fn transcendental_gradients_match_finite_differences() {
    assert_gradients_match([0.5], |[a]| a.tanh());
    assert_gradients_match([0.8], |[a]| a.exp());
    assert_gradients_match([1.7], |[a]| a.ln());
    assert_gradients_match([0.3, 1.2], |[a, b]| ((a * b).exp() + a.tanh()).ln());
}

#[test]
fn fan_out_gradients_match_finite_differences() {
    assert_gradients_match([1.5], |[a]| {
        let squared = a * a;
        squared * squared + squared + a
    });
}

#[test]
fn literal_sugar_gradients_match_finite_differences() {
    assert_gradients_match([0.4], |[x]| 1.0 / ((-x).exp() + 1.0));
    assert_gradients_match([3.0], |[x]| 2.0 * x + 1.0);
}

/// Returns `tensor` with the element at `position` shifted by `delta`.
fn nudged(tensor: &Tensor<f64>, position: usize, delta: f64) -> Tensor<f64> {
    let mut elements = tensor.elements().to_vec();
    elements[position] += delta;
    Tensor::new(tensor.shape().axes().iter().copied(), elements)
}

/// Checks the dense-layer expression per element: `matmul`, the
/// axis-wise bias broadcast, `tanh`, elementwise arithmetic, and the
/// full reduction — the exact shape of `Layer::express`.
#[test]
fn dense_layer_gradients_match_finite_differences() {
    let base: Vec<Tensor<f64>> = vec![
        Tensor::new([2, 3], [0.5, -1.0, 0.25, 1.5, 0.75, -0.5]),
        Tensor::new([3, 2], [1.0, 0.5, -0.75, 0.25, 0.5, 1.25]),
        Tensor::new([2], [0.35, -0.15]),
        Tensor::new([2, 2], [0.6, -0.5, 0.25, 0.75]),
    ];

    let loss_of = |tensors: &[Tensor<f64>]| -> f64 {
        let network = Network::new();
        let x = network.leaf(tensors[0].clone());
        let w = network.leaf(tensors[1].clone());
        let bias = network.leaf(tensors[2].clone());
        let y = network.leaf(tensors[3].clone());
        let product = x.matmul(w);
        let activated = (product + bias.broadcast_along(0, product)).tanh();
        let error = activated - y;
        let loss = (error * error).sum();
        network.forward().of(loss).elements()[0]
    };

    let network = Network::new();
    let x = network.leaf(base[0].clone());
    let w = network.leaf(base[1].clone());
    let bias = network.leaf(base[2].clone());
    let y = network.leaf(base[3].clone());
    let product = x.matmul(w);
    let activated = (product + bias.broadcast_along(0, product)).tanh();
    let error = activated - y;
    let loss = (error * error).sum();
    let evaluation = network.forward();
    let gradients = evaluation.backward(loss);
    let analytic = [
        gradients.of(x).clone(),
        gradients.of(w).clone(),
        gradients.of(bias).clone(),
        gradients.of(y).clone(),
    ];

    for (which, input) in base.iter().enumerate() {
        for position in 0..input.elements().len() {
            let mut up = base.clone();
            up[which] = nudged(input, position, STEP);
            let mut down = base.clone();
            down[which] = nudged(input, position, -STEP);
            let numeric = (loss_of(&up) - loss_of(&down)) / (2.0 * STEP);
            let value = analytic[which].elements()[position];
            assert!(
                (value - numeric).abs() <= TOLERANCE * (1.0 + numeric.abs()),
                "dense input {which} element {position} diverges: \
                 analytic {value}, numeric {numeric}"
            );
        }
    }
}

/// Checks the tensor-native operations the scalar harness cannot reach:
/// one expression covering `matmul`, `transposed`, `broadcast_like`,
/// elementwise arithmetic, and `sum`, differentiated per element.
#[test]
fn tensor_gradients_match_finite_differences() {
    let base: Vec<Tensor<f64>> = vec![
        Tensor::new([2, 3], [0.5, -1.0, 0.25, 1.5, 0.75, -0.5]),
        Tensor::new([3, 2], [1.0, 0.5, -0.75, 0.25, 0.5, 1.25]),
        Tensor::new([], [0.35]),
        Tensor::new([2, 2], [1.0, -0.5, 0.25, 0.75]),
    ];

    let loss_of = |tensors: &[Tensor<f64>]| -> f64 {
        let network = Network::new();
        let x = network.leaf(tensors[0].clone());
        let w = network.leaf(tensors[1].clone());
        let bias = network.leaf(tensors[2].clone());
        let y = network.leaf(tensors[3].clone());
        let product = x.matmul(w).transposed();
        let shifted = product + bias.broadcast_like(product);
        let error = shifted - y;
        let loss = (error * error).sum();
        network.forward().of(loss).elements()[0]
    };

    let network = Network::new();
    let x = network.leaf(base[0].clone());
    let w = network.leaf(base[1].clone());
    let bias = network.leaf(base[2].clone());
    let y = network.leaf(base[3].clone());
    let product = x.matmul(w).transposed();
    let shifted = product + bias.broadcast_like(product);
    let error = shifted - y;
    let loss = (error * error).sum();
    let evaluation = network.forward();
    let gradients = evaluation.backward(loss);
    let analytic = [
        gradients.of(x).clone(),
        gradients.of(w).clone(),
        gradients.of(bias).clone(),
        gradients.of(y).clone(),
    ];

    for (which, input) in base.iter().enumerate() {
        for position in 0..input.elements().len() {
            let mut up = base.clone();
            up[which] = nudged(input, position, STEP);
            let mut down = base.clone();
            down[which] = nudged(input, position, -STEP);
            let numeric = (loss_of(&up) - loss_of(&down)) / (2.0 * STEP);
            let value = analytic[which].elements()[position];
            assert!(
                (value - numeric).abs() <= TOLERANCE * (1.0 + numeric.abs()),
                "tensor input {which} element {position} diverges: \
                 analytic {value}, numeric {numeric}"
            );
        }
    }
}
