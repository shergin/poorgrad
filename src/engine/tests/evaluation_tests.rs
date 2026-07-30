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
fn nudge(tensor: &Tensor<f64>, position: usize, delta: f64) -> Tensor<f64> {
    let mut elements = tensor.to_vec();
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
        network.forward().of(loss).to_vec()[0]
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
        for position in 0..input.to_vec().len() {
            let mut up = base.clone();
            up[which] = nudge(input, position, STEP);
            let mut down = base.clone();
            down[which] = nudge(input, position, -STEP);
            let numeric = (loss_of(&up) - loss_of(&down)) / (2.0 * STEP);
            let value = analytic[which].to_vec()[position];
            assert!(
                (value - numeric).abs() <= TOLERANCE * (1.0 + numeric.abs()),
                "dense input {which} element {position} diverges: \
                 analytic {value}, numeric {numeric}"
            );
        }
    }
}

/// Checks the tensor-native operations the scalar harness cannot reach:
/// one expression covering `matmul`, `transpose`, `broadcast_like`,
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
        let product = x.matmul(w).transpose();
        let shifted = product + bias.broadcast_like(product);
        let error = shifted - y;
        let loss = (error * error).sum();
        network.forward().of(loss).to_vec()[0]
    };

    let network = Network::new();
    let x = network.leaf(base[0].clone());
    let w = network.leaf(base[1].clone());
    let bias = network.leaf(base[2].clone());
    let y = network.leaf(base[3].clone());
    let product = x.matmul(w).transpose();
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
        for position in 0..input.to_vec().len() {
            let mut up = base.clone();
            up[which] = nudge(input, position, STEP);
            let mut down = base.clone();
            down[which] = nudge(input, position, -STEP);
            let numeric = (loss_of(&up) - loss_of(&down)) / (2.0 * STEP);
            let value = analytic[which].to_vec()[position];
            assert!(
                (value - numeric).abs() <= TOLERANCE * (1.0 + numeric.abs()),
                "tensor input {which} element {position} diverges: \
                 analytic {value}, numeric {numeric}"
            );
        }
    }
}

#[test]
fn forward_materializes_every_value() {
    let network = Network::new();
    let a = network.leaf(2.0_f64);
    let b = network.leaf(3.0);
    let c = network.leaf(4.0);
    let expression = -((a + b) * c);

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(a), 2.0);
    assert_eq!(*evaluation.of(expression), -20.0);
}

#[test]
fn backward_accumulates_gradients_through_fan_out() {
    let network = Network::new();
    let a = network.leaf(2.0_f64);
    let b = network.leaf(3.0);
    let output = a * b + a;

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(output), 8.0);

    let gradients = evaluation.backward(output);
    assert_eq!(*gradients.of(output), 1.0);
    assert_eq!(*gradients.of(a), 4.0);
    assert_eq!(*gradients.of(b), 2.0);
}

#[test]
fn backward_routes_negation() {
    let network = Network::new();
    let input = network.leaf(2.0_f64);
    let output = -(input * input);

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(output), -4.0);
    assert_eq!(*evaluation.backward(output).of(input), -4.0);
}

#[test]
fn subtraction_routes_signed_gradients() {
    let network = Network::new();
    let left = network.leaf(5.0_f64);
    let right = network.leaf(3.0);
    let difference = left - right;

    let gradients = network.forward().backward(difference);
    assert_eq!(*gradients.of(left), 1.0);
    assert_eq!(*gradients.of(right), -1.0);
}

#[test]
fn division_reuses_its_output_in_backward() {
    let network = Network::new();
    let left = network.leaf(6.0_f64);
    let right = network.leaf(2.0);
    let quotient = left / right;

    let gradients = network.forward().backward(quotient);
    assert_eq!(*gradients.of(left), 0.5);
    assert_eq!(*gradients.of(right), -1.5);
}

#[test]
fn tanh_routes_gradient_through_its_output() {
    let network = Network::new();
    let input = network.leaf(0.5_f64);
    let output = input.tanh();

    let evaluation = network.forward();
    let expected = 1.0 - 0.5_f64.tanh().powi(2);
    assert!((evaluation.backward(output).of(input) - expected).abs() < 1e-12);
}

#[test]
fn exp_reuses_its_output_in_backward() {
    let network = Network::new();
    let input = network.leaf(1.0_f64);
    let output = input.exp();

    let evaluation = network.forward();
    let value = *evaluation.of(output);
    assert!((value - std::f64::consts::E).abs() < 1e-12);
    assert!((evaluation.backward(output).of(input) - value).abs() < 1e-12);
}

#[test]
fn ln_routes_gradient_through_its_operand() {
    let network = Network::new();
    let input = network.leaf(2.0_f64);
    let output = input.ln();

    let gradients = network.forward().backward(output);
    assert!((gradients.of(input) - 0.5).abs() < 1e-12);
}

#[test]
fn sigmoid_composes_from_primitives() {
    let network = Network::new();
    let input = network.leaf(0.0_f64);
    let one = network.leaf(1.0);
    let sigmoid = one / (one + (-input).exp());

    let evaluation = network.forward();
    assert!((evaluation.of(sigmoid) - 0.5).abs() < 1e-12);
    assert!((evaluation.backward(sigmoid).of(input) - 0.25).abs() < 1e-12);
}

#[test]
fn backward_survives_later_recordings() {
    let network = Network::new();
    let input = network.leaf(2.0_f64);
    let evaluation = network.forward();
    network.leaf(3.0);

    assert_eq!(*evaluation.backward(input).of(input), 1.0);
}

#[test]
fn backward_skips_disconnected_nodes() {
    let network = Network::new();
    let unrelated = network.leaf(0.0_f64);
    let quotient = unrelated / unrelated;
    let input = network.leaf(2.0);
    let target = input * input;

    let gradients = network.forward().backward(target);
    assert_eq!(*gradients.of(input), 4.0);
    assert_eq!(*gradients.of(unrelated), 0.0);
    assert_eq!(*gradients.of(quotient), 0.0);
}

#[test]
fn backward_ignores_singular_paths_through_shared_leaves() {
    let network = Network::new();
    let input = network.leaf(0.0_f64);
    let _quotient = input / input;
    let target = input * input;

    assert_eq!(*network.forward().backward(target).of(input), 0.0);
}

#[test]
fn backward_skips_nodes_recorded_after_the_target() {
    let network = Network::new();
    let input = network.leaf(2.0_f64);
    let target = input * input;
    let late = network.leaf(0.0);
    let quotient = late / late;

    let gradients = network.forward().backward(target);
    assert_eq!(*gradients.of(input), 4.0);
    assert_eq!(*gradients.of(late), 0.0);
    assert_eq!(*gradients.of(quotient), 0.0);
}

#[test]
#[should_panic(expected = "allocated after")]
fn backward_rejects_later_targets() {
    let network = Network::new();
    let _ = network.leaf(2.0_f64);
    let evaluation = network.forward();
    let late = network.leaf(3.0);
    evaluation.backward(late);
}

#[test]
#[should_panic(expected = "different network")]
fn backward_rejects_foreign_targets() {
    let first = Network::new();
    let second = Network::new();
    let _ = first.leaf(1.0_f64);
    let foreign = second.leaf(2.0);
    first.forward().backward(foreign);
}
