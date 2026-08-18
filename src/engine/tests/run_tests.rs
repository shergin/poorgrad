use crate::{Differentiable, Tape, Tensor, Value};

/// The half-width of the central difference.
const STEP: f64 = 1e-5;

/// The mixed absolute and relative tolerance of a comparison.
const TOLERANCE: f64 = 1e-6;

/// Asserts that the analytic gradients of `expression` match central
/// finite differences at `inputs`, for every input.
///
/// Each numeric probe rebuilds the graph on a fresh tape with one
/// input nudged, so the check exercises recording, forward, and backward
/// exactly as a user would.
fn assert_gradients_match<const INPUTS: usize>(
    inputs: [f64; INPUTS],
    expression: impl for<'tape> Fn([Value<'tape, f64>; INPUTS]) -> Value<'tape, f64>,
) {
    let evaluate = |point: [f64; INPUTS]| -> f64 {
        let tape = Tape::new();
        let target = expression(point.map(|value| tape.leaf(value))).symbol();
        let network = tape.into_network();
        *network.forward(&network.parameters(), []).of(target)
    };

    let tape = Tape::new();
    let leaves = inputs.map(|value| tape.leaf(value));
    let target = expression(leaves).symbol();
    let leaves = leaves.map(|leaf| leaf.symbol());
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(target);

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
    Tensor::new(tensor.shape(), elements)
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
        let tape = Tape::new();
        let x = tape.leaf(tensors[0].clone());
        let w = tape.leaf(tensors[1].clone());
        let bias = tape.leaf(tensors[2].clone());
        let y = tape.leaf(tensors[3].clone());
        let product = x.matmul(w);
        let activated = (product + bias.broadcast_along(0, product)).tanh();
        let error = activated - y;
        let loss = (error * error).sum().symbol();
        let network = tape.into_network();
        network.forward(&network.parameters(), []).of(loss).to_vec()[0]
    };

    let tape = Tape::new();
    let x = tape.leaf(base[0].clone());
    let w = tape.leaf(base[1].clone());
    let bias = tape.leaf(base[2].clone());
    let y = tape.leaf(base[3].clone());
    let product = x.matmul(w);
    let activated = (product + bias.broadcast_along(0, product)).tanh();
    let error = activated - y;
    let loss = (error * error).sum();
    let (x, w, bias, y, loss) = (
        x.symbol(),
        w.symbol(),
        bias.symbol(),
        y.symbol(),
        loss.symbol(),
    );
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(loss);
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
        let tape = Tape::new();
        let x = tape.leaf(tensors[0].clone());
        let w = tape.leaf(tensors[1].clone());
        let bias = tape.leaf(tensors[2].clone());
        let y = tape.leaf(tensors[3].clone());
        let product = x.matmul(w).transpose();
        let shifted = product + bias.broadcast_like(product);
        let error = shifted - y;
        let loss = (error * error).sum().symbol();
        let network = tape.into_network();
        network.forward(&network.parameters(), []).of(loss).to_vec()[0]
    };

    let tape = Tape::new();
    let x = tape.leaf(base[0].clone());
    let w = tape.leaf(base[1].clone());
    let bias = tape.leaf(base[2].clone());
    let y = tape.leaf(base[3].clone());
    let product = x.matmul(w).transpose();
    let shifted = product + bias.broadcast_like(product);
    let error = shifted - y;
    let loss = (error * error).sum();
    let (x, w, bias, y, loss) = (
        x.symbol(),
        w.symbol(),
        bias.symbol(),
        y.symbol(),
        loss.symbol(),
    );
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(loss);
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
    let tape = Tape::new();
    let a = tape.leaf(2.0_f64);
    let b = tape.leaf(3.0);
    let c = tape.leaf(4.0);
    let expression = -((a + b) * c);
    let (a, expression) = (a.symbol(), expression.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(*run.of(a), 2.0);
    assert_eq!(*run.of(expression), -20.0);
}

#[test]
fn backward_accumulates_gradients_through_fan_out() {
    let tape = Tape::new();
    let a = tape.leaf(2.0_f64);
    let b = tape.leaf(3.0);
    let output = a * b + a;
    let (a, b, output) = (a.symbol(), b.symbol(), output.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(*run.of(output), 8.0);

    let gradients = run.backward(output);
    assert_eq!(*gradients.of(output), 1.0);
    assert_eq!(*gradients.of(a), 4.0);
    assert_eq!(*gradients.of(b), 2.0);
}

#[test]
fn backward_routes_negation() {
    let tape = Tape::new();
    let input = tape.leaf(2.0_f64);
    let output = -(input * input);
    let (input, output) = (input.symbol(), output.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(*run.of(output), -4.0);
    assert_eq!(*run.backward(output).of(input), -4.0);
}

#[test]
fn subtraction_routes_signed_gradients() {
    let tape = Tape::new();
    let left = tape.leaf(5.0_f64);
    let right = tape.leaf(3.0);
    let difference = left - right;
    let (left, right, difference) = (left.symbol(), right.symbol(), difference.symbol());
    let network = tape.into_network();

    let gradients = network
        .forward(&network.parameters(), [])
        .backward(difference);
    assert_eq!(*gradients.of(left), 1.0);
    assert_eq!(*gradients.of(right), -1.0);
}

#[test]
fn division_reuses_its_output_in_backward() {
    let tape = Tape::new();
    let left = tape.leaf(6.0_f64);
    let right = tape.leaf(2.0);
    let quotient = left / right;
    let (left, right, quotient) = (left.symbol(), right.symbol(), quotient.symbol());
    let network = tape.into_network();

    let gradients = network
        .forward(&network.parameters(), [])
        .backward(quotient);
    assert_eq!(*gradients.of(left), 0.5);
    assert_eq!(*gradients.of(right), -1.5);
}

#[test]
fn tanh_routes_gradient_through_its_output() {
    let tape = Tape::new();
    let input = tape.leaf(0.5_f64);
    let output = input.tanh();
    let (input, output) = (input.symbol(), output.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    let expected = 1.0 - 0.5_f64.tanh().powi(2);
    assert!((run.backward(output).of(input) - expected).abs() < 1e-12);
}

#[test]
fn exp_reuses_its_output_in_backward() {
    let tape = Tape::new();
    let input = tape.leaf(1.0_f64);
    let output = input.exp();
    let (input, output) = (input.symbol(), output.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    let value = *run.of(output);
    assert!((value - std::f64::consts::E).abs() < 1e-12);
    assert!((run.backward(output).of(input) - value).abs() < 1e-12);
}

#[test]
fn ln_routes_gradient_through_its_operand() {
    let tape = Tape::new();
    let input = tape.leaf(2.0_f64);
    let output = input.ln();
    let (input, output) = (input.symbol(), output.symbol());
    let network = tape.into_network();

    let gradients = network.forward(&network.parameters(), []).backward(output);
    assert!((gradients.of(input) - 0.5).abs() < 1e-12);
}

#[test]
fn sigmoid_composes_from_primitives() {
    let tape = Tape::new();
    let input = tape.leaf(0.0_f64);
    let one = tape.leaf(1.0);
    let sigmoid = one / (one + (-input).exp());
    let (input, sigmoid) = (input.symbol(), sigmoid.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert!((run.of(sigmoid) - 0.5).abs() < 1e-12);
    assert!((run.backward(sigmoid).of(input) - 0.25).abs() < 1e-12);
}

#[test]
fn backward_survives_later_recordings() {
    let tape = Tape::new();
    let input = tape.leaf(2.0_f64).symbol();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);

    // Reopening the network and recording a later leaf must not disturb
    // a run taken before the reopen.
    let tape = network.into_tape();
    let _late = tape.leaf(3.0).symbol();

    assert_eq!(*run.backward(input).of(input), 1.0);
}

#[test]
fn backward_skips_disconnected_nodes() {
    let tape = Tape::new();
    let unrelated = tape.leaf(0.0_f64);
    let quotient = unrelated / unrelated;
    let input = tape.leaf(2.0);
    let target = input * input;
    let (unrelated, quotient, input, target) = (
        unrelated.symbol(),
        quotient.symbol(),
        input.symbol(),
        target.symbol(),
    );
    let network = tape.into_network();

    let gradients = network.forward(&network.parameters(), []).backward(target);
    assert_eq!(*gradients.of(input), 4.0);
    assert_eq!(*gradients.of(unrelated), 0.0);
    assert_eq!(*gradients.of(quotient), 0.0);
}

#[test]
fn backward_ignores_singular_paths_through_shared_leaves() {
    let tape = Tape::new();
    let input = tape.leaf(0.0_f64);
    let _quotient = input / input;
    let target = input * input;
    let (input, target) = (input.symbol(), target.symbol());
    let network = tape.into_network();

    let gradients = network.forward(&network.parameters(), []).backward(target);
    assert_eq!(*gradients.of(input), 0.0);
}

#[test]
fn backward_skips_singular_producers_of_broadcast_references() {
    let tape = Tape::new();
    let input = tape.leaf(0.0_f64);
    let singular_reference = input / input;
    let source = tape.leaf(2.0);
    let output = source.broadcast_like(singular_reference);
    let (input, singular_reference, source, output) = (
        input.symbol(),
        singular_reference.symbol(),
        source.symbol(),
        output.symbol(),
    );
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(*run.of(output), 2.0);

    // The reference contributes only its shape, so the target has no
    // differentiable dependence on `input`: its gradient is exactly
    // zero, never the NaN of the singular quotient's derivative rule.
    let gradients = run.backward(output);
    assert_eq!(*gradients.of(input), 0.0);
    assert_eq!(*gradients.of(singular_reference), 0.0);
    assert_eq!(*gradients.of(source), 1.0);
}

#[test]
fn backward_skips_singular_producers_of_axis_references() {
    let tape = Tape::new();
    let input = tape.leaf(Tensor::new([2, 2], [0.0_f64, 1.0, 2.0, 3.0]));
    let singular_reference = input / input;
    let source = tape.leaf(Tensor::new([2], [5.0_f64, 7.0]));
    let output = source.broadcast_along(0, singular_reference).sum();
    let (input, singular_reference, source, output) = (
        input.symbol(),
        singular_reference.symbol(),
        source.symbol(),
        output.symbol(),
    );
    let network = tape.into_network();

    let gradients = network.forward(&network.parameters(), []).backward(output);
    assert_eq!(gradients.of(input).to_vec(), [0.0; 4]);
    assert_eq!(gradients.of(singular_reference).to_vec(), [0.0; 4]);
    assert_eq!(gradients.of(source).to_vec(), [2.0, 2.0]);
}

#[test]
fn backward_skips_nodes_recorded_after_the_target() {
    let tape = Tape::new();
    let input = tape.leaf(2.0_f64);
    let target = input * input;
    let late = tape.leaf(0.0);
    let quotient = late / late;
    let (input, target, late, quotient) = (
        input.symbol(),
        target.symbol(),
        late.symbol(),
        quotient.symbol(),
    );
    let network = tape.into_network();

    let gradients = network.forward(&network.parameters(), []).backward(target);
    assert_eq!(*gradients.of(input), 4.0);
    assert_eq!(*gradients.of(late), 0.0);
    assert_eq!(*gradients.of(quotient), 0.0);
}

#[test]
#[should_panic(expected = "allocated after")]
fn backward_rejects_later_targets() {
    let tape = Tape::new();
    let _ = tape.leaf(2.0_f64);
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);

    // A symbol minted after the run, across a reopen, is outside the
    // run's coverage and must be rejected.
    let tape = network.into_tape();
    let late = tape.leaf(3.0).symbol();
    run.backward(late);
}

#[test]
#[should_panic(expected = "different network")]
fn backward_rejects_foreign_targets() {
    let first = Tape::new();
    let second = Tape::new();
    let _ = first.leaf(1.0_f64);
    let foreign = second.leaf(2.0).symbol();
    let network = first.into_network();
    network.forward(&network.parameters(), []).backward(foreign);
}

#[test]
fn pad_places_the_window_and_narrows_the_gradient() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]));
    let padded = x.pad(1, 1, 4);
    // Distinct weights make the narrowed-back gradient unambiguous.
    let weights = tape.leaf(Tensor::new(
        [2, 4],
        (1..=8).map(|v| v as f64).collect::<Vec<_>>(),
    ));
    let loss = (padded * weights).sum();
    let (x, padded, loss) = (x.symbol(), padded.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(
        run.of(padded).to_vec(),
        &[0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0]
    );

    let gradients = run.backward(loss);
    // The pad gradient is the weights with the zero lanes narrowed away.
    assert_eq!(gradients.of(x).to_vec(), &[2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn unfold_slides_windows_and_folds_the_gradient() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new(
        [8],
        (1..=8).map(|v| v as f64).collect::<Vec<_>>(),
    ));
    let windows = x.unfold(0, 3, 2, 1);
    let loss = windows.sum();
    let (x, windows, loss) = (x.symbol(), windows.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(
        run.of(windows).to_vec(),
        &[1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0]
    );

    let gradients = run.backward(loss);
    // Summing all windows grades each position by its window coverage;
    // position 7 is beyond the last window and receives zero.
    assert_eq!(
        gradients.of(x).to_vec(),
        &[1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn narrow_of_pad_roundtrips_the_value() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([3], [1.0_f64, 2.0, 3.0]));
    let roundtrip = x.pad(0, 2, 7).narrow(0, 2, 3);
    let loss = roundtrip.sum();
    let (x, roundtrip, loss) = (x.symbol(), roundtrip.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(roundtrip).to_vec(), &[1.0, 2.0, 3.0]);

    let gradients = run.backward(loss);
    assert_eq!(gradients.of(x).to_vec(), &[1.0, 1.0, 1.0]);
}

#[test]
#[should_panic(expected = "cannot take shape")]
fn scalar_reshape_to_rank_one_is_rejected_at_forward() {
    // Record-time inference cannot reject this (a rank-0 tensor may
    // legitimately take shape [1]), so the scalar payload rejects the
    // capability mismatch when the rule runs, keeping the recorded and
    // payload views of every shape coherent.
    let tape = Tape::new();
    let x = tape.leaf(2.0_f64);
    let _reshaped = x.reshape([1]);
    let network = tape.into_network();
    network.forward(&network.parameters(), []);
}
