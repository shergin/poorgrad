use crate::{Adam, AdamW, Differentiable, Optimizer, Sgd, Symbol, Tape, Tensor};

/// A tiny two-parameter model: `loss = (w * x + b)^2` summed over a
/// fixed batch, with a rank-2 weight and a rank-1 bias.
fn model(tape: &Tape<Tensor<f64>>) -> (Symbol, Symbol, Symbol) {
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f64, -0.25, 1.0, 0.75]));
    let bias = tape.parameter(Tensor::new([2], [0.1_f64, -0.2]));
    let x = tape.leaf(Tensor::new([3, 2], [1.0_f64, 2.0, -1.0, 0.5, 0.25, -2.0]));
    let product = x.matmul(weights);
    let shifted = product + bias.broadcast_along(0, product);
    let loss = (shifted * shifted).sum();
    (loss.symbol(), weights.symbol(), bias.symbol())
}

#[test]
fn sgd_matches_the_hand_written_rule_bitwise() {
    let tape = Tape::new();
    let (loss, weights, _) = model(&tape);
    let network = tape.into_network();
    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let gradients = run.backward(loss);
    let rate = Tensor::new([], [0.05_f64]);

    let by_hand = parameters.step(&gradients, |parameter, gradient| {
        parameter.clone() - gradient.clone() * Tensor::filled(gradient.shape(), 0.05)
    });
    let by_trait = Sgd.step(&parameters, &gradients, &rate);

    for (hand, stepped) in by_hand
        .of(weights)
        .to_vec()
        .iter()
        .zip(by_trait.of(weights).to_vec())
    {
        assert_eq!(hand.to_bits(), stepped.to_bits());
    }
}

#[test]
fn a_comparison_loop_runs_over_dynamic_optimizers() {
    // The trait is object-safe by design: a comparison loop steps
    // several strategies side by side through one dynamic slot.
    let rate = Tensor::new([], [0.01_f64]);
    let conventional = |value: f64| Tensor::new([], [value]);
    let mut sgd = Sgd;
    let mut adam = Adam::new(conventional(0.9), conventional(0.999), conventional(1e-8));
    let mut adamw = AdamW::new(
        conventional(0.9),
        conventional(0.999),
        conventional(1e-8),
        conventional(0.01),
    );
    let strategies: [&mut dyn Optimizer<Tensor<f64>>; 3] = [&mut sgd, &mut adam, &mut adamw];

    for strategy in strategies {
        let tape = Tape::new();
        let (loss, ..) = model(&tape);
        let network = tape.into_network();
        let mut parameters = network.parameters();
        let mut first = None;
        for _ in 0..25 {
            let run = network.forward(&parameters, []);
            let value = run.of(loss).to_vec()[0];
            first.get_or_insert(value);
            let gradients = run.backward(loss);
            parameters = strategy.step(&parameters, &gradients, &rate);
        }
        let run = network.forward(&parameters, []);
        let last = run.of(loss).to_vec()[0];
        let first = first.expect("the loop ran");
        assert!(
            last.is_finite() && last < first,
            "the strategy did not descend: {first} -> {last}"
        );
    }
}

#[test]
fn step_each_sees_every_parameter_with_its_identity() {
    let tape = Tape::new();
    let (loss, weights, bias) = model(&tape);
    let network = tape.into_network();
    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let gradients = run.backward(loss);

    let mut seen = Vec::new();
    let next = parameters.step_each(&gradients, |symbol, current, _| {
        seen.push((symbol, current.shape().rank()));
        current.clone()
    });
    assert_eq!(seen, vec![(weights, 2), (bias, 1)]);
    // The identity rule left every payload untouched.
    assert_eq!(next.of(weights).to_vec(), parameters.of(weights).to_vec());
}

#[test]
#[should_panic(expected = "single value")]
fn hyperparameters_must_hold_single_values() {
    Adam::new(
        Tensor::new([2], [0.9_f64, 0.9]),
        Tensor::new([], [0.999_f64]),
        Tensor::new([], [1e-8_f64]),
    );
}
