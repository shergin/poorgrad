use crate::{Adam, AdamW, Compile, Network, Optimizer, Sgd, Tensor};

/// The conventional hyperparameters as single-value payloads.
fn conventional() -> (Tensor<f64>, Tensor<f64>, Tensor<f64>) {
    (
        Tensor::new([], [0.9_f64]),
        Tensor::new([], [0.999_f64]),
        Tensor::new([], [1e-8_f64]),
    )
}

#[test]
fn two_steps_match_the_paper_trace() {
    // `loss = w * w` over a single scalar-shaped parameter: the
    // gradient is `2w`, and two Adam steps are traced by hand from
    // the paper's update rule.
    let network = Network::new();
    let w = network.parameter(Tensor::new([], [1.0_f64])).symbol();
    let loss = {
        let w = network.resolve(w);
        (w * w).sum().symbol()
    };
    let (beta1, beta2, epsilon) = conventional();
    let mut adam = Adam::new(beta1, beta2, epsilon);
    let rate = Tensor::new([], [0.1_f64]);

    let mut network = network;
    let mut expected_w = 1.0_f64;
    let (mut m, mut v) = (0.0_f64, 0.0);
    let (mut beta1_power, mut beta2_power) = (1.0_f64, 1.0);
    for _ in 0..2 {
        let run = network.forward();
        let gradients = run.backward(network.resolve(loss));
        network = adam.step(&network, &gradients, &rate);

        let gradient = 2.0 * expected_w;
        m = m * 0.9 + gradient * (1.0 - 0.9);
        v = v * 0.999 + gradient * gradient * (1.0 - 0.999);
        beta1_power *= 0.9;
        beta2_power *= 0.999;
        let corrected_m = m / (1.0 - beta1_power);
        let corrected_v = v / (1.0 - beta2_power);
        expected_w -= 0.1 * corrected_m / (corrected_v.sqrt() + 1e-8);

        let stepped = network.resolve(w).payload().unwrap().to_vec()[0];
        assert!(
            (stepped - expected_w).abs() < 1e-12,
            "stepped {stepped}, expected {expected_w}"
        );
    }
}

#[test]
fn identical_runs_are_bitwise_identical() {
    let run = || {
        let network = Network::new();
        let w = network
            .parameter(Tensor::new([2, 2], [1.0_f64, -0.5, 0.25, 2.0]))
            .symbol();
        let loss = {
            let w = network.resolve(w);
            (w * w).sum().symbol()
        };
        let (beta1, beta2, epsilon) = conventional();
        let mut adam = Adam::new(beta1, beta2, epsilon);
        let rate = Tensor::new([], [0.05_f64]);
        let mut network = network;
        for _ in 0..5 {
            let run = network.forward();
            let gradients = run.backward(network.resolve(loss));
            network = adam.step(&network, &gradients, &rate);
        }
        network.resolve(w).payload().unwrap().to_vec()
    };
    for (first, second) in run().iter().zip(run()) {
        assert_eq!(first.to_bits(), second.to_bits());
    }
}

#[test]
fn adamw_decays_weights_and_spares_biases() {
    let build = || {
        let network = Network::new();
        let (weights, bias, loss) = {
            let weights = network.parameter(Tensor::new([2, 2], [1.0_f64, -0.5, 0.25, 2.0]));
            let bias = network.parameter(Tensor::new([2], [0.5_f64, -1.0]));
            let x = network.leaf(Tensor::new([1, 2], [1.0_f64, -1.0]));
            let product = x.matmul(weights);
            let shifted = product + bias.broadcast_along(0, product);
            let loss = (shifted * shifted).sum();
            (weights.symbol(), bias.symbol(), loss.symbol())
        };
        (network, weights, bias, loss)
    };

    let (network, weights, bias, loss) = build();
    let (beta1, beta2, epsilon) = conventional();
    let mut plain = Adam::new(beta1.clone(), beta2.clone(), epsilon.clone());
    let mut decoupled = AdamW::new(beta1, beta2, epsilon, Tensor::new([], [0.1_f64]));
    let rate = Tensor::new([], [0.05_f64]);

    let run = network.forward();
    let gradients = run.backward(network.resolve(loss));
    let by_adam = plain.step(&network, &gradients, &rate);
    let by_adamw = decoupled.step(&network, &gradients, &rate);

    // The rank-1 bias is spared: both routes agree bitwise there.
    let adam_bias = by_adam.resolve(bias).payload().unwrap().to_vec();
    let adamw_bias = by_adamw.resolve(bias).payload().unwrap().to_vec();
    for (plain, decayed) in adam_bias.iter().zip(&adamw_bias) {
        assert_eq!(plain.to_bits(), decayed.to_bits());
    }

    // The rank-2 weight differs by exactly the decoupled decay term.
    let before = network.resolve(weights).payload().unwrap().to_vec();
    let adam_weights = by_adam.resolve(weights).payload().unwrap().to_vec();
    let adamw_weights = by_adamw.resolve(weights).payload().unwrap().to_vec();
    for ((plain, decayed), original) in adam_weights.iter().zip(&adamw_weights).zip(before) {
        let term = original * 0.1 * 0.05;
        assert!((plain - decayed - term).abs() < 1e-15);
    }
}

#[test]
fn step_where_overrides_the_structural_policy() {
    let network = Network::new();
    let weights = network
        .parameter(Tensor::new([2, 2], [1.0_f64, -0.5, 0.25, 2.0]))
        .symbol();
    let loss = {
        let weights = network.resolve(weights);
        (weights * weights).sum().symbol()
    };
    let (beta1, beta2, epsilon) = conventional();
    let mut plain = Adam::new(beta1.clone(), beta2.clone(), epsilon.clone());
    let mut decoupled = AdamW::new(beta1, beta2, epsilon, Tensor::new([], [0.1_f64]));
    let rate = Tensor::new([], [0.05_f64]);

    let run = network.forward();
    let gradients = run.backward(network.resolve(loss));
    let by_adam = plain.step(&network, &gradients, &rate);
    // Decay nothing: AdamW must reproduce Adam bitwise.
    let spared = decoupled.step_where(&network, &gradients, &rate, |_| false);

    let adam_payload = by_adam.resolve(weights).payload().unwrap();
    let spared_payload = spared.resolve(weights).payload().unwrap();
    for (plain, decayed) in adam_payload.to_vec().iter().zip(spared_payload.to_vec()) {
        assert_eq!(plain.to_bits(), decayed.to_bits());
    }
}

#[test]
fn recorded_gradients_feed_adam_bitwise() {
    // A field is a field: the compiled-training route and the engine's
    // backward produce the same Adam trajectory.
    let build = || {
        let network = Network::new();
        let (w, loss) = {
            let w = network.parameter(Tensor::new([2, 2], [1.0_f64, -0.5, 0.25, 2.0]));
            (w.symbol(), (w * w).sum().symbol())
        };
        (network, w, loss)
    };
    let (beta1, beta2, epsilon) = conventional();
    let rate = Tensor::new([], [0.05_f64]);

    let (engine_network, engine_w, engine_loss) = build();
    let mut engine_adam = Adam::new(beta1.clone(), beta2.clone(), epsilon.clone());
    let mut engine_network = engine_network;

    let (recorded_network, recorded_w, recorded_loss) = build();
    let gradient_symbols = recorded_network.differentiate(recorded_loss, [recorded_w]);
    let plan = recorded_network.compile(Compile::roots(
        std::iter::once(recorded_loss).chain(gradient_symbols.iter().copied()),
    ));
    let mut recorded_adam = Adam::new(beta1, beta2, epsilon);
    let mut recorded_network = recorded_network;

    for _ in 0..4 {
        let run = engine_network.forward();
        let gradients = run.backward(engine_network.resolve(engine_loss));
        engine_network = engine_adam.step(&engine_network, &gradients, &rate);

        let run = plan.forward(&recorded_network, []);
        let gradients = run.recorded_gradients([(
            recorded_network.resolve(recorded_w),
            recorded_network.resolve(gradient_symbols[0]),
        )]);
        recorded_network = recorded_adam.step(&recorded_network, &gradients, &rate);

        let engine_payload = engine_network.resolve(engine_w).payload().unwrap();
        let recorded_payload = recorded_network.resolve(recorded_w).payload().unwrap();
        for (engine, recorded) in engine_payload
            .to_vec()
            .iter()
            .zip(recorded_payload.to_vec())
        {
            assert_eq!(engine.to_bits(), recorded.to_bits());
        }
    }
}

#[test]
fn adam_descends_faster_than_sgd_on_a_skewed_bowl() {
    // A quadratic bowl with wildly different curvatures per axis: the
    // fixed problem where per-coordinate step normalization pays.
    let run = |strategy: &mut dyn Optimizer<Tensor<f64>>| {
        let network = Network::new();
        let loss = {
            let w = network.parameter(Tensor::new([2], [5.0_f64, 5.0]));
            let curvatures = network.leaf(Tensor::new([2], [100.0_f64, 0.01]));
            (w * w * curvatures).sum().symbol()
        };
        let rate = Tensor::new([], [0.01_f64]);
        let mut network = network;
        for _ in 0..100 {
            let run = network.forward();
            let gradients = run.backward(network.resolve(loss));
            network = strategy.step(&network, &gradients, &rate);
        }
        let run = network.forward();
        run.of(network.resolve(loss)).to_vec()[0]
    };

    let (beta1, beta2, epsilon) = conventional();
    let sgd_loss = run(&mut Sgd);
    let adam_loss = run(&mut Adam::new(beta1, beta2, epsilon));
    assert!(adam_loss.is_finite() && sgd_loss.is_finite());
    assert!(
        adam_loss < sgd_loss,
        "adam {adam_loss} should beat sgd {sgd_loss} on the skewed bowl"
    );
}
