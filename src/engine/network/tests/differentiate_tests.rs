use crate::{Compile, Differentiable, Network, Symbol, Tensor};

/// Asserts the closure contract for one recorded graph: the gradients
/// `differentiate` records must reproduce `Run::backward`
/// **bitwise** — same seed, same masking, same accumulation order —
/// for every `wrt` entry. This fixture family is the no-fork
/// guarantee: it fails if a rule uses an untraceable payload call, if
/// a variant ships without adjoint closure, or if the two scans'
/// arithmetic drifts apart.
fn assert_closure(network: &Network<Tensor<f64>>, loss: Symbol, wrt: &[Symbol]) {
    let gradients = network.differentiate(loss, wrt.iter().copied());
    let run = network.forward();
    let engine = run.backward(network.resolve(loss));
    for (&target, gradient) in wrt.iter().zip(gradients) {
        let recorded = run.of(network.resolve(gradient)).to_vec();
        let computed = engine.of(network.resolve(target)).to_vec();
        assert_eq!(recorded.len(), computed.len());
        for (recorded, computed) in recorded.iter().zip(&computed) {
            assert_eq!(
                recorded.to_bits(),
                computed.to_bits(),
                "recorded gradient {recorded} differs from the engine's {computed}"
            );
        }
    }
}

/// A small varied payload: values spread over both signs, no zeros.
fn varied(shape: impl Into<crate::Shape>, seed: usize) -> Tensor<f64> {
    let shape = shape.into();
    let volume = shape.volume();
    Tensor::new(
        shape,
        (0..volume)
            .map(|index| ((index * 7 + seed * 3) % 11) as f64 * 0.375 - 1.5)
            .collect::<Vec<_>>(),
    )
}

#[test]
fn add_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let b = network.parameter(varied([2, 3], 2));
    let loss = (a + b).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol(), b.symbol()]);
}

#[test]
fn sub_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let b = network.parameter(varied([2, 3], 2));
    let loss = (a - b).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol(), b.symbol()]);
}

#[test]
fn mul_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let b = network.parameter(varied([2, 3], 2));
    let loss = (a * b).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol(), b.symbol()]);
}

#[test]
fn div_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let b = network.parameter(Tensor::new(
        [2, 3],
        (0..6).map(|v| v as f64 * 0.5 + 1.0).collect::<Vec<_>>(),
    ));
    let loss = (a / b).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol(), b.symbol()]);
}

#[test]
fn neg_closes() {
    let network = Network::new();
    let a = network.parameter(varied([4], 1));
    let loss = (-a).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn tanh_closes() {
    let network = Network::new();
    let a = network.parameter(varied([4], 1));
    let loss = a.tanh().sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn exp_closes() {
    let network = Network::new();
    let a = network.parameter(varied([4], 1));
    let loss = a.exp().sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn ln_closes() {
    let network = Network::new();
    let a = network.parameter(Tensor::new(
        [4],
        (1..=4).map(|v| v as f64 * 0.75).collect::<Vec<_>>(),
    ));
    let loss = a.ln().sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn sqrt_closes() {
    let network = Network::new();
    let a = network.parameter(Tensor::new(
        [4],
        (1..=4).map(|v| v as f64 * 1.25).collect::<Vec<_>>(),
    ));
    let loss = a.sqrt().sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn powf_closes() {
    let network = Network::new();
    let base = network.parameter(Tensor::new(
        [3],
        (1..=3).map(|v| v as f64 * 0.5 + 0.25).collect::<Vec<_>>(),
    ));
    let exponent = network.parameter(Tensor::new([3], [2.0_f64, 0.5, 3.0]));
    let loss = base.powf(exponent).sum();
    assert_closure(&network, loss.symbol(), &[base.symbol(), exponent.symbol()]);
}

#[test]
fn maximum_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let b = network.parameter(varied([2, 3], 2));
    let loss = a.maximum(b).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol(), b.symbol()]);
}

#[test]
fn relu_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let loss = a.relu().sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn step_closes() {
    let network = Network::new();
    // The step's operands are data, so the gradient reaches `a` only
    // through the product's other path — with `a` on both sides, the
    // fan-out accumulation is exercised too.
    let a = network.parameter(varied([2, 3], 1));
    let threshold = network.leaf(Tensor::filled([2, 3], 0.0_f64));
    let loss = (a.step(threshold) * a).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn matmul_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let b = network.parameter(varied([3, 4], 2));
    let loss = a.matmul(b).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol(), b.symbol()]);
}

#[test]
fn transpose_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let weights = network.leaf(varied([3, 2], 2));
    let loss = (a.transpose() * weights).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn sum_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let loss = a.sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn sum_along_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let weights = network.leaf(varied([3], 2));
    let loss = (a.sum_along(0) * weights).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn broadcast_like_closes() {
    let network = Network::new();
    let a = network.parameter(Tensor::filled([], 1.25_f64));
    let reference = network.leaf(varied([2, 3], 2));
    let loss = (a.broadcast_like(reference) * reference).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn broadcast_along_closes() {
    let network = Network::new();
    let a = network.parameter(varied([3], 1));
    let reference = network.leaf(varied([2, 3], 2));
    let loss = (a.broadcast_along(0, reference) * reference).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn reshape_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let weights = network.leaf(varied([6], 2));
    let loss = (a.reshape([6]) * weights).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn permute_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3, 4], 1));
    let weights = network.leaf(varied([4, 2, 3], 2));
    let loss = (a.permute([2, 0, 1]) * weights).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn narrow_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 5], 1));
    let weights = network.leaf(varied([2, 3], 2));
    let loss = (a.narrow(1, 1, 3) * weights).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn pad_closes() {
    let network = Network::new();
    let a = network.parameter(varied([2, 3], 1));
    let weights = network.leaf(varied([2, 6], 2));
    let loss = (a.pad(1, 2, 6) * weights).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn unfold_closes() {
    let network = Network::new();
    let a = network.parameter(varied([8], 1));
    let weights = network.leaf(varied([3, 3], 2));
    let loss = (a.unfold(0, 3, 2, 1) * weights).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn fold_closes() {
    let network = Network::new();
    let a = network.parameter(varied([3, 3], 1));
    let weights = network.leaf(varied([8], 2));
    let loss = (a.fold(0, 3, 2, 1, 8) * weights).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn gather_closes() {
    let network = Network::new();
    let table = network.parameter(varied([3, 2], 1));
    let selection = network.input(Tensor::selection(vec![0_usize, 2, 0], 3, 1.0));
    let weights = network.leaf(varied([3, 2], 2));
    let loss = (table.gather(selection) * weights).sum();
    assert_closure(&network, loss.symbol(), &[table.symbol()]);
}

#[test]
fn scatter_closes() {
    let network = Network::new();
    let rows = network.parameter(varied([3, 2], 1));
    let selection = network.input(Tensor::selection(vec![0_usize, 2, 0], 3, 1.0));
    let weights = network.leaf(varied([3, 2], 2));
    let loss = (rows.scatter(selection, 3) * weights).sum();
    assert_closure(&network, loss.symbol(), &[rows.symbol()]);
}

#[test]
fn log_softmax_closes() {
    let network = Network::new();
    let logits = network.parameter(varied([2, 4], 1));
    let weights = network.leaf(varied([2, 4], 2));
    let loss = (logits.log_softmax(1) * weights).sum();
    assert_closure(&network, loss.symbol(), &[logits.symbol()]);
}

#[test]
fn logsumexp_closes() {
    let network = Network::new();
    let logits = network.parameter(varied([2, 4], 1));
    let weights = network.leaf(varied([2], 2));
    let loss = (logits.logsumexp(1) * weights).sum();
    assert_closure(&network, loss.symbol(), &[logits.symbol()]);
}

#[test]
fn fan_out_accumulates_in_engine_order() {
    let network = Network::new();
    // One parameter feeding three consumers: the recorded `Add` chain
    // must fold the contributions exactly as the engine's scan does.
    let a = network.parameter(varied([2, 3], 1));
    let loss = (a * a).sum() + a.tanh().sum() + (-a).sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
fn a_composed_loss_closes_through_a_plan() {
    let network = Network::new();
    // The end-to-end shape of E2: a small dense model's loss,
    // differentiated, compiled with its gradients into one forward-only
    // plan, and checked bitwise against the engine's backward.
    let x = network.input(varied([2, 3], 1));
    let weights = network.parameter(varied([3, 2], 2));
    let bias = network.parameter(varied([2], 3));
    let logits = x.matmul(weights) + bias.broadcast_along(0, x.matmul(weights));
    let loss = logits.tanh().sum();

    let gradients = network.differentiate(loss.symbol(), [weights.symbol(), bias.symbol()]);
    let plan = network.compile(Compile::roots(
        std::iter::once(loss.symbol()).chain(gradients.iter().copied()),
    ));
    let planned = plan.forward(&network, []);
    let engine = network.forward().backward(network.resolve(loss.symbol()));
    for (&target, gradient) in [weights.symbol(), bias.symbol()].iter().zip(gradients) {
        let recorded = planned.of(network.resolve(gradient)).to_vec();
        let computed = engine.of(network.resolve(target)).to_vec();
        for (recorded, computed) in recorded.iter().zip(&computed) {
            assert_eq!(recorded.to_bits(), computed.to_bits());
        }
    }
}

#[test]
fn non_ancestors_answer_recorded_zeros() {
    let network = Network::new();
    let a = network.parameter(varied([2], 1));
    let unrelated = network.parameter(varied([3], 2));
    let loss = a.sum();
    let gradients = network.differentiate(loss.symbol(), [unrelated.symbol()]);
    let run = network.forward();
    assert_eq!(run.of(network.resolve(gradients[0])).to_vec(), &[0.0; 3]);
}

#[test]
fn singular_disconnected_expressions_stay_masked() {
    let network = Network::new();
    // The PG-001 semantics carry over: a disconnected division by zero
    // must not poison a recorded gradient, because non-ancestors' rules
    // are never recorded at all.
    let a = network.parameter(varied([2], 1));
    let zero = network.leaf(Tensor::filled([2], 0.0_f64));
    let _poison = zero / zero;
    let loss = a.sum();
    assert_closure(&network, loss.symbol(), &[a.symbol()]);
}

#[test]
#[should_panic(expected = "scalar loss")]
fn differentiate_rejects_non_scalar_losses() {
    let network = Network::new();
    let a = network.parameter(varied([2], 1));
    let doubled = a + a;
    network.differentiate(doubled.symbol(), [a.symbol()]);
}

#[test]
fn second_derivative_of_a_cubic_is_exact() {
    let network = Network::new();
    let x = network.parameter(Tensor::new([3], [0.5_f64, -1.25, 2.0]));
    let loss = (x * x * x).sum();

    let first = network.differentiate(loss.symbol(), [x.symbol()]);
    let first_value = network.resolve(first[0]);
    let second = network.differentiate(first_value.sum().symbol(), [x.symbol()]);

    let run = network.forward();
    let computed = run.of(network.resolve(second[0])).to_vec();
    for (computed, x) in computed.iter().zip([0.5_f64, -1.25, 2.0]) {
        assert_eq!(*computed, 6.0 * x);
    }
}

#[test]
fn second_derivative_of_tanh_matches_finite_differences() {
    let probe = 0.65_f64;
    let network = Network::new();
    let x = network.parameter(Tensor::new([1], [probe]));
    let loss = x.tanh().sum();
    let first = network.differentiate(loss.symbol(), [x.symbol()]);
    let second = network.differentiate(network.resolve(first[0]).sum().symbol(), [x.symbol()]);

    let run = network.forward();
    let computed = run.of(network.resolve(second[0])).to_vec()[0];
    let step = 1e-6;
    let derivative_at = |x: f64| 1.0 - x.tanh().powi(2);
    let expected = (derivative_at(probe + step) - derivative_at(probe - step)) / (2.0 * step);
    assert!((computed - expected).abs() < 1e-6);
}

#[test]
fn relu_hessians_are_exact_zeros() {
    let network = Network::new();
    // The `Step` rule's `None` cotangents: differentiating a relu
    // gradient answers zero almost everywhere, never `NaN`.
    let x = network.parameter(Tensor::new([3], [-2.0_f64, 0.5, 3.0]));
    let loss = x.relu().sum();
    let first = network.differentiate(loss.symbol(), [x.symbol()]);
    let second = network.differentiate(network.resolve(first[0]).sum().symbol(), [x.symbol()]);

    let run = network.forward();
    assert_eq!(run.of(network.resolve(second[0])).to_vec(), &[0.0; 3]);
}

#[test]
fn tape_growth_stays_a_small_constant() {
    let network = Network::new();
    let x = network.input(varied([4, 3], 1));
    let weights = network.parameter(varied([3, 4], 2));
    let logits = x
        .matmul(weights)
        .tanh()
        .matmul(network.parameter(varied([4, 2], 3)));
    let loss = logits.log_softmax(1).sum();

    let before = network.len();
    network.differentiate(loss.symbol(), [weights.symbol()]);
    let after = network.len();
    // The design's expectation: a small constant per forward node.
    // The measured ratio is recorded in notes/differentiate.md.
    assert!(
        after - before <= before * 6,
        "differentiating {before} nodes appended {}",
        after - before
    );
}

#[test]
fn a_recorded_training_loop_matches_the_engine_bitwise() {
    // Two identical models under matched seeds: one trains through the
    // engine's backward, the other through a compiled
    // `[loss, gradients...]` forward-only plan and
    // `recorded_gradients` — every generation's parameters must agree
    // bit for bit, because both loops fold the same arithmetic.
    let build = |network: &Network<Tensor<f64>>| {
        let x = network.input(varied([4, 3], 1));
        let weights = network.parameter(varied([3, 4], 2));
        let bias = network.parameter(varied([4], 3));
        let product = x.matmul(weights);
        let hidden = (product + bias.broadcast_along(0, product)).tanh();
        let head = network.parameter(varied([4, 2], 4));
        let loss = hidden.matmul(head).log_softmax(1).sum();
        (
            x.symbol(),
            [weights.symbol(), bias.symbol(), head.symbol()],
            loss.symbol(),
        )
    };

    let engine_network = Network::new();
    let (engine_x, engine_params, engine_loss) = build(&engine_network);
    let recorded_network = Network::new();
    let (recorded_x, recorded_params, recorded_loss) = build(&recorded_network);
    let gradients = recorded_network.differentiate(recorded_loss, recorded_params.iter().copied());
    let plan = recorded_network.compile(Compile::roots(
        std::iter::once(recorded_loss).chain(gradients.iter().copied()),
    ));

    let mut engine_network = engine_network;
    let mut recorded_network = recorded_network;
    for step in 0..12 {
        let batch = varied([4, 3], 10 + step);

        let engine_run = engine_network.forward_with([(engine_x, batch.clone())]);
        let engine_gradients = engine_run.backward(engine_network.resolve(engine_loss));
        engine_network = engine_network.update(&engine_gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * Tensor::filled(gradient.shape(), 0.05)
        });

        let recorded_run = plan.forward(&recorded_network, [(recorded_x, batch)]);
        let recorded_field =
            recorded_run.recorded_gradients(recorded_params.iter().zip(&gradients).map(
                |(&parameter, &gradient)| {
                    (
                        recorded_network.resolve(parameter),
                        recorded_network.resolve(gradient),
                    )
                },
            ));
        recorded_network = recorded_network.update(&recorded_field, |parameter, gradient| {
            parameter.clone() - gradient.clone() * Tensor::filled(gradient.shape(), 0.05)
        });

        for (&engine_param, &recorded_param) in engine_params.iter().zip(&recorded_params) {
            let engine_payload = engine_network
                .resolve(engine_param)
                .payload()
                .expect("parameters carry payloads");
            let recorded_payload = recorded_network
                .resolve(recorded_param)
                .payload()
                .expect("parameters carry payloads");
            for (engine, recorded) in engine_payload
                .to_vec()
                .iter()
                .zip(recorded_payload.to_vec())
            {
                assert_eq!(engine.to_bits(), recorded.to_bits(), "step {step}");
            }
        }
    }
}

#[test]
#[should_panic(expected = "is not a parameter")]
fn recorded_gradients_reject_swapped_pairs() {
    let network = Network::new();
    let a = network.parameter(varied([2], 1));
    let loss = a.sum();
    let gradients = network.differentiate(loss.symbol(), [a.symbol()]);
    let run = network.forward();
    run.recorded_gradients([(network.resolve(gradients[0]), network.resolve(a.symbol()))]);
}
