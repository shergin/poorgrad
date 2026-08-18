use crate::{Compile, Differentiable, Symbol, Tape, Tensor};

/// Asserts the closure contract for one recorded graph: the gradients
/// `differentiate` records must reproduce `Run::backward`
/// **bitwise** — same seed, same masking, same accumulation order —
/// for every `wrt` entry. This fixture family is the no-fork
/// guarantee: it fails if a rule uses an untraceable payload call, if
/// a variant ships without adjoint closure, or if the two scans'
/// arithmetic drifts apart. Sealing the tape is part of the fixture,
/// so it consumes it.
fn assert_closure(loss: Symbol, wrt: &[Symbol], tape: Tape<Tensor<f64>>) {
    let gradients = tape.differentiate(loss, wrt.iter().copied());
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let engine = run.backward(loss);
    for (&target, gradient) in wrt.iter().zip(gradients) {
        let recorded = run.of(gradient).to_vec();
        let computed = engine.of(target).to_vec();
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
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let b = tape.parameter(varied([2, 3], 2));
    let loss = (a + b).sum();
    assert_closure(loss.symbol(), &[a.symbol(), b.symbol()], tape);
}

#[test]
fn sub_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let b = tape.parameter(varied([2, 3], 2));
    let loss = (a - b).sum();
    assert_closure(loss.symbol(), &[a.symbol(), b.symbol()], tape);
}

#[test]
fn mul_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let b = tape.parameter(varied([2, 3], 2));
    let loss = (a * b).sum();
    assert_closure(loss.symbol(), &[a.symbol(), b.symbol()], tape);
}

#[test]
fn div_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let b = tape.parameter(Tensor::new(
        [2, 3],
        (0..6).map(|v| v as f64 * 0.5 + 1.0).collect::<Vec<_>>(),
    ));
    let loss = (a / b).sum();
    assert_closure(loss.symbol(), &[a.symbol(), b.symbol()], tape);
}

#[test]
fn neg_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([4], 1));
    let loss = (-a).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn tanh_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([4], 1));
    let loss = a.tanh().sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn exp_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([4], 1));
    let loss = a.exp().sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn ln_closes() {
    let tape = Tape::new();
    let a = tape.parameter(Tensor::new(
        [4],
        (1..=4).map(|v| v as f64 * 0.75).collect::<Vec<_>>(),
    ));
    let loss = a.ln().sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn sqrt_closes() {
    let tape = Tape::new();
    let a = tape.parameter(Tensor::new(
        [4],
        (1..=4).map(|v| v as f64 * 1.25).collect::<Vec<_>>(),
    ));
    let loss = a.sqrt().sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn powf_closes() {
    let tape = Tape::new();
    let base = tape.parameter(Tensor::new(
        [3],
        (1..=3).map(|v| v as f64 * 0.5 + 0.25).collect::<Vec<_>>(),
    ));
    let exponent = tape.parameter(Tensor::new([3], [2.0_f64, 0.5, 3.0]));
    let loss = base.powf(exponent).sum();
    assert_closure(loss.symbol(), &[base.symbol(), exponent.symbol()], tape);
}

#[test]
fn maximum_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let b = tape.parameter(varied([2, 3], 2));
    let loss = a.maximum(b).sum();
    assert_closure(loss.symbol(), &[a.symbol(), b.symbol()], tape);
}

#[test]
fn relu_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let loss = a.relu().sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn step_closes() {
    let tape = Tape::new();
    // The step's operands are data, so the gradient reaches `a` only
    // through the product's other path — with `a` on both sides, the
    // fan-out accumulation is exercised too.
    let a = tape.parameter(varied([2, 3], 1));
    let threshold = tape.leaf(Tensor::filled([2, 3], 0.0_f64));
    let loss = (a.step(threshold) * a).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn matmul_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let b = tape.parameter(varied([3, 4], 2));
    let loss = a.matmul(b).sum();
    assert_closure(loss.symbol(), &[a.symbol(), b.symbol()], tape);
}

#[test]
fn batched_matmul_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 2, 3], 1));
    let b = tape.parameter(varied([2, 3, 4], 2));
    let loss = a.matmul(b).sum();
    assert_closure(loss.symbol(), &[a.symbol(), b.symbol()], tape);
}

#[test]
fn batched_matmul_matches_the_head_loop_bitwise() {
    // Two formulations of the same two-slice product: batched, and
    // rank-2 head loops over narrowed slices. Products and operand
    // gradients agree bit for bit, because each batch slice runs the
    // same gemm.
    let batched_tape = Tape::new();
    let a = batched_tape.parameter(varied([2, 3, 4], 1));
    let b = batched_tape.parameter(varied([2, 4, 5], 2));
    let product = a.matmul(b);
    let loss = product.sum().symbol();
    let (a, b, product) = (a.symbol(), b.symbol(), product.symbol());
    let batched = batched_tape.into_network();
    let batched_run = batched.forward(&batched.parameters(), []);
    let batched_gradients = batched_run.backward(loss);

    let looped_tape = Tape::new();
    let a2 = looped_tape.parameter(varied([2, 3, 4], 1));
    let b2 = looped_tape.parameter(varied([2, 4, 5], 2));
    let heads: Vec<_> = (0..2)
        .map(|head| {
            let left = a2.narrow(0, head, 1).reshape([3, 4]);
            let right = b2.narrow(0, head, 1).reshape([4, 5]);
            left.matmul(right)
        })
        .collect();
    let looped_loss = (heads[0].sum() + heads[1].sum()).symbol();
    let head_symbols: Vec<Symbol> = heads.iter().map(|head| head.symbol()).collect();
    let (a2, b2) = (a2.symbol(), b2.symbol());
    let looped = looped_tape.into_network();
    let looped_run = looped.forward(&looped.parameters(), []);
    let looped_gradients = looped_run.backward(looped_loss);

    let slices: Vec<f64> = head_symbols
        .iter()
        .flat_map(|&head| looped_run.of(head).to_vec())
        .collect();
    assert_eq!(batched_run.of(product).to_vec(), slices);
    assert_eq!(
        batched_gradients.of(a).to_vec(),
        looped_gradients.of(a2).to_vec()
    );
    assert_eq!(
        batched_gradients.of(b).to_vec(),
        looped_gradients.of(b2).to_vec()
    );
}

#[test]
fn transpose_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let weights = tape.leaf(varied([3, 2], 2));
    let loss = (a.transpose() * weights).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn sum_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let loss = a.sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn sum_along_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let weights = tape.leaf(varied([3], 2));
    let loss = (a.sum_along(0) * weights).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn broadcast_like_closes() {
    let tape = Tape::new();
    let a = tape.parameter(Tensor::filled([], 1.25_f64));
    let reference = tape.leaf(varied([2, 3], 2));
    let loss = (a.broadcast_like(reference) * reference).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn broadcast_along_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([3], 1));
    let reference = tape.leaf(varied([2, 3], 2));
    let loss = (a.broadcast_along(0, reference) * reference).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn reshape_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let weights = tape.leaf(varied([6], 2));
    let loss = (a.reshape([6]) * weights).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn permute_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3, 4], 1));
    let weights = tape.leaf(varied([4, 2, 3], 2));
    let loss = (a.permute([2, 0, 1]) * weights).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn narrow_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 5], 1));
    let weights = tape.leaf(varied([2, 3], 2));
    let loss = (a.narrow(1, 1, 3) * weights).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn pad_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2, 3], 1));
    let weights = tape.leaf(varied([2, 6], 2));
    let loss = (a.pad(1, 2, 6) * weights).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn unfold_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([8], 1));
    let weights = tape.leaf(varied([3, 3], 2));
    let loss = (a.unfold(0, 3, 2, 1) * weights).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn fold_closes() {
    let tape = Tape::new();
    let a = tape.parameter(varied([3, 3], 1));
    let weights = tape.leaf(varied([8], 2));
    let loss = (a.fold(0, 3, 2, 1, 8) * weights).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn gather_closes() {
    let tape = Tape::new();
    let table = tape.parameter(varied([3, 2], 1));
    let selection = tape.input(Tensor::selection(vec![0_usize, 2, 0], 3, 1.0));
    let weights = tape.leaf(varied([3, 2], 2));
    let loss = (table.gather(selection) * weights).sum();
    assert_closure(loss.symbol(), &[table.symbol()], tape);
}

#[test]
fn scatter_closes() {
    let tape = Tape::new();
    let rows = tape.parameter(varied([3, 2], 1));
    let selection = tape.input(Tensor::selection(vec![0_usize, 2, 0], 3, 1.0));
    let weights = tape.leaf(varied([3, 2], 2));
    let loss = (rows.scatter(selection, 3) * weights).sum();
    assert_closure(loss.symbol(), &[rows.symbol()], tape);
}

#[test]
fn log_softmax_closes() {
    let tape = Tape::new();
    let logits = tape.parameter(varied([2, 4], 1));
    let weights = tape.leaf(varied([2, 4], 2));
    let loss = (logits.log_softmax(1) * weights).sum();
    assert_closure(loss.symbol(), &[logits.symbol()], tape);
}

#[test]
fn logsumexp_closes() {
    let tape = Tape::new();
    let logits = tape.parameter(varied([2, 4], 1));
    let weights = tape.leaf(varied([2], 2));
    let loss = (logits.logsumexp(1) * weights).sum();
    assert_closure(loss.symbol(), &[logits.symbol()], tape);
}

#[test]
fn fan_out_accumulates_in_engine_order() {
    let tape = Tape::new();
    // One parameter feeding three consumers: the recorded `Add` chain
    // must fold the contributions exactly as the engine's scan does.
    let a = tape.parameter(varied([2, 3], 1));
    let loss = (a * a).sum() + a.tanh().sum() + (-a).sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
fn a_composed_loss_closes_through_a_plan() {
    let tape = Tape::new();
    // The end-to-end shape of E2: a small dense model's loss,
    // differentiated, compiled with its gradients into one forward-only
    // plan, and checked bitwise against the engine's backward.
    let x = tape.input(varied([2, 3], 1));
    let weights = tape.parameter(varied([3, 2], 2));
    let bias = tape.parameter(varied([2], 3));
    let logits = x.matmul(weights) + bias.broadcast_along(0, x.matmul(weights));
    let loss = logits.tanh().sum();

    let targets = [weights.symbol(), bias.symbol()];
    let gradients = tape.differentiate(loss.symbol(), targets);
    let loss = loss.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();
    let plan = network.compile(Compile::roots(
        std::iter::once(loss).chain(gradients.iter().copied()),
    ));
    let planned = plan.forward(&parameters, []);
    let engine = network.forward(&parameters, []).backward(loss);
    for (&target, gradient) in targets.iter().zip(gradients) {
        let recorded = planned.of(gradient).to_vec();
        let computed = engine.of(target).to_vec();
        for (recorded, computed) in recorded.iter().zip(&computed) {
            assert_eq!(recorded.to_bits(), computed.to_bits());
        }
    }
}

#[test]
fn non_ancestors_answer_recorded_zeros() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2], 1));
    let unrelated = tape.parameter(varied([3], 2));
    let loss = a.sum();
    let gradients = tape.differentiate(loss, [unrelated]);
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(gradients[0]).to_vec(), &[0.0; 3]);
}

#[test]
fn singular_disconnected_expressions_stay_masked() {
    let tape = Tape::new();
    // The PG-001 semantics carry over: a disconnected division by zero
    // must not poison a recorded gradient, because non-ancestors' rules
    // are never recorded at all.
    let a = tape.parameter(varied([2], 1));
    let zero = tape.leaf(Tensor::filled([2], 0.0_f64));
    let _poison = zero / zero;
    let loss = a.sum();
    assert_closure(loss.symbol(), &[a.symbol()], tape);
}

#[test]
#[should_panic(expected = "scalar loss")]
fn differentiate_rejects_non_scalar_losses() {
    let tape = Tape::new();
    let a = tape.parameter(varied([2], 1));
    let doubled = a + a;
    tape.differentiate(doubled, [a]);
}

#[test]
fn second_derivative_of_a_cubic_is_exact() {
    let tape = Tape::new();
    let x = tape.parameter(Tensor::new([3], [0.5_f64, -1.25, 2.0]));
    let loss = (x * x * x).sum();

    let first = tape.differentiate(loss, [x]);
    let first_value = tape.resolve(first[0]);
    let second = tape.differentiate(first_value.sum(), [x]);

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let computed = run.of(second[0]).to_vec();
    for (computed, x) in computed.iter().zip([0.5_f64, -1.25, 2.0]) {
        assert_eq!(*computed, 6.0 * x);
    }
}

#[test]
fn second_derivative_of_tanh_matches_finite_differences() {
    let probe = 0.65_f64;
    let tape = Tape::new();
    let x = tape.parameter(Tensor::new([1], [probe]));
    let loss = x.tanh().sum();
    let first = tape.differentiate(loss, [x]);
    let second = tape.differentiate(tape.resolve(first[0]).sum(), [x]);

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let computed = run.of(second[0]).to_vec()[0];
    let step = 1e-6;
    let derivative_at = |x: f64| 1.0 - x.tanh().powi(2);
    let expected = (derivative_at(probe + step) - derivative_at(probe - step)) / (2.0 * step);
    assert!((computed - expected).abs() < 1e-6);
}

#[test]
fn relu_hessians_are_exact_zeros() {
    let tape = Tape::new();
    // The `Step` rule's `None` cotangents: differentiating a relu
    // gradient answers zero almost everywhere, never `NaN`.
    let x = tape.parameter(Tensor::new([3], [-2.0_f64, 0.5, 3.0]));
    let loss = x.relu().sum();
    let first = tape.differentiate(loss, [x]);
    let second = tape.differentiate(tape.resolve(first[0]).sum(), [x]);

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(second[0]).to_vec(), &[0.0; 3]);
}

#[test]
fn tape_growth_stays_a_small_constant() {
    let tape = Tape::new();
    let x = tape.input(varied([4, 3], 1));
    let weights = tape.parameter(varied([3, 4], 2));
    let logits = x
        .matmul(weights)
        .tanh()
        .matmul(tape.parameter(varied([4, 2], 3)));
    let loss = logits.log_softmax(1).sum();

    let before = tape.len();
    tape.differentiate(loss, [weights]);
    let after = tape.len();
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
    let build = |tape: &Tape<Tensor<f64>>| {
        let x = tape.input(varied([4, 3], 1));
        let weights = tape.parameter(varied([3, 4], 2));
        let bias = tape.parameter(varied([4], 3));
        let product = x.matmul(weights);
        let hidden = (product + bias.broadcast_along(0, product)).tanh();
        let head = tape.parameter(varied([4, 2], 4));
        let loss = hidden.matmul(head).log_softmax(1).sum();
        (
            x.symbol(),
            [weights.symbol(), bias.symbol(), head.symbol()],
            loss.symbol(),
        )
    };

    let engine_tape = Tape::new();
    let (engine_x, engine_params, engine_loss) = build(&engine_tape);
    let engine_network = engine_tape.into_network();
    let recorded_tape = Tape::new();
    let (recorded_x, recorded_params, recorded_loss) = build(&recorded_tape);
    let gradients = recorded_tape.differentiate(recorded_loss, recorded_params.iter().copied());
    let recorded_network = recorded_tape.into_network();
    let plan = recorded_network.compile(Compile::roots(
        std::iter::once(recorded_loss).chain(gradients.iter().copied()),
    ));

    let mut engine_state = engine_network.parameters();
    let mut recorded_state = recorded_network.parameters();
    for step in 0..12 {
        let batch = varied([4, 3], 10 + step);

        let engine_run = engine_network.forward(&engine_state, [(engine_x, batch.clone())]);
        let engine_gradients = engine_run.backward(engine_loss);
        engine_state = engine_state.step(&engine_gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * Tensor::filled(gradient.shape(), 0.05)
        });

        let recorded_run = plan.forward(&recorded_state, [(recorded_x, batch)]);
        let recorded_field = recorded_run.recorded_gradients(
            recorded_params
                .iter()
                .copied()
                .zip(gradients.iter().copied()),
        );
        recorded_state = recorded_state.step(&recorded_field, |parameter, gradient| {
            parameter.clone() - gradient.clone() * Tensor::filled(gradient.shape(), 0.05)
        });

        for (&engine_param, &recorded_param) in engine_params.iter().zip(&recorded_params) {
            let engine_payload = engine_state.of(engine_param);
            let recorded_payload = recorded_state.of(recorded_param);
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
    let tape = Tape::new();
    let a = tape.parameter(varied([2], 1));
    let loss = a.sum();
    let a = a.symbol();
    let gradients = tape.differentiate(loss, [a]);
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    run.recorded_gradients([(gradients[0], a)]);
}
