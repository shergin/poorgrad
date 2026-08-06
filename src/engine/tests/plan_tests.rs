use crate::{Network, Symbol, Tensor};

use super::Plan;

/// The empty feed set, typed for the scalar tests.
fn no_feeds() -> std::iter::Empty<(Symbol, f64)> {
    std::iter::empty()
}

#[test]
fn plan_forward_matches_the_interpreter_bitwise() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]));
    let y = network.leaf(Tensor::new([2, 2], [0.5, -0.5, 1.5, -1.5]));
    let target = ((x.matmul(y) + x).tanh() * y).sum();

    let plan = network.compile([target.symbol()], []);
    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();

    assert_eq!(planned.of(target).to_vec(), interpreted.of(target).to_vec());
}

#[test]
fn plan_skips_what_the_targets_cannot_observe() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let wanted = x * x;
    let unwanted = x + x;

    let plan = network.compile([wanted.symbol()], []);
    let evaluation = plan.forward(&network, no_feeds());
    assert_eq!(*evaluation.of(wanted), 4.0);
    let _ = unwanted;
}

#[test]
#[should_panic(expected = "not evaluated by this target-sliced run")]
fn plan_reads_outside_the_readable_set_are_rejected() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let wanted = x * x;
    // An interior ancestor: computed, but not declared readable.
    let interior = x * x * x;
    let target = interior + wanted;

    let plan = network.compile([target.symbol()], []);
    let evaluation = plan.forward(&network, no_feeds());
    evaluation.of(interior);
}

#[test]
fn keep_makes_an_interior_value_readable() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let interior = x * x;
    let target = interior + x;

    let plan = network.compile([target.symbol()], [interior.symbol()]);
    let evaluation = plan.forward(&network, no_feeds());
    assert_eq!(*evaluation.of(target), 6.0);
    assert_eq!(*evaluation.of(interior), 4.0);
}

#[test]
#[should_panic(expected = "forward-only plan")]
fn forward_only_plans_refuse_backward() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let target = x * x;

    let plan = network.compile([target.symbol()], []);
    let evaluation = plan.forward(&network, no_feeds());
    evaluation.backward(target);
}

#[test]
fn training_plans_differentiate_like_the_interpreter() {
    let network = Network::new();
    let w = network.parameter(Tensor::new([2], [1.0_f64, -2.0]));
    let x = network.leaf(Tensor::new([2], [3.0, 4.0]));
    let loss = ((w * x).tanh() * x).sum();

    let plan = network.compile_training(loss.symbol(), []);
    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();

    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
    assert_eq!(
        planned.backward(loss).of(w).to_vec(),
        interpreted.backward(loss).of(w).to_vec()
    );
}

#[test]
fn one_plan_serves_every_generation() {
    // Compile once, train for several generations: the plan's runs
    // must match a freshly interpreted run at every step, bitwise.
    let network = Network::new();
    let w = network.parameter(Tensor::new([2], [0.0_f64, 0.0]));
    let x = network.leaf(Tensor::new([2], [3.0, 2.0]));
    let y = network.leaf(Tensor::new([2], [15.0, -6.0]));
    let error = w * x - y;
    let loss = (error * error).sum();
    let loss_symbol = loss.symbol();
    let w_symbol = w.symbol();

    let plan = network.compile_training(loss_symbol, []);
    let mut network = network;
    for _ in 0..5 {
        let loss_value = network.resolve(loss_symbol);
        let planned = plan.forward(&network, std::iter::empty());
        let interpreted = network.forward();
        assert_eq!(
            planned.of(loss_value).to_vec(),
            interpreted.of(loss_value).to_vec()
        );
        let gradients = planned.backward(loss_value);
        network = network.update(&gradients, |parameter: &Tensor<f64>, gradient| {
            parameter.clone() - gradient.clone() * Tensor::filled([2], 0.05)
        });
    }
    let learned = network.resolve(w_symbol).payload().unwrap();
    assert!(learned.to_vec()[0] > 1.0);
}

#[test]
fn liveness_frees_only_after_the_last_consumer() {
    // A diamond: `shared` feeds two later consumers, so freeing after
    // the first would corrupt the second. Bitwise agreement with the
    // interpreter is the proof.
    let network = Network::new();
    let x = network.leaf(Tensor::new([2], [1.5_f64, -2.5]));
    let shared = x.tanh();
    let early = shared * x;
    let late = (shared + early).sum();

    let plan = network.compile([late.symbol()], []);
    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();
    assert_eq!(planned.of(late).to_vec(), interpreted.of(late).to_vec());
}

#[test]
fn plan_forward_binds_feeds() {
    let network = Network::new();
    let x = network.input(Tensor::new([2], [0.0_f64, 0.0]));
    let doubled = x * Tensor::new([2], [2.0, 2.0]);

    let plan = network.compile([doubled.symbol()], []);
    let evaluation = plan.forward(&network, [(x.symbol(), Tensor::new([2], [4.0, 5.0]))]);
    assert_eq!(evaluation.of(doubled).to_vec(), &[8.0, 10.0]);
}

#[test]
#[should_panic(expected = "different network lineage")]
fn plans_reject_foreign_networks() {
    let first = Network::new();
    let second = Network::new();
    let x = first.leaf(2.0_f64);
    let target = x * x;
    let _ = second.leaf(1.0_f64);

    let plan = first.compile([target.symbol()], []);
    plan.forward(&second, no_feeds());
}

#[test]
fn plans_keep_serving_their_prefix_after_recording() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let target = x * x;

    let plan = network.compile([target.symbol()], []);
    // Later recordings grow the tape past the plan's prefix.
    let _later = x + x;
    let evaluation = plan.forward(&network, no_feeds());
    assert_eq!(*evaluation.of(target), 4.0);
}

#[test]
fn describe_reports_the_liveness_story() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([4], [1.0_f64, 2.0, 3.0, 4.0]));
    let target = (x.tanh() * x).sum();

    let plan = network.compile([target.symbol()], []);
    let description = plan.describe();
    assert!(description.contains("forward-only"));
    assert!(description.contains("Tanh"));
    assert!(description.contains("kept"));
    assert!(description.contains("peak"));
}

#[test]
fn training_liveness_matches_the_interpreter_on_a_convnet() {
    // The real consumer motif: conv, relu, pool, flatten, dense,
    // cross-entropy. Retention keeps what the derivative rules read
    // and frees the view chains and padded copies; the proof is
    // bitwise agreement of loss and every parameter gradient.
    use crate::{conv2d, cross_entropy, max_pool};

    let network = Network::new();
    let input = network.leaf(Tensor::new(
        [2, 1, 4, 4],
        (0..32).map(|v| (v as f64) / 16.0 - 1.0).collect::<Vec<_>>(),
    ));
    let weights = network.parameter(Tensor::new(
        [2, 1, 3, 3],
        (0..18)
            .map(|v| (v as f64) / 32.0 - 0.25)
            .collect::<Vec<_>>(),
    ));
    let bias = network.parameter(Tensor::new([2], [0.1, -0.1]));
    let head = network.parameter(Tensor::new(
        [8, 3],
        (0..24)
            .map(|v| (v as f64) / 48.0 - 0.25)
            .collect::<Vec<_>>(),
    ));
    let targets = network.leaf(Tensor::selection([1, 2], 3, 1.0));

    let pooled = max_pool(conv2d(input, weights, bias, 1, 1).relu(), 2, 2);
    let logits = pooled.reshape([2, 8]).matmul(head);
    let loss = cross_entropy(logits, targets);

    let mut plan = network.compile_training(loss.symbol(), []);
    // The retention analysis must license releases here: the conv view
    // chains and output permutes are all shape-only.
    assert!(plan.describe().contains("releasable after"));
    assert!(plan.describe().contains("retention floor"));
    // Force the analysis to execute (training runs hold everything by
    // default after the allocator measurements): gradients must stay
    // bit-identical even with every licensed release performed — the
    // guarantee rematerialization will later build on.
    plan.frees = plan.releases.clone();

    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();
    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());

    let planned_gradients = planned.backward(loss);
    let interpreted_gradients = interpreted.backward(loss);
    for parameter in [weights, bias, head] {
        assert_eq!(
            planned_gradients.of(parameter).to_vec(),
            interpreted_gradients.of(parameter).to_vec()
        );
    }
}

#[test]
fn training_liveness_matches_the_interpreter_on_every_value_reader() {
    // One scalar soup exercising every retention class: mul, tanh,
    // exp, ln, sqrt, powf, div, relu, maximum, with fan-out so freed
    // and retained slots interleave.
    let network = Network::new();
    let w = network.parameter(0.7_f64);
    let x = network.leaf(1.3_f64);

    let product = w * x;
    let squashed = product.tanh();
    let grown = squashed.exp();
    let logged = grown.ln();
    let rooted = grown.sqrt();
    let raised = product.powf(x);
    let divided = grown / product;
    let rectified = product.relu();
    let larger = product.maximum(x);
    let loss =
        (squashed + grown + logged + rooted + raised + divided + rectified + larger) * product;

    let plan = network.compile_training(loss.symbol(), []);
    let planned = plan.forward(&network, no_feeds());
    let interpreted = network.forward();

    assert_eq!(*planned.of(loss), *interpreted.of(loss));
    assert_eq!(
        *planned.backward(loss).of(w),
        *interpreted.backward(loss).of(w)
    );
}

#[test]
fn training_liveness_retains_the_gather_selection() {
    // The scatter in gather's backward reads the selection payload
    // itself; freeing it would panic, not merely corrupt. Bitwise
    // agreement proves retention kept it.
    let network = Network::new();
    let table = network.parameter(Tensor::new(
        [4, 2],
        (0..8).map(|v| v as f64 * 0.5).collect::<Vec<_>>(),
    ));
    let selection = network.input(Tensor::selection([2, 0, 3], 4, 1.0));
    let loss = table.gather(selection).sum();

    let plan = network.compile_training(loss.symbol(), []);
    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();

    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
    assert_eq!(
        planned.backward(loss).of(table).to_vec(),
        interpreted.backward(loss).of(table).to_vec()
    );
}

#[test]
fn remat_matches_the_interpreter_on_a_convnet() {
    // A tiny threshold forces deep drop chains through the conv motif;
    // backward rematerializes them and must agree with the interpreter
    // bit for bit on the loss and every parameter gradient.
    use crate::{conv2d, cross_entropy, max_pool};

    let network = Network::new();
    let input = network.leaf(Tensor::new(
        [2, 1, 4, 4],
        (0..32).map(|v| (v as f64) / 16.0 - 1.0).collect::<Vec<_>>(),
    ));
    let weights = network.parameter(Tensor::new(
        [2, 1, 3, 3],
        (0..18)
            .map(|v| (v as f64) / 32.0 - 0.25)
            .collect::<Vec<_>>(),
    ));
    let bias = network.parameter(Tensor::new([2], [0.1, -0.1]));
    let head = network.parameter(Tensor::new(
        [8, 3],
        (0..24)
            .map(|v| (v as f64) / 48.0 - 0.25)
            .collect::<Vec<_>>(),
    ));
    let targets = network.leaf(Tensor::selection([1, 2], 3, 1.0));

    let pooled = max_pool(conv2d(input, weights, bias, 1, 1).relu(), 2, 2);
    let logits = pooled.reshape([2, 8]).matmul(head);
    let loss = cross_entropy(logits, targets);

    let plan = Plan::new(&network, &[loss.symbol()], &[], true, 4);
    let description = plan.describe();
    assert!(description.contains("(remat)"));
    assert!(description.contains("remat drops"));

    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();
    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());

    let planned_gradients = planned.backward(loss);
    let interpreted_gradients = interpreted.backward(loss);
    for parameter in [weights, bias, head] {
        assert_eq!(
            planned_gradients.of(parameter).to_vec(),
            interpreted_gradients.of(parameter).to_vec()
        );
    }
}

#[test]
fn full_remat_matches_the_interpreter_on_every_value_reader() {
    // Threshold 1 drops every non-source, non-readable value: the
    // whole backward runs on rematerialized payloads across every
    // retention class.
    let network = Network::new();
    let w = network.parameter(0.7_f64);
    let x = network.leaf(1.3_f64);

    let product = w * x;
    let squashed = product.tanh();
    let grown = squashed.exp();
    let logged = grown.ln();
    let rooted = grown.sqrt();
    let raised = product.powf(x);
    let divided = grown / product;
    let rectified = product.relu();
    let larger = product.maximum(x);
    let loss =
        (squashed + grown + logged + rooted + raised + divided + rectified + larger) * product;

    let plan = Plan::new(&network, &[loss.symbol()], &[], true, 1);
    let planned = plan.forward(&network, no_feeds());
    let interpreted = network.forward();

    assert_eq!(*planned.of(loss), *interpreted.of(loss));
    assert_eq!(
        *planned.backward(loss).of(w),
        *interpreted.backward(loss).of(w)
    );
}

#[test]
fn remat_never_drops_the_gather_selection() {
    // Sources are never dropped, so the selection an input carries
    // stays genuine for the backward scatter even under full remat.
    let network = Network::new();
    let table = network.parameter(Tensor::new(
        [4, 2],
        (0..8).map(|v| v as f64 * 0.5).collect::<Vec<_>>(),
    ));
    let selection = network.input(Tensor::selection([2, 0, 3], 4, 1.0));
    let loss = table.gather(selection).sum();

    let plan = Plan::new(&network, &[loss.symbol()], &[], true, 1);
    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();

    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
    assert_eq!(
        planned.backward(loss).of(table).to_vec(),
        interpreted.backward(loss).of(table).to_vec()
    );
}

#[test]
fn compact_training_drops_where_the_default_retains() {
    // A value crossing the remat size class: the default plan keeps
    // it, the compact plan drops and rematerializes it.
    let network = Network::new();
    let w = network.parameter(Tensor::filled([300, 300], 0.5_f64));
    let x = network.leaf(Tensor::filled([300, 300], 2.0));
    let big = w * x;
    let loss = (big * big).sum();

    let default = network.compile_training(loss.symbol(), []);
    let compact = network.compile_training_compact(loss.symbol(), []);
    assert!(default.describe().contains("remat drops 0 slots"));
    assert!(!compact.describe().contains("remat drops 0 slots"));
    assert!(compact.describe().contains("(remat)"));

    let planned = compact.forward(&network, std::iter::empty());
    let interpreted = network.forward();
    assert_eq!(
        planned.backward(loss).of(w).to_vec(),
        interpreted.backward(loss).of(w).to_vec()
    );
}

#[test]
fn window_gemm_fusion_matches_the_interpreter() {
    // The conv facade's emission fuses; forward and backward stay
    // bitwise against the interpreter with the chain never
    // materialized.
    use crate::{conv2d, cross_entropy, max_pool};

    let network = Network::new();
    let input = network.leaf(Tensor::new(
        [2, 1, 4, 4],
        (0..32).map(|v| (v as f64) / 16.0 - 1.0).collect::<Vec<_>>(),
    ));
    let weights = network.parameter(Tensor::new(
        [2, 1, 3, 3],
        (0..18)
            .map(|v| (v as f64) / 32.0 - 0.25)
            .collect::<Vec<_>>(),
    ));
    let bias = network.parameter(Tensor::new([2], [0.1, -0.1]));
    let head = network.parameter(Tensor::new(
        [8, 3],
        (0..24)
            .map(|v| (v as f64) / 48.0 - 0.25)
            .collect::<Vec<_>>(),
    ));
    let targets = network.leaf(Tensor::selection([1, 2], 3, 1.0));
    let pooled = max_pool(conv2d(input, weights, bias, 1, 1).relu(), 2, 2);
    let loss = cross_entropy(pooled.reshape([2, 8]).matmul(head), targets);

    let plan = network.compile_training_compact(loss.symbol(), []);
    let description = plan.describe();
    assert!(description.contains("fused 1 window-gemm"));
    assert!(description.contains("fused (window-gemm)"));

    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();
    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
    let planned_gradients = planned.backward(loss);
    let interpreted_gradients = interpreted.backward(loss);
    for parameter in [weights, bias, head] {
        assert_eq!(
            planned_gradients.of(parameter).to_vec(),
            interpreted_gradients.of(parameter).to_vec()
        );
    }
}

#[test]
fn forward_only_plans_fuse_and_agree() {
    use crate::{conv2d, max_pool};

    let network = Network::new();
    let input = network.leaf(Tensor::new(
        [2, 1, 4, 4],
        (0..32).map(|v| (v as f64) / 10.0 - 1.5).collect::<Vec<_>>(),
    ));
    let weights = network.leaf(Tensor::new(
        [2, 1, 3, 3],
        (0..18)
            .map(|v| (v as f64) / 24.0 - 0.375)
            .collect::<Vec<_>>(),
    ));
    let bias = network.leaf(Tensor::new([2], [0.05, -0.05]));
    let output = max_pool(conv2d(input, weights, bias, 1, 1).relu(), 2, 2);

    let plan = network.compile([output.symbol()], []);
    assert!(plan.describe().contains("fused 1 window-gemm"));

    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();
    assert_eq!(planned.of(output).to_vec(), interpreted.of(output).to_vec());
}

#[test]
fn kept_interiors_bar_fusion() {
    // Keeping the im2col matrix readable is a fusion barrier: the
    // chain must materialize so the keep-set can answer.
    let network = Network::new();
    let x = network.leaf(Tensor::new(
        [1, 1, 4, 4],
        (0..16).map(|v| v as f64 * 0.3 - 2.0).collect::<Vec<_>>(),
    ));
    let kernel = network.leaf(Tensor::new(
        [4, 2],
        (0..8).map(|v| v as f64 * 0.25 - 0.75).collect::<Vec<_>>(),
    ));
    let patches = x
        .unfold(2, 2, 1, 1)
        .unfold(4, 2, 1, 1)
        .permute([0, 2, 4, 1, 3, 5])
        .reshape([9, 4]);
    let loss = patches.matmul(kernel).sum();

    let fused = network.compile_training_compact(loss.symbol(), []);
    assert!(fused.describe().contains("window-gemm"));
    // The default retain-all training plan does not fuse at all: its
    // memory contract stays exact.
    let default = network.compile_training(loss.symbol(), []);
    assert!(!default.describe().contains("window-gemm"));

    let barred = network.compile_training_compact(loss.symbol(), [patches.symbol()]);
    assert!(!barred.describe().contains("window-gemm"));
    let evaluation = barred.forward(&network, std::iter::empty());
    assert_eq!(
        evaluation.of(patches).to_vec(),
        network.forward().of(patches).to_vec()
    );
}

#[test]
fn shared_windows_bar_fusion() {
    // A second consumer inside the chain bars fusion, and results
    // stay bitwise either way.

    let network = Network::new();
    let x = network.leaf(Tensor::new(
        [1, 1, 4, 4],
        (0..16).map(|v| v as f64 * 0.5 - 3.0).collect::<Vec<_>>(),
    ));
    let kernel = network.leaf(Tensor::new(
        [4, 2],
        (0..8).map(|v| v as f64 * 0.125).collect::<Vec<_>>(),
    ));
    let windows = x.unfold(2, 2, 1, 1).unfold(4, 2, 1, 1);
    let patches = windows.permute([0, 2, 4, 1, 3, 5]).reshape([9, 4]);
    let loss = patches.matmul(kernel).sum() + windows.sum();

    let plan = network.compile_training_compact(loss.symbol(), []);
    assert!(!plan.describe().contains("window-gemm"));

    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();
    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
}
