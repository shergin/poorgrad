use crate::{Compile, Network, Symbol, Tensor};

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

    let plan = network.compile(Compile::roots([target.symbol()]));
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

    let plan = network.compile(Compile::roots([wanted.symbol()]));
    let run = plan.forward(&network, no_feeds());
    assert_eq!(*run.of(wanted), 4.0);
    let _ = unwanted;
}

#[test]
#[should_panic(expected = "not computed by this target-sliced run")]
fn plan_reads_outside_the_readable_set_are_rejected() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let wanted = x * x;
    // An interior ancestor: computed, but not declared readable.
    let interior = x * x * x;
    let target = interior + wanted;

    let plan = network.compile(Compile::roots([target.symbol()]));
    let run = plan.forward(&network, no_feeds());
    run.of(interior);
}

#[test]
fn keep_makes_an_interior_value_readable() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let interior = x * x;
    let target = interior + x;

    let plan = network.compile(Compile::roots([target.symbol()]).observe([interior.symbol()]));
    let run = plan.forward(&network, no_feeds());
    assert_eq!(*run.of(target), 6.0);
    assert_eq!(*run.of(interior), 4.0);
}

#[test]
#[should_panic(expected = "forward-only plan")]
fn forward_only_plans_refuse_backward() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let target = x * x;

    let plan = network.compile(Compile::roots([target.symbol()]));
    let run = plan.forward(&network, no_feeds());
    run.backward(target);
}

#[test]
fn training_plans_differentiate_like_the_interpreter() {
    let network = Network::new();
    let w = network.parameter(Tensor::new([2], [1.0_f64, -2.0]));
    let x = network.leaf(Tensor::new([2], [3.0, 4.0]));
    let loss = ((w * x).tanh() * x).sum();

    let plan = network.compile(Compile::roots([loss.symbol()]).engine_backward());
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

    let plan = network.compile(Compile::roots([loss_symbol]).engine_backward());
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

    let plan = network.compile(Compile::roots([late.symbol()]));
    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();
    assert_eq!(planned.of(late).to_vec(), interpreted.of(late).to_vec());
}

#[test]
fn plan_forward_binds_feeds() {
    let network = Network::new();
    let x = network.input(Tensor::new([2], [0.0_f64, 0.0]));
    let doubled = x * Tensor::new([2], [2.0, 2.0]);

    let plan = network.compile(Compile::roots([doubled.symbol()]));
    let run = plan.forward(&network, [(x.symbol(), Tensor::new([2], [4.0, 5.0]))]);
    assert_eq!(run.of(doubled).to_vec(), &[8.0, 10.0]);
}

#[test]
#[should_panic(expected = "different network lineage")]
fn plans_reject_foreign_networks() {
    let first = Network::new();
    let second = Network::new();
    let x = first.leaf(2.0_f64);
    let target = x * x;
    let _ = second.leaf(1.0_f64);

    let plan = first.compile(Compile::roots([target.symbol()]));
    plan.forward(&second, no_feeds());
}

#[test]
fn plans_keep_serving_their_prefix_after_recording() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let target = x * x;

    let plan = network.compile(Compile::roots([target.symbol()]));
    // Later recordings grow the tape past the plan's prefix.
    let _later = x + x;
    let run = plan.forward(&network, no_feeds());
    assert_eq!(*run.of(target), 4.0);
}

#[test]
fn describe_reports_the_liveness_story() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([4], [1.0_f64, 2.0, 3.0, 4.0]));
    let target = (x.tanh() * x).sum();

    let plan = network.compile(Compile::roots([target.symbol()]));
    let description = plan.describe();
    assert!(description.contains("plan: forward;"));
    assert!(description.contains("Tanh"));
    assert!(description.contains("kept"));
    assert!(description.contains("peak"));
}

#[test]
fn training_liveness_matches_the_interpreter_on_a_convnet() {
    // The real consumer motif: conv, relu, pool, flatten, dense,
    // cross-entropy. The read contract keeps what the derivative rules read
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

    let plan = network.compile(Compile::roots([loss.symbol()]).engine_backward());
    // The release analysis must license releases here: the conv view
    // chains and output permutes are all shape-only. Engine runs hold
    // everything (the graded posture), so the licensed set is the
    // reported floor, never executed.
    assert!(plan.describe().contains("releasable after"));
    assert!(plan.describe().contains("release floor"));

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

    let plan = network.compile(Compile::roots([loss.symbol()]).engine_backward());
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

    let plan = network.compile(Compile::roots([loss.symbol()]).engine_backward());
    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();

    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
    assert_eq!(
        planned.backward(loss).of(table).to_vec(),
        interpreted.backward(loss).of(table).to_vec()
    );
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

    let plan = network.compile(Compile::roots([output.symbol()]));
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

    let fused = network.compile(Compile::roots([loss.symbol()]));
    assert!(fused.describe().contains("window-gemm"));
    // Engine-backward plans do not fuse at all: their memory
    // contract stays exact for the reverse scan.
    let engine = network.compile(Compile::roots([loss.symbol()]).engine_backward());
    assert!(!engine.describe().contains("window-gemm"));

    let barred = network.compile(Compile::roots([loss.symbol()]).observe([patches.symbol()]));
    assert!(!barred.describe().contains("window-gemm"));
    let run = barred.forward(&network, std::iter::empty());
    assert_eq!(
        run.of(patches).to_vec(),
        network.forward().of(patches).to_vec()
    );
}

#[test]
fn batched_products_do_not_fuse() {
    // The window-GEMM matcher keys on the rank-2 im2col shape; a
    // batched product must leave it indifferent.
    let network = Network::new();
    let lhs = network
        .leaf(Tensor::new(
            [72],
            (0..72).map(|v| v as f64 * 0.05 - 1.8).collect::<Vec<_>>(),
        ))
        .reshape([2, 9, 4]);
    let rhs = network.leaf(Tensor::new(
        [2, 4, 2],
        (0..16).map(|v| v as f64 * 0.25 - 2.0).collect::<Vec<_>>(),
    ));
    let loss = lhs.matmul(rhs).sum();
    let plan = network.compile(Compile::roots([loss.symbol()]));
    assert!(!plan.describe().contains("window-gemm"));
    let planned = plan.forward(&network, std::iter::empty());
    assert_eq!(
        planned.of(loss).to_vec(),
        network.forward().of(loss).to_vec()
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

    let plan = network.compile(Compile::roots([loss.symbol()]));
    assert!(!plan.describe().contains("window-gemm"));

    let planned = plan.forward(&network, std::iter::empty());
    let interpreted = network.forward();
    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
}

/// Compiles a plan whose last node is minted by `extend` on one fork,
/// then calls it on the untouched, shorter sibling: the sibling's chain
/// attributes the whole range to the same branches, but it does not
/// carry the plan's nodes and must be rejected uniformly.
fn shorter_sibling_is_rejected(extend: impl FnOnce(&Network<f64>) -> Symbol) {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let shared = x * x;
    let sibling = network.clone();
    let late = extend(&network);
    let _ = shared;

    let plan = network.compile(Compile::roots([late]));
    plan.forward(&sibling, no_feeds());
}

#[test]
#[should_panic(expected = "does not contain")]
fn plans_reject_a_shorter_sibling_with_a_later_leaf() {
    shorter_sibling_is_rejected(|network| network.leaf(3.0_f64).symbol());
}

#[test]
#[should_panic(expected = "does not contain")]
fn plans_reject_a_shorter_sibling_with_a_later_input() {
    shorter_sibling_is_rejected(|network| network.input(3.0_f64).symbol());
}

#[test]
#[should_panic(expected = "does not contain")]
fn plans_reject_a_shorter_sibling_with_a_later_parameter() {
    shorter_sibling_is_rejected(|network| network.parameter(3.0_f64).symbol());
}

#[test]
#[should_panic(expected = "does not contain")]
fn plans_reject_a_shorter_sibling_with_a_later_computed_target() {
    shorter_sibling_is_rejected(|network| {
        let y = network.leaf(3.0_f64);
        (y * y).symbol()
    });
}
