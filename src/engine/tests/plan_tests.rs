use crate::{Numerics, Request, Symbol, Tape, Tensor};

/// The empty feed set, typed for the scalar tests.
fn no_feeds() -> std::iter::Empty<(Symbol, f64)> {
    std::iter::empty()
}

#[test]
fn plan_forward_matches_the_interpreter_bitwise() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]));
    let y = tape.leaf(Tensor::new([2, 2], [0.5, -0.5, 1.5, -1.5]));
    let target = ((x.matmul(y) + x).tanh() * y).sum().symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let plan = network.compile(Request::roots([target]));
    let planned = plan.forward(&parameters, std::iter::empty());
    let interpreted = network.forward(&parameters, []);

    assert_eq!(planned.of(target).to_vec(), interpreted.of(target).to_vec());
}

#[test]
fn plan_skips_what_the_targets_cannot_observe() {
    let tape = Tape::new();
    let x = tape.leaf(2.0_f64);
    let wanted = (x * x).symbol();
    let _unwanted = x + x;
    let network = tape.into_network();

    let plan = network.compile(Request::roots([wanted]));
    let run = plan.forward(&network.parameters(), no_feeds());
    assert_eq!(*run.of(wanted), 4.0);
}

#[test]
#[should_panic(expected = "not computed by this target-sliced run")]
fn plan_reads_outside_the_readable_set_are_rejected() {
    let tape = Tape::new();
    let x = tape.leaf(2.0_f64);
    let wanted = x * x;
    // An interior ancestor: computed, but not declared readable.
    let interior = x * x * x;
    let target = (interior + wanted).symbol();
    let interior = interior.symbol();
    let network = tape.into_network();

    let plan = network.compile(Request::roots([target]));
    let run = plan.forward(&network.parameters(), no_feeds());
    run.of(interior);
}

#[test]
fn keep_makes_an_interior_value_readable() {
    let tape = Tape::new();
    let x = tape.leaf(2.0_f64);
    let interior = x * x;
    let target = (interior + x).symbol();
    let interior = interior.symbol();
    let network = tape.into_network();

    let plan = network.compile(Request::roots([target]).observe([interior]));
    let run = plan.forward(&network.parameters(), no_feeds());
    assert_eq!(*run.of(target), 6.0);
    assert_eq!(*run.of(interior), 4.0);
}

#[test]
#[should_panic(expected = "forward-only plan")]
fn forward_only_plans_refuse_backward() {
    let tape = Tape::new();
    let x = tape.leaf(2.0_f64);
    let target = (x * x).symbol();
    let network = tape.into_network();

    let plan = network.compile(Request::roots([target]));
    let run = plan.forward(&network.parameters(), no_feeds());
    run.backward(target);
}

#[test]
fn training_plans_differentiate_like_the_interpreter() {
    let tape = Tape::new();
    let w = tape.parameter(Tensor::new([2], [1.0_f64, -2.0]));
    let x = tape.leaf(Tensor::new([2], [3.0, 4.0]));
    let loss = ((w * x).tanh() * x).sum().symbol();
    let w = w.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let plan = network.compile(Request::roots([loss]).backward());
    let planned = plan.forward(&parameters, std::iter::empty());
    let interpreted = network.forward(&parameters, []);

    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
    assert_eq!(
        planned.backward(loss).of(w).to_vec(),
        interpreted.backward(loss).of(w).to_vec()
    );
}

#[test]
fn one_plan_serves_every_training_step() {
    // Request once, train for several steps: the plan's runs must
    // match a freshly interpreted run at every step, bitwise — the
    // plan holds no state, so there is nothing for a step to
    // invalidate.
    let tape = Tape::new();
    let w = tape.parameter(Tensor::new([2], [0.0_f64, 0.0]));
    let x = tape.leaf(Tensor::new([2], [3.0, 2.0]));
    let y = tape.leaf(Tensor::new([2], [15.0, -6.0]));
    let error = w * x - y;
    let loss = (error * error).sum().symbol();
    let w = w.symbol();
    let network = tape.into_network();

    let plan = network.compile(Request::roots([loss]).backward());
    let mut parameters = network.parameters();
    for _ in 0..5 {
        let planned = plan.forward(&parameters, std::iter::empty());
        let interpreted = network.forward(&parameters, []);
        assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
        let gradients = planned.backward(loss);
        parameters = parameters.step(&gradients, |parameter: &Tensor<f64>, gradient| {
            parameter.clone() - gradient.clone() * Tensor::filled([2], 0.05)
        });
    }
    assert!(parameters.of(w).to_vec()[0] > 1.0);
}

#[test]
fn liveness_frees_only_after_the_last_consumer() {
    // A diamond: `shared` feeds two later consumers, so freeing after
    // the first would corrupt the second. Bitwise agreement with the
    // interpreter is the proof.
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2], [1.5_f64, -2.5]));
    let shared = x.tanh();
    let early = shared * x;
    let late = (shared + early).sum().symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let plan = network.compile(Request::roots([late]));
    let planned = plan.forward(&parameters, std::iter::empty());
    let interpreted = network.forward(&parameters, []);
    assert_eq!(planned.of(late).to_vec(), interpreted.of(late).to_vec());
}

#[test]
fn plan_forward_binds_feeds() {
    let tape = Tape::new();
    let x = tape.input(Tensor::new([2], [0.0_f64, 0.0]));
    let doubled = (x * Tensor::new([2], [2.0, 2.0])).symbol();
    let x = x.symbol();
    let network = tape.into_network();

    let plan = network.compile(Request::roots([doubled]));
    let run = plan.forward(&network.parameters(), [(x, Tensor::new([2], [4.0, 5.0]))]);
    assert_eq!(run.of(doubled).to_vec(), &[8.0, 10.0]);
}

#[test]
#[should_panic(expected = "parameters belong to a different network")]
fn plans_reject_foreign_parameters() {
    let tape = Tape::new();
    let x = tape.leaf(2.0_f64);
    let target = (x * x).symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([target]));

    let other = Tape::<f64>::new();
    let foreign = other.into_network().parameters();
    plan.forward(&foreign, no_feeds());
}

#[test]
#[should_panic(expected = "parameters do not cover the plan's parameter slots")]
fn plans_reject_uncarried_parameters_after_extension() {
    // A plan compiled after a reopen draws on the new parameter slot;
    // a state taken before the extension does not cover it and must be
    // rejected, pointing at `carried`.
    let tape = Tape::new();
    tape.parameter(1.0_f64);
    let network = tape.into_network();
    let stale = network.parameters();

    let tape = network.into_tape();
    let late = tape.parameter(2.0);
    let target = (late * late).symbol();
    let network = tape.into_network();
    let plan = network.compile(Request::roots([target]));
    plan.forward(&stale, no_feeds());
}

#[test]
fn plans_keep_serving_their_prefix_after_extension() {
    // Reopening the network and recording more does not disturb a
    // compiled plan: it executes its own frozen prefix, and a carried
    // state covers it.
    let tape = Tape::new();
    let w = tape.parameter(2.0_f64);
    let target = (w * w).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();
    let plan = network.compile(Request::roots([target]));

    let tape = network.into_tape();
    let late = tape.resolve(target) + tape.parameter(3.0);
    let late = late.symbol();
    let network = tape.into_network();
    let parameters = parameters.carried(&network);

    let run = plan.forward(&parameters, no_feeds());
    assert_eq!(*run.of(target), 4.0);
    assert_eq!(*network.forward(&parameters, []).of(late), 7.0);
}

#[test]
fn describe_reports_the_liveness_story() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([4], [1.0_f64, 2.0, 3.0, 4.0]));
    let target = (x.tanh() * x).sum().symbol();
    let network = tape.into_network();

    let plan = network.compile(Request::roots([target]));
    let description = plan.describe();
    assert!(description.contains("plan: forward;"));
    assert!(description.contains("Tanh"));
    assert!(description.contains("kept"));
    assert!(description.contains("peak"));
}

#[test]
fn training_liveness_matches_the_interpreter_on_a_convnet() {
    // The real consumer shape: conv, relu, pool, flatten, dense,
    // cross-entropy. The read contract keeps what the derivative rules read
    // and frees the view chains and padded copies; the proof is
    // bitwise agreement of loss and every parameter gradient.
    use crate::{conv2d, cross_entropy, max_pool};

    let tape = Tape::new();
    let input = tape.leaf(Tensor::new(
        [2, 1, 4, 4],
        (0..32).map(|v| (v as f64) / 16.0 - 1.0).collect::<Vec<_>>(),
    ));
    let weights = tape.parameter(Tensor::new(
        [2, 1, 3, 3],
        (0..18)
            .map(|v| (v as f64) / 32.0 - 0.25)
            .collect::<Vec<_>>(),
    ));
    let bias = tape.parameter(Tensor::new([2], [0.1, -0.1]));
    let head = tape.parameter(Tensor::new(
        [8, 3],
        (0..24)
            .map(|v| (v as f64) / 48.0 - 0.25)
            .collect::<Vec<_>>(),
    ));
    let targets = tape.leaf(Tensor::selection([1, 2], 3, 1.0));

    let pooled = max_pool(conv2d(input, weights, bias, 1, 1).relu(), 2, 2);
    let logits = pooled.reshape([2, 8]).matmul(head);
    let loss = cross_entropy(logits, targets).symbol();
    let parameter_symbols = [weights.symbol(), bias.symbol(), head.symbol()];
    let network = tape.into_network();
    let parameters = network.parameters();

    let plan = network.compile(Request::roots([loss]).backward());
    // The release analysis must license releases here: the conv view
    // chains and output permutes are all shape-only. Engine runs hold
    // everything (the graded posture), so the licensed set is the
    // reported floor, never executed.
    assert!(plan.describe().contains("releasable after"));
    assert!(plan.describe().contains("release floor"));

    let planned = plan.forward(&parameters, std::iter::empty());
    let interpreted = network.forward(&parameters, []);
    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());

    let planned_gradients = planned.backward(loss);
    let interpreted_gradients = interpreted.backward(loss);
    for parameter in parameter_symbols {
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
    let tape = Tape::new();
    let w = tape.parameter(0.7_f64);
    let x = tape.leaf(1.3_f64);

    let product = w * x;
    let squashed = product.tanh();
    let grown = squashed.exp();
    let logged = grown.ln();
    let rooted = grown.sqrt();
    let raised = product.powf(x);
    let divided = grown / product;
    let rectified = product.relu();
    let larger = product.maximum(x);
    let loss = ((squashed + grown + logged + rooted + raised + divided + rectified + larger)
        * product)
        .symbol();
    let w = w.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let plan = network.compile(Request::roots([loss]).backward());
    let planned = plan.forward(&parameters, no_feeds());
    let interpreted = network.forward(&parameters, []);

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
    let tape = Tape::new();
    let table = tape.parameter(Tensor::new(
        [4, 2],
        (0..8).map(|v| v as f64 * 0.5).collect::<Vec<_>>(),
    ));
    let selection = tape.input(Tensor::selection([2, 0, 3], 4, 1.0));
    let loss = table.gather(selection).sum().symbol();
    let table = table.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let plan = network.compile(Request::roots([loss]).backward());
    let planned = plan.forward(&parameters, std::iter::empty());
    let interpreted = network.forward(&parameters, []);

    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
    assert_eq!(
        planned.backward(loss).of(table).to_vec(),
        interpreted.backward(loss).of(table).to_vec()
    );
}

#[test]
fn forward_only_plans_fuse_and_agree() {
    use crate::{conv2d, max_pool};

    let tape = Tape::new();
    let input = tape.leaf(Tensor::new(
        [2, 1, 4, 4],
        (0..32).map(|v| (v as f64) / 10.0 - 1.5).collect::<Vec<_>>(),
    ));
    let weights = tape.leaf(Tensor::new(
        [2, 1, 3, 3],
        (0..18)
            .map(|v| (v as f64) / 24.0 - 0.375)
            .collect::<Vec<_>>(),
    ));
    let bias = tape.leaf(Tensor::new([2], [0.05, -0.05]));
    let output = max_pool(conv2d(input, weights, bias, 1, 1).relu(), 2, 2).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let plan = network.compile(Request::roots([output]));
    assert!(plan.describe().contains("fused 1 window-gemm"));

    let planned = plan.forward(&parameters, std::iter::empty());
    let interpreted = network.forward(&parameters, []);
    assert_eq!(planned.of(output).to_vec(), interpreted.of(output).to_vec());
}

#[test]
fn describe_snapshots_the_fused_forward_plan() {
    // The full wording of a fused plan's schedule, so the catalog
    // extract cannot silently reword what `contains` checks miss.
    use crate::{conv2d, max_pool};

    let tape = Tape::new();
    let input = tape.leaf(Tensor::new(
        [2, 1, 4, 4],
        (0..32).map(|v| (v as f64) / 10.0 - 1.5).collect::<Vec<_>>(),
    ));
    let weights = tape.leaf(Tensor::new(
        [2, 1, 3, 3],
        (0..18)
            .map(|v| (v as f64) / 24.0 - 0.375)
            .collect::<Vec<_>>(),
    ));
    let bias = tape.leaf(Tensor::new([2], [0.05, -0.05]));
    let output = max_pool(conv2d(input, weights, bias, 1, 1).relu(), 2, 2).symbol();
    let network = tape.into_network();

    let forward = network.compile(Request::roots([output]));
    let expected = "     0  Leaf           [2, 1, 4, 4]     freed after 11
     1  Leaf           [2, 1, 3, 3]     freed after 9
     2  Leaf           [2]              freed after 12
     3  Pad            [2, 1, 6, 4]     fused (window-gemm)
     4  Pad            [2, 1, 6, 6]     fused (window-gemm)
     5  Unfold         [2, 1, 4, 3, 6]  fused (window-gemm)
     6  Unfold         [2, 1, 4, 3, 4, 3] fused (window-gemm)
     7  Permute        [2, 4, 4, 1, 3, 3] fused (window-gemm)
     8  Reshape        [32, 9]          fused (window-gemm)
     9  Permute        [1, 3, 3, 2]     freed after 10
    10  Reshape        [9, 2]           freed after 11
    11  MatMul         [32, 2]          freed after 13
    12  BroadcastAlong [32, 2]          freed after 13
    13  Add            [32, 2]          freed after 14
    14  Reshape        [2, 4, 4, 2]     freed after 15
    15  Permute        [2, 2, 4, 4]     freed after 16
    16  Relu           [2, 2, 4, 4]     freed after 17
    17  Unfold         [2, 2, 2, 2, 4]  freed after 18
    18  Unfold         [2, 2, 2, 2, 2, 2] freed after 19
    19  Permute        [2, 2, 2, 2, 2, 2] freed after 20
    20  Reshape        [2, 2, 2, 2, 4]  freed after 26
    21  Narrow         [2, 2, 2, 2, 1]  freed after 23
    22  Narrow         [2, 2, 2, 2, 1]  freed after 23
    23  Maximum        [2, 2, 2, 2, 1]  freed after 25
    24  Narrow         [2, 2, 2, 2, 1]  freed after 25
    25  Maximum        [2, 2, 2, 2, 1]  freed after 27
    26  Narrow         [2, 2, 2, 2, 1]  freed after 27
    27  Maximum        [2, 2, 2, 2, 1]  freed after 28
    28  Reshape        [2, 2, 2, 2]     kept
plan: forward; 29 of 29 nodes evaluated, 1 readable
fused 1 window-gemm groups, 6 interior nodes skipped
live volume: peak 192 elements at node 13, retain-all 856
";
    assert_eq!(forward.describe(), expected);

    // The posture gate: the same tape compiled engine-backward stores
    // no window-GEMM group.
    assert_eq!(forward.home().groups(), 1);
    let backward = network.compile(Request::roots([output]).backward());
    assert_eq!(backward.home().groups(), 0);
}

#[test]
fn kept_interiors_bar_fusion() {
    // Keeping the im2col matrix readable is a fusion barrier: the
    // chain must materialize so the keep-set can answer.
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new(
        [1, 1, 4, 4],
        (0..16).map(|v| v as f64 * 0.3 - 2.0).collect::<Vec<_>>(),
    ));
    let kernel = tape.leaf(Tensor::new(
        [4, 2],
        (0..8).map(|v| v as f64 * 0.25 - 0.75).collect::<Vec<_>>(),
    ));
    let patches = x
        .unfold(2, 2, 1, 1)
        .unfold(4, 2, 1, 1)
        .permute([0, 2, 4, 1, 3, 5])
        .reshape([9, 4]);
    let loss = patches.matmul(kernel).sum().symbol();
    let patches = patches.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let fused = network.compile(Request::roots([loss]));
    assert!(fused.describe().contains("window-gemm"));
    // Engine-backward plans do not fuse at all: their memory
    // contract stays exact for the reverse scan.
    let engine = network.compile(Request::roots([loss]).backward());
    assert!(!engine.describe().contains("window-gemm"));

    let barred = network.compile(Request::roots([loss]).observe([patches]));
    assert!(!barred.describe().contains("window-gemm"));
    let run = barred.forward(&parameters, std::iter::empty());
    assert_eq!(
        run.of(patches).to_vec(),
        network.forward(&parameters, []).of(patches).to_vec()
    );
}

#[test]
fn batched_products_do_not_fuse() {
    // The window-GEMM matcher keys on the rank-2 im2col shape; a
    // batched product must leave it indifferent.
    let tape = Tape::new();
    let lhs = tape
        .leaf(Tensor::new(
            [72],
            (0..72).map(|v| v as f64 * 0.05 - 1.8).collect::<Vec<_>>(),
        ))
        .reshape([2, 9, 4]);
    let rhs = tape.leaf(Tensor::new(
        [2, 4, 2],
        (0..16).map(|v| v as f64 * 0.25 - 2.0).collect::<Vec<_>>(),
    ));
    let loss = lhs.matmul(rhs).sum().symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let plan = network.compile(Request::roots([loss]));
    assert!(!plan.describe().contains("window-gemm"));
    let planned = plan.forward(&parameters, std::iter::empty());
    assert_eq!(
        planned.of(loss).to_vec(),
        network.forward(&parameters, []).of(loss).to_vec()
    );
}

#[test]
fn shared_windows_bar_fusion() {
    // A second consumer inside the chain bars fusion, and results
    // stay bitwise either way.
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new(
        [1, 1, 4, 4],
        (0..16).map(|v| v as f64 * 0.5 - 3.0).collect::<Vec<_>>(),
    ));
    let kernel = tape.leaf(Tensor::new(
        [4, 2],
        (0..8).map(|v| v as f64 * 0.125).collect::<Vec<_>>(),
    ));
    let windows = x.unfold(2, 2, 1, 1).unfold(4, 2, 1, 1);
    let patches = windows.permute([0, 2, 4, 1, 3, 5]).reshape([9, 4]);
    let loss = (patches.matmul(kernel).sum() + windows.sum()).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let plan = network.compile(Request::roots([loss]));
    assert!(!plan.describe().contains("window-gemm"));

    let planned = plan.forward(&parameters, std::iter::empty());
    let interpreted = network.forward(&parameters, []);
    assert_eq!(planned.of(loss).to_vec(), interpreted.of(loss).to_vec());
}

#[test]
fn the_numerics_posture_is_a_value_on_the_plan() {
    // A 32 x 32 x 32 product (32768 flops) sits above every backend
    // threshold, so a backend build exercises the Exact branch for
    // real; the default build's chain is empty and both postures
    // agree trivially. The reference below is the naive in-order
    // definition the built-in path is bit-identical to.
    let a_data: Vec<f32> = (0..1024).map(|v| (v % 37) as f32 * 0.21 - 3.7).collect();
    let b_data: Vec<f32> = (0..1024).map(|v| (v % 29) as f32 * 0.17 - 2.3).collect();
    let tape = Tape::new();
    let a = tape.parameter(Tensor::new([32, 32], a_data.clone()));
    let b = tape.parameter(Tensor::new([32, 32], b_data.clone()));
    let product = a.matmul(b).symbol();
    let network = tape.into_network();

    let fast = network.compile(Request::roots([product]));
    let exact = network.compile(Request::roots([product]).numerics(Numerics::Exact));
    assert_eq!(fast.numerics(), Numerics::Fast);
    assert_eq!(exact.numerics(), Numerics::Exact);

    // The Exact run reproduces the reference definition bit for bit,
    // in every build: the one-process oracle.
    let exact_values = exact
        .forward(&network.parameters(), [])
        .of(product)
        .to_vec();
    for row in 0..32 {
        for column in 0..32 {
            let mut total = 0.0_f32;
            for inner in 0..32 {
                total += a_data[row * 32 + inner] * b_data[inner * 32 + column];
            }
            assert_eq!(
                exact_values[row * 32 + column].to_bits(),
                total.to_bits(),
                "exact runs must reproduce the reference product bitwise"
            );
        }
    }

    // The Fast run stays within a reassociation envelope of Exact —
    // and equals it bitwise wherever the chain is empty or declines.
    let fast_values = fast.forward(&network.parameters(), []).of(product).to_vec();
    for (fast_value, exact_value) in fast_values.iter().zip(&exact_values) {
        let envelope = 1e-4 * exact_value.abs().max(1.0);
        assert!(
            (fast_value - exact_value).abs() <= envelope,
            "fast {fast_value} strays past the envelope around exact {exact_value}"
        );
    }
}
