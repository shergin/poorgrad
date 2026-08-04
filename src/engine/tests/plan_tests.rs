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
