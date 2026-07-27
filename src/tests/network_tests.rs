use std::thread;

use crate::{Field, Function};

use super::Network;

#[test]
fn new_network_is_empty() {
    let network = Network::<f64>::new();
    assert!(network.is_empty());
    assert_eq!(network.len(), 0);
}

#[test]
fn try_resolve_probes_foreign_and_unrecorded_symbols() {
    let network = Network::<f64>::new();
    let foreign = Network::new().leaf(1.0).symbol();
    assert!(network.try_resolve(foreign).is_none());

    let fork = network.clone();
    let late = network.leaf(2.0).symbol();
    assert!(fork.try_resolve(late).is_none());
}

#[test]
#[should_panic(expected = "different network lineage")]
fn resolve_rejects_foreign_symbols() {
    let network = Network::<f64>::new();
    let other = Network::new();
    let foreign = other.leaf(1.0);
    network.resolve(foreign.symbol());
}

#[test]
#[should_panic(expected = "not allocated")]
fn resolve_rejects_unrecorded_symbols() {
    let network = Network::<f64>::new();
    let fork = network.clone();
    let late = network.leaf(1.0);
    fork.resolve(late.symbol());
}

#[test]
fn parameter_carries_payload_like_a_leaf() {
    let network = Network::new();
    let w = network.parameter(1.5_f64);
    let x = network.leaf(2.0);
    assert_eq!(w.data(), Some(1.5));

    let y = w * x;

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(y), 3.0);
}

#[test]
fn leaf_allocates_on_the_network() {
    let network = Network::new();
    let a = network.leaf(2.0_f64);
    let b = network.leaf(3.0);
    assert_eq!(network.len(), 2);
    assert_ne!(a.id(), b.id());
    assert_eq!(a.data(), Some(2.0));
    assert_eq!(b.data(), Some(3.0));
}

#[test]
fn operator_sugar_allocates_on_the_same_network() {
    let network = Network::new();
    let v1 = network.leaf(2.0_f64);
    let v2 = network.leaf(3.0);

    let x = v1 + v2;

    assert_eq!(network.len(), 3);
    assert_eq!(x.function(), Function::add(v1.id(), v2.id()));
    assert_eq!(x.data(), None);
}

#[test]
fn copy_values_are_reusable_across_expressions() {
    let network = Network::new();
    let v1 = network.leaf(2.0_f64);
    let v2 = network.leaf(3.0);

    let x = v1 * v2;
    let y = v1 + v2;
    let z = x + y;
    let negated = -z;

    assert_eq!(network.len(), 6);
    assert_eq!(z.function(), Function::add(x.id(), y.id()));
    assert_eq!(negated.function(), Function::neg(z.id()));
}

#[test]
fn expression_chain_allocates_intermediate_values() {
    let network = Network::new();
    let v1 = network.leaf(2.0_f64);
    let v2 = network.leaf(3.0);
    let v3 = network.leaf(4.0);

    let x = v1 * v2 + v3;

    assert_eq!(network.len(), 5);
    assert!(matches!(x.function(), Function::Add(_)));
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

    // `a` feeds both the product and the sum, so its gradient must
    // accumulate: d(a * b + a)/da = b + 1.
    let f = a * b + a;

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(f), 8.0);

    let gradients = evaluation.backward(f);
    assert_eq!(*gradients.of(f), 1.0);
    assert_eq!(*gradients.of(a), 4.0);
    assert_eq!(*gradients.of(b), 2.0);
}

#[test]
fn backward_routes_negation() {
    let network = Network::new();
    let a = network.leaf(2.0_f64);
    let f = -(a * a);

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(f), -4.0);

    let gradients = evaluation.backward(f);
    assert_eq!(*gradients.of(a), -4.0);
}

#[test]
fn subtraction_routes_signed_gradients() {
    let network = Network::new();
    let a = network.leaf(5.0_f64);
    let b = network.leaf(3.0);
    let difference = a - b;

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(difference), 2.0);

    let gradients = evaluation.backward(difference);
    assert_eq!(*gradients.of(a), 1.0);
    assert_eq!(*gradients.of(b), -1.0);
}

#[test]
fn division_reuses_its_output_in_backward() {
    let network = Network::new();
    let a = network.leaf(6.0_f64);
    let b = network.leaf(2.0);
    let quotient = a / b;

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(quotient), 3.0);

    let gradients = evaluation.backward(quotient);
    assert_eq!(*gradients.of(a), 0.5);
    assert_eq!(*gradients.of(b), -1.5);
}

#[test]
fn tanh_routes_gradient_through_its_output() {
    let network = Network::new();
    let x = network.leaf(0.5_f64);
    let y = x.tanh();

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(y), 0.5_f64.tanh());

    let gradients = evaluation.backward(y);
    let expected = 1.0 - 0.5_f64.tanh().powi(2);
    assert!((gradients.of(x) - expected).abs() < 1e-12);
}

#[test]
fn exp_reuses_its_output_in_backward() {
    let network = Network::new();
    let x = network.leaf(1.0_f64);
    let y = x.exp();

    let evaluation = network.forward();
    let value = *evaluation.of(y);
    assert!((value - std::f64::consts::E).abs() < 1e-12);

    // The derivative of the exponential is the output itself.
    let gradients = evaluation.backward(y);
    assert!((gradients.of(x) - value).abs() < 1e-12);
}

#[test]
fn ln_routes_gradient_through_its_operand() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let y = x.ln();

    let evaluation = network.forward();
    assert!((evaluation.of(y) - 2.0_f64.ln()).abs() < 1e-12);

    let gradients = evaluation.backward(y);
    assert!((gradients.of(x) - 0.5).abs() < 1e-12);
}

#[test]
fn sigmoid_composes_from_primitives() {
    let network = Network::new();
    let x = network.leaf(0.0_f64);
    let one = network.leaf(1.0);

    let sigmoid = one / (one + (-x).exp());

    let evaluation = network.forward();
    assert!((evaluation.of(sigmoid) - 0.5).abs() < 1e-12);

    // The classic identity: d sigmoid / dx = sigmoid * (1 - sigmoid).
    let gradients = evaluation.backward(sigmoid);
    assert!((gradients.of(x) - 0.25).abs() < 1e-12);
}

#[test]
fn backward_survives_later_recordings() {
    let network = Network::new();
    let a = network.leaf(2.0_f64);
    let evaluation = network.forward();
    network.leaf(3.0);

    // The evaluation carries its own snapshot, so differentiating it
    // stays coherent after the network grows; later values are simply
    // absent from the result.
    let gradients = evaluation.backward(a);
    assert_eq!(*gradients.of(a), 1.0);
}

#[test]
fn backward_skips_disconnected_nodes() {
    let network = Network::new();
    let unrelated = network.leaf(0.0_f64);
    // A singular expression the target does not depend on: its forward
    // value is NaN, but its derivative rule must never run.
    let quotient = unrelated / unrelated;
    let input = network.leaf(2.0);
    let target = input * input;

    let evaluation = network.forward();
    let gradients = evaluation.backward(target);
    assert_eq!(*gradients.of(input), 4.0);
    assert_eq!(*gradients.of(unrelated), 0.0);
    assert_eq!(*gradients.of(quotient), 0.0);
}

#[test]
fn backward_ignores_singular_paths_through_shared_leaves() {
    let network = Network::new();
    let x = network.leaf(0.0_f64);
    // `x` feeds both the target and a disconnected singular quotient; the
    // quotient must not poison the genuine gradient of `x`.
    let _quotient = x / x;
    let target = x * x;

    let evaluation = network.forward();
    let gradients = evaluation.backward(target);
    assert_eq!(*gradients.of(x), 0.0);
}

#[test]
fn backward_skips_nodes_recorded_after_the_target() {
    let network = Network::new();
    let input = network.leaf(2.0_f64);
    let target = input * input;
    let late = network.leaf(0.0);
    let quotient = late / late;

    let evaluation = network.forward();
    let gradients = evaluation.backward(target);
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
    let evaluation = first.forward();
    evaluation.backward(foreign);
}

#[test]
fn payload_literals_mix_into_expressions() {
    let network = Network::new();
    let x = network.leaf(3.0_f64);

    let y = 2.0 * x + 1.0;
    let z = 6.0 / x;

    // Every literal appearance records its own leaf: x, 2, the product,
    // 1, the sum, 6, and the quotient.
    assert_eq!(network.len(), 7);

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(y), 7.0);
    assert_eq!(*evaluation.of(z), 2.0);

    let gradients = evaluation.backward(y);
    assert_eq!(*gradients.of(x), 2.0);
}

#[test]
fn values_chain_inside_scoped_threads() {
    let network = Network::new();
    let v1 = network.leaf(1.0_f64);
    let v2 = network.leaf(2.0);

    thread::scope(|scope| {
        scope.spawn(move || {
            let _ = v1 + v2;
        });
        scope.spawn(move || {
            let _ = v1 * v2;
        });
    });

    assert_eq!(network.len(), 4);
}

#[test]
fn clone_forks_the_network() {
    let network = Network::new();
    let v1 = network.leaf(1.0_f64);

    let fork = network.clone();
    network.leaf(2.0);

    assert_eq!(network.len(), 2);
    assert_eq!(fork.len(), 1);
    let rebound = fork.resolve(v1.symbol());
    assert_eq!(rebound.data(), Some(1.0));
}

#[test]
fn updated_replaces_parameters_and_keeps_everything_else() {
    let network = Network::new();
    let w = network.parameter(1.0_f64);
    let x = network.leaf(2.0);
    let y = w * x;

    let evaluation = network.forward();
    let gradients = evaluation.backward(y);
    let updated = network.updated(gradients.as_field(), |parameter, gradient| {
        parameter - gradient
    });

    assert_eq!(updated.len(), network.len());
    // The gradient of `y` with respect to `w` is `x`, so the parameter
    // moves from 1 to -1; the plain leaf stays untouched.
    assert_eq!(updated.resolve(w.symbol()).data(), Some(-1.0));
    assert_eq!(updated.resolve(x.symbol()).data(), Some(2.0));
    // The old generation is untouched as well.
    assert_eq!(w.data(), Some(1.0));
}

#[test]
fn gradient_descent_converges() {
    // Minimizes `(w - 3)^2` starting from `w = 0`.
    let network = Network::new();
    let w = network.parameter(0.0_f64);
    let target = network.leaf(3.0);
    let error = w - target;
    let loss = error * error;

    let w_symbol = w.symbol();
    let loss_symbol = loss.symbol();

    let mut network = network;
    for _ in 0..30 {
        let loss = network.resolve(loss_symbol);
        let evaluation = network.forward();
        let gradients = evaluation.backward(loss);
        network = network.updated(gradients.as_field(), |parameter, gradient| {
            parameter - 0.3 * gradient
        });
    }

    let learned = network.resolve(w_symbol).data().unwrap();
    assert!((learned - 3.0).abs() < 1e-6);
}

#[test]
fn momentum_descent_converges() {
    // Minimizes `(w - 3)^2` with heavy-ball momentum, the velocity kept
    // as a field carried across generations.
    let network = Network::new();
    let w = network.parameter(0.0_f64);
    let target = network.leaf(3.0);
    let error = w - target;
    let loss = error * error;

    let w_symbol = w.symbol();
    let loss_symbol = loss.symbol();

    let mut network = network;
    let mut velocity: Option<Field<f64>> = None;
    for _ in 0..40 {
        let loss = network.resolve(loss_symbol);
        let evaluation = network.forward();
        let gradients = evaluation.backward(loss);
        let step = match velocity {
            Some(previous) => previous.scaled(0.5) + gradients.into_field(),
            None => gradients.into_field(),
        };
        network = network.updated(&step, |parameter, direction| parameter - 0.1 * direction);
        velocity = Some(step);
    }

    let learned = network.resolve(w_symbol).data().unwrap();
    assert!((learned - 3.0).abs() < 1e-3);
}

#[test]
#[should_panic(expected = "stale")]
fn updated_rejects_stale_gradients() {
    let network = Network::new();
    let w = network.parameter(1.0_f64);
    let evaluation = network.forward();
    let gradients = evaluation.backward(w);
    network.leaf(2.0);
    network.updated(gradients.as_field(), |parameter, _gradient| {
        parameter.clone()
    });
}

#[test]
#[should_panic(expected = "different network")]
fn updated_rejects_foreign_gradients() {
    let first = Network::new();
    let w = first.parameter(1.0_f64);
    let evaluation = first.forward();
    let gradients = evaluation.backward(w);
    let second = Network::<f64>::new();
    second.updated(gradients.as_field(), |parameter, _gradient| {
        parameter.clone()
    });
}

#[test]
#[should_panic(expected = "different networks")]
fn cross_network_operation_panics() {
    let first = Network::new();
    let second = Network::new();
    let a = first.leaf(1.0_f64);
    let b = second.leaf(2.0);
    let _ = a + b;
}
