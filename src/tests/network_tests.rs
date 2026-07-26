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
fn resolve_rejects_unallocated_symbols() {
    let network = Network::<f64>::new();
    let other = Network::new();
    let foreign = other.leaf(1.0);
    assert!(network.resolve(foreign.symbol()).is_none());
}

#[test]
fn parameter_carries_payload_like_a_leaf() {
    let network = Network::new();
    let w = network.parameter(1.5_f64);
    let x = network.leaf(2.0);
    assert_eq!(w.data(), Some(1.5));

    let y = w * x;

    let evaluation = network.forward();
    assert_eq!(*evaluation.value(y), 3.0);
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
    assert_eq!(*evaluation.value(a), 2.0);
    assert_eq!(*evaluation.value(expression), -20.0);
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
    assert_eq!(*evaluation.value(f), 8.0);

    let gradients = network.backward(&evaluation, f);
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
    assert_eq!(*evaluation.value(f), -4.0);

    let gradients = network.backward(&evaluation, f);
    assert_eq!(*gradients.of(a), -4.0);
}

#[test]
fn tanh_routes_gradient_through_its_output() {
    let network = Network::new();
    let x = network.leaf(0.5_f64);
    let y = x.tanh();

    let evaluation = network.forward();
    assert_eq!(*evaluation.value(y), 0.5_f64.tanh());

    let gradients = network.backward(&evaluation, y);
    let expected = 1.0 - 0.5_f64.tanh().powi(2);
    assert!((gradients.of(x) - expected).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "stale")]
fn backward_rejects_stale_evaluation() {
    let network = Network::new();
    let a = network.leaf(2.0_f64);
    let evaluation = network.forward();
    network.leaf(3.0);
    network.backward(&evaluation, a);
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
    let rebound = fork.resolve(v1.symbol()).unwrap();
    assert_eq!(rebound.data(), Some(1.0));
}

#[test]
fn updated_replaces_parameters_and_keeps_everything_else() {
    let network = Network::new();
    let w = network.parameter(1.0_f64);
    let x = network.leaf(2.0);
    let y = w * x;

    let evaluation = network.forward();
    let gradients = network.backward(&evaluation, y);
    let updated = network.updated(gradients.as_field(), |parameter, gradient| {
        parameter - gradient
    });

    assert_eq!(updated.len(), network.len());
    // The gradient of `y` with respect to `w` is `x`, so the parameter
    // moves from 1 to -1; the plain leaf stays untouched.
    assert_eq!(updated.resolve(w.symbol()).unwrap().data(), Some(-1.0));
    assert_eq!(updated.resolve(x.symbol()).unwrap().data(), Some(2.0));
    // The old generation is untouched as well.
    assert_eq!(w.data(), Some(1.0));
}

#[test]
fn gradient_descent_converges() {
    // Minimizes `(w - 3)^2` starting from `w = 0`.
    let network = Network::new();
    let w = network.parameter(0.0_f64);
    let target = network.leaf(3.0);
    let error = w + -target;
    let loss = error * error;

    let w_symbol = w.symbol();
    let loss_symbol = loss.symbol();

    let mut network = network;
    for _ in 0..30 {
        let loss = network.resolve(loss_symbol).unwrap();
        let evaluation = network.forward();
        let gradients = network.backward(&evaluation, loss);
        network = network.updated(gradients.as_field(), |parameter, gradient| {
            parameter - 0.3 * gradient
        });
    }

    let learned = network.resolve(w_symbol).unwrap().data().unwrap();
    assert!((learned - 3.0).abs() < 1e-6);
}

#[test]
fn momentum_descent_converges() {
    // Minimizes `(w - 3)^2` with heavy-ball momentum, the velocity kept
    // as a field carried across generations.
    let network = Network::new();
    let w = network.parameter(0.0_f64);
    let target = network.leaf(3.0);
    let error = w + -target;
    let loss = error * error;

    let w_symbol = w.symbol();
    let loss_symbol = loss.symbol();

    let mut network = network;
    let mut velocity: Option<Field<f64>> = None;
    for _ in 0..40 {
        let loss = network.resolve(loss_symbol).unwrap();
        let evaluation = network.forward();
        let gradients = network.backward(&evaluation, loss);
        let step = match velocity {
            Some(previous) => previous.scaled(0.5) + gradients.into_field(),
            None => gradients.into_field(),
        };
        network = network.updated(&step, |parameter, direction| parameter - 0.1 * direction);
        velocity = Some(step);
    }

    let learned = network.resolve(w_symbol).unwrap().data().unwrap();
    assert!((learned - 3.0).abs() < 1e-3);
}

#[test]
#[should_panic(expected = "stale")]
fn updated_rejects_stale_gradients() {
    let network = Network::new();
    let w = network.parameter(1.0_f64);
    let evaluation = network.forward();
    let gradients = network.backward(&evaluation, w);
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
    let gradients = first.backward(&evaluation, w);
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
