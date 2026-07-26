use std::thread;

use crate::Function;

use super::{Network, Value};

#[test]
fn new_network_is_empty() {
    let network = Network::<f64>::new();
    assert!(network.is_empty());
    assert_eq!(network.len(), 0);
}

#[test]
fn rebind_rejects_unallocated_nodes() {
    let network = Network::<f64>::new();
    let other = Network::new();
    let foreign = other.leaf(1.0);
    assert!(network.rebind(foreign).is_none());
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
    let rebound = fork.rebind(v1).unwrap();
    assert_eq!(rebound.data(), Some(1.0));
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

#[test]
fn network_and_value_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Network<f64>>();
    assert_send_sync::<Value<'static, f64>>();
}
