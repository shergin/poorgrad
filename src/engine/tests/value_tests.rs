use std::thread;

use crate::Network;
use crate::engine::Function;

#[test]
fn operator_sugar_allocates_on_the_same_network() {
    let network = Network::new();
    let v1 = network.leaf(2.0_f64);
    let v2 = network.leaf(3.0);

    let x = v1 + v2;

    assert_eq!(network.len(), 3);
    assert_eq!(x.function(), Function::add(v1.id(), v2.id()));
    assert_eq!(x.payload(), None);
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
#[should_panic(expected = "different networks")]
fn cross_network_operation_panics() {
    let first = Network::new();
    let second = Network::new();
    let a = first.leaf(1.0_f64);
    let b = second.leaf(2.0);
    let _ = a + b;
}
