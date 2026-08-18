use std::thread;

use crate::Tape;
use crate::function::Function;

#[test]
fn operator_sugar_allocates_on_the_same_tape() {
    let tape = Tape::new();
    let v1 = tape.leaf(2.0_f64);
    let v2 = tape.leaf(3.0);

    let x = v1 + v2;

    assert_eq!(tape.len(), 3);
    assert_eq!(x.function(), Function::add());
    assert_eq!(x.operands(), vec![v1.id(), v2.id()]);
    assert_eq!(x.payload(), None);
}

#[test]
fn copy_values_are_reusable_across_expressions() {
    let tape = Tape::new();
    let v1 = tape.leaf(2.0_f64);
    let v2 = tape.leaf(3.0);

    let x = v1 * v2;
    let y = v1 + v2;
    let z = x + y;
    let negated = -z;

    assert_eq!(tape.len(), 6);
    assert_eq!(z.function(), Function::add());
    assert_eq!(z.operands(), vec![x.id(), y.id()]);
    assert_eq!(negated.function(), Function::neg());
    assert_eq!(negated.operands(), vec![z.id()]);
}

#[test]
fn expression_chain_allocates_intermediate_values() {
    let tape = Tape::new();
    let v1 = tape.leaf(2.0_f64);
    let v2 = tape.leaf(3.0);
    let v3 = tape.leaf(4.0);

    let x = v1 * v2 + v3;

    assert_eq!(tape.len(), 5);
    assert!(matches!(x.function(), Function::Add(_)));
}

#[test]
fn payload_literals_mix_into_expressions() {
    let tape = Tape::new();
    let x = tape.leaf(3.0_f64);

    let y = 2.0 * x + 1.0;
    let z = 6.0 / x;

    // Every literal appearance records its own leaf: x, 2, the product,
    // 1, the sum, 6, and the quotient.
    assert_eq!(tape.len(), 7);

    let (x, y, z) = (x.symbol(), y.symbol(), z.symbol());
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(*run.of(y), 7.0);
    assert_eq!(*run.of(z), 2.0);

    let gradients = run.backward(y);
    assert_eq!(*gradients.of(x), 2.0);
}

#[test]
fn values_chain_inside_scoped_threads() {
    let tape = Tape::new();
    let v1 = tape.leaf(1.0_f64);
    let v2 = tape.leaf(2.0);

    thread::scope(|scope| {
        scope.spawn(move || {
            let _ = v1 + v2;
        });
        scope.spawn(move || {
            let _ = v1 * v2;
        });
    });

    assert_eq!(tape.len(), 4);
}

#[test]
#[should_panic(expected = "different tapes")]
fn cross_tape_operation_panics() {
    let first = Tape::new();
    let second = Tape::new();
    let a = first.leaf(1.0_f64);
    let b = second.leaf(2.0);
    let _ = a + b;
}
