//! Demonstrates how values allocated on a `Network` chain into an
//! expression graph that can be evaluated and differentiated.
//!
//! Run with: `cargo run --example chain`

use poorgrad::Network;

fn main() {
    let network = Network::new();

    // Leaves are the inputs of the graph: learnable parameters or data.
    // The network owns their state; the returned values are `Copy` proxies
    // borrowing it.
    let a = network.leaf(2.0_f64);
    let b = network.leaf(3.0);
    let c = network.leaf(4.0);
    println!("allocated {} leaves", network.len());

    // Operators record computed nodes on the same network. Proxies are
    // never consumed, so the same value can feed any number of expressions.
    let sum = a + b;
    let product = sum * c;
    let expression = -product + a * c;

    println!("chained -((a + b) * c) + a * c as {expression:?}");
    println!("the network now holds {} values", network.len());

    // The forward pass materializes every payload into per-run storage,
    // leaving the network untouched.
    let evaluation = network.forward();
    println!("forward: expression = {}", evaluation.of(expression));

    // The backward pass produces the gradient of the expression with
    // respect to every value. `a` feeds two subexpressions whose
    // contributions cancel exactly, hence its zero gradient.
    let gradients = network.backward(&evaluation, expression);
    println!(
        "gradients: d/da = {}, d/db = {}, d/dc = {}",
        gradients.of(a),
        gradients.of(b),
        gradients.of(c)
    );
}
