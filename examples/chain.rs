//! Demonstrates how values allocated on a `Network` chain into an
//! expression graph.
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
    println!(
        "leaf payloads: a = {:?}, b = {:?}, c = {:?}",
        a.data(),
        b.data(),
        c.data()
    );

    // Operators record computed nodes on the same network. Proxies are
    // never consumed, so the same value can feed any number of expressions.
    let sum = a + b;
    let product = sum * c;
    let expression = -product + a * c;

    println!("chained -((a + b) * c) + a * c as {expression:?}");
    println!("the network now holds {} values", network.len());

    // Computed values have no payload until the forward pass materializes
    // them into a per-run buffer.
    println!("computed payload before forward: {:?}", expression.data());
}
