//! Demonstrates how values allocated on a `Tape` chain into an
//! expression graph that can be evaluated and differentiated.
//!
//! Run with: `cargo run --example chain`

use topos::Tape;

fn main() {
    let tape = Tape::new();

    // Leaves are the inputs of the graph: learnable parameters or data.
    // The tape owns their state; the returned values are `Copy` proxies
    // borrowing it.
    let a = tape.leaf(2.0_f64);
    let b = tape.leaf(3.0);
    let c = tape.leaf(4.0);
    println!("allocated {} leaves", tape.len());

    // Operators record computed nodes on the same tape. Proxies are
    // never consumed, so the same value can feed any number of expressions.
    let sum = a + b;
    let product = sum * c;
    let expression = -product + a * c;

    println!("chained -((a + b) * c) + a * c as {expression:?}");
    println!("the tape now holds {} values", tape.len());

    // Symbols are the names every phase after recording speaks; the
    // seal consumes the tape and hands back the immutable spec.
    let (a, b, c, expression) = (a.symbol(), b.symbol(), c.symbol(), expression.symbol());
    let network = tape.into_network();
    let parameters = network.parameters();

    // The forward pass materializes every payload into per-run storage,
    // leaving the network untouched.
    let run = network.forward(&parameters, []);
    println!("forward: expression = {}", run.of(expression));

    // The backward pass produces the gradient of the expression with
    // respect to every value. `a` feeds two subexpressions whose
    // contributions cancel exactly, hence its zero gradient.
    let gradients = run.backward(expression);
    println!(
        "gradients: d/da = {}, d/db = {}, d/dc = {}",
        gradients.of(a),
        gradients.of(b),
        gradients.of(c)
    );
}
