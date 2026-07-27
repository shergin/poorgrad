//! Benchmarks of forward and backward runs over recorded graphs.
//!
//! The two backward cases fence the ancestor mask: `dense-cone` is its
//! worst case (the target depends on every node, the mask is pure
//! bookkeeping), `sparse-cone` its best (a per-sample loss touching a
//! handful of nodes in a large tape).

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use poorgrad::{Network, Tensor};

fn run(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("run");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    // A 10_000-node scalar chain whose tail depends on every node.
    let chain = Network::new();
    let increment = chain.leaf(0.000001_f64);
    let mut tail = chain.leaf(1.0);
    for _ in 0..(10_000 - 2) {
        tail = tail + increment;
    }

    group.bench_function("forward/scalar-chain-10k", |bencher| {
        bencher.iter(|| chain.forward());
    });

    let chain_evaluation = chain.forward();
    group.bench_function("backward/dense-cone-10k", |bencher| {
        bencher.iter(|| chain_evaluation.backward(tail));
    });

    // The per-sample training pattern: 1000 losses sharing one
    // parameter, each depending on a ~7-node cone of a ~6000-node tape.
    let samples = Network::new();
    let weight = samples.parameter(0.5_f64);
    let mut losses = Vec::new();
    for index in 0..1000 {
        let input = samples.leaf(index as f64);
        let target = samples.leaf(2.0 * index as f64);
        let error = weight * input - target;
        losses.push(error * error);
    }
    let first_loss = losses[0];

    let samples_evaluation = samples.forward();
    group.bench_function("backward/sparse-cone", |bencher| {
        bencher.iter(|| samples_evaluation.backward(first_loss));
    });

    // A small dense regression in matrix form: matmul, subtraction,
    // elementwise square, and full reduction over real tensor payloads.
    let regression = Network::new();
    let inputs = regression.leaf(Tensor::filled([64, 32], 0.5_f64));
    let weights = regression.parameter(Tensor::filled([32, 16], 0.1_f64));
    let targets = regression.leaf(Tensor::filled([64, 16], 1.0_f64));
    let error = inputs.matmul(weights) - targets;
    let loss = (error * error).sum();

    group.bench_function("forward/tensor-regression", |bencher| {
        bencher.iter(|| regression.forward());
    });

    let regression_evaluation = regression.forward();
    group.bench_function("backward/tensor-regression", |bencher| {
        bencher.iter(|| regression_evaluation.backward(loss));
    });

    group.finish();
}

criterion_group!(benches, run);
criterion_main!(benches);
