//! Benchmarks of forward and backward runs over recorded graphs.
//!
//! The two backward cases fence the ancestor mask: `dense-cone` is its
//! worst case (the target depends on every node, the mask is pure
//! bookkeeping), `sparse-cone` its best (a per-sample loss touching a
//! handful of nodes in a large tape).

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use topos::{Tape, Tensor};

fn run(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("run");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    // A 10_000-node scalar chain whose tail depends on every node.
    let chain_tape = Tape::new();
    let increment = chain_tape.leaf(0.000001_f64);
    let mut tail = chain_tape.leaf(1.0);
    for _ in 0..(10_000 - 2) {
        tail = tail + increment;
    }
    let tail = tail.symbol();
    let chain = chain_tape.into_network();
    let chain_parameters = chain.parameters();

    group.bench_function("forward/scalar-chain-10k", |bencher| {
        bencher.iter(|| chain.forward(&chain_parameters, []));
    });

    let chain_evaluation = chain.forward(&chain_parameters, []);
    group.bench_function("backward/dense-cone-10k", |bencher| {
        bencher.iter(|| chain_evaluation.backward(tail));
    });

    // The per-sample training pattern: 1000 losses sharing one
    // parameter, each depending on a ~7-node cone of a ~6000-node tape.
    let samples_tape = Tape::new();
    let weight = samples_tape.parameter(0.5_f64);
    let mut losses = Vec::new();
    for index in 0..1000 {
        let input = samples_tape.leaf(index as f64);
        let target = samples_tape.leaf(2.0 * index as f64);
        let error = weight * input - target;
        losses.push(error * error);
    }
    let first_loss = losses[0].symbol();
    let samples = samples_tape.into_network();

    let samples_evaluation = samples.forward(&samples.parameters(), []);
    group.bench_function("backward/sparse-cone", |bencher| {
        bencher.iter(|| samples_evaluation.backward(first_loss));
    });

    // A small dense regression in matrix form: matmul, subtraction,
    // elementwise square, and full reduction over real tensor payloads.
    let regression_tape = Tape::new();
    let inputs = regression_tape.leaf(Tensor::filled([64, 32], 0.5_f64));
    let weights = regression_tape.parameter(Tensor::filled([32, 16], 0.1_f64));
    let targets = regression_tape.leaf(Tensor::filled([64, 16], 1.0_f64));
    let error = inputs.matmul(weights) - targets;
    let loss = (error * error).sum().symbol();
    let regression = regression_tape.into_network();
    let regression_parameters = regression.parameters();

    group.bench_function("forward/tensor-regression", |bencher| {
        bencher.iter(|| regression.forward(&regression_parameters, []));
    });

    let regression_evaluation = regression.forward(&regression_parameters, []);
    group.bench_function("backward/tensor-regression", |bencher| {
        bencher.iter(|| regression_evaluation.backward(loss));
    });

    // The same regression over dense payloads: `filled` above stores
    // constants, which bypass the dense matmul and slice paths, so
    // these twin cases are the ones that price the accelerated tiers.
    let dense_tape = Tape::new();
    let dense_values = |len: usize, seed: u64| -> Vec<f64> {
        (0..len)
            .map(|index| {
                ((index as u64).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f64 / 1000.0
            })
            .collect()
    };
    let inputs = dense_tape.leaf(Tensor::new([64, 32], dense_values(64 * 32, 1)));
    let weights = dense_tape.parameter(Tensor::new([32, 16], dense_values(32 * 16, 2)));
    let targets = dense_tape.leaf(Tensor::new([64, 16], dense_values(64 * 16, 3)));
    let error = inputs.matmul(weights) - targets;
    let dense_loss = (error * error).sum().symbol();
    let dense = dense_tape.into_network();
    let dense_parameters = dense.parameters();

    group.bench_function("forward/tensor-regression-dense", |bencher| {
        bencher.iter(|| dense.forward(&dense_parameters, []));
    });

    let dense_evaluation = dense.forward(&dense_parameters, []);
    group.bench_function("backward/tensor-regression-dense", |bencher| {
        bencher.iter(|| dense_evaluation.backward(dense_loss));
    });

    group.finish();
}

criterion_group!(benches, run);
criterion_main!(benches);
