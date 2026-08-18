//! Benchmarks of the training-step state transition.
//!
//! `step` is benchmarked across graph sizes at a fixed parameter
//! count: it rebuilds only the caller-owned store, so the time must
//! stay flat as the graph grows. This bench is the regression fence
//! for that O(parameters) claim.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use topos::{Tape, Tensor, Tensorial};

const PARAMETERS: usize = 100;

fn training_step(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("train");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    // One full step (forward, backward, update) on a 100-sample scalar
    // loss over shared `w` and `b`.
    let scalar_tape = Tape::new();
    let w = scalar_tape.parameter(0.0_f64);
    let b = scalar_tape.parameter(0.0);
    let loss = (0..100)
        .map(|index| {
            let input = scalar_tape.leaf(index as f64);
            let target = scalar_tape.leaf(2.0 * index as f64 + 1.0);
            let error = w * input + b - target;
            error * error
        })
        .reduce(|total, sample| total + sample)
        .expect("at least one sample")
        .symbol();
    let scalar = scalar_tape.into_network();
    let scalar_parameters = scalar.parameters();

    group.bench_function("step/scalar-100-samples", |bencher| {
        bencher.iter(|| {
            let evaluation = scalar.forward(&scalar_parameters, []);
            let gradients = evaluation.backward(loss);
            scalar_parameters.step(&gradients, |parameter, gradient| {
                parameter - 0.01 * gradient
            })
        });
    });

    // One full step of the matrix-form regression.
    let tensor_tape = Tape::new();
    let inputs = tensor_tape.leaf(Tensor::filled([64, 32], 0.5_f64));
    let weights = tensor_tape.parameter(Tensor::filled([32, 16], 0.1_f64));
    let targets = tensor_tape.leaf(Tensor::filled([64, 16], 1.0_f64));
    let error = inputs.matmul(weights) - targets;
    let tensor_loss = (error * error).sum().symbol();
    let learning_rate = Tensor::new([], [0.01_f64]);
    let tensor = tensor_tape.into_network();
    let tensor_parameters = tensor.parameters();

    group.bench_function("step/tensor-regression", |bencher| {
        bencher.iter(|| {
            let evaluation = tensor.forward(&tensor_parameters, []);
            let gradients = evaluation.backward(tensor_loss);
            tensor_parameters.step(&gradients, |parameter, gradient| {
                parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
            })
        });
    });

    // `step` alone, with the parameter count fixed and the graph
    // padded to increasing sizes.
    for nodes in [1_000usize, 10_000, 100_000] {
        let tape = Tape::new();
        let target = (0..PARAMETERS)
            .map(|_| tape.parameter(1.0_f64))
            .reduce(|total, parameter| total + parameter)
            .expect("at least one parameter")
            .symbol();
        let mut padding = tape.leaf(1.0);
        while tape.len() < nodes {
            padding = padding * padding;
        }
        let network = tape.into_network();
        let parameters = network.parameters();
        let evaluation = network.forward(&parameters, []);
        let direction = evaluation.backward(target);

        group.throughput(Throughput::Elements(nodes as u64));
        group.bench_with_input(BenchmarkId::new("step", nodes), &nodes, |bencher, _| {
            bencher.iter(|| {
                parameters.step(&direction, |parameter, gradient| {
                    parameter - 0.01 * gradient
                })
            });
        });
    }

    group.finish();
}

criterion_group!(benches, training_step);
criterion_main!(benches);
