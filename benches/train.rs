//! Benchmarks of the training-step state transition.
//!
//! `update` is benchmarked across graph sizes at a fixed parameter
//! count: since the parameter store landed it rebuilds only the store,
//! so the time must stay flat as the graph grows. This bench is the
//! regression fence for that O(parameters) claim.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use topos::{Network, Tensor, Tensorial};

const PARAMETERS: usize = 100;

fn training_step(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("train");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    // One full step (forward, backward, update) on a 100-sample scalar
    // loss over shared `w` and `b`.
    let scalar = Network::new();
    let w = scalar.parameter(0.0_f64);
    let b = scalar.parameter(0.0);
    let loss = (0..100)
        .map(|index| {
            let input = scalar.leaf(index as f64);
            let target = scalar.leaf(2.0 * index as f64 + 1.0);
            let error = w * input + b - target;
            error * error
        })
        .reduce(|total, sample| total + sample)
        .expect("at least one sample");

    group.bench_function("step/scalar-100-samples", |bencher| {
        bencher.iter(|| {
            let evaluation = scalar.forward();
            let gradients = evaluation.backward(loss);
            scalar.update(&gradients, |parameter, gradient| {
                parameter - 0.01 * gradient
            })
        });
    });

    // One full step of the matrix-form regression.
    let tensor = Network::new();
    let inputs = tensor.leaf(Tensor::filled([64, 32], 0.5_f64));
    let weights = tensor.parameter(Tensor::filled([32, 16], 0.1_f64));
    let targets = tensor.leaf(Tensor::filled([64, 16], 1.0_f64));
    let error = inputs.matmul(weights) - targets;
    let tensor_loss = (error * error).sum();
    let learning_rate = Tensor::new([], [0.01_f64]);

    group.bench_function("step/tensor-regression", |bencher| {
        bencher.iter(|| {
            let evaluation = tensor.forward();
            let gradients = evaluation.backward(tensor_loss);
            tensor.update(&gradients, |parameter, gradient| {
                parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
            })
        });
    });

    // `update` alone, with the parameter count fixed and the graph
    // padded to increasing sizes.
    for nodes in [1_000usize, 10_000, 100_000] {
        let network = Network::new();
        let target = (0..PARAMETERS)
            .map(|_| network.parameter(1.0_f64))
            .reduce(|total, parameter| total + parameter)
            .expect("at least one parameter");
        let mut padding = network.leaf(1.0);
        while network.len() < nodes {
            padding = padding * padding;
        }
        let evaluation = network.forward();
        let direction = evaluation.backward(target);

        group.throughput(Throughput::Elements(nodes as u64));
        group.bench_with_input(BenchmarkId::new("update", nodes), &nodes, |bencher, _| {
            bencher.iter(|| {
                network.update(&direction, |parameter, gradient| {
                    parameter - 0.01 * gradient
                })
            });
        });
    }

    group.finish();
}

criterion_group!(benches, training_step);
criterion_main!(benches);
