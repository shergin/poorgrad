//! Benchmarks of multi-thread scaling.
//!
//! `parallel-backward` sweeps 1000 per-sample backwards over one shared
//! evaluation and should scale with threads (runs never lock the
//! network). `fork-training` gives every thread its own fork doing the
//! same fixed amount of training; ideal scaling is flat time as threads
//! grow. Training stopped touching the arena lock when the parameter
//! store landed, so any remaining rise measures allocator contention
//! from per-run buffers.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use topos::Network;

fn scale(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("scale");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    let samples = Network::new();
    let weight = samples.parameter(0.5_f64);
    let mut losses = Vec::new();
    for index in 0..1000 {
        let input = samples.leaf(index as f64);
        let target = samples.leaf(2.0 * index as f64);
        let error = weight * input - target;
        losses.push(error * error);
    }
    let evaluation = samples.forward();

    for threads in [1usize, 2, 4, 8] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool builds");
        group.bench_with_input(
            BenchmarkId::new("parallel-backward", threads),
            &threads,
            |bencher, _| {
                bencher.iter(|| {
                    pool.install(|| {
                        losses
                            .par_iter()
                            .map(|&loss| *evaluation.backward(loss).of(weight))
                            .sum::<f64>()
                    })
                });
            },
        );
    }

    let trainer = Network::new();
    let w = trainer.parameter(0.0_f64);
    let x = trainer.leaf(3.0);
    let y = trainer.leaf(15.0);
    let error = w * x - y;
    let loss = error * error;
    let loss_symbol = loss.symbol();

    for threads in [1usize, 2, 4] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool builds");
        group.bench_with_input(
            BenchmarkId::new("fork-training", threads),
            &threads,
            |bencher, &threads| {
                bencher.iter(|| {
                    pool.install(|| {
                        (0..threads).into_par_iter().for_each(|_| {
                            let mut fork = trainer.clone();
                            for _ in 0..20 {
                                let loss = fork.resolve(loss_symbol);
                                let evaluation = fork.forward();
                                let gradients = evaluation.backward(loss);
                                fork = fork.update(&gradients, |parameter, gradient| {
                                    parameter - 0.01 * gradient
                                });
                            }
                        })
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, scale);
criterion_main!(benches);
