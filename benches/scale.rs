//! Benchmarks of multi-thread scaling.
//!
//! `parallel-backward` sweeps 1000 per-sample backwards over one shared
//! evaluation and should scale with threads (runs borrow nothing).
//! `state-training` gives every thread its own cloned `Parameters`
//! state doing the same fixed amount of training over one shared
//! network; ideal scaling is flat time as threads grow. Training never
//! touches a lock, so any remaining rise measures allocator contention
//! from per-run buffers.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use topos::Tape;

fn scale(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("scale");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    let samples_tape = Tape::new();
    let weight = samples_tape.parameter(0.5_f64);
    let mut losses = Vec::new();
    for index in 0..1000 {
        let input = samples_tape.leaf(index as f64);
        let target = samples_tape.leaf(2.0 * index as f64);
        let error = weight * input - target;
        losses.push(error * error);
    }
    let weight = weight.symbol();
    let losses: Vec<topos::Symbol> = losses.iter().map(|loss| loss.symbol()).collect();
    let samples = samples_tape.into_network();
    let evaluation = samples.forward(&samples.parameters(), []);

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

    let trainer_tape = Tape::new();
    let w = trainer_tape.parameter(0.0_f64);
    let x = trainer_tape.leaf(3.0);
    let y = trainer_tape.leaf(15.0);
    let error = w * x - y;
    let loss_symbol = (error * error).symbol();
    let trainer = trainer_tape.into_network();
    let trainer_parameters = trainer.parameters();

    for threads in [1usize, 2, 4] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool builds");
        group.bench_with_input(
            BenchmarkId::new("state-training", threads),
            &threads,
            |bencher, &threads| {
                bencher.iter(|| {
                    pool.install(|| {
                        (0..threads).into_par_iter().for_each(|_| {
                            let mut state = trainer_parameters.clone();
                            for _ in 0..20 {
                                let evaluation = trainer.forward(&state, []);
                                let gradients = evaluation.backward(loss_symbol);
                                state = state.step(&gradients, |parameter, gradient| {
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
