//! Benchmarks of the dense elementwise paths.
//!
//! Cases mirror the passes a training step actually makes over big
//! tensors: a dense-times-dense multiply (the squares and gradient
//! products), a dense-plus-constant add (the gradient accumulator's
//! zero seed), and a negation map. Throughput is in elements per
//! second.

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use poorgrad::{Elementary, Tensor};

/// Builds a `[len]` tensor with a deterministic, cheap fill.
fn filled(len: usize, seed: u64) -> Tensor<f32> {
    let elements: Vec<f32> = (0..len)
        .map(|index| {
            ((index as u64).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32 / 500.0 - 1.0
        })
        .collect();
    Tensor::new([len], elements)
}

fn elementwise(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("elementwise");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    let len = 1 << 21;
    let left = filled(len, 1);
    let right = filled(len, 2);
    let zero = Tensor::filled([len], 0.0_f32);

    group.throughput(Throughput::Elements(len as u64));
    group.bench_function("f32/multiply-dense-2m", |bencher| {
        bencher.iter(|| left.clone() * right.clone());
    });
    group.throughput(Throughput::Elements(len as u64));
    group.bench_function("f32/add-constant-2m", |bencher| {
        bencher.iter(|| zero.clone() + left.clone());
    });
    group.throughput(Throughput::Elements(len as u64));
    group.bench_function("f32/negate-2m", |bencher| {
        bencher.iter(|| -left.clone());
    });
    // The transcendental seam: scalar without features, vForce under
    // `accelerate`, the GPU under `metal` at this size.
    group.throughput(Throughput::Elements(len as u64));
    group.bench_function("f32/tanh-2m", |bencher| {
        bencher.iter(|| left.tanh());
    });

    let left = Tensor::new([len / 4], vec![0.5_f64; len / 4]);
    let right = Tensor::new([len / 4], vec![0.25_f64; len / 4]);
    group.throughput(Throughput::Elements((len / 4) as u64));
    group.bench_function("f64/multiply-dense-512k", |bencher| {
        bencher.iter(|| left.clone() * right.clone());
    });

    group.finish();
}

criterion_group!(benches, elementwise);
criterion_main!(benches);
