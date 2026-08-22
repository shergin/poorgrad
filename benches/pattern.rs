//! Benchmarks of the fused pattern kernels against the interpreter.
//!
//! Each case records one canonical composition and compares the
//! interpreter (the recorded nodes, unfused) with a forward plan
//! (the elected home kernel). The batch-norm case fuses only where a
//! compiled backend can take the task, so its default-build row
//! measures plan overhead alone and its `--features accelerate` row
//! measures the vDSP kernel.

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use topos::{BatchNorm, Request, Tape, Tensor, conv2d, max_pool};

fn pattern(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("pattern");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    // Max pool over a conv-activation-sized value.
    let tape = Tape::new();
    let input = tape.leaf(Tensor::new(
        [64, 32, 28, 28],
        (0..64 * 32 * 28 * 28)
            .map(|index| ((index * 5 % 13) as f32 - 6.0) / 4.0)
            .collect::<Vec<_>>(),
    ));
    let pooled = max_pool(input, 2, 2).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();
    let plan = network.compile(Request::roots([pooled]));
    assert!(plan.describe().contains("fused 1 groups"));
    group.bench_function("pool/interpreted", |bencher| {
        bencher.iter(|| network.forward(&parameters, []));
    });
    group.bench_function("pool/fused-plan", |bencher| {
        bencher.iter(|| plan.forward(&parameters, std::iter::empty()));
    });

    // Convolution through the window-GEMM group.
    let tape = Tape::new();
    let input = tape.leaf(Tensor::new(
        [64, 16, 14, 14],
        (0..64 * 16 * 14 * 14)
            .map(|index| ((index * 7 % 23) as f32 - 11.0) / 8.0)
            .collect::<Vec<_>>(),
    ));
    let weights = tape.leaf(Tensor::new(
        [32, 16, 3, 3],
        (0..32 * 16 * 9)
            .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>(),
    ));
    let bias = tape.leaf(Tensor::filled([32], 0.01_f32));
    let features = conv2d(input, weights, bias, 1, 1).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();
    let plan = network.compile(Request::roots([features]));
    assert!(plan.describe().contains("fused 1 groups"));
    group.bench_function("conv/interpreted", |bencher| {
        bencher.iter(|| network.forward(&parameters, []));
    });
    group.bench_function("conv/fused-plan", |bencher| {
        bencher.iter(|| plan.forward(&parameters, std::iter::empty()));
    });

    // Batch normalization over a wide dense activation.
    let tape = Tape::new();
    let input = tape.leaf(Tensor::new(
        [256, 512],
        (0..256 * 512)
            .map(|index| ((index * 11 % 29) as f32 - 14.0) / 8.0)
            .collect::<Vec<_>>(),
    ));
    let layer = BatchNorm::new(
        &tape,
        Tensor::filled([512], 1.0_f32),
        Tensor::filled([512], 0.0),
        Tensor::filled([1], 1.0e-5),
    );
    let output = layer.express(&tape, input).output.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();
    let plan = network.compile(Request::roots([output]));
    // Build-adaptive: the group fuses exactly where a compiled
    // backend can take the task, so the default-build row measures
    // plan overhead and the accelerate row measures the kernel.
    let fed = topos::Precision::ALL.iter().any(|&precision| {
        topos::Formula::BatchNormTraining
            .chain(precision)
            .iter()
            .any(|backend| backend.compiled())
    });
    assert_eq!(plan.describe().contains("fused 1 groups"), fed);
    group.bench_function("batch-norm/interpreted", |bencher| {
        bencher.iter(|| network.forward(&parameters, []));
    });
    group.bench_function("batch-norm/fused-plan", |bencher| {
        bencher.iter(|| plan.forward(&parameters, std::iter::empty()));
    });

    group.finish();
}

criterion_group!(benches, pattern);
criterion_main!(benches);
