//! Reports the allocation behavior of training, in exact bytes rather
//! than timings: how much a step allocates in total, and how much of it
//! survives the step. The caller-owned parameter state keeps retention
//! at zero — replaced payloads drop with the previous state instead of
//! accumulating in the arena — and this report is the regression fence
//! keeping it there.
//!
//! Not a criterion benchmark: a counting global allocator gives
//! deterministic numbers where RSS sampling is noisy. The `unsafe` here
//! is the `GlobalAlloc` contract of this reporting harness, not part of
//! the library, which remains `#![forbid(unsafe_code)]`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use topos::{Tape, Tensor, Tensorial};

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static FREED: AtomicUsize = AtomicUsize::new(0);

/// It forwards every request to the system allocator while counting the
/// bytes that pass through.
struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        FREED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
        FREED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Bytes currently allocated and not yet freed.
fn retained() -> usize {
    ALLOCATED.load(Ordering::Relaxed) - FREED.load(Ordering::Relaxed)
}

/// Runs `steps` training steps of `step` and reports per-step averages.
///
/// Retention is signed: a training loop that reclaims its state can end
/// below its starting point (for example by replacing a growth-sized
/// store with exact-capacity ones).
fn report(label: &str, steps: usize, mut step: impl FnMut()) {
    let allocated_before = ALLOCATED.load(Ordering::Relaxed);
    let retained_before = retained() as i64;
    for _ in 0..steps {
        step();
    }
    let allocated = ALLOCATED.load(Ordering::Relaxed) - allocated_before;
    let kept = retained() as i64 - retained_before;
    println!("{label} ({steps} steps):");
    println!("  allocated per step: {:>8} bytes", allocated / steps);
    println!("  retained per step:  {:>8} bytes", kept / steps as i64);
}

fn main() {
    const STEPS: usize = 1000;

    // A 200-parameter quadratic bowl: the scalar training pattern.
    let scalar_tape = Tape::new();
    let target = (0..200)
        .map(|index| scalar_tape.parameter(index as f64))
        .reduce(|total, parameter| total + parameter * parameter)
        .expect("at least one parameter")
        .symbol();
    let scalar = scalar_tape.into_network();
    let mut scalar_parameters = scalar.parameters();
    report("scalar training, 200 parameters", STEPS, || {
        let evaluation = scalar.forward(&scalar_parameters, []);
        let gradients = evaluation.backward(target);
        scalar_parameters = scalar_parameters.step(&gradients, |parameter, gradient| {
            parameter - 0.01 * gradient
        });
    });

    // The matrix-form regression: one tensor parameter of 512 elements.
    let tensor_tape = Tape::new();
    let inputs = tensor_tape.leaf(Tensor::filled([64, 32], 0.5_f64));
    let weights = tensor_tape.parameter(Tensor::filled([32, 16], 0.1_f64));
    let targets = tensor_tape.leaf(Tensor::filled([64, 16], 1.0_f64));
    let error = inputs.matmul(weights) - targets;
    let loss = (error * error).sum().symbol();
    let learning_rate = Tensor::new([], [0.01_f64]);
    let tensor = tensor_tape.into_network();
    let mut tensor_parameters = tensor.parameters();
    report("tensor training, [32, 16] parameter", STEPS, || {
        let evaluation = tensor.forward(&tensor_parameters, []);
        let gradients = evaluation.backward(loss);
        tensor_parameters = tensor_parameters.step(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    });
}
