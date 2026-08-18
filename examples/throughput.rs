//! Measures training-step throughput on a deliberately wide dense
//! model: the workload hardware backends exist for.
//!
//! A `[1024 features -> 2048 tanh -> 512]` regression over batches of
//! 1024 costs about 19 GFLOP of matrix products per training step —
//! every one of them large enough for the whole acceleration ladder,
//! so the same binary spans two orders of magnitude depending on the
//! features it was built with:
//!
//! ```sh
//! cargo run --release --example throughput
//! cargo run --release --features accelerate --example throughput
//! cargo run --release --features accelerate,metal --example throughput
//! ```
//!
//! Without any backend the dimensions shrink eightfold so the run
//! still terminates; the printed GFLOP/s stays comparable.

use std::time::Instant;

use topos::{Backend, Shape, Tape, Tensor, Tensorial, init};

fn main() {
    let mut accelerated = false;
    for backend in Backend::ALL {
        let status = backend.status();
        accelerated |= status.is_ok();
        match status {
            Ok(()) => println!("{backend:?}: ready"),
            Err(reason) => println!("{backend:?}: {reason}"),
        }
    }

    // On the built-in paths alone, the full workload would take
    // minutes per step; an eightfold shrink keeps the run honest
    // and short while measuring the same code path.
    let scale = if accelerated { 1 } else { 8 };
    let batch_len = 1024 / scale;
    let feature_len = 1024 / scale;
    let hidden_len = 2048 / scale;
    let class_len = 512 / scale;
    let step_flops = (2 * batch_len * feature_len * hidden_len
        + 2 * batch_len * hidden_len * class_len) as f64
        * 3.0;

    let tape: Tape<Tensor<f32>> = Tape::new();
    let mut initializer = init::kaiming(7);
    let hidden_weights = tape.parameter(initializer(&Shape::new([feature_len, hidden_len])));
    let output_weights = tape.parameter(initializer(&Shape::new([hidden_len, class_len])));
    let features = tape.input(Tensor::filled([batch_len, feature_len], 0.0));
    let targets = tape.input(Tensor::filled([batch_len, class_len], 0.0));

    let hidden = features.matmul(hidden_weights).tanh();
    let prediction = hidden.matmul(output_weights);
    let error = prediction - targets;
    let loss = (error * error).sum();

    let (features, loss) = (features.symbol(), loss.symbol());
    let network = tape.into_network();
    let mut parameters = network.parameters();
    let learning_rate = Tensor::new([], [0.0001_f32]);

    let mut batch = init::normal(11, 1.0);
    let batch = batch(&Shape::new([batch_len, feature_len]));

    // Phase one: the raw product, the number the backend ladder is
    // about. One `matmul` at this size is a single backend call.
    let side = 2048 / scale;
    let left = initializer(&Shape::new([side, side]));
    let right = initializer(&Shape::new([side, side]));
    let product_flops = 2.0 * (side as f64).powi(3);
    let rounds = if accelerated { 20 } else { 2 };
    let started = Instant::now();
    for _ in 0..rounds {
        std::hint::black_box(left.matmul(&right));
    }
    let per_product = started.elapsed().as_secs_f64() / rounds as f64;
    println!(
        "raw product {side} x {side}: {:.1} GFLOP/s",
        product_flops / per_product / 1e9
    );

    // Phase two: whole training steps — the honest Amdahl statement.
    // Once the products are accelerated, the elementwise operations
    // (2M scalar `tanh` calls above all) own the step time; that gap
    // is the roadmap's next tier, not the backends'.
    let warmup = 2;
    let steps = 8;
    let mut started = Instant::now();
    for step in 0..warmup + steps {
        if step == warmup {
            started = Instant::now();
        }
        let run = network.forward(&parameters, [(features, batch.clone())]);
        let gradients = run.backward(loss);
        parameters = parameters.step(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }
    let elapsed = started.elapsed().as_secs_f64();
    let per_step = elapsed / steps as f64;
    println!(
        "training step, batch {batch_len}, {feature_len} -> {hidden_len} -> {class_len}: \
         {per_step:.3} s/step, {:.1} GFLOP/s of matrix products",
        step_flops / per_step / 1e9
    );
}
