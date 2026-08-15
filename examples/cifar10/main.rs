//! Trains a small VGG-style convolutional network on CIFAR-10 — the
//! plan tier's pressure consumer: three conv/relu/pool stages over
//! real 32x32 color images, a dense head, and `cross_entropy`,
//! compiled once into a training plan and a forward-only probe plan.
//!
//! The training plan retains what `backward` reads and survives every
//! generation; the probe plan's liveness pass frees each intermediate
//! after its last consumer, which is what keeps the 500-image probe's
//! footprint flat. Both print through `Plan::describe`'s summary at
//! startup, and every convolution runs as one im2col matrix product.
//!
//! The binary archive (~170 MB) downloads on first run into
//! `examples/cifar10/data/` and is cached.
//!
//! Run with: `cargo run --release --example cifar10`

mod chart;
mod dataset;

use std::time::Instant;

use poorgrad::{
    Compile, Conv2d, Linear, Memory, Module, Network, Plan, Shape, Symbol, Tensor, Tensorial,
    Value, cross_entropy, init, max_pool,
};

use chart::loss_chart;
use dataset::{Split, load, shuffle};

/// The image side length; CIFAR-10 images are `32 x 32`.
const IMAGE_SIDE: usize = 32;

/// How many values one image holds: three channel planes.
const PIXELS: usize = 3 * IMAGE_SIDE * IMAGE_SIDE;

/// How many classes the head scores.
const CLASSES: usize = 10;

/// How many samples each training step feeds.
const BATCH_LEN: usize = 64;

/// How many test images the accuracy probe feeds per run.
const PROBE_LEN: usize = 500;

/// How many filters the three convolution stages learn.
const FILTERS: [usize; 3] = [16, 32, 64];

/// The flattened feature length after three 2x2 pools: `64 * 4 * 4`.
const FLAT_LEN: usize = FILTERS[2] * (IMAGE_SIDE / 8) * (IMAGE_SIDE / 8);

/// How many training steps to run: about two and a half epochs.
const STEPS: usize = 2000;

/// The model's layers, holding parameter symbols across generations.
struct Model {
    conv_1: Conv2d<Tensor<f32>>,
    conv_2: Conv2d<Tensor<f32>>,
    conv_3: Conv2d<Tensor<f32>>,
    head: Linear<Tensor<f32>>,
}

impl Model {
    /// Allocates the parameters on `network`: three 3x3 same-padded
    /// convolution stages with Kaiming-scaled kernels and zero biases,
    /// and an affine classification head.
    fn new(network: &Network<Tensor<f32>>) -> Self {
        // Kaiming deviations by kernel fan-in: `sqrt(2 / (c * kh * kw))`.
        let conv_1_weights =
            init::normal(21, (2.0 / 27.0_f64).sqrt())(&Shape::new([FILTERS[0], 3, 3, 3]));
        let conv_2_weights =
            init::normal(22, (2.0 / 144.0_f64).sqrt())(&Shape::new([FILTERS[1], FILTERS[0], 3, 3]));
        let conv_3_weights =
            init::normal(23, (2.0 / 288.0_f64).sqrt())(&Shape::new([FILTERS[2], FILTERS[1], 3, 3]));
        let mut head_weights = init::kaiming(24);
        Self {
            conv_1: Conv2d::new(
                network,
                conv_1_weights,
                Tensor::filled([FILTERS[0]], 0.0),
                1,
                1,
            ),
            conv_2: Conv2d::new(
                network,
                conv_2_weights,
                Tensor::filled([FILTERS[1]], 0.0),
                1,
                1,
            ),
            conv_3: Conv2d::new(
                network,
                conv_3_weights,
                Tensor::filled([FILTERS[2]], 0.0),
                1,
                1,
            ),
            head: Linear::new(
                network,
                head_weights(&Shape::new([FLAT_LEN, CLASSES])),
                head_weights(&Shape::new([CLASSES])),
            ),
        }
    }

    /// Records the model's expression over `images` (`[rows, 3, 32,
    /// 32]`) and returns the `[rows, 10]` logits: conv, rectify, pool,
    /// three times, then flatten and score.
    fn express<'network>(
        &self,
        network: &'network Network<Tensor<f32>>,
        images: Value<'network, Tensor<f32>>,
        rows: usize,
    ) -> Value<'network, Tensor<f32>> {
        let stage_1 = max_pool(self.conv_1.express(network, images).relu(), 2, 2);
        let stage_2 = max_pool(self.conv_2.express(network, stage_1).relu(), 2, 2);
        let stage_3 = max_pool(self.conv_3.express(network, stage_2).relu(), 2, 2);
        self.head
            .express(network, stage_3.reshape([rows, FLAT_LEN]))
    }
}

/// Builds the image and one-hot label payloads for the sample `indices`.
fn batch_payloads(split: &Split, indices: &[usize]) -> (Tensor<f32>, Tensor<f32>) {
    let mut pixels = Vec::with_capacity(indices.len() * PIXELS);
    for &index in indices {
        pixels.extend_from_slice(&split.pixels[index * PIXELS..(index + 1) * PIXELS]);
    }
    let labels: Vec<usize> = indices.iter().map(|&index| split.labels[index]).collect();
    (
        Tensor::new([indices.len(), 3, IMAGE_SIDE, IMAGE_SIDE], pixels),
        Tensor::selection(labels, CLASSES, 1.0),
    )
}

/// Counts correct predictions over `PROBE_LEN` test images from `start`,
/// one probe-plan run.
fn probe_correct(
    network: &Network<Tensor<f32>>,
    probe_plan: &Plan<Tensor<f32>>,
    images_symbol: Symbol,
    logits_symbol: Symbol,
    test: &Split,
    start: usize,
) -> usize {
    let indices: Vec<usize> = (start..start + PROBE_LEN).collect();
    let (images, _) = batch_payloads(test, &indices);
    let run = probe_plan.forward(network, [(images_symbol, images)]);
    let logits = run.of(network.resolve(logits_symbol)).to_vec();
    let mut correct = 0;
    for (row, &index) in logits.chunks(CLASSES).zip(&indices) {
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are finite"))
            .map(|(class, _)| class)
            .expect("a logit row is never empty");
        if predicted == test.labels[index] {
            correct += 1;
        }
    }
    correct
}

fn main() {
    let (train, test) = load();
    println!(
        "loaded {} training and {} test images",
        train.len(),
        test.len()
    );

    let network = Network::new();
    let model = Model::new(&network);

    // The training expression: images and one-hot targets are fed per
    // run, the defaults only fix the shapes.
    let images = network.input(Tensor::filled(
        [BATCH_LEN, 3, IMAGE_SIDE, IMAGE_SIDE],
        0.0_f32,
    ));
    let targets = network.input(Tensor::selection(vec![0; BATCH_LEN], CLASSES, 1.0));
    let loss = cross_entropy(model.express(&network, images, BATCH_LEN), targets);

    // The accuracy twin over a probe of test images.
    let probe_images = network.input(Tensor::filled(
        [PROBE_LEN, 3, IMAGE_SIDE, IMAGE_SIDE],
        0.0_f32,
    ));
    let probe_logits = model.express(&network, probe_images, PROBE_LEN);

    let images_symbol = images.symbol();
    let targets_symbol = targets.symbol();
    let loss_symbol = loss.symbol();
    let probe_images_symbol = probe_images.symbol();
    let probe_logits_symbol = probe_logits.symbol();
    let recorded_nodes = network.len();
    println!("recorded {recorded_nodes} nodes for both expressions");

    // Compile once, run every generation.
    let training_plan =
        network.compile(Compile::roots([loss_symbol]).engine_backward(Memory::Retain));
    let probe_plan = network.compile(Compile::roots([probe_logits_symbol]));
    for line in probe_plan.describe().lines().filter(|line| {
        line.starts_with("plan:") || line.starts_with("live volume:") || line.starts_with("fused")
    }) {
        println!("probe {line}");
    }

    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut shuffle_state: u64 = 3;
    shuffle(&mut order, &mut shuffle_state);

    let fast = Tensor::new([], [0.05_f32]);
    let slow = Tensor::new([], [0.005_f32]);
    let mut network = network;
    let mut losses = Vec::new();
    let training = Instant::now();
    for step in 0..STEPS {
        let start = (step * BATCH_LEN) % (train.len() - BATCH_LEN);
        let batch = &order[start..start + BATCH_LEN];
        let (batch_images, batch_targets) = batch_payloads(&train, batch);

        let loss_value = network.resolve(loss_symbol);
        let run = training_plan.forward(
            &network,
            [
                (images_symbol, batch_images),
                (targets_symbol, batch_targets),
            ],
        );
        let batch_loss = run.of(loss_value).to_vec()[0];
        losses.push(batch_loss);
        if step == 0 {
            println!(
                "step 0: minibatch loss = {batch_loss:.4} (a uniform model costs ln 10 ~ 2.30)"
            );
        }
        let gradients = run.backward(loss_value);
        let learning_rate = if step < STEPS * 3 / 4 { &fast } else { &slow };
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });

        if (step + 1) % 250 == 0 {
            let correct = probe_correct(
                &network,
                &probe_plan,
                probe_images_symbol,
                probe_logits_symbol,
                &test,
                0,
            );
            println!(
                "step {:4}: minibatch loss = {batch_loss:.4}, probe accuracy = {:.1}%",
                step + 1,
                correct as f64 * 100.0 / PROBE_LEN as f64
            );
        }
    }
    let elapsed = training.elapsed().as_secs_f64();
    println!(
        "trained {STEPS} steps in {elapsed:.3}s ({:.1} ms/step)",
        elapsed * 1000.0 / STEPS as f64
    );

    // The full test set, probed chunk by chunk through the same plan.
    let correct: usize = (0..test.len() / PROBE_LEN)
        .map(|chunk| {
            probe_correct(
                &network,
                &probe_plan,
                probe_images_symbol,
                probe_logits_symbol,
                &test,
                chunk * PROBE_LEN,
            )
        })
        .sum();
    println!(
        "test accuracy: {:.2}% over {} images (chance is 10%)",
        correct as f64 * 100.0 / test.len() as f64,
        test.len()
    );

    assert_eq!(network.len(), recorded_nodes);
    println!("the tape held {recorded_nodes} nodes through every step");
    println!("{}", loss_chart("cifar-10 convnet training", &losses));
}
