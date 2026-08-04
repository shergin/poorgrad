//! Trains a LeNet-style convolutional network on MNIST — the first
//! consumer of `Conv2d`, `max_pool`, and the im2col-over-GEMM route:
//! two conv/relu/pool stages, a dense head, and `cross_entropy`, all
//! recorded once and fed minibatches per run.
//!
//! The tape carries two expressions of the same parameters: a
//! batch-shaped one for training and a probe-shaped twin for test
//! accuracy, because input shapes are baked in at recording time.
//! Every convolution runs as one rank-2 matrix product over its
//! im2col patches, so the FLOPs land on the GEMM fast path with no
//! backend feature required.
//!
//! The four IDX files download on first run into
//! `examples/mnist/data/` (about 55 MB unpacked) and are cached.
//!
//! Run with: `cargo run --release --example mnist`

mod chart;
mod dataset;

use std::time::Instant;

use poorgrad::{
    Activation, Conv2d, Layer, Network, Plan, Shape, Symbol, Tensor, Tensorial, Value,
    cross_entropy, init, max_pool,
};

use chart::loss_chart;
use dataset::{Split, load, shuffle};

/// The image side length; MNIST digits are `28 x 28`.
const IMAGE_SIDE: usize = 28;

/// How many pixels one image holds.
const PIXELS: usize = IMAGE_SIDE * IMAGE_SIDE;

/// How many digit classes the head scores.
const CLASSES: usize = 10;

/// How many samples each training step feeds.
const BATCH_LEN: usize = 64;

/// How many test images the accuracy probe feeds per run.
const PROBE_LEN: usize = 1000;

/// How many filters the first convolution stage learns.
const FILTERS_1: usize = 8;

/// How many filters the second convolution stage learns.
const FILTERS_2: usize = 16;

/// The flattened feature length after two 2x2 pools: `16 * 7 * 7`.
const FLAT_LEN: usize = FILTERS_2 * (IMAGE_SIDE / 4) * (IMAGE_SIDE / 4);

/// How many training steps to run: a bit over two epochs.
const STEPS: usize = 2000;

/// The model's layers, holding parameter symbols across generations.
struct Model {
    conv_1: Conv2d<Tensor<f32>>,
    conv_2: Conv2d<Tensor<f32>>,
    head: Layer<Tensor<f32>>,
}

impl Model {
    /// Allocates the parameters on `network`: two 3x3 same-padded
    /// convolution stages with Kaiming-scaled kernels and zero biases,
    /// and an affine classification head.
    fn new(network: &Network<Tensor<f32>>) -> Self {
        // Kaiming deviations by kernel fan-in: `sqrt(2 / (c * kh * kw))`.
        let conv_1_weights =
            init::normal(11, (2.0 / 9.0_f64).sqrt())(&Shape::new([FILTERS_1, 1, 3, 3]));
        let conv_2_weights =
            init::normal(12, (2.0 / 72.0_f64).sqrt())(&Shape::new([FILTERS_2, FILTERS_1, 3, 3]));
        let mut head_weights = init::kaiming(13);
        Self {
            conv_1: Conv2d::new(
                network,
                conv_1_weights,
                Tensor::filled([FILTERS_1], 0.0),
                1,
                1,
            ),
            conv_2: Conv2d::new(
                network,
                conv_2_weights,
                Tensor::filled([FILTERS_2], 0.0),
                1,
                1,
            ),
            head: Layer::new(
                network,
                head_weights(&Shape::new([FLAT_LEN, CLASSES])),
                head_weights(&Shape::new([CLASSES])),
                Activation::Identity,
            ),
        }
    }

    /// Records the model's expression over `images` (`[rows, 1, 28,
    /// 28]`) and returns the `[rows, 10]` logits: conv, rectify, pool,
    /// twice, then flatten and score.
    fn express<'network>(
        &self,
        network: &'network Network<Tensor<f32>>,
        images: Value<'network, Tensor<f32>>,
        rows: usize,
    ) -> Value<'network, Tensor<f32>> {
        let stage_1 = max_pool(self.conv_1.express(network, images).relu(), 2, 2);
        let stage_2 = max_pool(self.conv_2.express(network, stage_1).relu(), 2, 2);
        self.head
            .express(network, stage_2.reshape([rows, FLAT_LEN]))
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
        Tensor::new([indices.len(), 1, IMAGE_SIDE, IMAGE_SIDE], pixels),
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
    // The forward-only plan skips the training twin and frees every
    // probe intermediate right after its last consumer.
    let evaluation = probe_plan.forward(network, [(images_symbol, images)]);
    let logits = evaluation.of(network.resolve(logits_symbol)).to_vec();
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
        [BATCH_LEN, 1, IMAGE_SIDE, IMAGE_SIDE],
        0.0_f32,
    ));
    let targets = network.input(Tensor::selection(vec![0; BATCH_LEN], CLASSES, 1.0));
    let loss = cross_entropy(model.express(&network, images, BATCH_LEN), targets);

    // The accuracy twin: the same parameters expressed over a probe of
    // test images.
    let probe_images = network.input(Tensor::filled(
        [PROBE_LEN, 1, IMAGE_SIDE, IMAGE_SIDE],
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

    // Compile once, run every generation. The compact training plan
    // drops its large intermediates and rematerializes them during
    // backward: measured here at 9% less peak RSS for 22% more step
    // time. The probe plan frees as it goes.
    let training_plan = network.compile_training_compact(loss_symbol, []);
    let probe_plan = network.compile([probe_logits_symbol], []);
    for line in training_plan.describe().lines().filter(|line| {
        line.starts_with("plan:") || line.starts_with("live volume:") || line.starts_with("fused")
    }) {
        println!("training {line}");
    }
    for line in probe_plan.describe().lines().filter(|line| {
        line.starts_with("plan:") || line.starts_with("live volume:") || line.starts_with("fused")
    }) {
        println!("probe {line}");
    }

    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut shuffle_state: u64 = 5;
    shuffle(&mut order, &mut shuffle_state);

    let fast = Tensor::new([], [0.1_f32]);
    let slow = Tensor::new([], [0.01_f32]);
    let mut network = network;
    let mut losses = Vec::new();
    let training = Instant::now();
    for step in 0..STEPS {
        let start = (step * BATCH_LEN) % (train.len() - BATCH_LEN);
        let batch = &order[start..start + BATCH_LEN];
        let (batch_images, batch_targets) = batch_payloads(&train, batch);

        let loss_value = network.resolve(loss_symbol);
        // The training plan skips the probe expression and survives
        // every generation: compiled once, run per step.
        let evaluation = training_plan.forward(
            &network,
            [
                (images_symbol, batch_images),
                (targets_symbol, batch_targets),
            ],
        );
        let batch_loss = evaluation.of(loss_value).to_vec()[0];
        losses.push(batch_loss);
        if step == 0 {
            println!(
                "step 0: minibatch loss = {batch_loss:.4} (a uniform model costs ln 10 ~ 2.30)"
            );
        }
        let gradients = evaluation.backward(loss_value);
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

    // The full test set, probed chunk by chunk through the same tape.
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
        "test accuracy: {:.2}% over {} images",
        correct as f64 * 100.0 / test.len() as f64,
        test.len()
    );

    assert_eq!(network.len(), recorded_nodes);
    println!("the tape held {recorded_nodes} nodes through every step");
    println!("{}", loss_chart("mnist convnet training", &losses));
}
