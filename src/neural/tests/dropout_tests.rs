use crate::{Differentiable, Network, Shape, Tensor, Tensorial};

use super::super::{Module, init};
use super::Dropout;

#[test]
fn masks_are_deterministic_in_seed_and_step() {
    let mut first = init::dropout::<f64>(11, 0.75);
    let mut second = init::dropout::<f64>(11, 0.75);
    let shape = Shape::new([4, 8]);
    for _ in 0..3 {
        assert_eq!(first(&shape).to_vec(), second(&shape).to_vec());
    }
    let mut third = init::dropout::<f64>(12, 0.75);
    let mut fresh = init::dropout::<f64>(11, 0.75);
    assert_ne!(fresh(&shape).to_vec(), third(&shape).to_vec());
}

#[test]
fn masks_hold_the_inverted_scale() {
    let mut masks = init::dropout::<f64>(7, 0.8);
    let mask = masks(&Shape::new([64, 64])).to_vec();
    assert!(mask.iter().all(|&value| value == 0.0 || value == 1.0 / 0.8));
    // Deterministic for the fixed seed, and near the keep probability
    // over 4096 draws.
    let kept = mask.iter().filter(|&&value| value != 0.0).count();
    assert!((2900..3600).contains(&kept), "kept {kept} of 4096");
}

#[test]
#[should_panic(expected = "the keep probability must lie within (0, 1], got 0")]
fn a_zero_keep_probability_is_rejected() {
    init::dropout::<f64>(1, 0.0);
}

#[test]
fn an_unfed_run_is_the_identity() {
    // The mask input's default payload is all ones, so the masked
    // expression answers bitwise as the plain one — inference needs
    // no second expression and no mode flag.
    let network = Network::new();
    let x = network.parameter(Tensor::new(
        [2, 3],
        vec![0.5_f64, -1.0, 2.0, 3.0, -0.25, 1.5],
    ));
    let dropout = Dropout::new(&network, [2, 3]);
    let masked = dropout.express(&network, x + x);
    let plain = x + x;
    let run = network.forward();
    assert_eq!(run.of(masked).to_vec(), run.of(plain).to_vec());
}

#[test]
fn gradients_route_through_the_mask() {
    let network = Network::new();
    let x = network.parameter(Tensor::new([1, 4], [1.0_f64, 2.0, 3.0, 4.0]));
    let dropout = Dropout::new(&network, [1, 4]);
    let loss = dropout.express(&network, x).sum();
    let mask = Tensor::new([1, 4], [2.0_f64, 0.0, 2.0, 0.0]);
    let run = network.forward_with([(dropout.mask(), mask)]);
    assert_eq!(run.backward(loss).of(x).to_vec(), vec![2.0, 0.0, 2.0, 0.0]);
}

#[test]
fn training_replays_bitwise_under_matched_seeds() {
    // A tiny masked training loop, run twice from the same seeds:
    // the trajectories agree bit for bit, dropout masks included.
    fn final_loss() -> f64 {
        let network = Network::new();
        let w = network.parameter(Tensor::new([2, 2], [0.5_f64, -0.25, 0.75, 0.1]));
        let x = network.leaf(Tensor::new([2, 2], [1.0_f64, 2.0, -1.0, 0.5]));
        let dropout = Dropout::new(&network, [2, 2]);
        let product = dropout.express(&network, x.matmul(w));
        let loss = (product * product).sum();
        let loss_symbol = loss.symbol();
        let w_symbol = w.symbol();

        let mut masks = init::dropout::<f64>(21, 0.5);
        let rate = Tensor::new([], [0.05_f64]);
        let mut network = network;
        let mut last = 0.0;
        for _ in 0..8 {
            let run = network.forward_with([(dropout.mask(), masks(&Shape::new([2, 2])))]);
            last = run.of(loss_symbol).to_vec()[0];
            let gradients = run.backward(loss_symbol);
            network = network.update(&gradients, |parameter, gradient| {
                parameter.clone() - gradient.clone() * rate.broadcast_like(gradient)
            });
        }
        let _ = w_symbol;
        last
    }
    assert_eq!(final_loss().to_bits(), final_loss().to_bits());
}

#[test]
fn dropout_composes_as_a_module() {
    // The trait impl serves the record-time composition surface: an
    // object-safe module slot expresses identically to the inherent
    // method.
    let network = Network::new();
    let x = network.parameter(Tensor::new([2, 2], [1.0_f64, -2.0, 3.0, -4.0]));
    let dropout = Dropout::new(&network, [2, 2]);
    let module: &dyn Module<Tensor<f64>> = &dropout;
    let through_trait = module.express(&network, x);
    let mask = Tensor::new([2, 2], [2.0_f64, 0.0, 0.0, 2.0]);
    let run = network.forward_with([(dropout.mask(), mask)]);
    assert_eq!(run.of(through_trait).to_vec(), vec![2.0, 0.0, 0.0, -8.0]);
}
