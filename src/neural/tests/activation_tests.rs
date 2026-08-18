use crate::{Tape, Tensor};

use super::Activation;

/// Evaluates `activation` over `inputs` on a fresh tape and returns
/// the outputs.
fn evaluated(activation: Activation, inputs: &[f64]) -> Vec<f64> {
    let tape = Tape::new();
    let value = tape.leaf(Tensor::new([inputs.len()], inputs.to_vec()));
    let expressed = activation.express(value).symbol();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    run.of(expressed).to_vec()
}

#[test]
fn the_dedicated_variants_record_their_operations() {
    let squashed = evaluated(Activation::Tanh, &[0.0, 2.0]);
    assert_eq!(squashed[0], 0.0);
    assert!((squashed[1] - 2.0_f64.tanh()).abs() < 1e-12);

    let rectified = evaluated(Activation::Relu, &[3.0, -2.0, 0.0]);
    assert_eq!(rectified, &[3.0, 0.0, 0.0]);
}
