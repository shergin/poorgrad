use crate::{Network, Shape, Tensor};

use super::{average_pool, max_pool};

#[test]
fn max_pool_takes_window_maxima() {
    let network = Network::new();
    let input = network.leaf(Tensor::new(
        [1, 1, 4, 4],
        (1..=16).map(|v| v as f64).collect::<Vec<_>>(),
    ));

    let pooled = max_pool(input, 2, 2);
    assert_eq!(pooled.shape(), Shape::new([1, 1, 2, 2]));

    let evaluation = network.forward();
    assert_eq!(evaluation.of(pooled).to_vec(), &[6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn max_pool_routes_the_gradient_to_the_maximum() {
    let network = Network::new();
    let input = network.leaf(Tensor::new(
        [1, 1, 4, 4],
        (1..=16).map(|v| v as f64).collect::<Vec<_>>(),
    ));

    let pooled = max_pool(input, 2, 2);
    let loss = pooled.sum();

    let evaluation = network.forward();
    let gradients = evaluation.backward(loss);
    let mut expected = vec![0.0; 16];
    for position in [5, 7, 13, 15] {
        expected[position] = 1.0;
    }
    assert_eq!(gradients.of(input).to_vec(), expected);
}

#[test]
fn max_pool_ties_route_to_the_earliest_lane() {
    let network = Network::new();
    let input = network.leaf(Tensor::filled([1, 1, 2, 2], 5.0_f64));

    let pooled = max_pool(input, 2, 2);
    let loss = pooled.sum();

    let evaluation = network.forward();
    assert_eq!(evaluation.of(pooled).to_vec(), &[5.0]);

    // All four window elements tie; the left-biased `maximum` fold
    // hands the whole gradient to the first lane in window order.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(input).to_vec(), &[1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn average_pool_takes_window_means() {
    let network = Network::new();
    let input = network.leaf(Tensor::new(
        [1, 1, 4, 4],
        (1..=16).map(|v| v as f64).collect::<Vec<_>>(),
    ));

    let pooled = average_pool(input, 2, 2);
    let loss = pooled.sum();

    let evaluation = network.forward();
    assert_eq!(evaluation.of(pooled).to_vec(), &[3.5, 5.5, 11.5, 13.5]);

    // Every window element contributes `1 / (size * size)`.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(input).to_vec(), &[0.25; 16]);
}

#[test]
#[should_panic(expected = "must be rank 4")]
fn pooling_rejects_non_image_input() {
    let network = Network::new();
    let input = network.leaf(Tensor::filled([1, 4, 4], 0.0_f64));
    max_pool(input, 2, 2);
}
