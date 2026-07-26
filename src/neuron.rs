/// A single neuron: a set of weights, a bias, and an activation.
///
/// It is the smallest learnable building block, computing a weighted sum of
/// its inputs plus a bias as `Value`s so the whole unit participates in
/// automatic differentiation.
#[derive(Debug, Clone)]
pub struct Neuron {}
