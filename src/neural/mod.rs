mod activation;
// The one public module in the crate: initializer names (`uniform`,
// `normal`) are meaningless without the `init::` qualifier.
pub mod init;
mod layer;
mod loss;
mod mlp;
mod neuron;

pub use activation::Activation;
pub use layer::Layer;
pub use loss::cross_entropy;
pub use mlp::Mlp;
pub use neuron::Neuron;
