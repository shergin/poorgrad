mod activation;
mod adam;
mod batch_norm;
mod convolution;
// The one public module in the crate: initializer names (`uniform`,
// `normal`) are meaningless without the `init::` qualifier.
pub mod init;
mod layer;
mod layer_norm;
mod loss;
mod mlp;
mod neuron;
mod optimizer;
mod pooling;
mod rms_norm;

pub use activation::Activation;
pub use adam::{Adam, AdamW};
pub use batch_norm::{BatchNorm, Normalization};
pub use convolution::{Conv2d, conv2d};
pub use layer::Layer;
pub use layer_norm::LayerNorm;
pub use loss::cross_entropy;
pub use mlp::Mlp;
pub use neuron::Neuron;
pub use optimizer::{Optimizer, Sgd};
pub use pooling::{average_pool, max_pool};
pub use rms_norm::RmsNorm;
