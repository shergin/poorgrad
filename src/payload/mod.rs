mod differentiable;
mod elementary;
mod gemm;
mod layout;
mod shape;
mod storage;
mod tensor;
mod tensorial;

pub use differentiable::Differentiable;
pub use elementary::Elementary;
pub use gemm::GemmTask;
pub use shape::Shape;
pub use tensor::Tensor;
pub use tensorial::Tensorial;
