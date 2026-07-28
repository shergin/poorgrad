/// The nonlinearity applied to a neural building block's affine output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Leaves the affine output unchanged.
    Identity,
    /// Applies the hyperbolic tangent elementwise.
    Tanh,
}
