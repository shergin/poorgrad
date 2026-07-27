/// The nonlinearity applied to a neural building block's affine output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// No nonlinearity: the output stays affine, as befits output layers
    /// of regressions.
    Identity,
    /// The hyperbolic tangent, squashing the output into `(-1, 1)`.
    Tanh,
}
