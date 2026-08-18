//! 2-D convolution as a composed formula: windows, im2col, and one
//! rank-2 matrix product.
//!
//! There is no convolution primitive and no dedicated backward rule:
//! the formula records `pad`, two single-axis `unfold`s, `permute`,
//! `reshape`, and the existing `matmul`, so its gradient falls out of
//! the chain rule through those operations' adjoints. The one
//! deliberate cost is the im2col materialization: reshaping the
//! overlapping window view to a matrix copies it (the reshape fallback
//! fires), which converts the whole convolution into a single
//! contiguous GEMM on the accelerated seam.

use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Symbol, Tape, Tensorial, Value};

use super::{Module, Visitor};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Conv2d<f64>: Send, Sync);

/// Records the 2-D convolution of `input` by `weights` plus `bias` on
/// their network and returns the `[batch, filters, out_height,
/// out_width]` output value.
///
/// For output position `(i, f, y, x)` with stride `S` and symmetric
/// zero padding `P`:
///
/// ```text
/// output[i, f, y, x] = bias[f]
///     + sum_{c, dy, dx} padded[i, c, y * S + dy, x * S + dx]
///                       * weights[f, c, dy, dx]
/// ```
///
/// # Parameters
/// - `input`: The `[batch, channels, height, width]` value.
/// - `weights`: The `[filters, channels, kernel_height, kernel_width]`
///   kernel stack, torch-shaped; the formula records the `permute` +
///   `reshape` to the GEMM operand, whose per-run cost is one
///   weight-sized copy.
/// - `bias`: The `[filters]` value, broadcast across every output
///   position.
/// - `stride`: The window step along both spatial axes.
/// - `padding`: The symmetric zero padding of both spatial axes.
///
/// # Panics
/// Panics if the values belong to different networks, the ranks are
/// not 4/4/1, the channel or filter counts disagree, `stride` is zero,
/// or the padded extents cannot hold one kernel window.
pub fn conv2d<'tape, Data: Tensorial>(
    input: Value<'tape, Data>,
    weights: Value<'tape, Data>,
    bias: Value<'tape, Data>,
    stride: usize,
    padding: usize,
) -> Value<'tape, Data> {
    let input_shape = input.shape();
    let weights_shape = weights.shape();
    let bias_shape = bias.shape();
    assert_eq!(
        input_shape.rank(),
        4,
        "conv2d input must be rank 4 [batch, channels, height, width], got {input_shape}"
    );
    assert_eq!(
        weights_shape.rank(),
        4,
        "conv2d weights must be rank 4 [filters, channels, kernel_height, kernel_width], \
         got {weights_shape}"
    );
    assert_eq!(
        input_shape.axes()[1],
        weights_shape.axes()[1],
        "conv2d input {input_shape} and weights {weights_shape} disagree on channels"
    );
    assert_eq!(
        bias_shape.rank(),
        1,
        "conv2d bias must be rank 1 [filters], got {bias_shape}"
    );
    assert_eq!(
        bias_shape.axes()[0],
        weights_shape.axes()[0],
        "conv2d bias {bias_shape} and weights {weights_shape} disagree on filters"
    );
    assert!(stride > 0, "conv2d stride must be positive");

    let batch = input_shape.axes()[0];
    let channels = input_shape.axes()[1];
    let height = input_shape.axes()[2];
    let width = input_shape.axes()[3];
    let filters = weights_shape.axes()[0];
    let kernel_height = weights_shape.axes()[2];
    let kernel_width = weights_shape.axes()[3];

    // Padding enters as explicit data injection, `narrow`'s adjoint.
    let mut padded = input;
    if padding > 0 {
        padded = padded.pad(2, padding, height + 2 * padding);
        padded = padded.pad(3, padding, width + 2 * padding);
    }
    // Two single-axis unfolds make the 2-D windows:
    // `[batch, channels, out_h, kernel_h, out_w, kernel_w]`.
    let windows = padded
        .unfold(2, kernel_height, stride, 1)
        .unfold(4, kernel_width, stride, 1);
    let windows_shape = windows.shape();
    let out_height = windows_shape.axes()[2];
    let out_width = windows_shape.axes()[4];
    // The im2col matrix: window-major rows over `[channels, kernel]`
    // columns. The permuted view overlaps, so this reshape is the one
    // deliberate copy of the formula.
    let patches = windows.permute([0, 2, 4, 1, 3, 5]).reshape([
        batch * out_height * out_width,
        channels * kernel_height * kernel_width,
    ]);
    // The torch-shaped weights become the `[columns, filters]` GEMM
    // operand; the reshape of the permuted view materializes a dense,
    // weight-sized copy per run, keeping the GEMM on the fast path.
    let kernel = weights
        .permute([1, 2, 3, 0])
        .reshape([channels * kernel_height * kernel_width, filters]);
    let product = patches.matmul(kernel);
    let shifted = product + bias.broadcast_along(0, product);
    shifted
        .reshape([batch, out_height, out_width, filters])
        .permute([0, 3, 1, 2])
}

/// A 2-D convolution layer over `[batch, channels, height, width]`
/// values: the [`conv2d`] formula with its kernel stack and bias held
/// as parameters.
///
/// Parameters are stored torch-shaped (`[filters, channels,
/// kernel_height, kernel_width]`) so the parameter store keeps the
/// conceptual shape; `express` records the formula, including the
/// weight-side `permute` + `reshape` to the GEMM operand. Parameters
/// are stored as [`Symbol`]s and resolved when the expression is
/// recorded on the family's [`Tape`], like
/// [`Layer`](super::Layer).
#[derive(Debug, Clone)]
pub struct Conv2d<Data> {
    weights: Symbol,
    bias: Symbol,
    stride: usize,
    padding: usize,
    _marker: PhantomData<Data>,
}

impl<Data: Differentiable> Conv2d<Data> {
    /// Allocates the layer's parameters on `tape` from their initial
    /// payloads and returns the layer.
    ///
    /// `weights` is a rank-4 `[filters, channels, kernel_height,
    /// kernel_width]` payload and `bias` a rank-1 `[filters]` payload
    /// agreeing on `filters`. Callers own initialization; the layer
    /// records whatever it is given.
    ///
    /// # Panics
    /// Panics if `weights` is not rank 4, `bias` is not rank 1, the two
    /// disagree on filters, or `stride` is zero.
    pub fn new(
        tape: &Tape<Data>,
        weights: Data,
        bias: Data,
        stride: usize,
        padding: usize,
    ) -> Self {
        let weights_shape = weights.shape();
        let bias_shape = bias.shape();
        assert_eq!(
            weights_shape.rank(),
            4,
            "conv2d weights must be rank 4 [filters, channels, kernel_height, kernel_width], \
             got {weights_shape}"
        );
        assert_eq!(
            bias_shape.rank(),
            1,
            "conv2d bias must be rank 1 [filters], got {bias_shape}"
        );
        assert_eq!(
            bias_shape.axes()[0],
            weights_shape.axes()[0],
            "conv2d bias {bias_shape} and weights {weights_shape} disagree on filters"
        );
        assert!(stride > 0, "conv2d stride must be positive");
        Self {
            weights: tape.parameter(weights).symbol(),
            bias: tape.parameter(bias).symbol(),
            stride,
            padding,
            _marker: PhantomData,
        }
    }

    /// Returns the symbols of the layer's parameters: the weights, then
    /// the bias.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        [self.weights, self.bias].into_iter()
    }
}

impl<Data: Tensorial> Conv2d<Data> {
    /// Records the layer's expression over the `[batch, channels,
    /// height, width]` value `input` on `tape` and returns the
    /// `[batch, filters, out_height, out_width]` output value.
    ///
    /// # Panics
    /// Panics as documented on [`conv2d`], or if the layer's parameters
    /// or `input` are not allocated on `tape`.
    pub fn express<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        let weights = tape.resolve(self.weights);
        let bias = tape.resolve(self.bias);
        conv2d(input, weights, bias, self.stride, self.padding)
    }
}

#[cfg(test)]
#[path = "tests/convolution_tests.rs"]
mod tests;

impl<Data: Differentiable> Conv2d<Data> {
    /// Returns the symbol of the `[filters, channels, kernel_height,
    /// kernel_width]` weight bank.
    pub fn weights(&self) -> Symbol {
        self.weights
    }

    /// Returns the symbol of the `[filters]` bias vector.
    pub fn bias(&self) -> Symbol {
        self.bias
    }
}

impl<Data: Tensorial> Module<Data> for Conv2d<Data> {
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        Conv2d::express(self, tape, input)
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.parameter("weights", self.weights);
        visitor.parameter("bias", self.bias);
    }
}
