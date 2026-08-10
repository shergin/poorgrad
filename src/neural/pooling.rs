//! Spatial pooling as composed formulas over the sliding-window view.
//!
//! Both pools ride the same two single-axis `unfold`s as convolution
//! and need no reduce opcode of their own: the average is
//! `mean_along`, and the maximum is a left-biased fold of the existing
//! binary `maximum` over the window lanes, so ties route their
//! gradient deterministically to the earliest lane.

use crate::{Network, Tensorial, Value};

use super::Module;

/// Records the square windows of a pooling operation and returns them
/// as `[batch, channels, out_height, out_width, size * size]` lanes.
///
/// It records the shared head of both pools: two unfolds, the axis
/// permutation, and the lane-merging reshape (a copy, since the window
/// view overlaps for `stride < size`).
fn window_lanes<'network, Data: Tensorial>(
    input: Value<'network, Data>,
    size: usize,
    stride: usize,
) -> Value<'network, Data> {
    let shape = input.shape();
    assert_eq!(
        shape.rank(),
        4,
        "pooling input must be rank 4 [batch, channels, height, width], got {shape}"
    );
    assert!(size > 0, "pooling windows must hold at least one element");
    assert!(stride > 0, "pooling stride must be positive");
    let windows = input.unfold(2, size, stride, 1).unfold(4, size, stride, 1);
    let windows_shape = windows.shape();
    let axes = windows_shape.axes();
    windows
        .permute([0, 1, 2, 4, 3, 5])
        .reshape([axes[0], axes[1], axes[2], axes[4], size * size])
}

/// Records the `size x size` max pooling of the `[batch, channels,
/// height, width]` value `input` with `stride` and returns the pooled
/// `[batch, channels, out_height, out_width]` value.
///
/// The window maximum is a left-biased fold of [`Value::maximum`] over
/// the window lanes in row-major window order, so a tie routes its
/// gradient to the earliest tied position — deterministic, like every
/// tie rule in the crate.
///
/// # Panics
/// Panics if `input` is not rank 4, `size` or `stride` is zero, or a
/// window does not fit the spatial extents.
pub fn max_pool<'network, Data: Tensorial>(
    input: Value<'network, Data>,
    size: usize,
    stride: usize,
) -> Value<'network, Data> {
    let lanes = window_lanes(input, size, stride);
    let mut largest = lanes.narrow(4, 0, 1);
    for lane in 1..size * size {
        largest = largest.maximum(lanes.narrow(4, lane, 1));
    }
    largest.squeeze(4)
}

/// Records the `size x size` average pooling of the `[batch, channels,
/// height, width]` value `input` with `stride` and returns the pooled
/// `[batch, channels, out_height, out_width]` value.
///
/// # Panics
/// Panics if `input` is not rank 4, `size` or `stride` is zero, or a
/// window does not fit the spatial extents.
pub fn average_pool<'network, Data: Tensorial>(
    input: Value<'network, Data>,
    size: usize,
    stride: usize,
) -> Value<'network, Data> {
    window_lanes(input, size, stride).mean_along(4)
}

#[cfg(test)]
#[path = "tests/pooling_tests.rs"]
mod tests;

/// The module form of [`max_pool`]: a stateless stage carrying its
/// window geometry.
pub struct MaxPool {
    size: usize,
    stride: usize,
}

impl MaxPool {
    /// Creates the stage with the given window `size` and `stride`.
    pub fn new(size: usize, stride: usize) -> Self {
        Self { size, stride }
    }
}

impl<Data: Tensorial> Module<Data> for MaxPool {
    fn express<'network>(
        &self,
        _network: &'network Network<Data>,
        input: Value<'network, Data>,
    ) -> Value<'network, Data> {
        max_pool(input, self.size, self.stride)
    }
}

/// The module form of [`average_pool`]: a stateless stage carrying
/// its window geometry.
pub struct AveragePool {
    size: usize,
    stride: usize,
}

impl AveragePool {
    /// Creates the stage with the given window `size` and `stride`.
    pub fn new(size: usize, stride: usize) -> Self {
        Self { size, stride }
    }
}

impl<Data: Tensorial> Module<Data> for AveragePool {
    fn express<'network>(
        &self,
        _network: &'network Network<Data>,
        input: Value<'network, Data>,
    ) -> Value<'network, Data> {
        average_pool(input, self.size, self.stride)
    }
}
