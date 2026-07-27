use std::fmt;

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Shape: Send, Sync);

/// The shape of a payload: its extent along every axis.
///
/// Ranks are structurally tiny — a scalar is rank 0, a matrix rank 2, a
/// batched convolution rank 4 — so the axes are stored inline up to rank
/// 4 and spill to the heap only beyond that. Shapes are lineage-level
/// metadata: inferred once per node when an expression is recorded and
/// never mutated afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(SmallVec<[usize; 4]>);

impl Shape {
    /// Creates a shape from its axes, outermost first.
    pub fn new(axes: impl IntoIterator<Item = usize>) -> Self {
        Self(axes.into_iter().collect())
    }

    /// Creates the rank-0 shape of a scalar.
    pub fn scalar() -> Self {
        Self(SmallVec::new())
    }

    /// Returns the number of axes.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Returns the number of values a payload of this shape holds.
    ///
    /// # Panics
    /// Panics if the product of the axes overflows `usize`.
    pub fn volume(&self) -> usize {
        self.0
            .iter()
            .try_fold(1usize, |volume, &axis| volume.checked_mul(axis))
            .expect("shape volume overflows `usize`")
    }

    /// Returns the axes, outermost first.
    pub fn axes(&self) -> &[usize] {
        &self.0
    }

    /// Returns the shape with `axis` removed.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    pub fn without_axis(&self, axis: usize) -> Shape {
        assert!(axis < self.rank(), "axis {axis} is out of rank for {self}");
        Shape(
            self.0
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != axis)
                .map(|(_, &extent)| extent)
                .collect(),
        )
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "[")?;
        for (index, axis) in self.0.iter().enumerate() {
            if index > 0 {
                write!(formatter, ", ")?;
            }
            write!(formatter, "{axis}")?;
        }
        write!(formatter, "]")
    }
}

#[cfg(test)]
#[path = "tests/shape_tests.rs"]
mod tests;
