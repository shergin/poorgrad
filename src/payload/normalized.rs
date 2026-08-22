//! The batch-normalization task and its product: the composed
//! formula's whole-group face at the seam.

/// One training-mode batch normalization over a contiguous
/// `[batch, features]` buffer: the fused form of the recorded
/// formula — center by the batch mean, scale by the
/// epsilon-stabilized deviation, apply the learned affine — offered
/// to the backend chain as a single task.
///
/// Tasks are built by [`Tensorial::batch_normalized`]
/// (crate::Tensorial::batch_normalized) when every operand is a
/// contiguous dense buffer, and read by backend code and by
/// [`Elementary::batch_norm`](crate::Elementary::batch_norm)
/// implementations. Answering asserts the whole
/// [`Normalized`] product of exactly the described task, within the
/// envelope.
#[derive(Debug)]
pub struct BatchNormTask<'buffers, Element> {
    input: &'buffers [Element],
    scale: &'buffers [Element],
    shift: &'buffers [Element],
    epsilon: Element,
    batch: usize,
    features: usize,
}

impl<'buffers, Element> BatchNormTask<'buffers, Element> {
    /// Creates a validated task over the row-major `[batch, features]`
    /// input and the `[features]` affine operands.
    ///
    /// # Panics
    /// Panics if any extent is zero or a slice does not span its
    /// shape.
    pub(crate) fn new(
        input: &'buffers [Element],
        scale: &'buffers [Element],
        shift: &'buffers [Element],
        epsilon: Element,
        batch: usize,
        features: usize,
    ) -> Self {
        assert!(
            batch > 0 && features > 0,
            "a batch-norm task needs non-empty extents"
        );
        assert_eq!(
            input.len(),
            batch * features,
            "the input slice does not span its {batch} x {features} matrix"
        );
        assert_eq!(
            scale.len(),
            features,
            "the scale slice does not span its {features} features"
        );
        assert_eq!(
            shift.len(),
            features,
            "the shift slice does not span its {features} features"
        );
        Self {
            input,
            scale,
            shift,
            epsilon,
            batch,
            features,
        }
    }

    /// Returns the row-major `[batch, features]` input slice.
    pub fn input(&self) -> &'buffers [Element] {
        self.input
    }

    /// Returns the `[features]` learned scale.
    pub fn scale(&self) -> &'buffers [Element] {
        self.scale
    }

    /// Returns the `[features]` learned shift.
    pub fn shift(&self) -> &'buffers [Element] {
        self.shift
    }

    /// Returns the single stabilizing epsilon.
    pub fn epsilon(&self) -> &Element {
        &self.epsilon
    }

    /// Returns the batch extent the statistics reduce over.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Returns the feature extent.
    pub fn features(&self) -> usize {
        self.features
    }
}

/// A batch-normalization task's whole product: the normalized output
/// with the batch statistics it normalized by, mirroring the
/// recorded formula's root and named results.
#[derive(Debug)]
pub struct Normalized<Element> {
    /// The normalized, affine-transformed `[batch, features]` output,
    /// row-major.
    pub output: Vec<Element>,
    /// The batch's per-feature `[features]` mean.
    pub mean: Vec<Element>,
    /// The batch's per-feature `[features]` biased variance.
    pub variance: Vec<Element>,
}
