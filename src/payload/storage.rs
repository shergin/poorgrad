use std::sync::Arc;

use super::Shape;
use super::layout::Layout;

/// The buffer representation behind a [`Tensor`](super::Tensor).
///
/// This enum is the extension seam for element storage. Today it holds a
/// strided dense buffer and a non-allocating constant; future
/// representations, such as a sparse buffer or a SIMD-aligned one, become
/// new variants without disturbing the operations, which read every
/// representation through the same logical element access
/// ([`Tensor::iter`](super::Tensor::iter) and its internal `get`).
///
/// Each variant carries exactly the metadata its representation needs. The
/// strided addressing lives inside `Dense` as a [`Layout`], so a
/// representation that does not stride is not forced to model strides; the
/// logical shape is the one universal descriptor and every variant answers
/// for it.
///
/// `Dense` is an owned, `Arc`-shared row-major buffer that a [`Layout`]
/// addresses through strides and an offset; cloning shares it. `Constant`
/// is a single value that logically fills its shape without allocating a
/// buffer, which keeps `filled`, `zero_like`, `one_like`, and whole-shape
/// broadcasts O(1) and lets their algebra stay closed.
///
/// `Selection` is a one-hot `[count, vocab]` matrix stored as its `count`
/// row indices: `t[i, j]` is `one` when `indices[i] == j` and `zero`
/// otherwise. It keeps the token indices of an embedding lookup as `usize`
/// inside a homogeneous payload (no integer encoding, no separate index
/// type), lets the buffer stay O(count) instead of O(count * vocab), and is
/// what a [`Gather`](crate) reads directly. The stored `zero` and `one` are
/// the values the logical-access fallback hands out by reference, since a
/// computed representation has no buffer to borrow from.
#[derive(Debug, Clone)]
pub(crate) enum Storage<Element> {
    Dense {
        data: Arc<Vec<Element>>,
        layout: Layout,
    },
    Constant {
        shape: Shape,
        value: Element,
    },
    Selection {
        indices: Arc<Vec<usize>>,
        shape: Shape,
        zero: Element,
        one: Element,
    },
}
