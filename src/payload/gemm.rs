//! Slice-path kernels for dense rank-2 matrix multiplication.
//!
//! The logical-access path reads every element through
//! `Layout::storage_index` — a per-axis unravel on each access — which
//! costs two orders of magnitude at any size. These kernels read the
//! same elements through the backing slice with precomputed stride
//! arithmetic instead. They change memory access only, never
//! arithmetic: every output element accumulates its terms in
//! ascending inner-index order, seeded from the first term exactly
//! like the logical path, so the two paths answer bit-identically.
//!
//! There is no explicit SIMD and no dispatch to special instructions
//! here; the loops are shaped so the compiler's auto-vectorizer is
//! *allowed* to emit them. Every output element owns an independent
//! accumulator and the hot loop runs over plain slices, so
//! vectorizing across columns reorders no floating-point sum —
//! vectorization stays legal under strict IEEE semantics, with no
//! fast math and no loss of the bit parity above. On aarch64 the
//! contiguous arm compiles to unrolled NEON multiply/add over the
//! output row, which is why `f32` measures at twice the `f64` rate:
//! four lanes per vector register instead of two.

use super::Differentiable;

/// A dense rank-2 operand for the slice-path kernels: the whole
/// backing buffer plus the layout numbers that place element `(i, j)`
/// at `data[offset + i * row_stride + j * column_stride]`.
///
/// Any layout a rank-2 dense view can carry is representable: a
/// transposed view has a unit `row_stride`, a narrowed window a
/// `row_stride` wider than its column count, and a broadcast axis a
/// stride of zero.
pub(crate) struct Operand<'buffer, Element> {
    pub(crate) data: &'buffer [Element],
    pub(crate) offset: usize,
    pub(crate) row_stride: usize,
    pub(crate) column_stride: usize,
}

impl<Element> Operand<'_, Element> {
    /// Returns the flat index of this operand's element `(row, column)`.
    fn index(&self, row: usize, column: usize) -> usize {
        self.offset + row * self.row_stride + column * self.column_stride
    }
}

/// It computes the `[rows, inner] . [inner, columns]` product of two
/// dense operands into a contiguous row-major buffer.
///
/// The loops walk output rows with one independent accumulator per
/// element, folding inner steps in ascending order, so the
/// per-element summation matches the logical path bit for bit; when
/// the right operand's rows are contiguous the inner loop runs over
/// plain slices and vectorizes.
///
/// The dot-product form was rejected on purpose: with reassociation
/// forbidden by bit parity, a single accumulator over the inner axis
/// is a serial dependency chain the compiler can neither vectorize
/// nor pipeline, while per-column accumulators keep every update
/// independent.
pub(crate) fn multiply<Element: Differentiable>(
    a: &Operand<'_, Element>,
    b: &Operand<'_, Element>,
    rows: usize,
    inner: usize,
    columns: usize,
) -> Vec<Element> {
    let mut elements = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        // Seed the output row with the first term of every product:
        // a generic element has no zero to start from, and a float
        // zero would break parity — `0.0 + -0.0` answers `+0.0`, so
        // a zero-seeded accumulator flips the sign of an all
        // negative-zero sum that the logical path keeps negative.
        let a_first = a.data[a.index(row, 0)].clone();
        seed_row(&mut elements, &a_first, b, columns);
        // The freshly seeded suffix of the buffer is this row's
        // vector of accumulators.
        let output = &mut elements[row * columns..];
        for step in 1..inner {
            let a_value = a.data[a.index(row, step)].clone();
            accumulate_row(output, &a_value, b, step, columns);
        }
    }
    elements
}

/// It appends one seed row, `a_first * b[0, column]` per column, to
/// the output buffer.
fn seed_row<Element: Differentiable>(
    elements: &mut Vec<Element>,
    a_first: &Element,
    b: &Operand<'_, Element>,
    columns: usize,
) {
    if b.column_stride == 1 {
        let b_row = &b.data[b.offset..b.offset + columns];
        elements.extend(
            b_row
                .iter()
                .map(|b_element| a_first.clone() * b_element.clone()),
        );
        return;
    }
    elements
        .extend((0..columns).map(|column| a_first.clone() * b.data[b.index(0, column)].clone()));
}

/// It folds one inner step into an output row:
/// `output[column] += a_value * b[step, column]` for every column.
///
/// The contiguous arm hands the compiler two plain slices — the
/// accumulators and the operand row — and is the loop that
/// auto-vectorizes; the strided arm (a transposed right operand,
/// most often) reads through the stride and stays scalar, which is
/// the measured cost of that case.
fn accumulate_row<Element: Differentiable>(
    output: &mut [Element],
    a_value: &Element,
    b: &Operand<'_, Element>,
    step: usize,
    columns: usize,
) {
    if b.column_stride == 1 {
        let start = b.index(step, 0);
        let b_row = &b.data[start..start + columns];
        for (output_element, b_element) in output.iter_mut().zip(b_row) {
            *output_element = output_element.clone() + a_value.clone() * b_element.clone();
        }
        return;
    }
    for (column, output_element) in output.iter_mut().enumerate() {
        *output_element =
            output_element.clone() + a_value.clone() * b.data[b.index(step, column)].clone();
    }
}

#[cfg(test)]
#[path = "tests/gemm_tests.rs"]
mod tests;
