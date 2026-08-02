//! The dense matrix-multiplication job and its slice-path kernels.
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

/// One dense matrix-multiplication job: `m x k` times `k x n`, each
/// operand a spanning slice read through two strides.
///
/// Element `(i, j)` of `a` lives at
/// `a[i * a_strides[0] + j * a_strides[1]]`; a contiguous operand has
/// strides `[k, 1]`, a transposed view `[1, m]`, a narrowed window a
/// first stride wider than its column count, and a broadcast axis a
/// stride of zero — so views pass through without materializing. The
/// product is contiguous row-major `[m, n]`.
///
/// Tasks are built by `Tensor`'s `matmul` (the constructor validates
/// that each slice spans its matrix under its strides) and read by
/// backend code and by [`Elementary::gemm`](super::Elementary::gemm)
/// implementations.
#[derive(Debug)]
pub struct GemmTask<'buffers, Element> {
    a: &'buffers [Element],
    b: &'buffers [Element],
    m: usize,
    n: usize,
    k: usize,
    a_strides: [usize; 2],
    b_strides: [usize; 2],
}

impl<'buffers, Element> GemmTask<'buffers, Element> {
    /// Creates a validated task; the first logical element of each
    /// operand is the first element of its slice.
    ///
    /// # Panics
    /// Panics if any extent is zero or a slice does not span its
    /// matrix under its strides.
    pub(crate) fn new(
        a: &'buffers [Element],
        a_strides: [usize; 2],
        b: &'buffers [Element],
        b_strides: [usize; 2],
        m: usize,
        k: usize,
        n: usize,
    ) -> Self {
        assert!(
            m > 0 && k > 0 && n > 0,
            "a gemm task needs non-empty extents"
        );
        let a_span = 1 + (m - 1) * a_strides[0] + (k - 1) * a_strides[1];
        assert!(
            a.len() >= a_span,
            "the left operand slice does not span its {m} x {k} matrix"
        );
        let b_span = 1 + (k - 1) * b_strides[0] + (n - 1) * b_strides[1];
        assert!(
            b.len() >= b_span,
            "the right operand slice does not span its {k} x {n} matrix"
        );
        Self {
            a,
            b,
            m,
            n,
            k,
            a_strides,
            b_strides,
        }
    }

    /// Returns the left operand's spanning slice.
    pub fn a(&self) -> &'buffers [Element] {
        self.a
    }

    /// Returns the right operand's spanning slice.
    pub fn b(&self) -> &'buffers [Element] {
        self.b
    }

    /// Returns the number of product rows.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Returns the number of product columns.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the inner (contracted) extent.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the left operand's `(row, column)` element strides.
    pub fn a_strides(&self) -> [usize; 2] {
        self.a_strides
    }

    /// Returns the right operand's `(row, column)` element strides.
    pub fn b_strides(&self) -> [usize; 2] {
        self.b_strides
    }
}

/// It computes a task's product into a contiguous row-major buffer.
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
pub(crate) fn multiply<Element: Differentiable>(task: &GemmTask<'_, Element>) -> Vec<Element> {
    let mut elements = Vec::with_capacity(task.m * task.n);
    for row in 0..task.m {
        let a_row_start = row * task.a_strides[0];
        // Seed the output row with the first term of every product:
        // a generic element has no zero to start from, and a float
        // zero would break parity — `0.0 + -0.0` answers `+0.0`, so
        // a zero-seeded accumulator flips the sign of an all
        // negative-zero sum that the logical path keeps negative.
        let a_first = task.a[a_row_start].clone();
        seed_row(&mut elements, &a_first, task);
        // The freshly seeded suffix of the buffer is this row's
        // vector of accumulators.
        let output = &mut elements[row * task.n..];
        for step in 1..task.k {
            let a_value = task.a[a_row_start + step * task.a_strides[1]].clone();
            accumulate_row(output, &a_value, task, step);
        }
    }
    elements
}

/// It appends one seed row, `a_first * b[0, column]` per column, to
/// the output buffer.
fn seed_row<Element: Differentiable>(
    elements: &mut Vec<Element>,
    a_first: &Element,
    task: &GemmTask<'_, Element>,
) {
    if task.b_strides[1] == 1 {
        let b_row = &task.b[..task.n];
        elements.extend(
            b_row
                .iter()
                .map(|b_element| a_first.clone() * b_element.clone()),
        );
        return;
    }
    elements.extend(
        (0..task.n).map(|column| a_first.clone() * task.b[column * task.b_strides[1]].clone()),
    );
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
    task: &GemmTask<'_, Element>,
    step: usize,
) {
    let b_row_start = step * task.b_strides[0];
    if task.b_strides[1] == 1 {
        let b_row = &task.b[b_row_start..b_row_start + task.n];
        for (output_element, b_element) in output.iter_mut().zip(b_row) {
            *output_element = output_element.clone() + a_value.clone() * b_element.clone();
        }
        return;
    }
    for (column, output_element) in output.iter_mut().enumerate() {
        *output_element = output_element.clone()
            + a_value.clone() * task.b[b_row_start + column * task.b_strides[1]].clone();
    }
}

#[cfg(test)]
#[path = "tests/gemm_tests.rs"]
mod tests;
