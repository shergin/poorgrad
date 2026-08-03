//! Stride classification shared by the BLAS-shaped backends.
//!
//! Both cblas (accelerate) and cuBLAS (cuda) express an operand as a
//! transpose flag plus a leading dimension; this module holds the
//! one classification of `GemmTask` strides into that form, and each
//! backend maps the flag onto its own constants.

/// One BLAS-ready operand: whether the buffer holds the logical
/// matrix transposed, and the leading dimension.
pub(super) struct Operand {
    pub(super) transposed: bool,
    pub(super) leading: i32,
}

/// It classifies an operand's strides into BLAS form, or declines:
/// `None` for patterns BLAS cannot express (a stride-0 broadcast
/// axis of extent above one) and for dimensions beyond `i32`.
///
/// A unit column stride is untransposed with the row stride as the
/// leading dimension; a unit row stride is transposed with the
/// column stride leading. An extent-1 axis leaves its stride unused,
/// so a degenerate leading dimension is replaced by the smallest
/// value BLAS accepts rather than declined.
pub(super) fn classify(strides: [usize; 2], rows: usize, columns: usize) -> Option<Operand> {
    if strides[1] == 1 {
        let leading = if rows == 1 { columns } else { strides[0] };
        if leading < columns {
            return None;
        }
        return Some(Operand {
            transposed: false,
            leading: i32::try_from(leading).ok()?,
        });
    }
    if strides[0] == 1 {
        let leading = if columns == 1 { rows } else { strides[1] };
        if leading < rows {
            return None;
        }
        return Some(Operand {
            transposed: true,
            leading: i32::try_from(leading).ok()?,
        });
    }
    None
}

#[cfg(test)]
#[path = "tests/operand_tests.rs"]
mod tests;
