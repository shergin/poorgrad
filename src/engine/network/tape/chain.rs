use std::sync::Arc;

use super::Segment;

/// Returns whether two chains attribute the index range `[0, length)`
/// to the same branches.
///
/// Segments starting at or beyond `length` are ignored: they describe
/// nodes outside the compared range, so a longer tape stays kin with a
/// field taken before it grew.
pub(crate) fn chains_agree(
    left: &Arc<Vec<Segment>>,
    right: &Arc<Vec<Segment>>,
    length: usize,
) -> bool {
    if Arc::ptr_eq(left, right) {
        return true;
    }
    let trimmed = |chain: &[Segment]| {
        chain
            .iter()
            .take_while(|segment| segment.start < length)
            .count()
    };
    left[..trimmed(left)] == right[..trimmed(right)]
}
