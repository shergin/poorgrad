use super::classify;

#[test]
fn contiguous_rows_are_untransposed_with_the_row_stride_leading() {
    let operand = classify([8, 1], 4, 8).expect("a contiguous operand");
    assert!(!operand.transposed);
    assert_eq!(operand.leading, 8);
}

#[test]
fn unit_row_strides_are_transposed_with_the_column_stride_leading() {
    let operand = classify([1, 4], 4, 8).expect("a transposed operand");
    assert!(operand.transposed);
    assert_eq!(operand.leading, 4);
}

#[test]
fn narrowed_windows_keep_their_wide_leading_dimension() {
    let operand = classify([11, 1], 4, 8).expect("a narrowed operand");
    assert!(!operand.transposed);
    assert_eq!(operand.leading, 11);
}

#[test]
fn degenerate_extent_one_axes_repair_their_leading_dimension() {
    // A one-row operand leaves its row stride unused, so a stride
    // below the column count is repaired, not declined.
    let operand = classify([1, 1], 1, 8).expect("a single row");
    assert!(!operand.transposed);
    assert_eq!(operand.leading, 8);
    let operand = classify([1, 3], 5, 1).expect("a single column");
    assert!(operand.transposed);
    assert_eq!(operand.leading, 5);
}

#[test]
fn broadcasts_and_underslung_leads_decline() {
    assert!(classify([0, 1], 4, 8).is_none());
    assert!(classify([2, 3], 4, 8).is_none());
    // A leading dimension shorter than the row cannot be a real
    // row-major layout.
    assert!(classify([4, 1], 4, 8).is_none());
}
