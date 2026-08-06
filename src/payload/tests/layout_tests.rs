use crate::Shape;

use super::Layout;

#[test]
fn contiguous_layout_has_row_major_strides() {
    let layout = Layout::contiguous(Shape::new([2, 3, 4]));
    assert_eq!(layout.strides(), &[12, 4, 1]);
    assert_eq!(layout.offset(), 0);
    assert!(layout.is_contiguous());
}

#[test]
fn scalar_layout_addresses_a_single_element() {
    let layout = Layout::contiguous(Shape::scalar());
    assert_eq!(layout.volume(), 1);
    assert_eq!(layout.storage_index(0), 0);
    assert!(layout.is_contiguous());
}

#[test]
fn contiguous_storage_index_is_the_identity() {
    let layout = Layout::contiguous(Shape::new([2, 3]));
    for position in 0..6 {
        assert_eq!(layout.storage_index(position), position);
    }
}

#[test]
fn transpose_swaps_axes_and_strides() {
    let transposed = Layout::contiguous(Shape::new([2, 3])).transpose();
    assert_eq!(transposed.shape(), &Shape::new([3, 2]));
    assert_eq!(transposed.strides(), &[1, 3]);
    assert!(!transposed.is_contiguous());

    // Logical position `(row, column)` of the [3, 2] view reads the
    // original [2, 3] element `(column, row)`.
    assert_eq!(transposed.storage_index(1), 3);
    assert_eq!(transposed.storage_index(2), 1);
}

#[test]
fn transpose_below_rank_two_is_unchanged() {
    let vector = Layout::contiguous(Shape::new([4]));
    assert_eq!(vector.transpose(), vector);
}

#[test]
#[should_panic(expected = "rank 2 at most")]
fn transpose_rejects_rank_above_two() {
    Layout::contiguous(Shape::new([2, 3, 4])).transpose();
}

#[test]
fn broadcast_along_inserts_a_stride_zero_axis() {
    let row = Layout::contiguous(Shape::new([3]));
    let spread = row.broadcast_along(0, &Shape::new([2, 3]));
    assert_eq!(spread.shape(), &Shape::new([2, 3]));
    assert_eq!(spread.strides(), &[0, 1]);

    let indices: Vec<usize> = (0..6)
        .map(|position| spread.storage_index(position))
        .collect();
    assert_eq!(indices, [0, 1, 2, 0, 1, 2]);
}

#[test]
fn extent_one_axes_stay_contiguous() {
    let layout = Layout::contiguous(Shape::new([1, 3]));
    assert!(layout.is_contiguous());
}

#[test]
fn reshape_of_a_contiguous_layout_recomputes_strides() {
    let reshaped = Layout::contiguous(Shape::new([2, 3]))
        .reshape(Shape::new([3, 2]))
        .expect("a contiguous layout reshapes without a copy");
    assert_eq!(reshaped.shape(), &Shape::new([3, 2]));
    assert_eq!(reshaped.strides(), &[2, 1]);
}

#[test]
fn reshape_of_a_strided_layout_requires_a_copy() {
    let transposed = Layout::contiguous(Shape::new([2, 3])).transpose();
    assert!(transposed.reshape(Shape::new([6])).is_none());
}

#[test]
fn reshape_of_a_strided_layout_drops_a_unit_axis_in_place() {
    let spread = Layout::contiguous(Shape::new([1, 3])).broadcast_along(0, &Shape::new([5, 1, 3]));
    let squeezed = spread
        .reshape(Shape::new([5, 3]))
        .expect("a unit-axis reshape keeps the view");
    assert_eq!(squeezed.shape(), &Shape::new([5, 3]));
    assert_eq!(squeezed.strides(), &[0, 1]);
    assert_eq!(squeezed.offset(), 0);
}

#[test]
fn reshape_of_a_strided_layout_inserts_a_unit_axis_in_place() {
    let spread = Layout::contiguous(Shape::new([2, 3])).broadcast_along(0, &Shape::new([4, 2, 3]));
    let unsqueezed = spread
        .reshape(Shape::new([4, 2, 1, 3]))
        .expect("a unit-axis reshape keeps the view");
    assert_eq!(unsqueezed.shape(), &Shape::new([4, 2, 1, 3]));
    assert_eq!(unsqueezed.strides(), &[0, 3, 0, 1]);
}

#[test]
fn reshape_of_a_strided_layout_still_copies_when_non_unit_axes_change() {
    let spread = Layout::contiguous(Shape::new([2, 3])).broadcast_along(0, &Shape::new([4, 2, 3]));
    assert!(spread.reshape(Shape::new([4, 6])).is_none());
    assert!(spread.reshape(Shape::new([2, 4, 3])).is_none());
}

#[test]
fn permute_reorders_axes_and_strides() {
    let permuted = Layout::contiguous(Shape::new([2, 3, 4])).permute(&[2, 0, 1]);
    assert_eq!(permuted.shape(), &Shape::new([4, 2, 3]));
    assert_eq!(permuted.strides(), &[1, 12, 4]);
}

#[test]
fn narrow_shifts_offset_and_shrinks_the_axis() {
    let window = Layout::contiguous(Shape::new([2, 3])).narrow(1, 1, 2);
    assert_eq!(window.shape(), &Shape::new([2, 2]));
    assert_eq!(window.strides(), &[3, 1]);
    assert_eq!(window.offset(), 1);
}
