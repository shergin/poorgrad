use super::Shape;

#[test]
fn scalar_has_rank_zero_and_volume_one() {
    let shape = Shape::scalar();
    assert_eq!(shape.rank(), 0);
    assert_eq!(shape.volume(), 1);
    assert_eq!(shape.axes(), &[] as &[usize]);
}

#[test]
fn axes_determine_rank_and_volume() {
    let shape = Shape::new([2, 3, 4]);
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.volume(), 24);
    assert_eq!(shape.axes(), &[2, 3, 4]);
}

#[test]
fn display_lists_the_axes() {
    assert_eq!(Shape::new([3, 2]).to_string(), "[3, 2]");
    assert_eq!(Shape::scalar().to_string(), "[]");
}
