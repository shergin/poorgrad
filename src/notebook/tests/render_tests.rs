use super::*;
use crate::Tensor;

#[test]
fn numbers_drop_trailing_zeros_and_keep_ordinary_magnitudes_plain() {
    assert_eq!(number(0.0), "0");
    assert_eq!(number(1.0), "1");
    assert_eq!(number(0.5), "0.5");
    assert_eq!(number(-2.25), "-2.25");
    assert_eq!(number(1234.5), "1234.5");
}

#[test]
fn numbers_outside_the_readable_band_become_exponential() {
    assert_eq!(number(1e-9), "1.000e-9");
    assert_eq!(number(2.5e12), "2.500e12");
}

#[test]
fn numbers_name_the_non_finite_cases_instead_of_formatting_them() {
    assert_eq!(number(f64::NAN), "NaN");
    assert_eq!(number(f64::INFINITY), "inf");
    assert_eq!(number(f64::NEG_INFINITY), "-inf");
}

#[test]
fn shapes_render_as_axis_lists() {
    assert_eq!(shape_text(&Shape::new([2, 3])), "[2, 3]");
    assert_eq!(shape_text(&Shape::scalar()), "[]");
}

#[test]
fn a_scalar_payload_renders_as_one_number() {
    let (html, text) = body(Theme::DARK, &2.5_f64);
    assert!(html.contains("2.5"));
    assert_eq!(text, "2.5");
    assert!(!html.contains("<table"));
}

#[test]
fn a_small_matrix_renders_as_an_exact_table() {
    let tensor = Tensor::new([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let (html, text) = body(Theme::DARK, &tensor);
    assert!(html.starts_with("<table"));
    // Two rows of three, so exactly two row openers.
    assert_eq!(html.matches("<tr>").count(), 2);
    assert_eq!(html.matches("<td").count(), 6);
    assert_eq!(text, "1 2 3\n4 5 6");
}

#[test]
fn a_large_matrix_renders_as_a_chart_rather_than_a_table() {
    let elements: Vec<f64> = (0..(20 * 20)).map(|index| index as f64).collect();
    let tensor = Tensor::new([20, 20], elements);
    let (html, _) = body(Theme::DARK, &tensor);
    assert!(!html.contains("<table"));
    assert!(html.contains("<pre"));
}

#[test]
fn a_long_vector_renders_as_a_chart_rather_than_a_table() {
    let elements: Vec<f64> = (0..64).map(|index| index as f64).collect();
    let tensor = Tensor::new([64], elements);
    let (html, _) = body(Theme::DARK, &tensor);
    assert!(!html.contains("<table"));
}

#[test]
fn the_header_reports_shape_type_and_extremes() {
    let tensor = Tensor::new([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let cells = tensor.cells();
    let header = header::<Tensor<f64>>(&tensor.shape(), &cells);
    assert!(header.contains("[2, 2]"));
    assert!(header.contains("f64"));
    assert!(header.contains("min 1"));
    assert!(header.contains("max 4"));
    assert!(header.contains("mean 2.5"));
}

#[test]
fn non_finite_elements_are_counted_and_excluded_from_the_extremes() {
    let tensor = Tensor::new([3], vec![1.0_f64, f64::NAN, 3.0]);
    let cells = tensor.cells();
    let header = header::<Tensor<f64>>(&tensor.shape(), &cells);
    assert!(header.contains("1 non-finite"));
    assert!(header.contains("min 1"));
    assert!(header.contains("max 3"));
    assert!(header.contains("mean 2"));
}

#[test]
fn an_all_non_finite_payload_still_renders_without_extremes() {
    let tensor = Tensor::new([2], vec![f64::NAN, f64::NAN]);
    let cells = tensor.cells();
    let header = header::<Tensor<f64>>(&tensor.shape(), &cells);
    assert!(header.contains("2 non-finite"));
    assert!(!header.contains("mean"));
}

#[test]
fn a_flat_payload_gets_no_shading_because_there_is_no_variation() {
    let tensor = Tensor::new([2, 2], vec![3.0_f64; 4]);
    let (html, _) = body(Theme::DARK, &tensor);
    assert!(html.contains("transparent"));
    assert!(!html.contains("rgba("));
}

#[test]
fn rendering_is_deterministic_for_a_given_payload_and_theme() {
    let tensor = Tensor::new([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        payload_card(Theme::DARK, "value", &tensor),
        payload_card(Theme::DARK, "value", &tensor)
    );
}

#[test]
fn the_theme_changes_the_card_and_nothing_else() {
    let tensor = Tensor::new([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let dark = payload_card(Theme::DARK, "value", &tensor);
    let light = payload_card(Theme::LIGHT, "value", &tensor);
    assert_ne!(dark, light);
    assert!(dark.contains("#0d1117"));
    assert!(light.contains("#ffffff"));
}

#[test]
fn f32_and_bf16_payloads_name_their_element_type() {
    assert_eq!(<Tensor<f32> as Renderable>::element_name(), "f32");
    assert_eq!(<f64 as Renderable>::element_name(), "f64");
    assert_eq!(<crate::Bf16 as Renderable>::element_name(), "bf16");
}
