use super::*;

#[test]
fn a_small_tensor_card_shows_its_exact_values() {
    let tensor = Tensor::new([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let html = tensor.to_html(Theme::DARK);
    assert!(html.contains("tensor"));
    assert!(html.contains("[2, 2]"));
    assert!(html.contains("<table"));
    assert!(html.contains(">4<"));
}

#[test]
fn an_f32_tensor_names_its_element_type() {
    let tensor = Tensor::new([2], vec![1.0_f32, 2.0]);
    assert!(tensor.to_html(Theme::DARK).contains("f32"));
}

#[test]
fn a_large_tensor_card_charts_instead_of_tabulating() {
    let elements: Vec<f64> = (0..(16 * 16)).map(|index| index as f64).collect();
    let tensor = Tensor::new([16, 16], elements);
    let html = tensor.to_html(Theme::DARK);
    assert!(!html.contains("<table"));
    assert!(html.contains("<pre"));
}

#[test]
fn tensor_rendering_is_deterministic_and_theme_aware() {
    let tensor = Tensor::new([2], vec![1.0_f64, 2.0]);
    assert_eq!(tensor.to_html(Theme::DARK), tensor.to_html(Theme::DARK));
    assert!(tensor.to_html(Theme::DARK).contains("#0d1117"));
    assert!(tensor.to_html(Theme::LIGHT).contains("#ffffff"));
}
