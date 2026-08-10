use super::*;

#[test]
fn escaping_neutralizes_every_markup_character() {
    assert_eq!(escape("a < b & c > d"), "a &lt; b &amp; c &gt; d");
}

#[test]
fn escaping_leaves_ordinary_text_untouched() {
    assert_eq!(escape("shape [2, 3] mean 0.5"), "shape [2, 3] mean 0.5");
}

#[test]
fn a_mime_bundle_matches_the_evcxr_protocol() {
    // The protocol emitter is `malevich`'s; this pins the shape poorgrad
    // relies on rather than restating its unit tests.
    assert_eq!(
        mime_bundle(&[("text/html", "<b>x</b>"), ("text/plain", "x")]),
        "EVCXR_BEGIN_CONTENT text/html\n<b>x</b>\nEVCXR_END_CONTENT\n\
         EVCXR_BEGIN_CONTENT text/plain\nx\nEVCXR_END_CONTENT"
    );
}

#[test]
fn cards_carry_their_theme_colors_and_place_the_body_verbatim() {
    let dark = card(Theme::DARK, "header", "<i>body</i>");
    assert!(dark.contains("background-color:#0d1117"));
    assert!(dark.contains("color:#e6edf3"));
    assert!(dark.contains("<i>body</i>"));
    assert!(dark.contains("header"));

    let light = card(Theme::LIGHT, "header", "body");
    assert!(light.contains("background-color:#ffffff"));
    assert!(light.contains("color:#1f2328"));
}

#[test]
fn a_poorgrad_card_and_a_malevich_chart_share_one_background() {
    // The reason the colors come from `malevich` rather than a local
    // copy: a tensor table and a chart in one cell must not disagree.
    let plot = malevich::Plot::new().layer(malevich::Line::y(vec![1.0, 2.0]));
    for theme in [Theme::DARK, Theme::LIGHT] {
        let mut frame = malevich::Frame::plain(20, 6);
        frame.theme = theme;
        let (background, _) = malevich::evcxr::card_colors(theme);
        assert!(
            plot.to_html(&frame)
                .contains(&format!("background-color:{background}"))
        );
        assert!(card(theme, "header", "body").contains(&format!("background-color:{background}")));
    }
}
