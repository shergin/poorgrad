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
    assert_eq!(
        mime_bundle(&[("text/html", "<b>x</b>"), ("text/plain", "x")]),
        "EVCXR_BEGIN_CONTENT text/html\n<b>x</b>\nEVCXR_END_CONTENT\n\
         EVCXR_BEGIN_CONTENT text/plain\nx\nEVCXR_END_CONTENT"
    );
}

#[test]
fn an_empty_bundle_renders_nothing() {
    assert_eq!(mime_bundle(&[]), "");
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
fn card_colors_match_malevichs_own_so_the_two_crates_agree() {
    // `malevich` renders its plot cards with these exact values; a
    // chart and a tensor table in one notebook must not disagree.
    assert_eq!(card_colors(Theme::DARK), ("#0d1117", "#e6edf3"));
    assert_eq!(card_colors(Theme::LIGHT), ("#ffffff", "#1f2328"));
}
