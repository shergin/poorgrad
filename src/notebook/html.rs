//! The Evcxr output protocol and the shared card chrome.
//!
//! Evcxr reads mime-typed blocks from a cell's stdout, so emitting
//! rich output is printing text in a fixed envelope. The helpers here
//! build that envelope and the themed card every display draws into,
//! matching `malevich`'s card colors so a chart and a tensor table
//! rendered side by side look like one system.

use malevich::Theme;

/// The card background and foreground for a theme.
///
/// These duplicate `malevich`'s own card colors, which are crate
/// private there. Exporting them from `malevich` would remove the
/// duplication; until then the values are kept identical on purpose.
pub(crate) fn card_colors(theme: Theme) -> (&'static str, &'static str) {
    if theme == Theme::LIGHT {
        ("#ffffff", "#1f2328")
    } else {
        ("#0d1117", "#e6edf3")
    }
}

/// The muted foreground for headers and units, per theme.
pub(crate) fn muted_color(theme: Theme) -> &'static str {
    if theme == Theme::LIGHT {
        "#59636e"
    } else {
        "#8d96a0"
    }
}

/// Escapes text for HTML element content.
pub(crate) fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

/// Wraps `body` in a themed card carrying `header` as its caption.
///
/// The body is placed verbatim, so callers pass either escaped text or
/// their own markup.
pub(crate) fn card(theme: Theme, header: &str, body: &str) -> String {
    use std::fmt::Write as _;

    let (background, foreground) = card_colors(theme);
    let muted = muted_color(theme);
    let mut html = String::with_capacity(body.len() + 512);
    let _ = write!(
        html,
        "<div style=\"margin:0;padding:12px 16px;border:0;border-radius:8px;\
         box-sizing:border-box;display:inline-block;max-width:100%;overflow-x:auto;\
         font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;\
         font-size:13px;line-height:1.35;background-color:{background};color:{foreground}\">\
         <div style=\"color:{muted};margin-bottom:6px\">{header}</div>{body}</div>"
    );
    html
}

/// Wraps `blocks` in Evcxr's stdout protocol as alternative
/// representations of one value.
///
/// A frontend draws the richest form it supports, so a `text/plain`
/// block beside `text/html` yields the card in Jupyter and readable
/// text in the terminal REPL, which cannot draw HTML.
pub(crate) fn mime_bundle(blocks: &[(&str, &str)]) -> String {
    use std::fmt::Write as _;

    let mut bundle = String::new();
    for (mime, content) in blocks {
        if !bundle.is_empty() {
            bundle.push('\n');
        }
        let _ = write!(
            bundle,
            "EVCXR_BEGIN_CONTENT {mime}\n{content}\nEVCXR_END_CONTENT"
        );
    }
    bundle
}

/// Prints an HTML representation and its plain-text alternative as one
/// Evcxr value: the last statement of every `evcxr_display`.
pub(crate) fn show(html: &str, plain: &str) {
    println!(
        "{}",
        mime_bundle(&[("text/html", html), ("text/plain", plain)])
    );
}

#[cfg(test)]
#[path = "tests/html_tests.rs"]
mod tests;
