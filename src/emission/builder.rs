//! The typed text layer under StableHLO emission: element formatting,
//! tensor types, and dense literals.
//!
//! Emission stays in-crate as pure string building — zero heavy
//! dependencies, human-readable output — but never as loose `format!`
//! calls: every fragment of MLIR syntax is produced here, typed by the
//! recorded [`Shape`] and the element's [`Emittable`] contract, so SSA
//! plumbing and type trailers cannot drift per call site. Anything that
//! links an MLIR toolchain (parsing, bytecode, execution) lives outside
//! the crate; the text these helpers produce is the interchange.

use crate::{Differentiable, Shape};

/// An element type StableHLO emission can render: its MLIR type name
/// and the literal forms MLIR's float syntax accepts.
///
/// Finite values print in shortest-round-trip decimal (normalized to
/// carry a dot, which MLIR requires); non-finite values print as IEEE
/// bit-pattern hex, the only literal form MLIR has for them.
pub trait Emittable: Differentiable + PartialEq {
    /// The MLIR element type name, such as `f32`.
    const ELEMENT: &'static str;

    /// The literal of the additive identity, seeding sum reduces and
    /// zero pads.
    const ZERO: &'static str;

    /// The literal of negative infinity, seeding max reduces.
    const NEGATIVE_INFINITY: &'static str;

    /// Formats this element as an MLIR literal.
    fn literal(&self) -> String;
}

impl Emittable for f32 {
    const ELEMENT: &'static str = "f32";
    const ZERO: &'static str = "0.0";
    const NEGATIVE_INFINITY: &'static str = "0xFF800000";

    fn literal(&self) -> String {
        if self.is_finite() {
            return dotted(format!("{self:?}"));
        }
        format!("0x{:08X}", self.to_bits())
    }
}

impl Emittable for f64 {
    const ELEMENT: &'static str = "f64";
    const ZERO: &'static str = "0.0";
    const NEGATIVE_INFINITY: &'static str = "0xFFF0000000000000";

    fn literal(&self) -> String {
        if self.is_finite() {
            return dotted(format!("{self:?}"));
        }
        format!("0x{:016X}", self.to_bits())
    }
}

/// Returns the decimal float `rendered`, guaranteed to carry a dot:
/// MLIR's float syntax requires one, while Rust's shortest form may
/// print scientific notation with a bare mantissa (`1e-5`).
fn dotted(rendered: String) -> String {
    if rendered.contains('.') {
        return rendered;
    }
    match rendered.split_once('e') {
        Some((mantissa, exponent)) => format!("{mantissa}.0e{exponent}"),
        None => format!("{rendered}.0"),
    }
}

/// Returns the MLIR tensor type of `shape`: `tensor<2x3xf32>`, with the
/// scalar shape printing as `tensor<f32>`.
pub(crate) fn tensor_type<Element: Emittable>(shape: &Shape) -> String {
    let mut dimensions = String::new();
    for extent in shape.axes() {
        dimensions.push_str(&extent.to_string());
        dimensions.push('x');
    }
    format!("tensor<{dimensions}{}>", Element::ELEMENT)
}

/// Returns the dense literal of `elements` in row-major `shape`: the
/// splat form when every element agrees, the nested-bracket form
/// otherwise.
pub(crate) fn dense_literal<Element: Emittable>(shape: &Shape, elements: &[Element]) -> String {
    if let Some(first) = elements.first()
        && elements.iter().all(|element| element == first)
    {
        return format!("dense<{}>", first.literal());
    }
    format!("dense<{}>", nested(shape.axes(), elements))
}

/// Renders `elements` as nested brackets following `axes`, one bracket
/// level per axis; rank 0 renders the single element bare.
fn nested<Element: Emittable>(axes: &[usize], elements: &[Element]) -> String {
    match axes.split_first() {
        None => elements[0].literal(),
        Some((&extent, rest)) => {
            let stride = elements.len() / extent;
            let rows: Vec<String> = (0..extent)
                .map(|row| nested(rest, &elements[row * stride..(row + 1) * stride]))
                .collect();
            format!("[{}]", rows.join(", "))
        }
    }
}

#[cfg(test)]
#[path = "tests/builder_tests.rs"]
mod tests;
