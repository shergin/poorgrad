# Rules for Comments in Code

- Only add comments in the following cases:
  - To clarify non-obvious or tricky logic.
  - To explain uncommon strategies or design choices.
  - For complex or hard-to-read conditions (e.g. long `if` statements).
  - To describe higher-level components such as files, classes, or functions.
- Do **not** add comments for steps that are obvious from the code.
- Most comments (except short inline ones) should be full English sentences ending with a period.
- Place comments on separate lines, not at the end of code lines (unless they are short inline comments).
- Use only English and ASCII characters.
- Do not use pseudographics or decorative characters.
- When referencing identifiers, wrap them in backticks (`` ` ``).
- Keep comments concise and to the point.
- When documenting a function, assume the sentence starts with "It", and use the third-person singular form of the verb.
- Use /// or //! for comments; use markdown for formatting.

Here is an example:
```rust
/// Fooify a `Foo` with a label
///
/// # Parameters
/// - `label`: A string labelling the foo
/// - `magic`: A `Foo` that will be labeled
///
/// # Returns
/// A `Result` which is:
/// - `Ok`: A `Bar` that is the labeled `Foo` and thus lives as long as the
///     `Foo` given in `magic`.
/// - `Err`: Returns the number of gravely appalled people (per half-century
///     per country) if you were to use that label *and* `Foo`'s acceptance
///     indicator is less than it.
///
/// # Type parameters
/// - `T`: A type that can be converted into a `FooLabel`, e.g. a `String`, a
///     `BananaPeelRope`, or a `Cow<str>`.
///
/// # Lifetimes
/// - `floof`: The life time of the given foo as determined by the floof source
///     it was originally loaded from.
///
/// # Examples
/// ```rust
/// assert_eq!(fooify("lorem", Foo::extract_from_global_floof_resource()).label(),
///            Bar::with_label("lorem"))
/// ```
///
/// # See also
/// - [`Bar::from_foo`]
/// - [Foobar](https://en.m.wikipedia.org/wiki/Foobar)
fn fooify<'floof, T>(label: T, magic: Foo<'floof>) -> Result<Bar<'floof>, i32>
    where T: Into<FooLabel>
{
    unimplemented!();
}
```

# General rules for writing code
- Do not abbreviate variable names (and its parts) unless it's extremely common or ideomatic for the particular language (e.g. variables like `i`, `ctx`, etc).
- Pay especial attention to the variables name, they must be logical and short, usually nouns pointing to exact nature of the object.
- Prefer early returns (fail-fast approach) over nested conditionals to reduce indentation and improve readability. Return early for error conditions and edge cases instead of wrapping main logic in nested `if` statements.
- For error types, use `thiserror` library with `#[from]` attribute for automatic error conversions. Prefer `?` operator and early returns over deeply nested `match` or `if let` statements.
- When handling errors from external types, add a variant to your error enum with `#[from]` instead of using `.map_err()`. This enables automatic error conversion via the `?` operator.
- Always run `cargo fmt` after making code changes, before running `cargo check` or `cargo build`.
- Dynamic dispatch is not allowed in the main API design or on any hot path: dispatch is static by construction — a plain enum `match` for the operation set, monomorphized generics for payloads and rules, `impl Fn`/`impl FnMut` for every closure parameter, and never `Box<dyn Fn>` or trait objects in engine loops. A public trait may be object-safe as a capability (`Optimizer` is), but no API may *require* a trait object, and no engine code may call through one. Sanctioned exceptions, on record: platform-mandated indirection (Metal's Objective-C protocol objects, the dlopen'd backend function pointers — both amortized over kernel launches), and caller-side `dyn` in tests or examples where a comparison loop iterates strategies.

# Rules for facade design

- Facades (the neural tier: activations, layers, losses, optimizers, initializers) compose exclusively through the public operation surface — no privileged engine access — so a facade's internal spelling can change without breaking anyone, and a hand-rolled equivalent behaves identically.
- Pick the facade's shape by what it is:
  - A *stateless closed set of alternatives* is a plain `Copy` enum with an `express` method (`Activation`).
  - A *stateful or user-extensible strategy* is an open, object-safe trait with the built-ins as ordinary implementations (`Optimizer`); never a closed enum, so custom implementations have equal standing.
  - An *operand-asymmetric formula* (no natural `self`) is a free function in a domain module (`cross_entropy`, `conv2d`).
  - A *factory* returns `impl FnMut` closures with explicit seeds (`init`).
- Initialization and hyperparameters are caller-owned and visible at the call site: facades never choose distributions, learning rates, or float constants. Facade-internal constants are integer `counted` ratios only; arbitrary float constants stay caller territory.
- Facade state (running estimates, optimizer moments) lives in explicit caller-held structs, never hidden in the graph or in globals; `Field` is the designed carrier for value-aligned state.
- Every stateful or tie-breaking choice a facade makes (decay policy, tie direction, default constants) is documented on the item with its rationale, and any escape hatch is a parameter or predicate, not a fork of the facade.

# Rules for writing Rust tests
- Tests for a particular module (file) should be in a separate file (placed inside `tests` folder) with `_tests` suffix.
- The test file should be included in the code file as module `tests` with specified path to the test file, like this:
    ```rust
    #[cfg(test)]
    #[path = "tests/full_math_tests.rs"]
    mod tests;
    ```
- Note that in this model `*_tests.rs` files must *not* have another `test` module defined inside because it's redundant.
- Also note that we put all the test files into `tests` folder which is placed next to the source code file.

# Rules for Module Organization
- The folder is the module, not the individual files.
- Each file in the module folder contains one main concept/type.
- All files are declared as private modules in `mod.rs` using `mod filename;`.
- Public API is exposed through explicit `pub use` re-exports in `mod.rs`.
- The `mod.rs` file serves as the single public interface for the entire module.
- File names should be lowercase and match their main concept (e.g., `array.rs` for `StorageArray`).
- Re-exported items should use their proper type names (e.g., `pub use array::StorageArray;`).
- Files within the same module import each other using `super::` (e.g., `use super::primitives::*;`).
- For types from the same module, use `super::` even if they're re-exported in `mod.rs`.
- External dependencies use `crate::` or absolute paths.
- All types from within the crate should be imported at the beginning of the file using `use` statements; do not use fully qualified `crate::` paths inline in the code.
- Avoid glob imports (`use module::*`) from internal modules and external crates; import specific items explicitly instead.
- Organize imports in the following order, with each group separated by a blank line:
    1. Standard library (`std::`)
    2. External crates (third-party dependencies)
    3. Internal crate (`crate::`, `super::`, `self::`)
- Sort imports alphabetically within each group.
- Group multiple imports from the same module using braces (e.g., `use std::collections::{HashMap, HashSet};`).
