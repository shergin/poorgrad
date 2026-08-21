use super::formula::Precision;

/// The certification bar a kernel clears against the oracle.
///
/// Every implementer is a shortcut over the reference
/// implementation, and the bar states how faithful the shortcut is
/// proven to be. Admission is one comparison — a kernel serves a
/// run when its bar [`meets`](Bar::meets) the bar the run's
/// [`Numerics`](crate::Numerics) posture demands — so `Exact`
/// excluding reordering kernels is a consequence, not a special
/// case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bar {
    /// Certified to answer the reference implementation's exact
    /// bits, proven by differential test.
    BitIdentical,
    /// Certified to answer within the documented error envelope;
    /// the kernel may reorder floating-point math.
    Envelope,
}

impl Bar {
    /// Whether a kernel certified at `self` may serve where
    /// `required` is demanded: bit-identity serves everywhere, an
    /// envelope serves only envelope demands.
    pub fn meets(self, required: Bar) -> bool {
        match (self, required) {
            (Bar::BitIdentical, _) => true,
            (Bar::Envelope, Bar::Envelope) => true,
            (Bar::Envelope, Bar::BitIdentical) => false,
        }
    }
}

/// The forwarding precisions a cell's kernel accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precisions {
    /// Payload-generic: the kernel computes for every element type,
    /// as the in-crate fused kernels and the translation library do.
    Any,
    /// Bound to these forwarding precisions, as hardware kernels
    /// are.
    Only(&'static [Precision]),
}

impl Precisions {
    /// Whether the kernel accepts jobs at this precision.
    pub fn admit(self, precision: Precision) -> bool {
        match self {
            Precisions::Any => true,
            Precisions::Only(list) => list.contains(&precision),
        }
    }
}

/// One cell of the coverage matrix: whether an implementer has a
/// kernel for a formula, and under what terms.
///
/// A cell declares *may*; whether a kernel *will* take a concrete
/// job stays a run-time decline inside the offer (thresholds,
/// stride mappings, device presence). The reference implementation
/// is not a cell — it is the substrate every `Absent` answer falls
/// to. Each implementer declares its own row in its module through
/// the `Implementer` contract; the whole matrix answers through
/// [`Backend::coverage`](crate::Backend::coverage).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// A kernel or translation exists, certified at `bar`, for the
    /// precisions `precisions` admits.
    Serves {
        /// The certification bar the kernel clears.
        bar: Bar,
        /// The forwarding precisions the kernel accepts.
        precisions: Precisions,
    },
    /// No kernel; the formula computes in its composed form.
    Absent,
}

impl Cell {
    /// Whether a kernel exists at all.
    pub fn serves(self) -> bool {
        matches!(self, Cell::Serves { .. })
    }

    /// Whether a kernel exists and its bar meets the demand.
    pub fn serves_at(self, required: Bar) -> bool {
        match self {
            Cell::Serves { bar, .. } => bar.meets(required),
            Cell::Absent => false,
        }
    }

    /// Whether a kernel exists and accepts jobs at this precision.
    pub fn admits(self, precision: Precision) -> bool {
        match self {
            Cell::Serves { precisions, .. } => precisions.admit(precision),
            Cell::Absent => false,
        }
    }
}

/// How an implementer's kernels are reached: the execution-context
/// attribute that replaced the home/abroad dichotomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatch {
    /// Offered buffer jobs down a formula's chain at run time.
    Offered,
    /// Elected onto the plan at compile time, executing in-process.
    Elected,
    /// Elected at emission time, translating the group into a
    /// foreign module another runtime executes.
    Translated,
}

#[cfg(test)]
#[path = "tests/coverage_tests.rs"]
mod tests;
