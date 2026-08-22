use super::formula::Precision;

/// The certified fidelity a kernel meets against the oracle.
///
/// Every implementer is a shortcut over the reference
/// implementation, and its fidelity states how faithful the shortcut is
/// proven to be. Admission is one comparison — a kernel serves a
/// run when its fidelity [`meets`](Fidelity::meets) the fidelity the run's
/// [`Numerics`](crate::Numerics) posture demands — so `Exact`
/// excluding reordering kernels is a consequence, not a special
/// case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fidelity {
    /// Certified to answer the reference implementation's exact
    /// bits, proven by differential test.
    BitIdentical,
    /// Certified to answer within the documented error envelope;
    /// the kernel may reorder floating-point math.
    Envelope,
}

impl Fidelity {
    /// Whether a kernel certified at `self` may serve where
    /// `required` is demanded: bit-identity serves everywhere, an
    /// envelope serves only envelope demands.
    pub fn meets(self, required: Fidelity) -> bool {
        match (self, required) {
            (Fidelity::BitIdentical, _) => true,
            (Fidelity::Envelope, Fidelity::Envelope) => true,
            (Fidelity::Envelope, Fidelity::BitIdentical) => false,
        }
    }
}

/// The forwarding precisions a covered kernel accepts.
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
    /// Whether the kernel accepts tasks at this precision.
    pub fn admit(self, precision: Precision) -> bool {
        match self {
            Precisions::Any => true,
            Precisions::Only(list) => list.contains(&precision),
        }
    }
}

/// One backend's coverage of one formula: whether it has a kernel,
/// and under what terms.
///
/// Coverage declares *may*; whether a kernel *will* take a concrete
/// task stays a run-time decline inside the offer (thresholds,
/// stride mappings, device presence). The reference implementation
/// has no coverage — it is the substrate every `Absent` answer
/// falls to. Each backend declares its own coverage in its module
/// through the `Manifest` contract; the whole matrix answers
/// through [`Backend::coverage`](crate::Backend::coverage).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coverage {
    /// A kernel or translation exists, certified at a fidelity, for the
    /// precisions `precisions` admits.
    Serves {
        /// The certified fidelity the kernel meets.
        fidelity: Fidelity,
        /// The forwarding precisions the kernel accepts.
        precisions: Precisions,
    },
    /// No kernel; the formula computes in its composed form.
    Absent,
}

impl Coverage {
    /// Whether a kernel exists at all.
    pub fn serves(self) -> bool {
        matches!(self, Coverage::Serves { .. })
    }

    /// Whether a kernel exists and its fidelity meets the demand.
    pub fn meets(self, required: Fidelity) -> bool {
        match self {
            Coverage::Serves { fidelity, .. } => fidelity.meets(required),
            Coverage::Absent => false,
        }
    }

    /// Whether a kernel exists and accepts tasks at this precision.
    pub fn admits(self, precision: Precision) -> bool {
        match self {
            Coverage::Serves { precisions, .. } => precisions.admit(precision),
            Coverage::Absent => false,
        }
    }
}

/// How a backend's kernels are reached: the execution-context
/// attribute that replaced the home/abroad dichotomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatch {
    /// Offered buffer tasks down a formula's chain at run time.
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
