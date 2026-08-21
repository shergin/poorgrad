use super::backend::Backend;
use super::formula::{Formula, Precision};

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
/// to.
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

/// Both precisions, for the hardware kernels that take either.
const BOTH: Precisions = Precisions::Only(Precision::ALL);

/// `f32` only, for Metal — the API has no `f64`.
const F32_ONLY: Precisions = Precisions::Only(&[Precision::F32]);

impl Backend {
    /// The coverage matrix: whether this implementer has a kernel
    /// for the formula, at what bar, for which precisions.
    ///
    /// This is the single declared truth the stack reads — offer
    /// chains agree with it by test, the plan's election consults
    /// [`Backend::Fused`]'s column, emission consults
    /// [`Backend::StableHlo`]'s — and both matches are exhaustive on
    /// purpose: a new formula or implementer cannot compile until
    /// every new cell is decided.
    pub fn coverage(self, formula: Formula) -> Cell {
        match self {
            Backend::Accelerate => match formula {
                Formula::Gemm | Formula::Map => Cell::Serves {
                    bar: Bar::Envelope,
                    precisions: BOTH,
                },
                Formula::WindowProduct
                | Formula::ReduceWindow
                | Formula::BatchNormTraining
                | Formula::BatchNormInference => Cell::Absent,
            },
            Backend::Metal => match formula {
                Formula::Gemm | Formula::Map => Cell::Serves {
                    bar: Bar::Envelope,
                    precisions: F32_ONLY,
                },
                Formula::WindowProduct
                | Formula::ReduceWindow
                | Formula::BatchNormTraining
                | Formula::BatchNormInference => Cell::Absent,
            },
            Backend::Cuda => match formula {
                Formula::Gemm => Cell::Serves {
                    bar: Bar::Envelope,
                    precisions: BOTH,
                },
                // A cuda map would be PCIe-bound: copies alone sink
                // an elementwise pass.
                Formula::Map
                | Formula::WindowProduct
                | Formula::ReduceWindow
                | Formula::BatchNormTraining
                | Formula::BatchNormInference => Cell::Absent,
            },
            Backend::Simd => match formula {
                Formula::Gemm => Cell::Serves {
                    bar: Bar::Envelope,
                    precisions: BOTH,
                },
                // `matrixmultiply` is GEMM-only.
                Formula::Map
                | Formula::WindowProduct
                | Formula::ReduceWindow
                | Formula::BatchNormTraining
                | Formula::BatchNormInference => Cell::Absent,
            },
            Backend::Fused => match formula {
                // `windowed_product` computes through the gemm seam
                // in the recorded accumulation order: bit-identical
                // under both postures, proven by the plan snapshots.
                Formula::WindowProduct => Cell::Serves {
                    bar: Bar::BitIdentical,
                    precisions: Precisions::Any,
                },
                // The pool kernel waits on a profile (`max` is
                // associative, so it could clear even the
                // bit-identity bar); the batch-norm kernels would
                // reassociate reductions and arrive envelope-only.
                Formula::Gemm
                | Formula::Map
                | Formula::ReduceWindow
                | Formula::BatchNormTraining
                | Formula::BatchNormInference => Cell::Absent,
            },
            // The translation column is total: every formula lowers,
            // leaf entries as single operations and composed entries
            // as raised library calls, under the envelope bar —
            // nobody controls the foreign runtime's kernels.
            Backend::StableHlo => match formula {
                Formula::Gemm
                | Formula::Map
                | Formula::WindowProduct
                | Formula::ReduceWindow
                | Formula::BatchNormTraining
                | Formula::BatchNormInference => Cell::Serves {
                    bar: Bar::Envelope,
                    precisions: Precisions::Any,
                },
            },
        }
    }

    /// Whether this implementer has a kernel for the formula that
    /// accepts jobs at this precision — designed coverage, the same
    /// answer in every build; [`status`](Backend::status) answers
    /// the orthogonal availability question.
    pub fn serves(self, formula: Formula, precision: Precision) -> bool {
        self.coverage(formula).admits(precision)
    }

    /// How this implementer's kernels are reached.
    pub fn dispatch(self) -> Dispatch {
        match self {
            Backend::Accelerate | Backend::Metal | Backend::Cuda | Backend::Simd => {
                Dispatch::Offered
            }
            Backend::Fused => Dispatch::Elected,
            Backend::StableHlo => Dispatch::Translated,
        }
    }
}

#[cfg(test)]
#[path = "tests/coverage_tests.rs"]
mod tests;
