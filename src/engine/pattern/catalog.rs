use super::candidates::Candidates;
use super::pattern::Pattern;

/// One consumer's election over the plan's candidate pool: the pattern
/// this consumer acts on at each root, and the skip mask of interior
/// and named nodes those actions replace.
///
/// A catalog serves exactly one consumer. The home consumer (a
/// forward run) elects the patterns it fuses; the emission consumer
/// elects the patterns it raises. Electing is claiming: entries never
/// overlap within one catalog, while two catalogs may elect the same
/// pool differently.
#[derive(Debug, Clone)]
pub(crate) struct Catalog {
    at: Vec<Option<Pattern>>,
    interior: Vec<bool>,
}

impl Catalog {
    /// Elects a consumer's catalog from the pool: walks the candidates
    /// in priority order and claims, first-wins, every candidate the
    /// repertoire supports whose nodes are all still free.
    ///
    /// `supports` is the consumer's repertoire — the patterns it can
    /// act on. Unsupported candidates do not claim, so their regions
    /// stay free for later supported candidates; an unelected region
    /// simply runs or lowers its recorded primitives, which is always
    /// sound. Electing an entry commits the consumer to producing the
    /// root and every named result while skipping the whole claim set.
    pub(crate) fn elect(candidates: &Candidates, supports: impl Fn(&Pattern) -> bool) -> Self {
        let length = candidates.length();
        let mut catalog = Self {
            at: vec![None; length],
            interior: vec![false; length],
        };
        let mut claimed = vec![false; length];
        for candidate in candidates.iter() {
            if !supports(&candidate.pattern) || claimed[candidate.root] {
                continue;
            }
            if candidate
                .interiors
                .iter()
                .chain(candidate.named.iter())
                .any(|&node| claimed[node])
            {
                continue;
            }
            claimed[candidate.root] = true;
            for &node in candidate.interiors.iter().chain(candidate.named.iter()) {
                claimed[node] = true;
                catalog.interior[node] = true;
            }
            catalog.at[candidate.root] = Some(candidate.pattern.clone());
        }
        catalog
    }

    /// Returns the pattern this consumer acts on at `index`, if any.
    /// Consumers read elected entries and never rematch.
    pub(crate) fn at(&self, index: usize) -> Option<&Pattern> {
        self.at[index].as_ref()
    }

    /// Returns whether this consumer's actions replace node `index`:
    /// an interior or named result of some elected entry.
    pub(crate) fn interior(&self, index: usize) -> bool {
        self.interior[index]
    }

    /// Returns how many entries this consumer elected.
    pub(crate) fn groups(&self) -> usize {
        self.at.iter().flatten().count()
    }

    /// Pins `last_consumer` so the reads of a fusing entry outlive the
    /// skipped chain. Only the home catalog needs this: raising
    /// consumers never touch liveness, since their chains actually
    /// run.
    pub(crate) fn pin_liveness(&self, last_consumer: &mut [Option<usize>]) {
        for (index, pattern) in self.at.iter().enumerate() {
            let Some(group) = pattern.as_ref().and_then(Pattern::fused) else {
                continue;
            };
            for slot in group.reads() {
                let latest = last_consumer[slot].unwrap_or(0).max(index);
                last_consumer[slot] = Some(latest);
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/catalog_tests.rs"]
mod tests;
