//! The shared makemore machinery: the names corpus, its token alphabet,
//! windowed training samples, and the seeded helpers for shuffling and
//! sampling that every makemore example uses.

use std::iter;

/// Stands in for the characters outside a name: the left padding of the
/// first contexts and the terminator that ends every name.
pub const PADDING: char = '.';

/// How many distinct tokens the corpus uses: `PADDING` plus the
/// lowercase ASCII alphabet.
pub const VOCABULARY_LEN: usize = 27;

/// Loads the training corpus, one name per line, embedded at compile
/// time so the examples run from any working directory.
pub fn load_names() -> Vec<&'static str> {
    include_str!("names.txt")
        .lines()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect()
}

/// Maps a character to its token: `PADDING` is `0` and `'a'..='z'` are
/// `1..=26`, so a token doubles as a row index into an embedding or
/// bigram table.
///
/// # Panics
/// Panics if `character` is neither `PADDING` nor a lowercase ASCII
/// letter.
pub fn to_token(character: char) -> usize {
    if character == PADDING {
        return 0;
    }
    assert!(
        character.is_ascii_lowercase(),
        "the corpus holds lowercase ASCII names only, got {character:?}"
    );
    character as usize - 'a' as usize + 1
}

/// Maps a token back to the character it stands for, inverting
/// [`to_token`].
///
/// # Panics
/// Panics if `token` is not below `VOCABULARY_LEN`.
pub fn from_token(token: usize) -> char {
    if token == 0 {
        return PADDING;
    }
    assert!(
        token < VOCABULARY_LEN,
        "the vocabulary holds {VOCABULARY_LEN} tokens, got {token}"
    );
    (b'a' + (token - 1) as u8) as char
}

/// Turns every name into training samples over its padded form: each
/// window of `CONTEXT_LEN` tokens predicts the token that follows it,
/// with `PADDING` filling the first contexts and terminating the name.
/// The bigram model is the `CONTEXT_LEN = 1` case.
pub fn training_samples<const CONTEXT_LEN: usize>(
    names: &[&str],
) -> Vec<([usize; CONTEXT_LEN], usize)> {
    let mut samples = Vec::new();
    for name in names {
        let tokens: Vec<usize> = iter::repeat_n(0, CONTEXT_LEN)
            .chain(name.chars().map(to_token))
            .chain(iter::once(0))
            .collect();
        for window in tokens.windows(CONTEXT_LEN + 1) {
            let (context, next) = window.split_at(CONTEXT_LEN);
            samples.push((
                context.try_into().expect("window has context length"),
                next[0],
            ));
        }
    }
    samples
}

/// Advances `state` and returns the next value uniformly distributed in
/// `[0, 1)`.
///
/// Run-time randomness (batch order, sampling) is data rather than
/// initialization, so the examples carry their own tiny splitmix64 the
/// same way `init` does, keeping runs reproducible.
pub fn unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^= mixed >> 31;
    (mixed >> 11) as f64 / (1u64 << 53) as f64
}

/// Draws one token from a probability `row` by walking its cumulative
/// distribution.
pub fn draw(row: &[f64], state: &mut u64) -> usize {
    let mut threshold = unit(state);
    for (token, probability) in row.iter().enumerate() {
        if threshold < *probability {
            return token;
        }
        threshold -= probability;
    }
    row.len() - 1
}

/// Shuffles `samples` in place with a seeded Fisher-Yates walk, so
/// minibatches mix the frequency-sorted corpus instead of feeding
/// correlated neighbors.
pub fn shuffle<T>(samples: &mut [T], state: &mut u64) {
    for index in (1..samples.len()).rev() {
        let other = (unit(state) * (index + 1) as f64) as usize;
        samples.swap(index, other);
    }
}
