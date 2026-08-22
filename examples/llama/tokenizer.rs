//! TinyLlama's SentencePiece-style BPE tokenizer, built from the
//! released `tokenizer.json`. The algorithm is hand-rolled and in
//! view; only the file's JSON syntax is read by `serde_json`.
//!
//! Encoding follows the release's own pipeline: the text normalizes
//! under the metaspace convention (a `U+2581` lower-block prepends,
//! and every space becomes one — there is no pretokenizer, the whole
//! text is one merge arena), the learned merges apply lowest rank
//! first over the character sequence, and any symbol left outside the
//! vocabulary falls back to its UTF-8 bytes' `<0xXX>` tokens.
//! Decoding is the reverse: pieces join as bytes so multi-byte
//! characters reassemble across byte tokens, the metaspace becomes a
//! space again, and the one prepended space strips.

use std::collections::HashMap;

use serde::Deserialize;

/// The metaspace character (`U+2581`) standing for a space inside
/// every piece.
const METASPACE: char = '\u{2581}';

/// The released `tokenizer.json`, reduced to the fields the algorithm
/// needs: the vocabulary and the ranked merges.
#[derive(Deserialize)]
struct File {
    model: Model,
}

/// The BPE model inside the tokenizer file.
#[derive(Deserialize)]
struct Model {
    vocab: HashMap<String, usize>,
    merges: Vec<String>,
}

/// The tokenizer: vocabulary and merge ranks.
pub struct Tokenizer {
    ids: HashMap<String, usize>,
    tokens: Vec<String>,
    ranks: HashMap<(String, String), usize>,
}

impl Tokenizer {
    /// Builds the tokenizer from the released `tokenizer.json` text.
    pub fn new(text: &str) -> Self {
        let file: File = serde_json::from_str(text).expect("the tokenizer file parses");
        let ids = file.model.vocab;
        let mut tokens = vec![String::new(); ids.len()];
        for (token, &id) in &ids {
            tokens[id] = token.clone();
        }

        let mut ranks = HashMap::new();
        for (rank, line) in file.model.merges.iter().enumerate() {
            let (left, right) = line.split_once(' ').expect("merges pair per entry");
            ranks.insert((left.to_string(), right.to_string()), rank);
        }

        Self { ids, tokens, ranks }
    }

    /// Encodes `text` into token ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let normalized = format!("{METASPACE}{text}").replace(' ', "\u{2581}");
        let symbols: Vec<String> = normalized
            .chars()
            .map(|character| character.to_string())
            .collect();
        let mut encoded = Vec::new();
        for symbol in self.merged(symbols) {
            match self.ids.get(&symbol) {
                Some(&id) => encoded.push(id),
                // Byte fallback: a symbol outside the vocabulary
                // encodes as its UTF-8 bytes' `<0xXX>` tokens.
                None => {
                    for byte in symbol.bytes() {
                        let name = format!("<0x{byte:02X}>");
                        encoded.push(
                            *self
                                .ids
                                .get(&name)
                                .expect("the byte tokens cover every byte"),
                        );
                    }
                }
            }
        }
        encoded
    }

    /// Decodes token ids back into text, stripping the space the
    /// normalizer prepended.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            self.emit(id, &mut bytes);
        }
        let text = String::from_utf8_lossy(&bytes);
        text.strip_prefix(' ').unwrap_or(&text).to_string()
    }

    /// Returns one token's printable piece, for streaming: the
    /// metaspace becomes a space and nothing strips.
    pub fn piece(&self, id: usize) -> String {
        let mut bytes = Vec::new();
        self.emit(id, &mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Appends the bytes of one token's piece: a `<0xXX>` token is its
    /// raw byte, any other token its text with metaspaces as spaces.
    fn emit(&self, id: usize, bytes: &mut Vec<u8>) {
        let token = &self.tokens[id];
        if let Some(byte) = fallback_byte(token) {
            bytes.push(byte);
            return;
        }
        for character in token.chars() {
            if character == METASPACE {
                bytes.push(b' ');
                continue;
            }
            let mut encoded = [0u8; 4];
            bytes.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
        }
    }

    /// Applies the learned merges to the symbols, lowest rank first.
    fn merged(&self, mut symbols: Vec<String>) -> Vec<String> {
        while symbols.len() > 1 {
            let best = (0..symbols.len() - 1)
                .filter_map(|at| {
                    self.ranks
                        .get(&(symbols[at].clone(), symbols[at + 1].clone()))
                        .map(|&rank| (rank, at))
                })
                .min();
            let Some((_, at)) = best else {
                break;
            };
            let joined = format!("{}{}", symbols[at], symbols[at + 1]);
            symbols.splice(at..at + 2, [joined]);
        }
        symbols
    }
}

/// Parses a `<0xXX>` byte-fallback token into its byte.
fn fallback_byte(token: &str) -> Option<u8> {
    let inner = token.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(inner, 16).ok()
}
