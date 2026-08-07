//! GPT-2's byte-level BPE tokenizer, built from the released
//! `vocab.json` and `merges.txt` with no dependencies.
//!
//! Text splits under GPT-2's pretokenizer rule (contractions, then
//! space-prefixed letter, number, and punctuation runs, then
//! whitespace, with a multi-space run yielding its last space to the
//! following word); each pretoken's UTF-8 bytes map through the
//! byte-to-unicode table into merge symbols; and the learned merges
//! apply lowest rank first until no adjacent pair remains. Decoding
//! is the reverse table walk.

use std::collections::HashMap;

use super::json::{Json, parse};

/// The tokenizer: vocabulary, merge ranks, and the byte tables.
pub struct Tokenizer {
    ids: HashMap<String, usize>,
    tokens: Vec<String>,
    ranks: HashMap<(String, String), usize>,
    byte_to_unicode: [char; 256],
    unicode_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    /// Builds the tokenizer from the released vocabulary and merges.
    pub fn new(vocabulary: &str, merges: &str) -> Self {
        let mut ids = HashMap::new();
        let mut tokens = vec![String::new(); 50257];
        if let Json::Object(fields) = parse(vocabulary) {
            for (token, id) in fields {
                let id = id.count();
                tokens[id] = token.clone();
                ids.insert(token, id);
            }
        } else {
            panic!("the vocabulary is a JSON object");
        }

        let mut ranks = HashMap::new();
        for (rank, line) in merges.lines().skip(1).enumerate() {
            if line.is_empty() {
                continue;
            }
            let (left, right) = line.split_once(' ').expect("merges pair per line");
            ranks.insert((left.to_string(), right.to_string()), rank);
        }

        // The byte-to-unicode table: printable latin bytes map to
        // themselves, every other byte to a codepoint above 255, so
        // merge symbols stay printable single characters.
        let mut byte_to_unicode = ['\0'; 256];
        let mut shifted = 0u32;
        for byte in 0..=255u8 {
            let printable = (b'!'..=b'~').contains(&byte)
                || (0xA1..=0xAC).contains(&byte)
                || (0xAE..=0xFF).contains(&byte);
            byte_to_unicode[byte as usize] = if printable {
                byte as char
            } else {
                shifted += 1;
                char::from_u32(255 + shifted).expect("shifted codepoint")
            };
        }
        let unicode_to_byte = byte_to_unicode
            .iter()
            .enumerate()
            .map(|(byte, &symbol)| (symbol, byte as u8))
            .collect();

        Self {
            ids,
            tokens,
            ranks,
            byte_to_unicode,
            unicode_to_byte,
        }
    }

    /// Encodes `text` into token ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut encoded = Vec::new();
        for pretoken in pretokenize(text) {
            let symbols: Vec<String> = pretoken
                .bytes()
                .map(|byte| self.byte_to_unicode[byte as usize].to_string())
                .collect();
            for symbol in self.merged(symbols) {
                encoded.push(
                    *self
                        .ids
                        .get(&symbol)
                        .unwrap_or_else(|| panic!("symbol `{symbol}` is not in the vocabulary")),
                );
            }
        }
        encoded
    }

    /// Decodes token ids back into text, replacing invalid UTF-8.
    pub fn decode(&self, ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.tokens[id].chars())
            .map(|symbol| self.unicode_to_byte[&symbol])
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Applies the learned merges to one pretoken's symbols, lowest
    /// rank first.
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

/// Splits `text` under GPT-2's pretokenizer rule.
fn pretokenize(text: &str) -> Vec<String> {
    let characters: Vec<char> = text.chars().collect();
    let mut pretokens = Vec::new();
    let mut at = 0;
    while at < characters.len() {
        // Contractions bind to the word before them.
        if characters[at] == '\'' {
            let rest: String = characters[at..].iter().take(3).collect();
            let contraction = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
                .iter()
                .filter(|c| rest.starts_with(**c))
                .max_by_key(|c| c.len());
            if let Some(contraction) = contraction {
                pretokens.push(contraction.to_string());
                at += contraction.chars().count();
                continue;
            }
        }
        // A single leading space joins the run that follows it.
        let lead = usize::from(characters[at] == ' ' && at + 1 < characters.len());
        let start = at + lead;
        if start < characters.len() {
            let class = |c: char| {
                if c.is_alphabetic() {
                    1
                } else if c.is_numeric() {
                    2
                } else if !c.is_whitespace() {
                    3
                } else {
                    0
                }
            };
            let kind = class(characters[start]);
            if kind != 0 {
                let mut end = start;
                while end < characters.len() && class(characters[end]) == kind {
                    end += 1;
                }
                pretokens.push(characters[at..end].iter().collect());
                at = end;
                continue;
            }
        }
        // Whitespace: a run followed by more text yields its last
        // space to the next pretoken; a trailing run stays whole.
        let mut end = at;
        while end < characters.len() && characters[end].is_whitespace() {
            end += 1;
        }
        if end > at {
            let held = usize::from(end < characters.len() && characters[end - 1] == ' ');
            if end - at - held > 0 {
                pretokens.push(characters[at..end - held].iter().collect());
            }
            // The held space fronts the next iteration's run.
            at = end - held;
            continue;
        }
        pretokens.push(characters[at].to_string());
        at += 1;
    }
    pretokens
}
