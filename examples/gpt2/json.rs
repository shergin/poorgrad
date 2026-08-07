//! A minimal JSON reader for the example's two fixed inputs — the
//! GPT-2 vocabulary and the safetensors header — kept dependency-free
//! the way the IDX and CIFAR readers are. It parses the full JSON
//! grammar (escapes and surrogate pairs included) but trades graceful
//! error recovery for panics, which is the right posture for cached
//! artifacts whose shape is known.

use std::collections::HashMap;

/// One parsed JSON value.
#[derive(Debug, Clone)]
pub enum Json {
    Null,
    Bool,
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(HashMap<String, Json>),
}

impl Json {
    /// Returns the object's field, panicking when absent.
    pub fn field(&self, name: &str) -> &Json {
        match self {
            Json::Object(fields) => fields
                .get(name)
                .unwrap_or_else(|| panic!("JSON object has no field `{name}`")),
            _ => panic!("expected a JSON object"),
        }
    }

    /// Returns the object's fields.
    pub fn fields(&self) -> &HashMap<String, Json> {
        match self {
            Json::Object(fields) => fields,
            _ => panic!("expected a JSON object"),
        }
    }

    /// Returns the array's elements.
    pub fn elements(&self) -> &[Json] {
        match self {
            Json::Array(elements) => elements,
            _ => panic!("expected a JSON array"),
        }
    }

    /// Returns the number as a `usize`.
    pub fn count(&self) -> usize {
        match self {
            Json::Number(value) => *value as usize,
            _ => panic!("expected a JSON number"),
        }
    }

    /// Returns the string's text.
    pub fn text(&self) -> &str {
        match self {
            Json::String(text) => text,
            _ => panic!("expected a JSON string"),
        }
    }
}

/// Parses one JSON document.
pub fn parse(source: &str) -> Json {
    let mut reader = Reader {
        bytes: source.as_bytes(),
        at: 0,
    };
    reader.skip_whitespace();
    let value = reader.value();
    reader.skip_whitespace();
    assert!(reader.at == reader.bytes.len(), "trailing JSON content");
    value
}

/// The parser's cursor over the document bytes.
struct Reader<'source> {
    bytes: &'source [u8],
    at: usize,
}

impl Reader<'_> {
    fn peek(&self) -> u8 {
        self.bytes[self.at]
    }

    fn advance(&mut self) -> u8 {
        let byte = self.bytes[self.at];
        self.at += 1;
        byte
    }

    fn skip_whitespace(&mut self) {
        while self.at < self.bytes.len() && self.bytes[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }

    fn expect(&mut self, byte: u8) {
        assert_eq!(self.advance(), byte, "malformed JSON near byte {}", self.at);
    }

    fn literal(&mut self, text: &str) {
        for &byte in text.as_bytes() {
            self.expect(byte);
        }
    }

    fn value(&mut self) -> Json {
        match self.peek() {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Json::String(self.string()),
            b't' => {
                self.literal("true");
                Json::Bool
            }
            b'f' => {
                self.literal("false");
                Json::Bool
            }
            b'n' => {
                self.literal("null");
                Json::Null
            }
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Json {
        self.expect(b'{');
        let mut fields = HashMap::new();
        self.skip_whitespace();
        if self.peek() == b'}' {
            self.advance();
            return Json::Object(fields);
        }
        loop {
            self.skip_whitespace();
            let name = self.string();
            self.skip_whitespace();
            self.expect(b':');
            self.skip_whitespace();
            fields.insert(name, self.value());
            self.skip_whitespace();
            match self.advance() {
                b',' => continue,
                b'}' => return Json::Object(fields),
                _ => panic!("malformed JSON object near byte {}", self.at),
            }
        }
    }

    fn array(&mut self) -> Json {
        self.expect(b'[');
        let mut elements = Vec::new();
        self.skip_whitespace();
        if self.peek() == b']' {
            self.advance();
            return Json::Array(elements);
        }
        loop {
            self.skip_whitespace();
            elements.push(self.value());
            self.skip_whitespace();
            match self.advance() {
                b',' => continue,
                b']' => return Json::Array(elements),
                _ => panic!("malformed JSON array near byte {}", self.at),
            }
        }
    }

    fn string(&mut self) -> String {
        self.expect(b'"');
        let mut text = String::new();
        loop {
            match self.advance() {
                b'"' => return text,
                b'\\' => match self.advance() {
                    b'"' => text.push('"'),
                    b'\\' => text.push('\\'),
                    b'/' => text.push('/'),
                    b'b' => text.push('\u{8}'),
                    b'f' => text.push('\u{c}'),
                    b'n' => text.push('\n'),
                    b'r' => text.push('\r'),
                    b't' => text.push('\t'),
                    b'u' => {
                        let unit = self.code_unit();
                        // A high surrogate must pair with the following
                        // escaped low surrogate.
                        let scalar = if (0xD800..0xDC00).contains(&unit) {
                            self.expect(b'\\');
                            self.expect(b'u');
                            let low = self.code_unit();
                            0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                        } else {
                            unit
                        };
                        text.push(char::from_u32(scalar).expect("valid JSON escape"));
                    }
                    _ => panic!("unknown JSON escape near byte {}", self.at),
                },
                byte => {
                    // Multi-byte UTF-8 passes through verbatim; the
                    // source is a `str`, so the bytes are valid.
                    let start = self.at - 1;
                    let width = match byte {
                        0x00..=0x7F => 1,
                        0xC0..=0xDF => 2,
                        0xE0..=0xEF => 3,
                        _ => 4,
                    };
                    self.at = start + width;
                    text.push_str(std::str::from_utf8(&self.bytes[start..self.at]).unwrap());
                }
            }
        }
    }

    fn code_unit(&mut self) -> u32 {
        let mut unit = 0;
        for _ in 0..4 {
            let digit = (self.advance() as char)
                .to_digit(16)
                .expect("hex digit in JSON escape");
            unit = unit * 16 + digit;
        }
        unit
    }

    fn number(&mut self) -> Json {
        let start = self.at;
        while self.at < self.bytes.len()
            && matches!(
                self.bytes[self.at],
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
            )
        {
            self.at += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).unwrap();
        Json::Number(text.parse().expect("valid JSON number"))
    }
}
