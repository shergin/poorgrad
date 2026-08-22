//! The released TinyLlama (1.1B) checkpoint: download-and-cache of
//! the safetensors file and the tokenizer, plus a hand-rolled reader
//! for the safetensors format — an 8-byte little-endian header
//! length, a JSON header mapping tensor names to dtype, shape, and
//! byte offsets, then the raw data section. Unlike GPT-2's f32-only
//! release, checkpoints in this family ship as f32, bf16, or f16, so
//! the reader widens all three element encodings.
//!
//! The cache lives outside the repository, under
//! `$XDG_CACHE_HOME/topos/tinyllama` (`~/.cache` by default), so
//! every checkout and worktree shares one 4.1 GB download and git
//! never sees it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use serde::de::IgnoredAny;
use topos::Tensor;

/// The published artifacts of the 1.1B base model (the 3T-token
/// intermediate release, the final base checkpoint of the run).
const SOURCE: &str =
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/resolve/main";

/// Returns the cached text of `name`, downloading on first use.
pub fn cached_text(name: &str) -> String {
    let path = cached(name);
    std::fs::read_to_string(&path).expect("the cached file reads")
}

/// Returns the machine-level cache directory, shared by every
/// checkout and worktree.
fn cache_directory() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").expect("a home directory");
            Path::new(&home).join(".cache")
        });
    let directory = base.join("topos").join("tinyllama");
    std::fs::create_dir_all(&directory).expect("the cache directory exists");
    directory
}

/// Returns the cache path of `name`, downloading on first use.
fn cached(name: &str) -> PathBuf {
    let path = cache_directory().join(name);
    if !path.exists() {
        download(name, &path);
    }
    path
}

/// Downloads `name` into the cache.
fn download(name: &str, path: &Path) {
    println!("downloading {name} from {SOURCE} ...");
    let command = format!(
        "curl -fSL {SOURCE}/{name} -o {path}.tmp && mv {path}.tmp {path}",
        path = path.display()
    );
    let status = Command::new("sh").arg("-c").arg(&command).status();
    assert!(
        matches!(status, Ok(status) if status.success()) && path.exists(),
        "could not download {name}; place it at {} by hand",
        path.display()
    );
}

/// The checkpoint: every tensor by its released name.
pub struct Weights {
    tensors: HashMap<String, Tensor<f32>>,
}

/// The safetensors header: one entry per tensor, plus the release's
/// `__metadata__` field, which is not a tensor and is discarded.
#[derive(Deserialize)]
struct Header {
    #[serde(rename = "__metadata__")]
    _metadata: Option<IgnoredAny>,
    #[serde(flatten)]
    entries: HashMap<String, Entry>,
}

/// One tensor's entry in the safetensors header.
#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl Weights {
    /// Loads the checkpoint, downloading on first use.
    pub fn load() -> Self {
        let path = cached("model.safetensors");
        let bytes = std::fs::read(&path).expect("the checkpoint reads");
        let header_length =
            u64::from_le_bytes(bytes[..8].try_into().expect("the header length")) as usize;
        let header: Header = serde_json::from_slice(&bytes[8..8 + header_length])
            .expect("the header describes every tensor");
        let data = &bytes[8 + header_length..];

        let mut tensors = HashMap::new();
        for (name, entry) in header.entries {
            let [start, end] = entry.data_offsets;
            let elements = widened(&entry.dtype, &data[start..end]);
            tensors.insert(name, Tensor::new(entry.shape, elements));
        }
        Self { tensors }
    }

    /// Returns the named tensor, panicking when absent.
    pub fn tensor(&self, name: &str) -> Tensor<f32> {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("the checkpoint has no tensor `{name}`"))
            .clone()
    }
}

/// Widens one tensor's raw little-endian `bytes` of `dtype` into f32
/// elements.
fn widened(dtype: &str, bytes: &[u8]) -> Vec<f32> {
    let halves = || {
        bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("two bytes")))
    };
    match dtype {
        "F32" => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect(),
        // A bf16 value is an f32 with the low sixteen mantissa bits
        // dropped, so widening is one shift.
        "BF16" => halves()
            .map(|bits| f32::from_bits((bits as u32) << 16))
            .collect(),
        "F16" => halves().map(from_half).collect(),
        other => panic!("the reader does not widen dtype `{other}`"),
    }
}

/// Widens one IEEE 754 binary16 value to f32, by field arithmetic
/// rather than bit surgery: five exponent bits biased by 15 around a
/// ten-bit mantissa.
fn from_half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0f32 } else { -1.0 };
    let exponent = ((bits >> 10) & 0x1F) as i32;
    let mantissa = bits & 0x03FF;
    match (exponent, mantissa) {
        // Subnormals scale the bare mantissa by the smallest exponent.
        (0, _) => sign * mantissa as f32 * (2.0f32).powi(-24),
        (0x1F, 0) => sign * f32::INFINITY,
        (0x1F, _) => f32::NAN,
        _ => sign * (1.0 + mantissa as f32 / 1024.0) * (2.0f32).powi(exponent - 15),
    }
}
