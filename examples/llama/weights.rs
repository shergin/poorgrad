//! The released checkpoints: download-and-cache of a family member's
//! safetensors shards and tokenizer, plus a hand-rolled reader for
//! the safetensors format — an 8-byte little-endian header length, a
//! JSON header mapping tensor names to dtype, shape, and byte
//! offsets, then the raw data section. Checkpoints in this family
//! ship as f32, bf16, or f16, so the reader widens all three element
//! encodings, and a checkpoint may span several shards, so the
//! reader streams: one shard's bytes resident at a time, each tensor
//! handed to the caller as it is widened.
//!
//! The cache lives outside the repository, under
//! `$XDG_CACHE_HOME/topos/<family>` (`~/.cache` by default), so
//! every checkout and worktree shares one download and git never
//! sees it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use serde::de::IgnoredAny;
use topos::Tensor;

use crate::family::Family;

/// Returns the cached text of `family`'s artifact `name`, downloading
/// on first use.
pub fn cached_text(family: &Family, name: &str) -> String {
    let path = cached(family, name);
    std::fs::read_to_string(&path).expect("the cached file reads")
}

/// Returns `family`'s cache directory, shared by every checkout and
/// worktree.
fn cache_directory(family: &Family) -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").expect("a home directory");
            Path::new(&home).join(".cache")
        });
    let directory = base.join("topos").join(family.name);
    std::fs::create_dir_all(&directory).expect("the cache directory exists");
    directory
}

/// Returns the cache path of `family`'s artifact `name`, downloading
/// on first use.
fn cached(family: &Family, name: &str) -> PathBuf {
    let path = cache_directory(family).join(name);
    if !path.exists() {
        download(family, name, &path);
    }
    path
}

/// Downloads `family`'s artifact `name` into the cache.
fn download(family: &Family, name: &str, path: &Path) {
    println!(
        "downloading {name} from {source} ...",
        source = family.source
    );
    let command = format!(
        "curl -fSL {source}/{name} -o {path}.tmp && mv {path}.tmp {path}",
        source = family.source,
        path = path.display()
    );
    let status = Command::new("sh").arg("-c").arg(&command).status();
    assert!(
        matches!(status, Ok(status) if status.success()) && path.exists(),
        "could not download {name}; place it at {} by hand",
        path.display()
    );
}

/// The checkpoint: the cached shard paths, read lazily so a 7B
/// restore never holds more than one shard's bytes.
pub struct Weights {
    shards: Vec<PathBuf>,
}

/// The safetensors header: one entry per tensor, plus the optional
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
    /// Ensures every shard of `family`'s checkpoint is cached,
    /// downloading on first use, and returns the handle.
    pub fn open(family: &Family) -> Self {
        Self {
            shards: family
                .shards
                .iter()
                .map(|name| cached(family, name))
                .collect(),
        }
    }

    /// Streams the checkpoint tensor by tensor: each shard's bytes
    /// are read, every tensor in it is widened to f32 and handed to
    /// `consume` under its released name, and the bytes drop before
    /// the next shard loads.
    pub fn for_each(&self, mut consume: impl FnMut(&str, Tensor<f32>)) {
        for shard in &self.shards {
            let bytes = std::fs::read(shard).expect("the checkpoint shard reads");
            let header_length =
                u64::from_le_bytes(bytes[..8].try_into().expect("the header length")) as usize;
            let header: Header = serde_json::from_slice(&bytes[8..8 + header_length])
                .expect("the header describes every tensor");
            let data = &bytes[8 + header_length..];
            for (name, entry) in header.entries {
                let [start, end] = entry.data_offsets;
                let elements = widened(&entry.dtype, &data[start..end]);
                consume(&name, Tensor::new(entry.shape, elements));
            }
        }
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
