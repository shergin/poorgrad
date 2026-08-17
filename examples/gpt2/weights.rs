//! The released GPT-2 (124M) checkpoint: download-and-cache of the
//! safetensors file and the tokenizer pair, plus a hand-rolled reader
//! for the safetensors format — an 8-byte little-endian header
//! length, a JSON header mapping tensor names to dtype, shape, and
//! byte offsets, then the raw data section.
//!
//! The cache lives outside the repository, under
//! `$XDG_CACHE_HOME/topos/gpt2` (`~/.cache` by default), so every
//! checkout and worktree shares one 548 MB download and git never
//! sees it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use serde::de::IgnoredAny;
use topos::Tensor;

/// The published artifacts of the 124M model.
const SOURCE: &str = "https://huggingface.co/openai-community/gpt2/resolve/main";

/// Returns the cached text of `name`, downloading on first use.
pub fn cached_text(name: &str) -> String {
    let path = cached(name);
    std::fs::read_to_string(&path).expect("the cached file reads")
}

/// Returns the machine-level cache directory, shared by every
/// checkout and worktree.
pub fn cache_directory() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").expect("a home directory");
            Path::new(&home).join(".cache")
        });
    let directory = base.join("topos").join("gpt2");
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
            assert_eq!(entry.dtype, "F32", "the 124M release is f32");
            let [start, end] = entry.data_offsets;
            let elements: Vec<f32> = data[start..end]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
                .collect();
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
