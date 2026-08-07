//! The released GPT-2 (124M) checkpoint: download-and-cache of the
//! safetensors file and the tokenizer pair, plus a dependency-free
//! reader for the safetensors format — an 8-byte little-endian header
//! length, a JSON header mapping tensor names to dtype, shape, and
//! byte offsets, then the raw data section.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use poorgrad::Tensor;

use super::json::parse;

/// The published artifacts of the 124M model.
const SOURCE: &str = "https://huggingface.co/openai-community/gpt2/resolve/main";

/// Returns the cached text of `name`, downloading on first use.
pub fn cached_text(name: &str) -> String {
    let path = cached(name);
    std::fs::read_to_string(&path).expect("the cached file reads")
}

/// Returns the cache path of `name`, downloading on first use.
fn cached(name: &str) -> PathBuf {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("gpt2")
        .join("data");
    std::fs::create_dir_all(&directory).expect("the cache directory exists");
    let path = directory.join(name);
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

impl Weights {
    /// Loads the checkpoint, downloading on first use.
    pub fn load() -> Self {
        let path = cached("model.safetensors");
        let bytes = std::fs::read(&path).expect("the checkpoint reads");
        let header_length =
            u64::from_le_bytes(bytes[..8].try_into().expect("the header length")) as usize;
        let header =
            parse(std::str::from_utf8(&bytes[8..8 + header_length]).expect("the header is UTF-8"));
        let data = &bytes[8 + header_length..];

        let mut tensors = HashMap::new();
        for (name, entry) in header.fields() {
            if name == "__metadata__" {
                continue;
            }
            assert_eq!(
                entry.field("dtype").text(),
                "F32",
                "the 124M release is f32"
            );
            let shape: Vec<usize> = entry
                .field("shape")
                .elements()
                .iter()
                .map(|extent| extent.count())
                .collect();
            let offsets = entry.field("data_offsets").elements();
            let (start, end) = (offsets[0].count(), offsets[1].count());
            let elements: Vec<f32> = data[start..end]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
                .collect();
            tensors.insert(name.clone(), Tensor::new(shape, elements));
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
