//! The released Llama-family checkpoints this example generates with:
//! one descriptor per model, carrying where its artifacts download
//! from and the dimensions that shape the module tree. The
//! architecture is one; a family member is data.

/// One released checkpoint: its download source, its safetensors
/// shards, and the dimensions the module tree is built from.
#[derive(Clone, Copy)]
pub struct Family {
    /// The cache directory name under `~/.cache/topos`.
    pub name: &'static str,
    /// The Hugging Face `resolve/main` base URL.
    pub source: &'static str,
    /// The safetensors file names, in release order; a single-file
    /// checkpoint is a one-shard list.
    pub shards: &'static [&'static str],
    /// How many dimensions the residual stream has.
    pub embed_dim: usize,
    /// How many query heads split the stream.
    pub head_count: usize,
    /// How many key/value heads the query heads share, in groups;
    /// equal to `head_count` for plain multi-head attention.
    pub key_value_head_count: usize,
    /// How many dimensions the MLP's hidden layer has.
    pub hidden_dim: usize,
    /// How many transformer blocks the model stacks.
    pub layer_count: usize,
}

impl Family {
    /// Returns how many dimensions each head reads and writes.
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.head_count
    }

    /// Returns how many query heads read each key/value head.
    pub fn group_size(&self) -> usize {
        self.head_count / self.key_value_head_count
    }
}

/// TinyLlama 1.1B, the base release at three trillion tokens: an f32
/// single-file checkpoint (4.1 GB) with grouped-query attention.
pub const TINYLLAMA: Family = Family {
    name: "tinyllama",
    source: "https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/resolve/main",
    shards: &["model.safetensors"],
    embed_dim: 2048,
    head_count: 32,
    key_value_head_count: 4,
    hidden_dim: 5632,
    layer_count: 22,
};

/// Llama 2 7B, Meta's released base model through an ungated mirror
/// of the converted layout: an f16 two-shard checkpoint (13.5 GB)
/// with plain multi-head attention.
pub const LLAMA2_7B: Family = Family {
    name: "llama2-7b",
    source: "https://huggingface.co/NousResearch/Llama-2-7b-hf/resolve/main",
    shards: &[
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ],
    embed_dim: 4096,
    head_count: 32,
    key_value_head_count: 32,
    hidden_dim: 11008,
    layer_count: 32,
};
