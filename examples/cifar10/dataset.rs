//! The CIFAR-10 corpus: download-and-cache of the binary batches,
//! parsing, per-channel standardization, and a deterministic shuffle.
//!
//! The archive (~170 MB) lands untarred in `examples/cifar10/data/` on
//! the first run and is reused afterwards. Records are 3073 bytes: a
//! label byte, then 3072 channel-planar pixels — exactly the
//! `[3, 32, 32]` layout the network wants, so parsing is a copy.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Where the binary batches live, next to the example and git-ignored.
const DATA_DIR: &str = "examples/cifar10/data";

/// The canonical archive.
const ARCHIVE: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

/// The unpacked batch directory inside the archive.
const BATCHES: &str = "cifar-10-batches-bin";

/// How many pixels one image holds: three channel planes of `32 x 32`.
const PIXELS: usize = 3 * 32 * 32;

/// The per-channel means of the training corpus, the standard constants.
const CHANNEL_MEAN: [f32; 3] = [0.4914, 0.4822, 0.4465];

/// The per-channel deviations of the training corpus.
const CHANNEL_DEVIATION: [f32; 3] = [0.2470, 0.2435, 0.2616];

/// One split of the corpus: standardized pixels and their labels.
pub struct Split {
    /// The images, `count * 3 * 32 * 32` standardized values in
    /// channel-planar order.
    pub pixels: Vec<f32>,
    /// The class labels, one per image.
    pub labels: Vec<usize>,
}

impl Split {
    /// Returns the number of images in the split.
    pub fn len(&self) -> usize {
        self.labels.len()
    }
}

/// Loads the training and test splits, downloading on first use.
pub fn load() -> (Split, Split) {
    fetch();
    let mut train = Split {
        pixels: Vec::new(),
        labels: Vec::new(),
    };
    for batch in 1..=5 {
        parse_into(&mut train, &format!("data_batch_{batch}.bin"));
    }
    let mut test = Split {
        pixels: Vec::new(),
        labels: Vec::new(),
    };
    parse_into(&mut test, "test_batch.bin");
    (train, test)
}

/// Downloads and unpacks the archive unless the batches are present.
fn fetch() {
    let marker = PathBuf::from(DATA_DIR).join(BATCHES).join("test_batch.bin");
    if marker.exists() {
        return;
    }
    fs::create_dir_all(DATA_DIR).expect("cannot create the CIFAR-10 data directory");
    println!("downloading CIFAR-10 (~170 MB) from {ARCHIVE} ...");
    let command = format!("curl -fsSL {ARCHIVE} | tar -xz -C {DATA_DIR}");
    let status = Command::new("sh").arg("-c").arg(&command).status();
    assert!(
        matches!(status, Ok(status) if status.success()) && marker.exists(),
        "could not download CIFAR-10; untar {ARCHIVE} into {DATA_DIR} by hand"
    );
}

/// Parses one binary batch file into `split`, standardizing pixels.
fn parse_into(split: &mut Split, name: &str) {
    let path = PathBuf::from(DATA_DIR).join(BATCHES).join(name);
    let bytes =
        fs::read(&path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    assert_eq!(bytes.len() % (PIXELS + 1), 0, "truncated batch {name}");
    for record in bytes.chunks_exact(PIXELS + 1) {
        split.labels.push(record[0] as usize);
        for (position, &byte) in record[1..].iter().enumerate() {
            let channel = position / (32 * 32);
            let value = byte as f32 / 255.0;
            split
                .pixels
                .push((value - CHANNEL_MEAN[channel]) / CHANNEL_DEVIATION[channel]);
        }
    }
}

/// Shuffles `indices` in place, Fisher-Yates over a splitmix64 stream,
/// so runs stay reproducible.
pub fn shuffle(indices: &mut [usize], state: &mut u64) {
    for i in (1..indices.len()).rev() {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        indices.swap(i, (z as usize) % (i + 1));
    }
}
