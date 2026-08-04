//! The MNIST corpus: download-and-cache of the four IDX files, their
//! parsing, and a deterministic shuffle.
//!
//! Files land unpacked in `examples/mnist/data/` (about 55 MB) on the
//! first run and are reused afterwards. Two public mirrors are tried
//! in order; if both fail, the panic names the path to fill by hand.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Where the IDX files live, next to the example and git-ignored.
const DATA_DIR: &str = "examples/mnist/data";

/// The mirrors tried in order for each file.
const MIRRORS: [&str; 2] = [
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
];

/// One split of the corpus: normalized pixels and their labels.
pub struct Split {
    /// The images, `count * 28 * 28` pixels scaled to `0.0 ..= 1.0`.
    pub pixels: Vec<f32>,
    /// The digit labels, one per image.
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
    let train = split("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    let test = split("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    (train, test)
}

/// Loads one split from its image and label files.
fn split(images: &str, labels: &str) -> Split {
    let pixels = parse_images(&fetched(images));
    let labels = parse_labels(&fetched(labels));
    assert_eq!(
        pixels.len() / (28 * 28),
        labels.len(),
        "image and label counts disagree"
    );
    Split { pixels, labels }
}

/// Returns the bytes of `name`, downloading into the cache if absent.
fn fetched(name: &str) -> Vec<u8> {
    let path = PathBuf::from(DATA_DIR).join(name);
    if !path.exists() {
        fs::create_dir_all(DATA_DIR).expect("cannot create the MNIST data directory");
        download(name, &path);
    }
    fs::read(&path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

/// Downloads and unpacks `name.gz` to `path`, trying each mirror.
fn download(name: &str, path: &Path) {
    for mirror in MIRRORS {
        println!("downloading {name} from {mirror} ...");
        let command = format!(
            "curl -fsSL {mirror}/{name}.gz | gunzip > {path}.tmp && mv {path}.tmp {path}",
            path = path.display()
        );
        let status = Command::new("sh").arg("-c").arg(&command).status();
        if matches!(status, Ok(status) if status.success()) && path.exists() {
            return;
        }
    }
    panic!(
        "could not download {name}; place the unpacked IDX file at {} by hand",
        path.display()
    );
}

/// Reads the big-endian `u32` at `at`.
fn read_be_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(bytes[at..at + 4].try_into().expect("truncated IDX header"))
}

/// Parses an IDX image file into normalized pixels.
fn parse_images(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(read_be_u32(bytes, 0), 2051, "not an IDX image file");
    let count = read_be_u32(bytes, 4) as usize;
    let rows = read_be_u32(bytes, 8) as usize;
    let columns = read_be_u32(bytes, 12) as usize;
    assert_eq!((rows, columns), (28, 28), "unexpected MNIST geometry");
    let pixels = &bytes[16..];
    assert_eq!(pixels.len(), count * rows * columns, "truncated image file");
    pixels.iter().map(|&byte| byte as f32 / 255.0).collect()
}

/// Parses an IDX label file.
fn parse_labels(bytes: &[u8]) -> Vec<usize> {
    assert_eq!(read_be_u32(bytes, 0), 2049, "not an IDX label file");
    let count = read_be_u32(bytes, 4) as usize;
    let labels = &bytes[8..];
    assert_eq!(labels.len(), count, "truncated label file");
    labels.iter().map(|&byte| byte as usize).collect()
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
