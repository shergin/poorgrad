use std::collections::HashMap;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// How many bytes of idle buffers the pool holds before returned
/// buffers are dropped instead of parked.
const CAPACITY_BYTES: usize = 256 * 1024 * 1024;

/// A size-classed free list of shared-mode buffers (tinygrad's
/// allocator lesson): buffer creation is the expensive part, so a
/// steady-state training loop reuses instead of allocating. Classes
/// are next-power-of-two byte sizes; eviction is by the capacity cap.
pub(super) struct Pool {
    state: Mutex<PoolState>,
}

struct PoolState {
    free: HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    held_bytes: usize,
}

impl Pool {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(PoolState {
                free: HashMap::new(),
                held_bytes: 0,
            }),
        }
    }

    /// Checks out a shared-mode buffer of at least `bytes`, reusing a
    /// parked one when its size class matches.
    pub(super) fn take(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let class = bytes.max(16).next_power_of_two();
        {
            let mut state = self.state.lock().expect("the buffer pool is poisoned");
            if let Some(buffer) = state.free.get_mut(&class).and_then(Vec::pop) {
                state.held_bytes -= class;
                return Ok(buffer);
            }
        }
        device
            .newBufferWithLength_options(class, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| format!("allocating a {class}-byte buffer failed"))
    }

    /// Returns a buffer to its size class, or drops it once the pool
    /// holds its capacity.
    pub(super) fn give(&self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        let class = buffer.length();
        let mut state = self.state.lock().expect("the buffer pool is poisoned");
        if state.held_bytes + class > CAPACITY_BYTES {
            return;
        }
        state.held_bytes += class;
        state.free.entry(class).or_default().push(buffer);
    }
}
