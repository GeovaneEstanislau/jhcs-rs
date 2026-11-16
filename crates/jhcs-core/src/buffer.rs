//! Memory buffer abstractions and handles for heterogeneous compute.

extern crate alloc;
use alloc::vec::Vec;

/// Opaque identifier for a memory buffer managed by JHCS or host platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferHandle(pub u32);

impl BufferHandle {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub const fn id(&self) -> u32 {
        self.0
    }
}

/// Physical or logical storage tier where a buffer resides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferLocation {
    /// Resides in CPU host RAM.
    HostMemory,
    /// Resides in GPU device VRAM or local accelerator memory.
    DeviceMemory,
    /// Resides in Unified / Zero-Copy shared memory (e.g. KDM / JDA Shmem / BAR MMIO).
    SharedUnifiedMemory,
}

/// Numerical precision or representation format of buffer elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    Float32,
    Float64,
    Int32,
    Uint8,
    RawBytes,
}

impl Precision {
    pub const fn element_size(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Int32 => 4,
            Self::Uint8 => 1,
            Self::RawBytes => 1,
        }
    }
}

/// Metadata description of a buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferDescriptor {
    pub handle: BufferHandle,
    pub size_bytes: usize,
    pub location: BufferLocation,
    pub precision: Precision,
    /// Optional dimension tensor shape: [rows, cols, depth, batch]
    pub shape: [usize; 4],
}

impl BufferDescriptor {
    pub const fn new(
        handle: BufferHandle,
        size_bytes: usize,
        location: BufferLocation,
        precision: Precision,
    ) -> Self {
        Self {
            handle,
            size_bytes,
            location,
            precision,
            shape: [size_bytes, 1, 1, 1],
        }
    }

    pub const fn with_2d_shape(
        handle: BufferHandle,
        rows: usize,
        cols: usize,
        location: BufferLocation,
        precision: Precision,
    ) -> Self {
        let size_bytes = rows * cols * precision.element_size();
        Self {
            handle,
            size_bytes,
            location,
            precision,
            shape: [rows, cols, 1, 1],
        }
    }
}

/// Collection of buffer handles associated with a task input or output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BufferSet {
    handles: Vec<BufferHandle>,
}

impl BufferSet {
    pub const fn new() -> Self {
        Self { handles: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            handles: Vec::with_capacity(capacity),
        }
    }

    pub fn from_handles(handles: &[BufferHandle]) -> Self {
        Self {
            handles: handles.to_vec(),
        }
    }

    pub fn add(&mut self, handle: BufferHandle) {
        if !self.contains(handle) {
            self.handles.push(handle);
        }
    }

    pub fn contains(&self, handle: BufferHandle) -> bool {
        self.handles.contains(&handle)
    }

    pub fn len(&self) -> usize {
        self.handles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    pub fn handles(&self) -> &[BufferHandle] {
        &self.handles
    }
}
