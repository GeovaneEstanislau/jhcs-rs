//! # Memory buffer abstractions and handles for heterogeneous compute.

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

/// Memory buffer ownership semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferOwnership {
    /// Buffer is owned exclusively by the current process or task.
    Owned,
    /// Buffer is temporarily borrowed (read or write slice).
    Borrowed,
    /// Buffer is shared across processes or execution units via zero-copy shmem.
    Shared,
}

/// Read/write access rights for a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferAccessMode {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// Synchronization and cache coherency status between host and accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferSyncState {
    /// Host and device representations are strictly identical.
    Clean,
    /// Modified locally; pending flush or device commit.
    Modified,
    /// Currently being transferred via DMA or GPU asynchronous pipeline.
    InFlight,
    /// Incoherent or invalidated; requires fresh re-fetch before access.
    Incoherent,
}

/// Universal trait representing a managed compute buffer without exposing raw pointers.
pub trait ComputeBuffer: Send + Sync {
    fn handle(&self) -> BufferHandle;
    fn size_bytes(&self) -> usize;
    fn precision(&self) -> Precision;
    fn location(&self) -> BufferLocation;
    fn alignment(&self) -> usize;
    fn ownership(&self) -> BufferOwnership;
    fn access_mode(&self) -> BufferAccessMode;
    fn sync_state(&self) -> BufferSyncState;
    fn is_valid(&self) -> bool;
}

/// Metadata description of a buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferDescriptor {
    pub handle: BufferHandle,
    pub size_bytes: usize,
    pub location: BufferLocation,
    pub precision: Precision,
    pub alignment: usize,
    pub ownership: BufferOwnership,
    pub access_mode: BufferAccessMode,
    pub sync_state: BufferSyncState,
    pub is_valid: bool,
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
            alignment: 64, // Default cache line / AVX512 alignment
            ownership: BufferOwnership::Owned,
            access_mode: BufferAccessMode::ReadWrite,
            sync_state: BufferSyncState::Clean,
            is_valid: true,
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
            alignment: 64,
            ownership: BufferOwnership::Owned,
            access_mode: BufferAccessMode::ReadWrite,
            sync_state: BufferSyncState::Clean,
            is_valid: true,
            shape: [rows, cols, 1, 1],
        }
    }

    pub fn with_alignment(mut self, align: usize) -> Self {
        self.alignment = align;
        self
    }

    pub fn with_ownership(mut self, ownership: BufferOwnership) -> Self {
        self.ownership = ownership;
        self
    }

    pub fn with_access_mode(mut self, mode: BufferAccessMode) -> Self {
        self.access_mode = mode;
        self
    }

    pub fn with_sync_state(mut self, state: BufferSyncState) -> Self {
        self.sync_state = state;
        self
    }
}

impl ComputeBuffer for BufferDescriptor {
    fn handle(&self) -> BufferHandle {
        self.handle
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    fn precision(&self) -> Precision {
        self.precision
    }

    fn location(&self) -> BufferLocation {
        self.location
    }

    fn alignment(&self) -> usize {
        self.alignment
    }

    fn ownership(&self) -> BufferOwnership {
        self.ownership
    }

    fn access_mode(&self) -> BufferAccessMode {
        self.access_mode
    }

    fn sync_state(&self) -> BufferSyncState {
        self.sync_state
    }

    fn is_valid(&self) -> bool {
        self.is_valid
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
