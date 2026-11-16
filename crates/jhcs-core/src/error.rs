//! Error types for the JHCS core domain.

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    /// A cycle was detected in the compute graph.
    CycleDetected,
    /// A referenced buffer was not found in the registry or set.
    BufferNotFound(u32),
    /// An input or output buffer format or dimension is invalid.
    InvalidBufferDescriptor,
    /// An operation requires more memory than available or permitted.
    OutOfMemory { requested: usize, limit: usize },
    /// A requested operation is not supported by the selected or available backends.
    UnsupportedOperation(String),
    /// A required predecessor task failed, preventing downstream execution.
    DependencyFailed(u64),
    /// The specified backend is offline, failed to initialize, or unavailable.
    BackendUnavailable,
    /// Fallback to CPU/SIMD was disallowed by task requirements upon backend failure.
    FallbackRejected,
    /// Task execution was explicitly cancelled.
    Cancelled,
    /// Numerical or dimension mismatch (e.g. matrix multiplication shape incompatibility).
    DimensionMismatch { expected: (usize, usize), actual: (usize, usize) },
    /// Internal execution failure with description.
    ExecutionFailed(String),
}

#[cfg(feature = "std")]
impl std::error::Error for ComputeError {}

impl core::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "Cycle detected in compute graph"),
            Self::BufferNotFound(id) => write!(f, "Buffer handle {} not found", id),
            Self::InvalidBufferDescriptor => write!(f, "Invalid buffer descriptor"),
            Self::OutOfMemory { requested, limit } => {
                write!(f, "Out of memory: requested {} bytes, limit is {}", requested, limit)
            }
            Self::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            Self::DependencyFailed(id) => write!(f, "Predecessor task {} failed", id),
            Self::BackendUnavailable => write!(f, "Compute backend is unavailable"),
            Self::FallbackRejected => write!(f, "Fallback rejected by task policy"),
            Self::Cancelled => write!(f, "Compute task was cancelled"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {:?}, actual {:?}", expected, actual)
            }
            Self::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
        }
    }
}
