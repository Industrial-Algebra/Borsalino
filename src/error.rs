// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Error types for Borsalino GPU operations.
//!
//! All fallible operations return [`Result<T>`], which is an alias for
//! `std::result::Result<T, GpuError>`. Errors are structured with
//! context-rich variants using `thiserror`.

use thiserror::Error;

/// Errors that can occur in GPU operations.
///
/// Each variant carries contextual information: what operation
/// failed, which buffer or shader was involved, and the underlying
/// platform error message where available.
#[derive(Error, Debug)]
pub enum GpuError {
    /// No GPU backend is available for the current platform.
    ///
    /// Enable the `metal` feature on macOS or the `vulkan` feature
    /// on Linux/Windows.
    #[error("no GPU backend available for current platform")]
    NoBackend,

    /// Failed to initialise the GPU device.
    ///
    /// On Metal, this means `MTLCreateSystemDefaultDevice` returned
    /// null — no Metal-capable GPU is present.
    #[error("failed to initialise GPU device: {0}")]
    InitFailed(String),

    /// Shader compilation failed.
    ///
    /// The Metal Shading Language source could not be compiled.
    /// The message contains the compiler error output.
    #[error("shader compilation failed for '{entry}': {message}")]
    CompileFailed {
        /// The kernel function name.
        entry: String,
        /// The compiler error message.
        message: String,
    },

    /// Pipeline creation failed.
    ///
    /// The compiled shader library loaded but a compute pipeline
    /// could not be created for the kernel function.
    #[error("pipeline creation failed for '{entry}': {message}")]
    PipelineFailed {
        /// The kernel function name.
        entry: String,
        /// The platform error message.
        message: String,
    },

    /// Buffer creation failed.
    ///
    /// The GPU could not allocate a buffer of the requested type
    /// and size.
    #[error("buffer creation failed: {message}")]
    BufferCreationFailed {
        /// The platform error message.
        message: String,
    },

    /// Buffer readback failed.
    ///
    /// The GPU buffer contents could not be mapped back to CPU
    /// memory.
    #[error("buffer readback failed: {message}")]
    BufferReadFailed {
        /// The platform error message.
        message: String,
    },

    /// Dispatch failed.
    ///
    /// The compute dispatch could not be submitted or completed.
    #[error("dispatch failed: {message}")]
    DispatchFailed {
        /// The platform error message.
        message: String,
    },

    /// Invalid buffer binding — wrong size, null handle, or
    /// type mismatch.
    #[error("invalid buffer binding: {message}")]
    InvalidBinding {
        /// What was wrong with the binding.
        message: String,
    },

    /// Internal error — should not occur in normal operation.
    #[error("internal GPU error: {0}")]
    Internal(String),

    /// I/O error from the platform layer.
    #[error("platform I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type alias for Borsalino operations.
pub type Result<T> = std::result::Result<T, GpuError>;
