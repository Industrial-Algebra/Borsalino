// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Vulkan compute backend via ash raw FFI, on the Zunesha device
//! substrate.
//!
//! # Architecture
//!
//! The Vulkan **instance, physical-device selection, logical device,
//! queues, and buffer memory** are owned by
//! [`zunesha::vulkan::VulkanDevice`][zdv] — the shared substrate (Zunesha
//! ADR 0001/0003). This backend holds it behind an [`Arc`] and builds
//! Borsalino's own layer on top via the raw escape hatches
//! (`raw_device`, `queues`, `limits`): pipelines (WGSL → SPIR-V via
//! `naga`), descriptor sets, command pools, dispatch, timestamps, and the
//! epoch tracker. All dispatch is synchronous: command buffer → submit →
//! wait_idle. Every queue submission joins the substrate's protocol
//! ([`VulkanDevice::with_compute_queue`]) — the compute queue is an
//! externally synchronized Vulkan object shared with the substrate's
//! staging transfers.
//!
//! # Ownership and lifetime
//!
//! `Arc<VulkanDevice>` replaces the former `SharedDevice`/`SharedInstance`
//! wrappers: buffers, pipelines, and pulses hold clones, so the device is
//! destroyed only when the **last** holder drops — a `GpuBuffer` or `Pulse`
//! may outlive the backend (P1 review contract, preserved).
//! [`VulkanBackend::from_zunesha`] / [`VulkanBackend::into_zunesha`] let a
//! caller share one device between Borsalino and e.g. Goldenweek (ADR 0001
//! interop).
//!
//! # Pre-allocated resources
//!
//! To keep the dispatch hot path allocation-free, the following are
//! created once at `init()` and reused for every dispatch:
//!
//! - 1× `VkPipelineLayout` with N storage buffer bindings (universal)
//! - 1× `VkDescriptorPool` with N sets
//! - N× `VkDescriptorSet` (pre-allocated, updated per dispatch)
//! - 1× `VkCommandPool` with `RESET_COMMAND_BUFFER_BIT`
//!
//! [zdv]: https://docs.rs/zunesha

use naga::back::spv;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

use ash::vk;
use zunesha::Device as _;

use std::ffi::CString;
use std::sync::Arc;

use crate::{
    ComputePipeline, DispatchSpec, GpuBackend, GpuBuffer, GpuError, MemoryStrategy, Pulse, Result,
};

/// Map Borsalino's public strategy onto the substrate's.
fn to_zunesha_strategy(s: MemoryStrategy) -> zunesha::MemoryStrategy {
    match s {
        MemoryStrategy::Auto => zunesha::MemoryStrategy::Auto,
        MemoryStrategy::Unified => zunesha::MemoryStrategy::Unified,
        MemoryStrategy::DeviceLocal => zunesha::MemoryStrategy::DeviceLocal,
    }
}

// ═══════════════════════════════════════════════════════════════════
// VulkanBackend
// ═══════════════════════════════════════════════════════════════════

/// Vulkan compute backend for Linux and Windows, built on a
/// [`zunesha::vulkan::VulkanDevice`].
///
/// Holds the shared substrate device behind an [`Arc`], the compute queue
/// it resolved, and Borsalino's pre-allocated dispatch resources (pipeline
/// layout, descriptor pool/set, command pools, timestamp query pool).
/// Created via [`VulkanBackend::init`] (own device) or
/// [`VulkanBackend::from_zunesha`] (share a caller's device).
///
/// # Platform
///
/// Available on Linux and Windows with the `vulkan` feature enabled.
/// Requires a Vulkan-capable driver with compute support.
pub struct VulkanBackend {
    /// The substrate device, shared with buffers/pipelines/pulses so the
    /// Vulkan device outlives the backend when they do (P1 review contract;
    /// replaces the former `SharedDevice`/`SharedInstance` wrappers —
    /// Zunesha owns the instance/loader lifetime by construction).
    z: Arc<zunesha::vulkan::VulkanDevice>,
    /// Queue family index for the compute queue (the raw handle lives in
    /// the substrate; submissions go through
    /// [`VulkanDevice::with_compute_queue`]).
    #[allow(dead_code)]
    queue_family_index: u32,
    /// Universal pipeline layout — N storage buffer bindings, shared by all pipelines.
    pipeline_layout: vk::PipelineLayout,
    /// Descriptor set layout for N storage buffers.
    #[allow(dead_code)]
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Descriptor pool for storage buffer descriptor sets.
    descriptor_pool: vk::DescriptorPool,
    /// Pre-allocated descriptor set with N storage buffer bindings.
    descriptor_set: vk::DescriptorSet,
    /// Command pool with `RESET_COMMAND_BUFFER_BIT`.
    command_pool: vk::CommandPool,
    /// Query pool for GPU timestamps (None if unsupported).
    timestamp_pool: Option<vk::QueryPool>,
    /// GPU timestamp period in nanoseconds (from device limits).
    timestamp_period: f32,
    /// Epoch tracker for GC safety — counts in-flight dispatches.
    /// Shared with async `Pulse`s so they cannot outlive it (P1 review
    /// finding — same defect as the Metal backend, same fix). Stays on
    /// Borsalino's side until Zunesha wires consumer dispatch accounting
    /// (ADR 0003's end state; honest intermediate state per the migration
    /// plan).
    epoch: Arc<crate::epoch::GpuEpochTracker>,
}

impl VulkanBackend {
    /// Maximum number of storage buffer bindings per pipeline layout.
    const MAX_BUFFER_BINDINGS: u32 = 8;

    /// The raw logical-device handle — every Borsalino-side Vulkan call
    /// goes through this (the substrate owns the real object).
    fn vk(&self) -> ash::Device {
        self.z.raw_device()
    }

    /// The `VkBuffer` handle behind a [`GpuBuffer`] — recovered via the
    /// substrate's raw-buffer escape hatch. Since Phase 2 every buffer is
    /// substrate-backed (all four create paths delegate).
    ///
    /// # Safety (caller)
    ///
    /// The returned handle is valid while `buf` lives and must only be used
    /// on this backend's device.
    fn buffer_handle(&self, buf: &GpuBuffer) -> vk::Buffer {
        // Phase 2: every buffer is substrate-backed. Safety: `raw` was
        // produced by `Box::into_raw::<ZuneshaBufferInner>` and is still
        // valid (buffer not dropped).
        let inner = unsafe { &*(buf.raw as *const ZuneshaBufferInner) };
        self.z.raw_buffer(&inner.zbuf)
    }

    /// Wrap a caller-owned substrate device (ADR 0001 interop: share one
    /// device between Borsalino compute and e.g. Goldenweek rendering).
    ///
    /// The backend holds an `Arc` clone; dropping it tears down only
    /// Borsalino's dispatch plumbing — the device survives while the
    /// caller (or any buffer/pipeline/pulse) holds the `Arc`.
    ///
    /// # Errors
    ///
    /// Fails if Borsalino's dispatch resources (descriptor pool, command
    /// pools, timestamp pool) cannot be created on the device.
    pub fn from_zunesha(z: Arc<zunesha::vulkan::VulkanDevice>) -> Result<Self> {
        Self::build(z)
    }

    /// Hand the substrate device onward, tearing down Borsalino's dispatch
    /// plumbing (descriptor sets, command pools, timestamp pool).
    ///
    /// The returned `Arc` shares ownership with any buffers/pipelines/pulses
    /// this backend created — the Vulkan device is destroyed only when the
    /// **last** holder drops.
    pub fn into_zunesha(self) -> Arc<zunesha::vulkan::VulkanDevice> {
        self.z.clone()
        // `self` drops here: destroys Borsalino's own pools only.
    }

    /// Build the dispatch plumbing on a substrate device.
    ///
    /// Failure at any step releases everything created so far (review
    /// round 2: construction used to leak all prior resources) via the
    /// armed guard below; success disarms it.
    fn build(z: Arc<zunesha::vulkan::VulkanDevice>) -> Result<Self> {
        /// Destroys registered resources in reverse creation order when
        /// dropped while still armed — i.e. on any `?` exit of `build`.
        struct Cleanup<'a> {
            fns: Vec<Box<dyn FnOnce() + 'a>>,
            armed: bool,
        }
        impl Drop for Cleanup<'_> {
            fn drop(&mut self) {
                if self.armed {
                    for f in self.fns.drain(..).rev() {
                        f();
                    }
                }
            }
        }
        let device = z.raw_device();
        // Shared by-ref copy for the guard closures (a &Device is Copy;
        // passing &device inline would move device into the closure).
        // Declared BEFORE `cleanup` so `device` outlives the guard.
        let dev = &device;
        let mut cleanup = Cleanup {
            fns: Vec::new(),
            armed: true,
        };

        // `device` is passed by the caller's tokens: identifiers written
        // in this macro body resolve at the definition site and would not
        // see `build`'s locals (macro_rules hygiene).
        macro_rules! guard {
            ($dev:expr, $destroy:expr) => {
                cleanup
                    .fns
                    .push(Box::new(move || unsafe { $destroy($dev) }));
            };
        }
        let compute = z.queues().compute;
        let queue_family_index = compute.family_index;

        // Timestamp support comes from the physical-device limits (the
        // substrate does not expose these — R5.3 in the Metal requirements
        // proposes adding `timestamp_period` to `DeviceLimits` later).
        let device_props = unsafe {
            z.raw_instance()
                .get_physical_device_properties(z.physical_device())
        };

        // ── Descriptor set layout (N storage buffers) ──────────────

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..Self::MAX_BUFFER_BINDINGS)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&dsl_info, None)
                .map_err(|e| GpuError::InitFailed(format!("create descriptor set layout: {e}")))?
        };
        guard!(dev, |d: &ash::Device| d
            .destroy_descriptor_set_layout(descriptor_set_layout, None));

        // ── Pipeline layout ────────────────────────────────────────

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|e| GpuError::InitFailed(format!("create pipeline layout: {e}")))?
        };
        guard!(dev, |d: &ash::Device| d
            .destroy_pipeline_layout(pipeline_layout, None));

        // ── Descriptor pool ────────────────────────────────────────

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(Self::MAX_BUFFER_BINDINGS)];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(Self::MAX_BUFFER_BINDINGS);

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| GpuError::InitFailed(format!("create descriptor pool: {e}")))?
        };
        // Destroying the pool frees the descriptor sets allocated from it.
        guard!(dev, |d: &ash::Device| d
            .destroy_descriptor_pool(descriptor_pool, None));

        // ── Pre-allocate descriptor set ───────────────────────────

        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&set_info)
                .map_err(|e| GpuError::InitFailed(format!("allocate descriptor set: {e}")))?
        }[0];

        // ── Command pool ───────────────────────────────────────────

        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            device
                .create_command_pool(&cmd_pool_info, None)
                .map_err(|e| GpuError::InitFailed(format!("create command pool: {e}")))?
        };
        guard!(dev, |d: &ash::Device| d
            .destroy_command_pool(command_pool, None));

        // ── Timestamp query pool ─────────────────────────────────

        let timestamp_pool = if device_props.limits.timestamp_compute_and_graphics == vk::TRUE {
            let pool_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(1);
            let pool = unsafe {
                device
                    .create_query_pool(&pool_info, None)
                    .map_err(|e| GpuError::InitFailed(format!("create timestamp pool: {e}")))?
            };
            guard!(dev, |d: &ash::Device| d.destroy_query_pool(pool, None));
            Some(pool)
        } else {
            None
        };
        let timestamp_period = device_props.limits.timestamp_period;

        cleanup.armed = false;
        Ok(Self {
            z,
            queue_family_index,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            command_pool,
            timestamp_pool,
            timestamp_period,
            epoch: Arc::new(crate::epoch::GpuEpochTracker::new()),
        })
    }

    /// Create a compute pipeline from pre-compiled SPIR-V.
    fn create_pipeline_from_spv(
        &self,
        entry_point: &str,
        spv_words: &[u32],
    ) -> Result<ComputePipeline> {
        let shader_info = vk::ShaderModuleCreateInfo::default().code(spv_words);

        let shader_module = unsafe {
            self.vk()
                .create_shader_module(&shader_info, None)
                .map_err(|e| GpuError::CompileFailed {
                    entry: entry_point.into(),
                    message: format!("vkCreateShaderModule: {e}"),
                })?
        };

        let entry_name = CString::new(entry_point).map_err(|_| GpuError::CompileFailed {
            entry: entry_point.into(),
            message: "entry point name contains null byte".into(),
        })?;

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .module(shader_module)
            .name(&entry_name)
            .stage(vk::ShaderStageFlags::COMPUTE);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.pipeline_layout);

        let pipelines = unsafe {
            self.vk()
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_pipelines, err)| GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: format!("vkCreateComputePipelines: {err}"),
                })?
        };

        unsafe {
            self.vk().destroy_shader_module(shader_module, None);
        }

        let inner = Box::new(VulkanPipelineInner {
            pipeline: pipelines[0],
            device: Arc::clone(&self.z),
        });

        Ok(ComputePipeline {
            raw: Box::into_raw(inner) as *mut std::ffi::c_void,
            drop_fn: drop_vulkan_pipeline,
        })
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            // Defensive quiescence: a pulse may still hold an in-flight
            // dispatch (its command buffer comes from our command pool,
            // its set from our descriptor pool). Destroying pools a
            // pending submission references is UB — idle the device
            // first, but ONLY when something is actually in flight (a
            // blanket wait_idle on every backend drop stampedes the
            // driver's internal locks under parallel test load).
            if self.epoch.in_flight() > 0 {
                let _ = self.vk().device_wait_idle();
            }
            if let Some(pool) = self.timestamp_pool {
                self.vk().destroy_query_pool(pool, None);
            }
            self.vk().destroy_command_pool(self.command_pool, None);
            self.vk()
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.vk()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.vk()
                .destroy_pipeline_layout(self.pipeline_layout, None);
            // Device and instance destruction happen in the substrate's
            // `VulkanDevice::Drop`, when the last `Arc` holder (backend or
            // pulse/buffer/pipeline inner) goes away — not here.
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Buffer inner type
// ═══════════════════════════════════════════════════════════════════

/// Internal state for a substrate-allocated buffer (from
/// `z.create_buffer` / `create_buffer_uninit`), stored behind the opaque
/// `GpuBuffer.raw` pointer. The `Arc` keeps the Vulkan device alive past
/// backend drop; the `zunesha::Buffer` destroys its own resources.
struct ZuneshaBufferInner {
    zbuf: zunesha::Buffer,
    _device: Arc<zunesha::vulkan::VulkanDevice>,
}

unsafe impl Send for ZuneshaBufferInner {}
unsafe impl Sync for ZuneshaBufferInner {}

// ═══════════════════════════════════════════════════════════════════
// Pipeline inner type
// ═══════════════════════════════════════════════════════════════════

/// Internal state for a Vulkan compute pipeline, stored behind the opaque
/// `ComputePipeline.raw` pointer.
struct VulkanPipelineInner {
    pipeline: vk::Pipeline,
    /// Substrate device share — keeps the Vulkan device alive past backend
    /// drop (destroy in `drop`).
    device: Arc<zunesha::vulkan::VulkanDevice>,
}

/// Drop function stored in [`ComputePipeline`] — destroys the Vulkan pipeline.
fn drop_vulkan_pipeline(raw: *mut std::ffi::c_void) {
    if !raw.is_null() {
        unsafe {
            let inner = Box::from_raw(raw as *mut VulkanPipelineInner);
            inner
                .device
                .raw_device()
                .destroy_pipeline(inner.pipeline, None);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pulse inner type
// ═══════════════════════════════════════════════════════════════════

/// Internal state for an async dispatch, stored behind the opaque
/// `Pulse.raw` pointer.
struct VulkanPulseInner {
    fence: vk::Fence,
    /// Substrate device share — the pulse owns a reference, so
    /// wait/drop keep a LIVE device even after the backend is gone
    /// (P1 review finding).
    device: Arc<zunesha::vulkan::VulkanDevice>,
    epoch: Arc<crate::epoch::GpuEpochTracker>,
    /// Tracks whether end_dispatch has been called (prevents double-decrement
    /// when wait() + drop() both fire).
    epoch_completed: std::sync::atomic::AtomicBool,
}

fn wait_vulkan_pulse(raw: *mut std::ffi::c_void) {
    if !raw.is_null() {
        let inner = unsafe { &*(raw as *const VulkanPulseInner) };
        unsafe {
            let _ = inner.device.raw_device().wait_for_fences(
                std::slice::from_ref(&inner.fence),
                true,
                u64::MAX,
            );
        }
        // Mark this dispatch as complete (balances the begin_dispatch at
        // dispatch_async submit time). Only decrements once even if wait()
        // is called multiple times.
        if !inner
            .epoch_completed
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            inner.epoch.end_dispatch();
        }
    }
}

fn drop_vulkan_pulse(raw: *mut std::ffi::c_void) {
    if !raw.is_null() {
        let inner = unsafe { Box::from_raw(raw as *mut VulkanPulseInner) };
        let device = inner.device.raw_device();
        unsafe {
            // Ensure GPU completes before destroying fence — a fence
            // that is still in use must not be destroyed.
            let _ = device.wait_for_fences(std::slice::from_ref(&inner.fence), true, u64::MAX);
            // Balance the begin_dispatch from dispatch_async (only if
            // wait() hasn't already done so).
            if !inner
                .epoch_completed
                .swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                inner.epoch.end_dispatch();
            }
            device.destroy_fence(inner.fence, None);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Buffer drop/contents functions
// ═══════════════════════════════════════════════════════════════════

/// Drop function stored in [`GpuBuffer`] — drops the boxed substrate
/// buffer (it destroys its own Vulkan resources).
fn drop_zunesha_buffer(raw: *mut std::ffi::c_void) {
    if !raw.is_null() {
        unsafe {
            drop(Box::from_raw(raw as *mut ZuneshaBufferInner));
        }
    }
}

/// Contents function for substrate-backed buffers — returns the
/// [`SUBSTRATE_BUFFER_TAG`] sentinel instead of a mapped pointer (reads go
/// through the substrate's `read_buffer`, which needs the device — not a
/// bare pointer). Real Vulkan mappings are page-aligned, so the sentinel
/// can never collide with a mapped address. Phase 2 removed the last
/// runtime dispatch on this tag (every buffer is substrate-backed now);
/// it survives as the public `contents()` value for substrate buffers
/// and as the delegation marker the Phase-2 tests assert.
const SUBSTRATE_BUFFER_TAG: *const std::ffi::c_void =
    std::ptr::without_provenance::<std::ffi::c_void>(1);

fn contents_zunesha_buffer(_raw: *mut std::ffi::c_void) -> *const std::ffi::c_void {
    SUBSTRATE_BUFFER_TAG
}

/// FNV-1a hash (deterministic across runs, used for cache keys).
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Get the cache directory (respects XDG).
fn cache_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        std::path::PathBuf::from(dir)
    } else if let Ok(home) = std::env::var("HOME") {
        std::path::PathBuf::from(home).join(".cache")
    } else {
        std::path::PathBuf::from(".cache")
    }
}

// ═══════════════════════════════════════════════════════════════════
// Staging helpers
// ═══════════════════════════════════════════════════════════════════

// GpuBackend implementation
// ═══════════════════════════════════════════════════════════════════

impl VulkanBackend {
    /// Wrap a substrate buffer as a [`GpuBuffer`] (all four create paths
    /// delegate since Phase 2 — Borsalino no longer allocates Vulkan
    /// memory itself).
    fn wrap_zunesha_buffer(
        &self,
        zbuf: zunesha::Buffer,
        len: usize,
        element_size: usize,
    ) -> GpuBuffer {
        GpuBuffer {
            raw: Box::into_raw(Box::new(ZuneshaBufferInner {
                zbuf,
                _device: Arc::clone(&self.z),
            })) as *mut std::ffi::c_void,
            len,
            element_size,
            drop_fn: drop_zunesha_buffer,
            contents_fn: contents_zunesha_buffer,
        }
    }
}

impl GpuBackend for VulkanBackend {
    fn init() -> Result<Self> {
        Self::init_with_strategy(MemoryStrategy::Auto)
    }

    fn init_with_strategy(strategy: MemoryStrategy) -> Result<Self> {
        let z = zunesha::vulkan::VulkanDevice::init_with_strategy(to_zunesha_strategy(strategy))
            .map_err(|e| GpuError::InitFailed(format!("zunesha device init: {e}")))?;
        Self::build(std::sync::Arc::new(z))
    }

    fn compile(&self, entry_point: &str, wgsl_source: &str) -> Result<ComputePipeline> {
        // Step 1: Parse WGSL → naga IR
        let module = wgsl::parse_str(wgsl_source).map_err(|e| GpuError::CompileFailed {
            entry: entry_point.into(),
            message: e.emit_to_string(wgsl_source),
        })?;

        // Step 2: Validate the module
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator
            .validate(&module)
            .map_err(|e| GpuError::CompileFailed {
                entry: entry_point.into(),
                message: e.emit_to_string(wgsl_source),
            })?;

        // Step 3: Emit SPIR-V
        let spv_words =
            spv::write_vec(&module, &info, &spv::Options::default(), None).map_err(|e| {
                GpuError::CompileFailed {
                    entry: entry_point.into(),
                    message: format!("SPIR-V emission failed: {e}"),
                }
            })?;

        // Step 4: Create Vulkan shader module
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);

        let shader_module = unsafe {
            self.vk()
                .create_shader_module(&shader_info, None)
                .map_err(|e| GpuError::CompileFailed {
                    entry: entry_point.into(),
                    message: format!("vkCreateShaderModule: {e}"),
                })?
        };

        // Step 5: Create compute pipeline
        let entry_name = CString::new(entry_point).map_err(|_| GpuError::CompileFailed {
            entry: entry_point.into(),
            message: "entry point name contains null byte".into(),
        })?;

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .module(shader_module)
            .name(&entry_name)
            .stage(vk::ShaderStageFlags::COMPUTE);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.pipeline_layout);

        let pipelines = unsafe {
            self.vk()
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_pipelines, err)| GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: format!("vkCreateComputePipelines: {err}"),
                })?
        };

        // Step 6: Destroy the shader module (pipeline owns the compiled code)
        unsafe {
            self.vk().destroy_shader_module(shader_module, None);
        }

        // Step 7: Wrap in opaque handle
        let inner = Box::new(VulkanPipelineInner {
            pipeline: pipelines[0],
            device: Arc::clone(&self.z),
        });

        Ok(ComputePipeline {
            raw: Box::into_raw(inner) as *mut std::ffi::c_void,
            drop_fn: drop_vulkan_pipeline,
        })
    }

    fn compile_cached(&self, entry_point: &str, wgsl_source: &str) -> Result<ComputePipeline> {
        // Determine cache path
        let cache_dir = cache_dir().join("borsalino");
        let _ = std::fs::create_dir_all(&cache_dir);
        let cache_key = fnv1a(wgsl_source.as_bytes());
        let cache_path = cache_dir.join(format!("{entry_point}_{cache_key:016x}.spv"));

        // Try loading from cache
        if let Ok(spv_bytes) = std::fs::read(&cache_path) {
            if !spv_bytes.is_empty() && spv_bytes.len() % 4 == 0 {
                let spv_words: Vec<u32> = spv_bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                if let Ok(pipeline) = self.create_pipeline_from_spv(entry_point, &spv_words) {
                    return Ok(pipeline);
                }
            }
        }

        // Cache miss: compile from source, then save SPIR-V
        let module = wgsl::parse_str(wgsl_source).map_err(|e| GpuError::CompileFailed {
            entry: entry_point.into(),
            message: e.emit_to_string(wgsl_source),
        })?;

        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator
            .validate(&module)
            .map_err(|e| GpuError::CompileFailed {
                entry: entry_point.into(),
                message: e.emit_to_string(wgsl_source),
            })?;

        let spv_words =
            spv::write_vec(&module, &info, &spv::Options::default(), None).map_err(|e| {
                GpuError::CompileFailed {
                    entry: entry_point.into(),
                    message: format!("SPIR-V emission failed: {e}"),
                }
            })?;

        // Save to cache (best-effort)
        let spv_bytes: Vec<u8> = spv_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let _ = std::fs::write(&cache_path, &spv_bytes);

        self.create_pipeline_from_spv(entry_point, &spv_words)
    }

    fn create_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        let zbuf = self
            .z
            .create_buffer(data)
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("zunesha create_buffer: {e}"),
            })?;
        Ok(self.wrap_zunesha_buffer(zbuf, data.len(), std::mem::size_of::<T>()))
    }

    fn create_buffer_uninit<T: bytemuck::Pod>(&self, len: usize) -> Result<GpuBuffer> {
        let zbuf =
            self.z
                .create_buffer_uninit::<T>(len)
                .map_err(|e| GpuError::BufferCreationFailed {
                    message: format!("zunesha create_buffer_uninit: {e}"),
                })?;
        Ok(self.wrap_zunesha_buffer(zbuf, len, std::mem::size_of::<T>()))
    }

    /// Phase 2 (§5.1, aligned contract): delegates to the substrate's
    /// forced device-local + persistent-staging path — including under
    /// forced `Unified` on discrete hardware (the documented trait
    /// contract; Borsalino's former host-visible divergence is gone).
    fn create_device_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        let zbuf =
            self.z
                .create_device_buffer(data)
                .map_err(|e| GpuError::BufferCreationFailed {
                    message: format!("zunesha create_device_buffer: {e}"),
                })?;
        Ok(self.wrap_zunesha_buffer(zbuf, data.len(), std::mem::size_of::<T>()))
    }
    /// Phase 2: the uninit variant delegates to the substrate identically
    /// (forced device-local + persistent staging).
    fn create_device_buffer_uninit<T: bytemuck::Pod>(&self, len: usize) -> Result<GpuBuffer> {
        let zbuf = self.z.create_device_buffer_uninit::<T>(len).map_err(|e| {
            GpuError::BufferCreationFailed {
                message: format!("zunesha create_device_buffer_uninit: {e}"),
            }
        })?;
        Ok(self.wrap_zunesha_buffer(zbuf, len, std::mem::size_of::<T>()))
    }
    fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        self.dispatch_ex(pipeline, buffers, workgroups, (256, 1, 1))
    }

    fn dispatch_ex(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        workgroups: (u32, u32, u32),
        _threads_per_group: (u32, u32, u32),
    ) -> Result<()> {
        let nbuffers = buffers.len();
        if nbuffers > Self::MAX_BUFFER_BINDINGS as usize {
            return Err(GpuError::InvalidBinding {
                message: format!(
                    "{nbuffers} buffers exceeds max {}",
                    Self::MAX_BUFFER_BINDINGS
                ),
            });
        }

        // ── Allocate command buffer ───────────────────────────────

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            self.vk()
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkAllocateCommandBuffers: {e}"),
                })?
        }[0];

        // ── Begin command buffer ──────────────────────────────────

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.vk()
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkBeginCommandBuffer: {e}"),
                })?;
        }

        // ── Bind pipeline ─────────────────────────────────────────

        unsafe {
            let vk_pipeline = (*(pipeline.raw as *const VulkanPipelineInner)).pipeline;
            self.vk()
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, vk_pipeline);
        }

        // ── Update descriptor set + bind ──────────────────────────

        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(nbuffers);
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(nbuffers);

        // Keep buffer_infos alive on the heap — the writes reference them
        for buf in buffers.iter() {
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(self.buffer_handle(buf))
                    .offset(0)
                    .range(vk::WHOLE_SIZE),
            );
        }

        for (i, buf_info) in buffer_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(i as u32)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(buf_info)),
            );
        }

        unsafe {
            self.vk().update_descriptor_sets(&writes, &[]);
        }

        unsafe {
            self.vk().cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_set),
                &[],
            );
        }

        // ── Dispatch ──────────────────────────────────────────────

        unsafe {
            self.vk()
                .cmd_dispatch(cmd, workgroups.0, workgroups.1, workgroups.2);
        }

        // ── Memory barrier (shader write → host read) ─────────────

        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);

        unsafe {
            self.vk().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }

        // ── End command buffer ────────────────────────────────────

        unsafe {
            self.vk()
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkEndCommandBuffer: {e}"),
                })?;
        }

        // ── Submit + wait ─────────────────────────────────────────

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

        self.epoch.begin_dispatch();

        // Join the substrate's submission protocol: the compute queue is
        // externally synchronized — submit AND the immediate wait run inside
        // `with_compute_queue` so concurrent substrate/consumer submissions
        // serialize (review round 1 P1).
        self.z.with_compute_queue(|queue| unsafe {
            self.vk()
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkQueueSubmit: {e}"),
                })?;

            self.vk()
                .queue_wait_idle(queue)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkQueueWaitIdle: {e}"),
                })
        })?;

        self.epoch.end_dispatch();

        // ── Cleanup ───────────────────────────────────────────────

        unsafe {
            self.vk()
                .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
        }

        Ok(())
    }

    fn dispatch_many(&self, dispatches: &[DispatchSpec<'_>]) -> Result<()> {
        if dispatches.is_empty() {
            return Ok(());
        }

        // ── Allocate ONE command buffer for all dispatches ──────

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            self.vk()
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkAllocateCommandBuffers: {e}"),
                })?
        }[0];

        // ── Begin ───────────────────────────────────────────────

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.vk()
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkBeginCommandBuffer: {e}"),
                })?;
        }

        // ── Process each dispatch ───────────────────────────────

        for spec in dispatches {
            let nbuffers = spec.buffers.len();
            if nbuffers > Self::MAX_BUFFER_BINDINGS as usize {
                return Err(GpuError::InvalidBinding {
                    message: format!(
                        "{nbuffers} buffers exceeds max {}",
                        Self::MAX_BUFFER_BINDINGS
                    ),
                });
            }

            // Bind pipeline
            unsafe {
                let vk_pipeline = (*(spec.pipeline.raw as *const VulkanPipelineInner)).pipeline;
                self.vk()
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, vk_pipeline);
            }

            // Update descriptor set + bind
            let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(nbuffers);
            let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(nbuffers);

            for buf in spec.buffers.iter() {
                buffer_infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(self.buffer_handle(buf))
                        .offset(0)
                        .range(vk::WHOLE_SIZE),
                );
            }

            for (i, buf_info) in buffer_infos.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(buf_info)),
                );
            }

            unsafe {
                self.vk().update_descriptor_sets(&writes, &[]);
                self.vk().cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    std::slice::from_ref(&self.descriptor_set),
                    &[],
                );

                self.vk().cmd_dispatch(
                    cmd,
                    spec.workgroups.0,
                    spec.workgroups.1,
                    spec.workgroups.2,
                );
            }
        }

        // ── Memory barrier (all dispatches → host) ──────────────

        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);

        unsafe {
            self.vk().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }

        // ── End, submit, wait ───────────────────────────────────

        unsafe {
            self.vk()
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkEndCommandBuffer: {e}"),
                })?;
        }

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

        self.epoch.begin_dispatch();

        // Join the substrate's submission protocol: the compute queue is
        // externally synchronized — submit AND the immediate wait run inside
        // `with_compute_queue` so concurrent substrate/consumer submissions
        // serialize (review round 1 P1).
        self.z.with_compute_queue(|queue| unsafe {
            self.vk()
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkQueueSubmit: {e}"),
                })?;

            self.vk()
                .queue_wait_idle(queue)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkQueueWaitIdle: {e}"),
                })
        })?;

        self.epoch.end_dispatch();

        // ── Cleanup ─────────────────────────────────────────────

        unsafe {
            self.vk()
                .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
        }

        Ok(())
    }

    fn dispatch_async(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        workgroups: (u32, u32, u32),
    ) -> Result<Pulse> {
        let nbuffers = buffers.len();
        if nbuffers > Self::MAX_BUFFER_BINDINGS as usize {
            return Err(GpuError::InvalidBinding {
                message: format!(
                    "{nbuffers} buffers exceeds max {}",
                    Self::MAX_BUFFER_BINDINGS
                ),
            });
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            self.vk()
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkAllocateCommandBuffers: {e}"),
                })?
        }[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.vk()
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkBeginCommandBuffer: {e}"),
                })?;
        }

        // Bind pipeline
        unsafe {
            let vk_pipeline = (*(pipeline.raw as *const VulkanPipelineInner)).pipeline;
            self.vk()
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, vk_pipeline);
        }

        // Descriptor set + bind
        let mut buffer_infos = Vec::with_capacity(nbuffers);
        let mut writes = Vec::with_capacity(nbuffers);
        for buf in buffers.iter() {
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(self.buffer_handle(buf))
                    .offset(0)
                    .range(vk::WHOLE_SIZE),
            );
        }
        for (i, bi) in buffer_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(i as u32)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(bi)),
            );
        }

        unsafe {
            self.vk().update_descriptor_sets(&writes, &[]);
            self.vk().cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_set),
                &[],
            );
            self.vk()
                .cmd_dispatch(cmd, workgroups.0, workgroups.1, workgroups.2);

            // Barrier: shader write → host read (applied when waited)
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            self.vk().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );

            self.vk()
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkEndCommandBuffer: {e}"),
                })?;
        }

        // Create fence for async completion signal
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            self.vk()
                .create_fence(&fence_info, None)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkCreateFence: {e}"),
                })?
        };

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

        self.epoch.begin_dispatch();

        // Submit inside the substrate's submission protocol; the fence (not
        // the queue) tracks completion, so Pulse waits need no lock.
        self.z.with_compute_queue(|queue| unsafe {
            self.vk()
                .queue_submit(queue, &[submit_info], fence)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkQueueSubmit: {e}"),
                })
        })?;

        // Free command buffer (work is submitted, fence tracks completion)
        unsafe {
            self.vk()
                .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
        }

        let inner = Box::new(VulkanPulseInner {
            fence,
            device: Arc::clone(&self.z),
            epoch: std::sync::Arc::clone(&self.epoch),
            epoch_completed: std::sync::atomic::AtomicBool::new(false),
        });

        Ok(Pulse {
            raw: Box::into_raw(inner) as *mut std::ffi::c_void,
            wait_fn: wait_vulkan_pulse,
            drop_fn: drop_vulkan_pulse,
        })
    }

    /// Phase 2: every buffer is substrate-backed, so readback is a pure
    /// delegation — the substrate's locked transfer + staging-copy protocol
    /// (device-local: persistent staging under the submission lock;
    /// host-visible: direct mapping read).
    fn read_buffer<T: bytemuck::Pod>(&self, buffer: &GpuBuffer) -> Result<Vec<T>> {
        // Safety: `raw` was produced by `Box::into_raw::<ZuneshaBufferInner>`
        // and is still valid (buffer not dropped).
        let inner = unsafe { &*(buffer.raw as *const ZuneshaBufferInner) };
        self.z
            .read_buffer(&inner.zbuf)
            .map_err(|e| GpuError::BufferReadFailed {
                message: format!("zunesha read_buffer: {e}"),
            })
    }
    fn timestamp(&self) -> Result<u64> {
        let Some(pool) = self.timestamp_pool else {
            // Fall back to CPU timestamp if GPU timestamps unsupported
            return Ok(std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos() as u64);
        };

        unsafe {
            // Allocate a one-shot command buffer
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let cmd = self
                .vk()
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| GpuError::Internal(format!("timestamp alloc: {e}")))?[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.vk()
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| GpuError::Internal(format!("timestamp begin: {e}")))?;

            // Reset query pool before use
            self.vk().reset_query_pool(pool, 0, 1);

            // Write GPU timestamp
            self.vk()
                .cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_COMMANDS, pool, 0);

            self.vk()
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::Internal(format!("timestamp end: {e}")))?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            self.z.with_compute_queue(|queue| {
                self.vk()
                    .queue_submit(queue, &[submit_info], vk::Fence::null())
                    .map_err(|e| GpuError::Internal(format!("timestamp submit: {e}")))?;
                self.vk()
                    .queue_wait_idle(queue)
                    .map_err(|e| GpuError::Internal(format!("timestamp wait: {e}")))
            })?;

            // Read back timestamp
            let mut ts_data = [0u64];
            self.vk()
                .get_query_pool_results(
                    pool,
                    0,
                    &mut ts_data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|e| GpuError::Internal(format!("timestamp read: {e}")))?;

            self.vk()
                .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));

            // Convert ticks to nanoseconds
            Ok((ts_data[0] as f64 * self.timestamp_period as f64) as u64)
        }
    }

    fn in_flight(&self) -> u64 {
        self.epoch.in_flight()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Substrate integration ─────────────────────────────────────

    /// `init()` builds on a substrate device; the returned backend hands
    /// that same device back via `into_zunesha` (still usable after the
    /// backend's plumbing is torn down).
    #[test]
    #[serial_test::serial]
    fn into_zunesha_returns_live_device_after_backend_drop() {
        let z = match VulkanBackend::init() {
            Ok(b) => b.into_zunesha(),
            Err(e) => {
                eprintln!("skipping: no Vulkan device ({e})");
                return;
            }
        };
        // The backend is dropped — its descriptor/command pools are gone,
        // but the substrate device must still allocate and read buffers.
        let buf = z.create_buffer(&[1.0f32, 2.0, 3.0]).unwrap();
        let back: Vec<f32> = z.read_buffer(&buf).unwrap();
        assert_eq!(back, vec![1.0, 2.0, 3.0]);
    }

    /// ADR 0001 interop: an externally-created substrate device wrapped via
    /// `from_zunesha` carries full Borsalino compute (compile → dispatch →
    /// read), and the external holder can keep using the device after the
    /// backend is dropped (the Arc share outlives the plumbing).
    #[test]
    #[serial_test::serial]
    fn from_zunesha_shares_device_with_external_holder() {
        let z = std::sync::Arc::new(match zunesha::vulkan::VulkanDevice::init() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Vulkan device ({e})");
                return;
            }
        });
        let backend = match VulkanBackend::from_zunesha(std::sync::Arc::clone(&z)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping: plumbing build failed ({e})");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if (i >= 4u) { return; }
                output[i] = input[i] + 1.0;
            }
        "#;
        let pipeline = backend.compile("add_one", wgsl).unwrap();
        let input = backend.create_buffer(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output = backend.create_buffer_uninit::<f32>(4).unwrap();
        backend
            .dispatch(&pipeline, &[&input, &output], (1, 1, 1))
            .unwrap();
        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);

        drop(backend);
        // The external holder's Arc keeps the device alive post-drop.
        let buf = z.create_buffer(&[9.0f32]).unwrap();
        let back: Vec<f32> = z.read_buffer(&buf).unwrap();
        assert_eq!(back, vec![9.0]);
    }

    #[test]
    #[serial_test::serial]
    fn device_init() {
        match VulkanBackend::init() {
            Ok(_) => {}
            Err(GpuError::InitFailed(msg)) => {
                eprintln!("Vulkan init failed (expected in CI/headless): {msg}");
            }
            Err(GpuError::NoBackend) => {
                eprintln!("no Vulkan backend (expected on macOS)");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    #[serial_test::serial]
    fn add_one_kernel() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                // Bound the access: 256 threads over a 4-element buffer
                // (review finding on the Metal twin; naga's default
                // bounds-check policies are Unchecked for SPIR-V too).
                if (i >= 4u) { return; }
                output[i] = input[i] + 1.0;
            }
        "#;

        let pipeline = backend.compile("add_one", wgsl).unwrap();
        let input = backend.create_buffer(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output = backend.create_buffer_uninit::<f32>(4).unwrap();
        backend
            .dispatch(&pipeline, &[&input, &output], (1, 1, 1))
            .unwrap();

        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[serial_test::serial]
    fn vector_scale_1024() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                output[i] = input[i] * 2.5;
            }
        "#;

        let n: usize = 1024;
        let input_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let expected: Vec<f32> = input_data.iter().map(|x| x * 2.5).collect();

        let pipeline = backend.compile("scale", wgsl).unwrap();
        let input = backend.create_buffer(&input_data).unwrap();
        let output = backend.create_buffer_uninit::<f32>(n).unwrap();

        backend
            .dispatch(&pipeline, &[&input, &output], (4, 1, 1))
            .unwrap();

        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-6,
                "mismatch at index {i}: got {r}, expected {e}"
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn compile_error() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };

        let bad_wgsl = "@compute fn broken( @storage(0) x: array<f32> ) { x[0] = ; }";
        let result = backend.compile("broken", bad_wgsl);
        assert!(result.is_err(), "expected compile error for invalid WGSL");
        match result.unwrap_err() {
            GpuError::CompileFailed { .. } => {}
            e => panic!("expected CompileFailed, got {e:?}"),
        }
    }

    #[test]
    #[serial_test::serial]
    fn roundtrip_empty() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };

        let buf = backend.create_buffer_uninit::<f32>(16).unwrap();
        let result: Vec<f32> = backend.read_buffer(&buf).unwrap();
        assert_eq!(result.len(), 16);
        // Uninitialised — all zeroes is typical for fresh device memory
    }

    /// Phase 2 (§5.1, aligned contract): `create_device_buffer` delegates
    /// to the substrate — device-local + persistent staging per Zunesha's
    /// documented contract, including under forced `Unified` on discrete
    /// hardware. Observable as the substrate tag from `contents()` and the
    /// substrate drop function (Borsalino's own allocation paths are gone).
    #[test]
    #[serial_test::serial]
    fn create_device_buffer_delegates_to_substrate() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let buf = backend.create_device_buffer(&data).unwrap();

        // Substrate tag, not a Borsalino mapping and not null.
        assert!(
            std::ptr::eq((buf.contents_fn)(buf.raw), SUBSTRATE_BUFFER_TAG),
            "device buffer must be substrate-backed (sentinel tag)"
        );
        // Substrate drop path (the deleted Borsalino-allocated kind's
        // drop function must NOT be in use anymore).
        assert!(
            std::ptr::eq(buf.drop_fn as *const (), drop_zunesha_buffer as *const ()),
            "device buffer must use the substrate drop function"
        );
        // Round-trip through the substrate's locked staging read.
        let back: Vec<f32> = backend.read_buffer(&buf).unwrap();
        assert_eq!(back, data.to_vec());
    }

    /// Same delegation for the uninit variant: substrate-backed, correct
    /// length/element_size, readable (contents unspecified until written).
    #[test]
    #[serial_test::serial]
    fn create_device_buffer_uninit_delegates_to_substrate() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };
        let buf = backend.create_device_buffer_uninit::<f32>(16).unwrap();
        assert!(
            std::ptr::eq((buf.contents_fn)(buf.raw), SUBSTRATE_BUFFER_TAG),
            "uninit device buffer must be substrate-backed"
        );
        assert_eq!(buf.len, 16);
        assert_eq!(buf.element_size, std::mem::size_of::<f32>());
        let back: Vec<f32> = backend.read_buffer(&buf).unwrap();
        assert_eq!(back.len(), 16);
    }

    /// Under forced `Unified` on discrete hardware the delegated contract
    /// forces device-local + persistent staging (the documented trait
    /// contract — Borsalino's old host-visible divergence is gone).
    /// Placement itself is the substrate's contract and is tested there
    /// (Zunesha: `device_buffer_forces_device_local_under_unified_strategy`)
    /// — per-buffer placement is not observable through Borsalino's public
    /// surface, so here we pin the delegation identity under the forced
    /// strategy (the only honest observable) plus the round-trip.
    #[test]
    #[serial_test::serial]
    fn device_buffer_roundtrips_under_forced_unified() {
        let backend = match VulkanBackend::init_with_strategy(MemoryStrategy::Unified) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };
        let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.25).collect();
        let buf = backend.create_device_buffer(&data).unwrap();
        assert!(
            std::ptr::eq((buf.contents_fn)(buf.raw), SUBSTRATE_BUFFER_TAG),
            "delegation holds under forced Unified (placement forced by substrate)"
        );
        let back: Vec<f32> = backend.read_buffer(&buf).unwrap();
        assert_eq!(back, data);
    }

    /// Phase 2 behavior change (changelog-noted): mixed-size reads are
    /// byte-based. `read_buffer::<u8>` returns one element per byte of the
    /// buffer's storage (the substrate's `len` is a byte length), where
    /// Borsalino's old path returned the element count captured at
    /// creation. Reading 2 × u32 as u8 now yields 8 elements (was 2).
    #[test]
    #[serial_test::serial]
    fn device_buffer_mixed_size_reads_are_byte_based() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };
        let data = [0x3f80_0000u32, 0x4000_0000]; // 1.0f32, 2.0f32
        let buf = backend.create_device_buffer(&data).unwrap();
        assert_eq!(buf.len, 2, "GpuBuffer.len stays the element count");
        let bytes: Vec<u8> = backend.read_buffer(&buf).unwrap();
        assert_eq!(bytes.len(), 8, "byte-based: 2 u32 = 8 u8 elements");
        let words: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();
        assert_eq!(words, data.to_vec(), "byte view reassembles exactly");
    }

    /// Large device buffer through the staging path (upload + readback
    /// exactness at a size well past any alignment slop).
    #[test]
    #[serial_test::serial]
    fn device_buffer_large_roundtrip() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };
        let data: Vec<f32> = (0..65_536).map(|i| (i % 97) as f32).collect();
        let buf = backend.create_device_buffer(&data).unwrap();
        let back: Vec<f32> = backend.read_buffer(&buf).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    #[serial_test::serial]
    fn timestamp_works() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };

        let t0 = backend.timestamp().unwrap();
        let t1 = backend.timestamp().unwrap();
        assert!(t1 >= t0, "timestamps should be monotonic");
        assert!(t1 > 0, "timestamp should be non-zero");
    }

    #[test]
    #[serial_test::serial]
    fn shader_caching() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> out: array<f32>;
            @compute @workgroup_size(1)
            fn cache_test(@builtin(global_invocation_id) gid: vec3<u32>) {
                out[gid.x] = 42.0;
            }
        "#;

        // First call: compile from source
        let p1 = backend.compile_cached("cache_test", wgsl).unwrap();

        // Second call: should load from cache
        let p2 = backend.compile_cached("cache_test", wgsl).unwrap();

        // Both pipelines should work
        let out = backend.create_buffer_uninit::<f32>(1).unwrap();
        backend.dispatch(&p1, &[&out], (1, 1, 1)).unwrap();
        let result: Vec<f32> = backend.read_buffer(&out).unwrap();
        assert!((result[0] - 42.0).abs() < 0.001);

        backend.dispatch(&p2, &[&out], (1, 1, 1)).unwrap();
        let result2: Vec<f32> = backend.read_buffer(&out).unwrap();
        assert!((result2[0] - 42.0).abs() < 0.001);
    }

    #[test]
    #[serial_test::serial]
    fn async_dispatch() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(256)
            fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
                // 256 threads over a 4-element buffer — bound the access.
                if (gid.x >= 4u) { return; }
                output[gid.x] = input[gid.x] + 1.0;
            }
        "#;
        let pipeline = backend.compile("add_one", wgsl).unwrap();
        let input = backend.create_buffer(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output = backend.create_device_buffer_uninit::<f32>(4).unwrap();

        let pulse = backend
            .dispatch_async(&pipeline, &[&input, &output], (1, 1, 1))
            .unwrap();

        pulse.wait();

        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
    }

    /// Regression (P1 review finding): a pulse, its buffers, and its
    /// pipeline must survive the backend being dropped first. The old
    /// code stored a cloned `ash::Device` handle, and the backend's
    /// `drop` destroyed the device — the pulse's wait/drop then called
    /// into a destroyed device. Shared ownership (`SharedDevice`) keeps
    /// the device (and the loader) alive until the last holder drops.
    #[test]
    #[serial_test::serial]
    fn pulse_outlives_backend() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(256)
            fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
                if (gid.x >= 4u) { return; }
                output[gid.x] = input[gid.x] + 1.0;
            }
        "#;
        let pipeline = backend.compile("add_one", wgsl).unwrap();
        let input = backend.create_buffer(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output = backend.create_buffer_uninit::<f32>(4).unwrap();
        let pulse = backend
            .dispatch_async(&pipeline, &[&input, &output], (1, 1, 1))
            .unwrap();

        // Wait first: the submission retires, so the backend's pool
        // destruction below is legal (backend drop also idles the
        // device defensively, but we keep the deterministic order).
        pulse.wait();

        // The backend (and its epoch tracker) go away next; everything
        // below runs through shared ownership only.
        drop(backend);

        drop(pulse);
        drop(output);
        drop(input);
        drop(pipeline);
    }

    #[test]
    #[serial_test::serial]
    fn persistent_buffer_multi_dispatch() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Vulkan device");
                return;
            }
        };

        let weights = backend
            .create_device_buffer(&[2.0f32, 3.0, 4.0, 5.0])
            .unwrap();
        let output = backend.create_device_buffer_uninit::<f32>(4).unwrap();

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> w: array<f32>;
            @group(0) @binding(1) var<storage, read_write> out: array<f32>;
            @compute @workgroup_size(4)
            fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
                out[gid.x] = w[gid.x] * 10.0;
            }
        "#;
        let pipeline = backend.compile("scale", wgsl).unwrap();

        for _ in 0..5 {
            backend
                .dispatch(&pipeline, &[&weights, &output], (1, 1, 1))
                .unwrap();
        }

        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result.len(), 4);
        for (i, &r) in result.iter().enumerate() {
            let expected = (2.0 + i as f32) * 10.0;
            assert!(
                (r - expected).abs() < 1e-5,
                "mismatch at {i}: {r} vs {expected}"
            );
        }
    }

    /// Miri-compatible: exercises buffer create → read → drop lifecycle.
    /// Run: `cargo +nightly miri test --features vulkan buffer_lifecycle`
    #[test]
    #[serial_test::serial]
    fn buffer_lifecycle_safety() {
        let backend = match VulkanBackend::init() {
            Ok(b) => b,
            Err(_) => return,
        };

        let buf = backend.create_buffer(&[1.0f32, 2.0, 3.0]).unwrap();
        let _ = backend.read_buffer::<f32>(&buf).unwrap();
        drop(buf);

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> out: array<f32>;
            @compute @workgroup_size(4)
            fn fill(@builtin(global_invocation_id) gid: vec3<u32>) {
                out[gid.x] = f32(gid.x);
            }
        "#;
        let p = backend.compile("fill", wgsl).unwrap();
        let buf2 = backend.create_buffer_uninit::<f32>(4).unwrap();
        backend.dispatch(&p, &[&buf2], (1, 1, 1)).unwrap();
        let result = backend.read_buffer::<f32>(&buf2).unwrap();
        assert_eq!(result.len(), 4);
        drop(buf2);
        drop(p);

        let dev_buf = backend.create_device_buffer(&[4.0f32, 5.0, 6.0]).unwrap();
        let _ = backend.read_buffer::<f32>(&dev_buf).unwrap();
        drop(dev_buf);

        let noop_wgsl = r#"
            @group(0) @binding(0) var<storage, read_write> out: array<f32>;
            @compute @workgroup_size(1)
            fn noop(@builtin(global_invocation_id) gid: vec3<u32>) {}
        "#;
        let p2 = backend.compile("noop", noop_wgsl).unwrap();
        drop(p2);
    }
}
