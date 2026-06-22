// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Vulkan compute backend via ash raw FFI.
//!
//! # Architecture
//!
//! Uses `ash` for raw Vulkan 1.3 bindings — the `objc_msgSend` equivalent
//! for Vulkan. Shaders are authored in WGSL and translated to SPIR-V via
//! `naga` at compile time. All dispatch is synchronous: command buffer →
//! submit → wait_idle.
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

use naga::back::spv;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

use ash::Entry;
use ash::vk;

use std::ffi::CString;

use crate::{
    ComputePipeline, DispatchSpec, GpuBackend, GpuBuffer, GpuError, MemoryStrategy, Pulse, Result,
};

// ═══════════════════════════════════════════════════════════════════
// VulkanBackend
// ═══════════════════════════════════════════════════════════════════

/// Vulkan compute backend for Linux and Windows.
///
/// Holds a Vulkan instance, logical device, compute queue, and
/// pre-allocated resources for pipeline layout, descriptor sets,
/// and command buffers. Created via [`VulkanBackend::init`].
///
/// # Platform
///
/// Available on Linux and Windows with the `vulkan` feature enabled.
/// Requires a Vulkan 1.3-capable driver with compute support.
pub struct VulkanBackend {
    /// Vulkan entry (loader). Kept alive for the lifetime of the instance.
    _entry: Entry,
    /// Vulkan instance handle.
    instance: ash::Instance,
    /// Logical device handle.
    device: ash::Device,
    /// Compute queue handle.
    queue: vk::Queue,
    /// Queue family index for the compute queue.
    #[allow(dead_code)]
    queue_family_index: u32,
    /// Minimum storage buffer offset alignment (from device limits).
    min_storage_buffer_offset_alignment: vk::DeviceSize,
    /// Physical device memory properties (for buffer memory type selection).
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Memory strategy: auto-detected or explicitly configured.
    #[allow(dead_code)]
    memory_strategy: MemoryStrategy,
    /// Whether to use device-local memory with staging transfers.
    uses_device_local: bool,
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
    /// Command pool for staging transfers (device ↔ host).
    #[allow(dead_code)]
    transfer_command_pool: vk::CommandPool,
    /// Query pool for GPU timestamps (None if unsupported).
    timestamp_pool: Option<vk::QueryPool>,
    /// GPU timestamp period in nanoseconds (from device limits).
    timestamp_period: f32,
}

impl VulkanBackend {
    /// Maximum number of storage buffer bindings per pipeline layout.
    const MAX_BUFFER_BINDINGS: u32 = 8;

    /// Create a compute pipeline from pre-compiled SPIR-V.
    fn create_pipeline_from_spv(
        &self,
        entry_point: &str,
        spv_words: &[u32],
    ) -> Result<ComputePipeline> {
        let shader_info = vk::ShaderModuleCreateInfo::default().code(spv_words);

        let shader_module = unsafe {
            self.device
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
            self.device
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
            self.device.destroy_shader_module(shader_module, None);
        }

        let inner = Box::new(VulkanPipelineInner {
            pipeline: pipelines[0],
            device: self.device.clone(),
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
            if let Some(pool) = self.timestamp_pool {
                self.device.destroy_query_pool(pool, None);
            }
            self.device
                .destroy_command_pool(self.transfer_command_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Buffer inner type
// ═══════════════════════════════════════════════════════════════════

/// Internal state for a Vulkan GPU buffer, stored behind the opaque
/// `GpuBuffer.raw` pointer.
struct VulkanBufferInner {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    _size: vk::DeviceSize,
    /// Persistently mapped host pointer. For unified memory, points to the
    /// buffer's own mapping. For device-local, points to the staging buffer.
    mapped: *mut std::ffi::c_void,
    /// Staging buffer for device-local memory (None if unified).
    staging_buffer: Option<vk::Buffer>,
    /// Staging buffer memory (None if unified).
    staging_memory: Option<vk::DeviceMemory>,
    /// Clone of the logical device, used for destroy / unmap in drop.
    device: ash::Device,
}

unsafe impl Send for VulkanBufferInner {}
unsafe impl Sync for VulkanBufferInner {}

impl Drop for VulkanBufferInner {
    fn drop(&mut self) {
        unsafe {
            if let Some(sb) = self.staging_buffer {
                self.device.destroy_buffer(sb, None);
            }
            if let Some(sm) = self.staging_memory {
                self.device.free_memory(sm, None);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline inner type
// ═══════════════════════════════════════════════════════════════════

/// Internal state for a Vulkan compute pipeline, stored behind the opaque
/// `ComputePipeline.raw` pointer.
struct VulkanPipelineInner {
    pipeline: vk::Pipeline,
    /// Clone of the logical device, used for destroy in drop.
    device: ash::Device,
}

/// Drop function stored in [`ComputePipeline`] — destroys the Vulkan pipeline.
fn drop_vulkan_pipeline(raw: *mut std::ffi::c_void) {
    if !raw.is_null() {
        unsafe {
            let inner = Box::from_raw(raw as *mut VulkanPipelineInner);
            inner.device.destroy_pipeline(inner.pipeline, None);
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
    device: ash::Device,
}

fn wait_vulkan_pulse(raw: *mut std::ffi::c_void) {
    if !raw.is_null() {
        let inner = unsafe { &*(raw as *const VulkanPulseInner) };
        unsafe {
            let _ =
                inner
                    .device
                    .wait_for_fences(std::slice::from_ref(&inner.fence), true, u64::MAX);
        }
    }
}

fn drop_vulkan_pulse(raw: *mut std::ffi::c_void) {
    if !raw.is_null() {
        let inner = unsafe { Box::from_raw(raw as *mut VulkanPulseInner) };
        unsafe {
            inner.device.destroy_fence(inner.fence, None);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Buffer drop/contents functions
// ═══════════════════════════════════════════════════════════════════

/// Drop function stored in [`GpuBuffer`] — drops the `Box<VulkanBufferInner>`.
fn drop_vulkan_buffer(raw: *mut std::ffi::c_void) {
    if !raw.is_null() {
        unsafe {
            drop(Box::from_raw(raw as *mut VulkanBufferInner));
        }
    }
}

/// Contents function stored in [`GpuBuffer`] — returns the persistently
/// mapped host pointer.
fn contents_vulkan_buffer(raw: *mut std::ffi::c_void) -> *const std::ffi::c_void {
    if raw.is_null() {
        return std::ptr::null();
    }
    let inner = unsafe { &*(raw as *const VulkanBufferInner) };
    inner.mapped
}

// ═══════════════════════════════════════════════════════════════════
// Initialisation helpers
// ═══════════════════════════════════════════════════════════════════

/// Score a device type for preference ordering.
fn device_type_score(ty: vk::PhysicalDeviceType) -> i32 {
    match ty {
        vk::PhysicalDeviceType::DISCRETE_GPU => 100,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 30,
        vk::PhysicalDeviceType::CPU => 20,
        _ => 10,
    }
}

/// Pick the best physical device with a compute-capable queue family.
///
/// Returns `(physical_device, queue_family_index)`, preferring discrete GPUs
/// over integrated, virtual, or CPU devices.
unsafe fn pick_physical_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| GpuError::InitFailed(format!("vkEnumeratePhysicalDevices: {e}")))?
    };

    let mut best_score: i32 = -1;
    let mut best: Option<(vk::PhysicalDevice, u32)> = None;

    for &device in &devices {
        let props = unsafe { instance.get_physical_device_properties(device) };
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(device) };

        // Find a compute-capable queue family
        let qf_index = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|i| i as u32);

        let Some(qf_index) = qf_index else {
            continue;
        };

        let score = device_type_score(props.device_type);
        if score > best_score {
            best_score = score;
            best = Some((device, qf_index));
        }
    }

    best.ok_or_else(|| GpuError::InitFailed("no Vulkan device with compute queue found".into()))
}

/// Create a logical device and retrieve the compute queue.
unsafe fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<(ash::Device, vk::Queue)> {
    let queue_priority = 1.0f32;
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(std::slice::from_ref(&queue_priority));

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info));

    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .map_err(|e| GpuError::InitFailed(format!("vkCreateDevice: {e}")))?
    };

    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    Ok((device, queue))
}

// ═══════════════════════════════════════════════════════════════════
// Utility helpers
// ═══════════════════════════════════════════════════════════════════

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

/// Execute a one-shot command buffer for staging transfers.
unsafe fn one_shot_transfer(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    record: impl FnOnce(vk::CommandBuffer),
) -> Result<()> {
    unsafe {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = device.allocate_command_buffers(&alloc_info).map_err(|e| {
            GpuError::BufferCreationFailed {
                message: format!("transfer allocate: {e}"),
            }
        })?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(cmd, &begin_info).map_err(|e| {
            GpuError::BufferCreationFailed {
                message: format!("transfer begin: {e}"),
            }
        })?;

        record(cmd);

        device
            .end_command_buffer(cmd)
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("transfer end: {e}"),
            })?;

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("transfer submit: {e}"),
            })?;
        device
            .queue_wait_idle(queue)
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("transfer wait: {e}"),
            })?;

        device.free_command_buffers(command_pool, std::slice::from_ref(&cmd));
        Ok(())
    }
}

/// Detect whether device-local memory should be used.
fn detect_device_local(
    device_type: vk::PhysicalDeviceType,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> bool {
    if device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
        return true;
    }
    for i in 0..memory_properties.memory_heap_count {
        if memory_properties.memory_heaps[i as usize]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            && memory_properties.memory_heaps[i as usize].size > 1024 * 1024 * 1024
        {
            return true;
        }
    }
    false
}

// ═══════════════════════════════════════════════════════════════════
// Buffer helpers
// ═══════════════════════════════════════════════════════════════════

/// Round `val` up to the nearest multiple of `alignment`.
fn align_up(val: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    val.div_ceil(alignment) * alignment
}

/// Find a memory type index that satisfies the required type filter and
/// desired property flags.
fn find_memory_type_index(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required_properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    for i in 0..memory_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && memory_properties.memory_types[i as usize]
                .property_flags
                .contains(required_properties)
        {
            return Ok(i);
        }
    }
    Err(GpuError::BufferCreationFailed {
        message: "no suitable memory type found".into(),
    })
}

/// Allocate a Vulkan buffer with the given size and usage flags.
///
/// Returns `(buffer, memory, mapped_ptr)`.
unsafe fn allocate_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut std::ffi::c_void)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("vkCreateBuffer: {e}"),
            })?
    };

    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

    // Prefer cached memory on discrete GPUs (avoids PCIe round-trips).
    // Fall back to uncached coherent if not available (e.g. integrated GPUs).
    let mut flags = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::HOST_CACHED;
    let mem_type_index =
        find_memory_type_index(memory_properties, mem_reqs.memory_type_bits, flags).or_else(
            |_| {
                flags =
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                find_memory_type_index(memory_properties, mem_reqs.memory_type_bits, flags)
            },
        )?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(mem_type_index);

    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("vkAllocateMemory: {e}"),
            })?
    };

    unsafe {
        device.bind_buffer_memory(buffer, memory, 0).map_err(|e| {
            GpuError::BufferCreationFailed {
                message: format!("vkBindBufferMemory: {e}"),
            }
        })?;
    }

    let mapped = unsafe {
        device
            .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("vkMapMemory: {e}"),
            })?
    };

    Ok((buffer, memory, mapped))
}

/// Allocate a device-local buffer (VRAM on discrete GPUs).
/// Not mapped — data transfers require staging buffers.
unsafe fn allocate_device_local_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("vkCreateBuffer(device-local): {e}"),
            })?
    };

    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mem_type_index = find_memory_type_index(
        memory_properties,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(mem_type_index);

    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("vkAllocateMemory(device-local): {e}"),
            })?
    };

    unsafe {
        device.bind_buffer_memory(buffer, memory, 0).map_err(|e| {
            GpuError::BufferCreationFailed {
                message: format!("vkBindBufferMemory(device-local): {e}"),
            }
        })?;
    }

    Ok((buffer, memory))
}

// ═══════════════════════════════════════════════════════════════════
// GpuBackend implementation
// ═══════════════════════════════════════════════════════════════════

impl GpuBackend for VulkanBackend {
    fn init() -> Result<Self> {
        Self::init_with_strategy(MemoryStrategy::Auto)
    }

    fn init_with_strategy(strategy: MemoryStrategy) -> Result<Self> {
        let entry = unsafe { Entry::load().map_err(|e| GpuError::InitFailed(format!("{e}")))? };

        let app_name = std::ffi::CString::new("borsalino").unwrap();
        let engine_name = std::ffi::CString::new("borsalino").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(vk::API_VERSION_1_3);

        let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .map_err(|e| GpuError::InitFailed(format!("vkCreateInstance: {e}")))?
        };

        let (physical_device, queue_family_index) = unsafe { pick_physical_device(&instance)? };

        // Query device properties before creating logical device
        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let min_storage_buffer_offset_alignment =
            device_props.limits.min_storage_buffer_offset_alignment;

        // Auto-detect or use explicit memory strategy
        let uses_device_local = match strategy {
            MemoryStrategy::DeviceLocal => true,
            MemoryStrategy::Unified => false,
            MemoryStrategy::Auto => {
                detect_device_local(device_props.device_type, &memory_properties)
            }
        };

        let (device, queue) =
            unsafe { create_device(&instance, physical_device, queue_family_index)? };

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

        // ── Pipeline layout ────────────────────────────────────────

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|e| GpuError::InitFailed(format!("create pipeline layout: {e}")))?
        };

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

        // ── Transfer command pool ─────────────────────────────────

        let transfer_cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let transfer_command_pool = unsafe {
            device
                .create_command_pool(&transfer_cmd_pool_info, None)
                .map_err(|e| GpuError::InitFailed(format!("create transfer pool: {e}")))?
        };

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
            Some(pool)
        } else {
            None
        };
        let timestamp_period = device_props.limits.timestamp_period;

        Ok(Self {
            _entry: entry,
            instance,
            device,
            queue,
            queue_family_index,
            min_storage_buffer_offset_alignment,
            memory_properties,
            memory_strategy: strategy,
            uses_device_local,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            command_pool,
            transfer_command_pool,
            timestamp_pool,
            timestamp_period,
        })
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
            self.device
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
            self.device
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
            self.device.destroy_shader_module(shader_module, None);
        }

        // Step 7: Wrap in opaque handle
        let inner = Box::new(VulkanPipelineInner {
            pipeline: pipelines[0],
            device: self.device.clone(),
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
        let element_size = std::mem::size_of::<T>();
        let byte_len = std::mem::size_of_val(data) as vk::DeviceSize;

        let aligned_size = if byte_len == 0 {
            self.min_storage_buffer_offset_alignment
        } else {
            align_up(byte_len, self.min_storage_buffer_offset_alignment)
        };

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        let (mapped, buffer, memory, staging_buffer, staging_memory) = if self.uses_device_local {
            // Allocate device-local buffer + staging buffer
            let (dev_buf, dev_mem) = unsafe {
                allocate_device_local_buffer(
                    &self.device,
                    &self.memory_properties,
                    aligned_size,
                    usage,
                )?
            };

            // Allocate staging buffer (host-visible)
            let (stg_buf, stg_mem, stg_mapped) = unsafe {
                allocate_buffer(
                    &self.device,
                    &self.memory_properties,
                    aligned_size,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                )?
            };

            // Copy data to staging, then staging → device
            if byte_len > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const std::ffi::c_void,
                        stg_mapped,
                        byte_len as usize,
                    );
                }
                unsafe {
                    one_shot_transfer(
                        &self.device,
                        self.transfer_command_pool,
                        self.queue,
                        |cmd| {
                            let copy = vk::BufferCopy::default().size(aligned_size);
                            self.device.cmd_copy_buffer(
                                cmd,
                                stg_buf,
                                dev_buf,
                                std::slice::from_ref(&copy),
                            );
                        },
                    )?;
                }
            }

            (stg_mapped, dev_buf, dev_mem, Some(stg_buf), Some(stg_mem))
        } else {
            // Unified memory: single host-visible buffer
            let (buf, mem, mapped) = unsafe {
                allocate_buffer(&self.device, &self.memory_properties, aligned_size, usage)?
            };
            if byte_len > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const std::ffi::c_void,
                        mapped,
                        byte_len as usize,
                    );
                }
            }
            (mapped, buf, mem, None, None)
        };

        let inner = Box::new(VulkanBufferInner {
            buffer,
            memory,
            _size: aligned_size,
            mapped,
            staging_buffer,
            staging_memory,
            device: self.device.clone(),
        });

        Ok(GpuBuffer {
            raw: Box::into_raw(inner) as *mut std::ffi::c_void,
            len: data.len(),
            element_size,
            drop_fn: drop_vulkan_buffer,
            contents_fn: contents_vulkan_buffer,
        })
    }

    fn create_buffer_uninit<T: bytemuck::Pod>(&self, len: usize) -> Result<GpuBuffer> {
        let element_size = std::mem::size_of::<T>();
        let byte_len = (len * element_size) as vk::DeviceSize;

        let aligned_size = if byte_len == 0 {
            self.min_storage_buffer_offset_alignment
        } else {
            align_up(byte_len, self.min_storage_buffer_offset_alignment)
        };

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        let (mapped, buffer, memory, staging_buffer, staging_memory) = if self.uses_device_local {
            let (dev_buf, dev_mem) = unsafe {
                allocate_device_local_buffer(
                    &self.device,
                    &self.memory_properties,
                    aligned_size,
                    usage,
                )?
            };
            let (stg_buf, stg_mem, stg_mapped) = unsafe {
                allocate_buffer(
                    &self.device,
                    &self.memory_properties,
                    aligned_size,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                )?
            };
            (stg_mapped, dev_buf, dev_mem, Some(stg_buf), Some(stg_mem))
        } else {
            let (buf, mem, mapped) = unsafe {
                allocate_buffer(&self.device, &self.memory_properties, aligned_size, usage)?
            };
            (mapped, buf, mem, None, None)
        };

        let inner = Box::new(VulkanBufferInner {
            buffer,
            memory,
            _size: aligned_size,
            mapped,
            staging_buffer,
            staging_memory,
            device: self.device.clone(),
        });

        Ok(GpuBuffer {
            raw: Box::into_raw(inner) as *mut std::ffi::c_void,
            len,
            element_size,
            drop_fn: drop_vulkan_buffer,
            contents_fn: contents_vulkan_buffer,
        })
    }

    fn create_device_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        let element_size = std::mem::size_of::<T>();
        let byte_len = std::mem::size_of_val(data) as vk::DeviceSize;

        let aligned_size = if byte_len == 0 {
            self.min_storage_buffer_offset_alignment
        } else {
            align_up(byte_len, self.min_storage_buffer_offset_alignment)
        };

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        // Device-local buffer: VRAM on discrete, host-visible on unified
        let (dev_buf, dev_mem) = if self.uses_device_local {
            unsafe {
                allocate_device_local_buffer(
                    &self.device,
                    &self.memory_properties,
                    aligned_size,
                    usage,
                )?
            }
        } else {
            let (buf, mem, mapped) = unsafe {
                allocate_buffer(&self.device, &self.memory_properties, aligned_size, usage)?
            };
            // Upload data directly (unified memory)
            if byte_len > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const std::ffi::c_void,
                        mapped,
                        byte_len as usize,
                    );
                }
            }
            let inner = Box::new(VulkanBufferInner {
                buffer: buf,
                memory: mem,
                _size: aligned_size,
                mapped,
                staging_buffer: None,
                staging_memory: None,
                device: self.device.clone(),
            });

            return Ok(GpuBuffer {
                raw: Box::into_raw(inner) as *mut std::ffi::c_void,
                len: data.len(),
                element_size,
                drop_fn: drop_vulkan_buffer,
                contents_fn: contents_vulkan_buffer,
            });
        };

        // On discrete GPU: upload via temp staging buffer, then free it
        let (stg_buf, stg_mem, stg_mapped) = unsafe {
            allocate_buffer(
                &self.device,
                &self.memory_properties,
                aligned_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?
        };

        if byte_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const std::ffi::c_void,
                    stg_mapped,
                    byte_len as usize,
                );
            }
            unsafe {
                one_shot_transfer(
                    &self.device,
                    self.transfer_command_pool,
                    self.queue,
                    |cmd| {
                        let copy = vk::BufferCopy::default().size(aligned_size);
                        self.device.cmd_copy_buffer(
                            cmd,
                            stg_buf,
                            dev_buf,
                            std::slice::from_ref(&copy),
                        );
                    },
                )?;
            }
        }

        // Free temporary staging buffer
        unsafe {
            self.device.destroy_buffer(stg_buf, None);
            self.device.free_memory(stg_mem, None);
        }

        let inner = Box::new(VulkanBufferInner {
            buffer: dev_buf,
            memory: dev_mem,
            _size: aligned_size,
            mapped: std::ptr::null_mut(),
            staging_buffer: None,
            staging_memory: None,
            device: self.device.clone(),
        });

        Ok(GpuBuffer {
            raw: Box::into_raw(inner) as *mut std::ffi::c_void,
            len: data.len(),
            element_size,
            drop_fn: drop_vulkan_buffer,
            contents_fn: contents_vulkan_buffer,
        })
    }

    fn create_device_buffer_uninit<T: bytemuck::Pod>(&self, len: usize) -> Result<GpuBuffer> {
        let element_size = std::mem::size_of::<T>();
        let byte_len = (len * element_size) as vk::DeviceSize;

        let aligned_size = if byte_len == 0 {
            self.min_storage_buffer_offset_alignment
        } else {
            align_up(byte_len, self.min_storage_buffer_offset_alignment)
        };

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        let (buf, mem) = if self.uses_device_local {
            unsafe {
                allocate_device_local_buffer(
                    &self.device,
                    &self.memory_properties,
                    aligned_size,
                    usage,
                )?
            }
        } else {
            let (buf, mem, mapped) = unsafe {
                allocate_buffer(&self.device, &self.memory_properties, aligned_size, usage)?
            };
            let inner = Box::new(VulkanBufferInner {
                buffer: buf,
                memory: mem,
                _size: aligned_size,
                mapped,
                staging_buffer: None,
                staging_memory: None,
                device: self.device.clone(),
            });
            return Ok(GpuBuffer {
                raw: Box::into_raw(inner) as *mut std::ffi::c_void,
                len,
                element_size,
                drop_fn: drop_vulkan_buffer,
                contents_fn: contents_vulkan_buffer,
            });
        };

        let inner = Box::new(VulkanBufferInner {
            buffer: buf,
            memory: mem,
            _size: aligned_size,
            mapped: std::ptr::null_mut(),
            staging_buffer: None,
            staging_memory: None,
            device: self.device.clone(),
        });

        Ok(GpuBuffer {
            raw: Box::into_raw(inner) as *mut std::ffi::c_void,
            len,
            element_size,
            drop_fn: drop_vulkan_buffer,
            contents_fn: contents_vulkan_buffer,
        })
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
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkAllocateCommandBuffers: {e}"),
                })?
        }[0];

        // ── Begin command buffer ──────────────────────────────────

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkBeginCommandBuffer: {e}"),
                })?;
        }

        // ── Bind pipeline ─────────────────────────────────────────

        unsafe {
            let vk_pipeline = (*(pipeline.raw as *const VulkanPipelineInner)).pipeline;
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, vk_pipeline);
        }

        // ── Update descriptor set + bind ──────────────────────────

        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(nbuffers);
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(nbuffers);

        // Keep buffer_infos alive on the heap — the writes reference them
        for buf in buffers.iter() {
            let inner = unsafe { &*(buf.raw as *const VulkanBufferInner) };
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(inner.buffer)
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
            self.device.update_descriptor_sets(&writes, &[]);
        }

        unsafe {
            self.device.cmd_bind_descriptor_sets(
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
            self.device
                .cmd_dispatch(cmd, workgroups.0, workgroups.1, workgroups.2);
        }

        // ── Memory barrier (shader write → host read) ─────────────

        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);

        unsafe {
            self.device.cmd_pipeline_barrier(
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
            self.device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkEndCommandBuffer: {e}"),
                })?;
        }

        // ── Submit + wait ─────────────────────────────────────────

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], vk::Fence::null())
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkQueueSubmit: {e}"),
                })?;

            self.device
                .queue_wait_idle(self.queue)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkQueueWaitIdle: {e}"),
                })?;
        }

        // ── Cleanup ───────────────────────────────────────────────

        unsafe {
            self.device
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
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkAllocateCommandBuffers: {e}"),
                })?
        }[0];

        // ── Begin ───────────────────────────────────────────────

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
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
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, vk_pipeline);
            }

            // Update descriptor set + bind
            let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(nbuffers);
            let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(nbuffers);

            for buf in spec.buffers.iter() {
                let inner = unsafe { &*(buf.raw as *const VulkanBufferInner) };
                buffer_infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(inner.buffer)
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
                self.device.update_descriptor_sets(&writes, &[]);
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    std::slice::from_ref(&self.descriptor_set),
                    &[],
                );

                self.device.cmd_dispatch(
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
            self.device.cmd_pipeline_barrier(
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
            self.device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkEndCommandBuffer: {e}"),
                })?;
        }

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], vk::Fence::null())
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkQueueSubmit: {e}"),
                })?;

            self.device
                .queue_wait_idle(self.queue)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkQueueWaitIdle: {e}"),
                })?;
        }

        // ── Cleanup ─────────────────────────────────────────────

        unsafe {
            self.device
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
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkAllocateCommandBuffers: {e}"),
                })?
        }[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkBeginCommandBuffer: {e}"),
                })?;
        }

        // Bind pipeline
        unsafe {
            let vk_pipeline = (*(pipeline.raw as *const VulkanPipelineInner)).pipeline;
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, vk_pipeline);
        }

        // Descriptor set + bind
        let mut buffer_infos = Vec::with_capacity(nbuffers);
        let mut writes = Vec::with_capacity(nbuffers);
        for buf in buffers.iter() {
            let inner = unsafe { &*(buf.raw as *const VulkanBufferInner) };
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(inner.buffer)
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
            self.device.update_descriptor_sets(&writes, &[]);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_set),
                &[],
            );
            self.device
                .cmd_dispatch(cmd, workgroups.0, workgroups.1, workgroups.2);

            // Barrier: shader write → host read (applied when waited)
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );

            self.device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkEndCommandBuffer: {e}"),
                })?;
        }

        // Create fence for async completion signal
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            self.device
                .create_fence(&fence_info, None)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkCreateFence: {e}"),
                })?
        };

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], fence)
                .map_err(|e| GpuError::DispatchFailed {
                    message: format!("vkQueueSubmit: {e}"),
                })?;
        }

        // Free command buffer (work is submitted, fence tracks completion)
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
        }

        let inner = Box::new(VulkanPulseInner {
            fence,
            device: self.device.clone(),
        });

        Ok(Pulse {
            raw: Box::into_raw(inner) as *mut std::ffi::c_void,
            wait_fn: wait_vulkan_pulse,
            drop_fn: drop_vulkan_pulse,
        })
    }

    fn read_buffer<T: bytemuck::Pod>(&self, buffer: &GpuBuffer) -> Result<Vec<T>> {
        let inner = unsafe { &*(buffer.raw as *const VulkanBufferInner) };

        if self.uses_device_local {
            // Device-local with persistent staging (from create_buffer)
            if inner.staging_buffer.is_some() {
                if let (Some(stg_buf), Some(_stg_mem)) =
                    (inner.staging_buffer, inner.staging_memory)
                {
                    unsafe {
                        one_shot_transfer(
                            &self.device,
                            self.transfer_command_pool,
                            self.queue,
                            |cmd| {
                                let copy = vk::BufferCopy::default().size(inner._size);
                                self.device.cmd_copy_buffer(
                                    cmd,
                                    inner.buffer,
                                    stg_buf,
                                    std::slice::from_ref(&copy),
                                );
                            },
                        )?;
                    }
                }
            } else if inner.mapped.is_null() {
                // Device-local without staging (from create_device_buffer).
                // Allocate temp staging, copy, read, free.
                let (stg_buf, stg_mem, stg_mapped) = unsafe {
                    allocate_buffer(
                        &self.device,
                        &self.memory_properties,
                        inner._size,
                        vk::BufferUsageFlags::TRANSFER_DST,
                    )?
                };
                unsafe {
                    one_shot_transfer(
                        &self.device,
                        self.transfer_command_pool,
                        self.queue,
                        |cmd| {
                            let copy = vk::BufferCopy::default().size(inner._size);
                            self.device.cmd_copy_buffer(
                                cmd,
                                inner.buffer,
                                stg_buf,
                                std::slice::from_ref(&copy),
                            );
                        },
                    )?;
                }

                let contents = stg_mapped as *const T;
                if contents.is_null() {
                    unsafe {
                        self.device.destroy_buffer(stg_buf, None);
                        self.device.free_memory(stg_mem, None);
                    }
                    return Err(GpuError::BufferReadFailed {
                        message: "staging buffer map failed".into(),
                    });
                }
                let slice = unsafe { std::slice::from_raw_parts(contents, buffer.len) };
                let result = slice.to_vec();

                unsafe {
                    self.device.destroy_buffer(stg_buf, None);
                    self.device.free_memory(stg_mem, None);
                }
                return Ok(result);
            }
        }

        let contents = (buffer.contents_fn)(buffer.raw) as *const T;
        if contents.is_null() {
            return Err(GpuError::BufferReadFailed {
                message: "buffer contents pointer is null".into(),
            });
        }
        let slice = unsafe { std::slice::from_raw_parts(contents, buffer.len) };
        Ok(slice.to_vec())
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
                .device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| GpuError::Internal(format!("timestamp alloc: {e}")))?[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| GpuError::Internal(format!("timestamp begin: {e}")))?;

            // Reset query pool before use
            self.device.reset_query_pool(pool, 0, 1);

            // Write GPU timestamp
            self.device
                .cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_COMMANDS, pool, 0);

            self.device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::Internal(format!("timestamp end: {e}")))?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            self.device
                .queue_submit(self.queue, &[submit_info], vk::Fence::null())
                .map_err(|e| GpuError::Internal(format!("timestamp submit: {e}")))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|e| GpuError::Internal(format!("timestamp wait: {e}")))?;

            // Read back timestamp
            let mut ts_data = [0u64];
            self.device
                .get_query_pool_results(
                    pool,
                    0,
                    &mut ts_data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|e| GpuError::Internal(format!("timestamp read: {e}")))?;

            self.device
                .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));

            // Convert ticks to nanoseconds
            Ok((ts_data[0] as f64 * self.timestamp_period as f64) as u64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
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

    #[test]
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

    #[test]
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
