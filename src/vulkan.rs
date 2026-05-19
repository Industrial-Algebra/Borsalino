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

use ash::vk;
use ash::Entry;

use crate::{ComputePipeline, GpuBuffer, GpuBackend, GpuError, Result};

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
    /// Selected physical device.
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    /// Logical device handle.
    device: ash::Device,
    /// Compute queue handle.
    #[allow(dead_code)]
    queue: vk::Queue,
    /// Queue family index for the compute queue.
    #[allow(dead_code)]
    queue_family_index: u32,
    /// Universal pipeline layout — N storage buffer bindings, shared by all pipelines.
    #[allow(dead_code)]
    pipeline_layout: vk::PipelineLayout,
    /// Descriptor set layout for N storage buffers.
    #[allow(dead_code)]
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Descriptor pool for storage buffer descriptor sets.
    #[allow(dead_code)]
    descriptor_pool: vk::DescriptorPool,
    /// Pre-allocated descriptor sets (one per buffer binding index).
    #[allow(dead_code)]
    descriptor_sets: Vec<vk::DescriptorSet>,
    /// Command pool with `RESET_COMMAND_BUFFER_BIT`.
    #[allow(dead_code)]
    command_pool: vk::CommandPool,
}

impl VulkanBackend {
    /// Maximum number of storage buffer bindings per pipeline layout.
    const MAX_BUFFER_BINDINGS: u32 = 8;
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
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
unsafe fn pick_physical_device(
    instance: &ash::Instance,
) -> Result<(vk::PhysicalDevice, u32)> {
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

    best.ok_or_else(|| {
        GpuError::InitFailed("no Vulkan device with compute queue found".into())
    })
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
// GpuBackend implementation
// ═══════════════════════════════════════════════════════════════════

impl GpuBackend for VulkanBackend {
    fn init() -> Result<Self> {
        let entry = unsafe {
            Entry::load().map_err(|e| GpuError::InitFailed(format!("{e}")))?
        };

        let app_name = std::ffi::CString::new("borsalino").unwrap();
        let engine_name = std::ffi::CString::new("borsalino").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(vk::API_VERSION_1_3);

        let instance_create_info =
            vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .map_err(|e| GpuError::InitFailed(format!("vkCreateInstance: {e}")))?
        };

        let (physical_device, queue_family_index) =
            unsafe { pick_physical_device(&instance)? };

        let (device, queue) = unsafe {
            create_device(&instance, physical_device, queue_family_index)?
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

        let dsl_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&dsl_info, None)
                .map_err(|e| {
                    GpuError::InitFailed(format!("create descriptor set layout: {e}"))
                })?
        };

        // ── Pipeline layout ────────────────────────────────────────

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|e| {
                    GpuError::InitFailed(format!("create pipeline layout: {e}"))
                })?
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
                .map_err(|e| {
                    GpuError::InitFailed(format!("create descriptor pool: {e}"))
                })?
        };

        // ── Pre-allocate descriptor sets ───────────────────────────

        let set_layouts: Vec<vk::DescriptorSetLayout> =
            (0..Self::MAX_BUFFER_BINDINGS)
                .map(|_| descriptor_set_layout)
                .collect();

        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&set_info)
                .map_err(|e| {
                    GpuError::InitFailed(format!("allocate descriptor sets: {e}"))
                })?
        };

        // ── Command pool ───────────────────────────────────────────

        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            device
                .create_command_pool(&cmd_pool_info, None)
                .map_err(|e| {
                    GpuError::InitFailed(format!("create command pool: {e}"))
                })?
        };

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            command_pool,
        })
    }

    fn compile(&self, _entry_point: &str, _wgsl_source: &str) -> Result<ComputePipeline> {
        todo!("Task 5: shader compilation")
    }

    fn create_buffer<T: bytemuck::Pod>(&self, _data: &[T]) -> Result<GpuBuffer> {
        todo!("Task 4: buffer lifecycle")
    }

    fn create_buffer_uninit<T: bytemuck::Pod>(&self, _len: usize) -> Result<GpuBuffer> {
        todo!("Task 4: buffer lifecycle")
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
        _pipeline: &ComputePipeline,
        _buffers: &[&GpuBuffer],
        _workgroups: (u32, u32, u32),
        _threads_per_group: (u32, u32, u32),
    ) -> Result<()> {
        todo!("Task 6: dispatch")
    }

    fn read_buffer<T: bytemuck::Pod>(&self, _buffer: &GpuBuffer) -> Result<Vec<T>> {
        todo!("Task 4: buffer lifecycle")
    }
}
