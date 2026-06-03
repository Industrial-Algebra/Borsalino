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

use ash::vk;
use ash::Entry;

use std::ffi::CString;

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
// Buffer inner type
// ═══════════════════════════════════════════════════════════════════

/// Internal state for a Vulkan GPU buffer, stored behind the opaque
/// `GpuBuffer.raw` pointer.
struct VulkanBufferInner {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    _size: vk::DeviceSize,
    /// Persistently mapped host pointer. Valid until the buffer is dropped.
    mapped: *mut std::ffi::c_void,
    /// Clone of the logical device, used for destroy / unmap in drop.
    device: ash::Device,
}

unsafe impl Send for VulkanBufferInner {}
unsafe impl Sync for VulkanBufferInner {}

impl Drop for VulkanBufferInner {
    fn drop(&mut self) {
        unsafe {
            // vkFreeMemory implicitly unmaps
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
    let mut flags =
        vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
    let mem_type_index = find_memory_type_index(
        memory_properties,
        mem_reqs.memory_type_bits,
        flags,
    )
    .or_else(|_| {
        flags = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        find_memory_type_index(memory_properties, mem_reqs.memory_type_bits, flags)
    })?;

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
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("vkBindBufferMemory: {e}"),
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

        // Query device properties before creating logical device
        let device_props =
            unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };
        let min_storage_buffer_offset_alignment =
            device_props.limits.min_storage_buffer_offset_alignment;

        let (device, queue) = unsafe {
            create_device(&instance, physical_device, queue_family_index)?
        };

        // ── Descriptor set layout (N storage buffers) ──────────────

        let bindings: Vec<vk::DescriptorSetLayoutBinding> =
            (0..Self::MAX_BUFFER_BINDINGS)
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

        // ── Pre-allocate descriptor set ───────────────────────────

        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&set_info)
                .map_err(|e| {
                    GpuError::InitFailed(format!("allocate descriptor set: {e}"))
                })?
        }[0];

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
            device,
            queue,
            queue_family_index,
            min_storage_buffer_offset_alignment,
            memory_properties,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            command_pool,
        })
    }

    fn compile(&self, entry_point: &str, wgsl_source: &str) -> Result<ComputePipeline> {
        // Step 1: Parse WGSL → naga IR
        let module = wgsl::parse_str(wgsl_source).map_err(|e| {
            GpuError::CompileFailed {
                entry: entry_point.into(),
                message: e.emit_to_string(wgsl_source),
            }
        })?;

        // Step 2: Validate the module
        let mut validator =
            Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator.validate(&module).map_err(|e| {
            GpuError::CompileFailed {
                entry: entry_point.into(),
                message: e.emit_to_string(wgsl_source),
            }
        })?;

        // Step 3: Emit SPIR-V
        let spv_words = spv::write_vec(
            &module,
            &info,
            &spv::Options::default(),
            None,
        )
        .map_err(|e| GpuError::CompileFailed {
            entry: entry_point.into(),
            message: format!("SPIR-V emission failed: {e}"),
        })?;

        // Step 4: Create Vulkan shader module
        let shader_info = vk::ShaderModuleCreateInfo::default()
            .code(&spv_words);

        let shader_module = unsafe {
            self.device
                .create_shader_module(&shader_info, None)
                .map_err(|e| GpuError::CompileFailed {
                    entry: entry_point.into(),
                    message: format!("vkCreateShaderModule: {e}"),
                })?
        };

        // Step 5: Create compute pipeline
        let entry_name =
            CString::new(entry_point).map_err(|_| GpuError::CompileFailed {
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

    fn create_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        let element_size = std::mem::size_of::<T>();
        let byte_len = std::mem::size_of_val(data) as vk::DeviceSize;

        // Pad to satisfy minStorageBufferOffsetAlignment
        let aligned_size = if byte_len == 0 {
            self.min_storage_buffer_offset_alignment
        } else {
            align_up(byte_len, self.min_storage_buffer_offset_alignment)
        };

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        let (buffer, memory, mapped) = unsafe {
            allocate_buffer(
                &self.device,
                &self.memory_properties,
                aligned_size,
                usage,
            )?
        };

        // Upload initial data
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
            buffer,
            memory,
            _size: aligned_size,
            mapped,
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

        let (buffer, memory, mapped) = unsafe {
            allocate_buffer(
                &self.device,
                &self.memory_properties,
                aligned_size,
                usage,
            )?
        };

        let inner = Box::new(VulkanBufferInner {
            buffer,
            memory,
            _size: aligned_size,
            mapped,
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
            let vk_pipeline =
                (*(pipeline.raw as *const VulkanPipelineInner)).pipeline;
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                vk_pipeline,
            );
        }

        // ── Update descriptor set + bind ──────────────────────────

        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> =
            Vec::with_capacity(nbuffers);
        let mut writes: Vec<vk::WriteDescriptorSet> =
            Vec::with_capacity(nbuffers);

        // Keep buffer_infos alive on the heap — the writes reference them
        for buf in buffers.iter() {
            let inner = unsafe {
                &*(buf.raw as *const VulkanBufferInner)
            };
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
            self.device
                .update_descriptor_sets(&writes, &[]);
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
            self.device.cmd_dispatch(
                cmd,
                workgroups.0,
                workgroups.1,
                workgroups.2,
            );
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

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cmd));

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
                .free_command_buffers(
                    self.command_pool,
                    std::slice::from_ref(&cmd),
                );
        }

        Ok(())
    }

    fn read_buffer<T: bytemuck::Pod>(&self, buffer: &GpuBuffer) -> Result<Vec<T>> {
        let contents = (buffer.contents_fn)(buffer.raw) as *const T;
        if contents.is_null() {
            return Err(GpuError::BufferReadFailed {
                message: "buffer contents pointer is null".into(),
            });
        }
        let slice =
            unsafe { std::slice::from_raw_parts(contents, buffer.len) };
        Ok(slice.to_vec())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

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
}
