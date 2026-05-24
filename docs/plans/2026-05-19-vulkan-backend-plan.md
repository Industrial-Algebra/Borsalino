# Vulkan Backend Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Implement a full Vulkan compute backend for Borsalino via ash (raw FFI) and naga (WGSL→SPIR-V), switching the shader language from MSL to WGSL across the crate.

**Architecture:** `GpuBackend` trait gets WGSL input. Metal backend: naga WGSL→MSL. Vulkan backend: naga WGSL→SPIR-V + ash FFI. Opaque handles unchanged. Synchronous dispatch via cmd buffer → submit → wait_idle.

**Tech Stack:** naga 27 (wgsl-in, spv-out, msl-out), ash 0.38 (Vulkan 1.3), std only

---

### Task 1: Update dependencies and trait signature

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/lib.rs` (trait signature + docs)
- Modify: `src/metal.rs` (stub compile param rename)
- Modify: `src/error.rs` (if needed)

**Step 1: Update Cargo.toml**

Replace vulkano deps with ash + naga:
```toml
[dependencies]
thiserror = "2"
bytemuck = { version = "1", features = ["derive"] }
naga = { version = "27", features = ["wgsl-in", "spv-out", "msl-out"] }

[features]
default = []
metal = []
vulkan = ["dep:ash"]

[target.'cfg(target_os = "linux")'.dependencies]
ash = { version = "0.38", optional = true }

[target.'cfg(target_os = "windows")'.dependencies]
ash = { version = "0.38", optional = true }
```

Remove: `vulkano`, `vulkano-shaders` optional deps entirely.

**Step 2: Rename trait parameter**

In `src/lib.rs`, rename `msl_source` → `wgsl_source` in the `compile()` method signature and all doc comments. Update the module-level doc to say "Write WGSL compute kernels" instead of "Write Metal Shading Language kernels."

**Step 3: Update stub backends**

Rename the unused parameter in VulkanStub and NoBackendStub `compile()` impls from `_msl` to `_wgsl`.

**Step 4: Update metal.rs parameter name**

Rename `msl_source` → `wgsl_source` in `MetalBackend::compile()`. This is just the parameter name — the actual naga WGSL→MSL translation comes in Task 2.

**Step 5: Verify builds**

```bash
cargo check && cargo check --features metal
```

Expected: clean compile.

**Step 6: Commit**

```bash
git add Cargo.toml src/lib.rs src/metal.rs
git commit -m "refactor: switch shader language from MSL to WGSL, add naga+ash deps"
```

---

### Task 2: Metal backend WGSL→MSL translation

**Files:**
- Modify: `src/metal.rs` (compile method)
- Modify: `src/main.rs` (MSL → WGSL)

**Step 1: Add naga WGSL→MSL translation to MetalBackend::compile()**

In `metal.rs`, before the existing Metal compilation steps, add:

```rust
use naga::front::wgsl;
use naga::back::msl;
use naga::valid::Validator;

// Inside compile():
// Step 0: Parse WGSL → naga IR
let module = wgsl::parse_str(wgsl_source).map_err(|e| {
    GpuError::CompileFailed {
        entry: entry_point.into(),
        message: e.emit_to_string(wgsl_source),
    }
})?;

// Validate
let mut validator = Validator::new(
    naga::valid::ValidationFlags::all(),
    naga::valid::Capabilities::all(),
);
let info = validator.validate(&module).map_err(|e| {
    GpuError::CompileFailed {
        entry: entry_point.into(),
        message: e.emit_to_string(wgsl_source),
    }
})?;

// Emit MSL
let (msl_source, _) = msl::write_string(
    &module, &info,
    &msl::Options::default(),
    &msl::PipelineOptions::default(),
).map_err(|e| GpuError::CompileFailed {
    entry: entry_point.into(),
    message: format!("MSL emission failed: {e}"),
})?;
```

Then feed `msl_source` into the existing Metal compile pipeline (unchanged).

**Step 2: Update main.rs to WGSL**

Replace the SAXPY MSL kernel with WGSL equivalent:
```wgsl
@compute @workgroup_size(256)
fn saxpy(@builtin(global_invocation_id) id: vec3<u32>,
          @storage(0) x: array<f32>,
          @storage(1) y: array<f32>,
          @storage(2) out: array<f32>) {
    let i = id.x;
    out[i] = 2.5 * x[i] + y[i];
}
```

**Step 3: Update metal.rs tests to WGSL**

Convert `add_one_kernel` and `vector_scale_1024` tests from MSL to WGSL.

**Step 4: Verify**

```bash
cargo check --features metal && cargo clippy --features metal
```

**Step 5: Commit**

```bash
git add src/metal.rs src/main.rs
git commit -m "feat: naga WGSL-to-MSL translation in Metal backend"
```

---

### Task 3: VulkanBackend scaffolding

**Files:**
- Create: `src/vulkan.rs`
- Modify: `src/lib.rs` (add module, update init())

**Step 1: Create src/vulkan.rs with struct and init()**

```rust
// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Vulkan compute backend via ash raw FFI.

use std::ffi::c_void;
use crate::{ComputePipeline, GpuBuffer, GpuBackend, GpuError, Result};

use ash::vk;
use ash::Entry;

pub struct VulkanBackend {
    _entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    // Pre-allocated (Task 5-6)
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    command_pool: vk::CommandPool,
}

impl VulkanBackend {
    /// Maximum number of storage buffer bindings per pipeline layout.
    const MAX_BUFFER_BINDINGS: u32 = 8;
}

impl GpuBackend for VulkanBackend {
    fn init() -> Result<Self> {
        // To be filled in
        todo!()
    }
    fn compile(&self, _entry: &str, _wgsl: &str) -> Result<ComputePipeline> {
        todo!()
    }
    fn create_buffer<T: bytemuck::Pod>(&self, _data: &[T]) -> Result<GpuBuffer> {
        todo!()
    }
    fn create_buffer_uninit<T: bytemuck::Pod>(&self, _len: usize) -> Result<GpuBuffer> {
        todo!()
    }
    fn dispatch(&self, pipeline: &ComputePipeline, buffers: &[&GpuBuffer], workgroups: (u32, u32, u32)) -> Result<()> {
        self.dispatch_ex(pipeline, buffers, workgroups, (256, 1, 1))
    }
    fn dispatch_ex(&self, _pipeline: &ComputePipeline, _buffers: &[&GpuBuffer], _workgroups: (u32, u32, u32), _threads_per_group: (u32, u32, u32)) -> Result<()> {
        todo!()
    }
    fn read_buffer<T: bytemuck::Pod>(&self, _buffer: &GpuBuffer) -> Result<Vec<T>> {
        todo!()
    }
}
```

**Step 2: Implement init() — instance creation**

```rust
fn init() -> Result<Self> {
    let entry = unsafe { Entry::load().map_err(|e| GpuError::InitFailed(format!("{e}")))? };

    let app_name = std::ffi::CString::new("borsalino").unwrap();
    let engine_name = std::ffi::CString::new("borsalino").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .engine_name(&engine_name)
        .api_version(vk::API_VERSION_1_3);

    let instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info);

    let instance = unsafe {
        entry.create_instance(&instance_create_info, None)
            .map_err(|e| GpuError::InitFailed(format!("vkCreateInstance: {e}")))?
    };

    // Physical device and device creation in a helper
    todo!() // picks device, creates logical device + queue
}
```

**Step 3: Implement pick_compute_device() helper**

Enumerate physical devices, score by type (discrete=100, integrated=50, other=10), pick the highest-scoring device with a compute-capable queue family.

**Step 4: Implement create_device() helper**

Create logical device with compute queue, no extensions, no validation layers.

**Step 5: Wire into lib.rs**

Add `mod vulkan;` gated on `#[cfg(all(feature = "vulkan", not(target_os = "macos")))]`.
Update the corresponding `init()` to return `vulkan::VulkanBackend`:
```rust
#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
pub fn init() -> Result<vulkan::VulkanBackend> {
    vulkan::VulkanBackend::init()
}
```

**Step 6: Verify**

```bash
cargo check --features vulkan
```

**Step 7: Commit**

```bash
git add src/vulkan.rs src/lib.rs
git commit -m "feat: VulkanBackend scaffolding with instance/device init"
```

---

### Task 4: Vulkan buffer lifecycle

**Files:**
- Modify: `src/vulkan.rs` (buffer inner struct, create, read, drop)

**Step 1: Add VulkanBufferInner**

```rust
struct VulkanBufferInner {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    mapped: *mut c_void,
    device: ash::Device, // cloned for drop_fn
}

unsafe impl Send for VulkanBufferInner {}
unsafe impl Sync for VulkanBufferInner {}

impl Drop for VulkanBufferInner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
```

**Step 2: Implement create_buffer()**

1. Compute size: `align_to(data.len() * size_of::<T>(), min_storage_buffer_offset_alignment)`
2. `vkCreateBuffer` with STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST usage
3. `vkAllocateMemory` with HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
4. `vkBindBufferMemory`
5. `vkMapMemory` → store persistent mapping in `mapped`
6. `ptr::copy_nonoverlapping(data, mapped, byte_len)`
7. Wrap in `GpuBuffer { raw, len, element_size, drop_fn, contents_fn }`

Drop fn: `|raw| drop(Box::from_raw(raw as *mut VulkanBufferInner))`
Contents fn: `|raw| (*(raw as *const VulkanBufferInner)).mapped`

**Step 3: Implement create_buffer_uninit()**

Same as create_buffer but skip step 6 (no initial data copy).

**Step 4: Implement read_buffer()**

`contents_fn` returns `inner.mapped`. Convert to `&[T]` → `to_vec()`.

**Step 5: Verify**

```bash
cargo check --features vulkan
```

**Step 6: Commit**

```bash
git add src/vulkan.rs
git commit -m "feat: Vulkan buffer lifecycle"
```

---

### Task 5: Vulkan shader compilation

**Files:**
- Modify: `src/vulkan.rs` (compile method + pipeline layout creation)

**Step 1: Implement compile() — naga WGSL→SPIR-V + vk pipeline**

1. `naga::front::wgsl::parse_str(source)` → Module
2. `Validator::validate()` → ModuleInfo
3. `spv::write_vec()` → Vec<u32>
4. `device.create_shader_module()` → VkShaderModule
5. `device.create_compute_pipelines()` with universal pipeline layout → VkPipeline
6. Destroy shader module (pipeline owns the compiled code)
7. Wrap in `ComputePipeline { raw, drop_fn }`

SPIR-V options: use `spv::Options::default()` with entry point specified.

Drop fn: `|raw| device.destroy_pipeline(raw as VkPipeline, None)`

**Step 2: Create pipeline layout in init()**

After device creation, create one universal `VkPipelineLayout` with N storage buffer bindings:
```rust
let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..Self::MAX_BUFFER_BINDINGS)
    .map(|i| vk::DescriptorSetLayoutBinding::default()
        .binding(i)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE))
    .collect();
```

**Step 3: Create descriptor pool + pre-allocate sets in init()**

Single pool with MAX_BUFFER_BINDINGS sets × 1 storage buffer each. Pre-allocate descriptor sets.

**Step 4: Update descriptor sets per dispatch**

In dispatch (Task 6), update descriptor sets with the actual buffer handles via `device.update_descriptor_sets()`.

**Step 5: Verify**

```bash
cargo check --features vulkan
```

**Step 6: Commit**

```bash
git add src/vulkan.rs
git commit -m "feat: Vulkan shader compilation via naga+ash"
```

---

### Task 6: Vulkan dispatch

**Files:**
- Modify: `src/vulkan.rs` (dispatch_ex, command pool)

**Step 1: Create command pool in init()**

Add to init(): `device.create_command_pool()` with `RESET_COMMAND_BUFFER_BIT` and the compute queue family index.

**Step 2: Implement dispatch_ex()**

Full synchronous dispatch:

1. Allocate one command buffer from pool: `device.allocate_command_buffers()`
2. Begin: `device.begin_command_buffer(cmd, ONE_TIME_SUBMIT)`
3. Bind pipeline: `device.cmd_bind_pipeline(cmd, COMPUTE, pipeline.raw as VkPipeline)`
4. For each buffer index i:
   a. Update descriptor set: `vkWriteDescriptorSet` with buffer info
   b. Bind: `device.cmd_bind_descriptor_sets(cmd, COMPUTE, layout, &[descriptor_sets[i]])`
5. Dispatch: `device.cmd_dispatch(cmd, groups_x, groups_y, groups_z)`
6. Memory barrier: `device.cmd_pipeline_barrier()` with source=COMPUTE_SHADER_BIT, dest=HOST_BIT, buffer memory barrier
7. End: `device.end_command_buffer(cmd)`
8. Submit: `device.queue_submit(queue, &[cmd], fence)` with a fence for completion
9. Wait: `device.wait_for_fences(&[fence], true, u64::MAX)`
10. Free command buffer: `device.free_command_buffers(pool, &[cmd])`
11. Destroy fence

**Step 3: Verify**

```bash
cargo check --features vulkan
```

**Step 4: Commit**

```bash
git add src/vulkan.rs
git commit -m "feat: Vulkan dispatch via command buffer lifecycle"
```

---

### Task 7: Tests, main.rs, and final integration

**Files:**
- Modify: `src/vulkan.rs` (add tests)
- Modify: `src/main.rs` (update to WGSL — already done in Task 2 if applicable)
- Modify: `src/lib.rs` (final fixes, dead code cleanup)

**Step 1: Write Vulkan tests**

In `vulkan.rs`, add `#[cfg(test)] mod tests { ... }` with:
1. `device_init` — `VulkanBackend::init()` succeeds or skips gracefully
2. `add_one_kernel` (WGSL) — compile, dispatch on [1,2,3,4], assert [2,3,4,5]
3. `vector_scale_1024` — 1024 elements, 4 workgroups of 256, scale by 2.5
4. `compile_error` — invalid WGSL → CompileFailed
5. `roundtrip_empty` — zero-init buffer survives create/read roundtrip

**Step 2: Update main.rs to WGSL**

If not done in Task 2, update `main.rs` SAXPY kernel from MSL to WGSL.

**Step 3: Build + lint + test**

```bash
cargo check --features vulkan
cargo clippy --features vulkan
cargo test --features vulkan
```

Fix all warnings. Tests should pass or skip gracefully on headless.

**Step 4: Cross-check Metal still works**

```bash
cargo check --features metal
cargo clippy --features metal
```

**Step 5: Final commit**

```bash
git add src/vulkan.rs src/main.rs src/lib.rs
git commit -m "feat: Vulkan tests and final integration"
```
