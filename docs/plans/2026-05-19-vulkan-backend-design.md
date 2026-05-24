# Borsalino Vulkan Backend — Design

**Date:** 2026-05-19
**Branch:** `feature/vulkan-backend`
**Status:** Design validated, ready for implementation

---

## 1. What This Builds

A Vulkan compute backend implementing `GpuBackend` for Linux/Windows. Users write WGSL compute kernels, Borsalino translates to SPIR-V via naga and dispatches synchronously on Vulkan.

## 2. Architecture

```
GpuBackend trait (lib.rs, unchanged)
    │
    ├── MetalBackend (metal.rs, unchanged)
    │       └── raw objc_msgSend FFI, MSL native
    │
    └── VulkanBackend (vulkan.rs, new)
            ├── naga: WGSL → SPIR-V compilation
            └── ash: raw Vulkan 1.3 FFI
```

Users write WGSL. Metal path: naga emits MSL → Metal compiles. Vulkan path: naga emits SPIR-V → vkCreateComputePipelines.

## 3. Shader Pipeline

```
compile(wgsl_source) {
    1. naga::front::wgsl::parse_str()       → Module
    2. naga::valid::Validator::validate()   → ModuleInfo
    3. naga::back::spv::write_vec()         → Vec<u32> (SPIR-V)
    4. device.create_shader_module()        → VkShaderModule
    5. device.create_compute_pipelines()    → VkPipeline
    6. wrap in ComputePipeline { raw, drop_fn }
}
```

Errors at steps 1-2 become `GpuError::CompileFailed` with naga's source-span messages.

## 4. Buffer Lifecycle

Opaque inner struct behind `GpuBuffer.raw`:

```rust
struct VulkanBufferInner {
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    size: VkDeviceSize,
    mapped: *mut c_void,       // persistent host mapping
    device: ash::Device,       // cloned for drop_fn/contents_fn
}
```

- **create_buffer:** alloc + bind + map + copy data
- **create_buffer_uninit:** alloc + bind + map (no copy)
- **read_buffer:** `contents_fn` returns `inner.mapped` — O(1), always mapped
- **Drop:** destroy buffer + free memory (unmap implicit)

## 5. Dispatch Flow

Synchronous, mirrors Metal's encoder → dispatch → end → submit → wait:

```
dispatch_ex(pipeline, buffers, workgroups, threads_per_group) {
    allocate_cmd() → begin() → bind_pipeline() → bind_buffers()
    → dispatch() → barrier(HOST_READ) → end() → submit() → wait_idle()
}
```

Pre-allocated resources (created once at init):
- 1× VkCommandPool (RESET_COMMAND_BUFFER_BIT)
- 1× VkDescriptorPool (8 sets × 1 storage buffer binding)
- 8× VkDescriptorSet (pre-allocated, re-bound per dispatch)
- 1× universal VkPipelineLayout (8 storage buffer bindings)

## 6. Initialization

```rust
VulkanBackend::init() {
    entry = Entry::linked()                     // libvulkan.so
    instance = create_instance()
    (physical_device, queue_family) = pick_compute_device()
    device = create_device()
    queue = get_queue()
    pipeline_layout = create_layout(8 buffers)
    descriptor_pool = create_pool(8 sets)
    descriptor_sets = allocate_sets(8)
    command_pool = create_pool(RESET)
}
```

Device picker: prefers discrete GPU, falls back to integrated, then any.

## 7. Dependencies

```toml
naga = { version = "27", features = ["wgsl-in", "spv-out", "msl-out"] }
ash = { version = "0.38", optional = true }

[features]
vulkan = ["dep:ash", "dep:naga"]   # naga needed for WGSL→SPIR-V
```

Remove `vulkano` and `vulkano-shaders` from Cargo.toml.

## 8. Backward Compatibility

- `GpuBackend` trait: unchanged
- `ComputePipeline`, `GpuBuffer`: unchanged (opaque handle pattern)
- Metal backend: receives naga-emitted MSL instead of raw user MSL — behaviour identical
- Stub backends: unchanged
- `init()`: new cfg-gated path for Vulkan returning `VulkanBackend`

## 9. Testing

All tests in `vulkan.rs`, `#[cfg(all(feature = "vulkan", not(target_os = "macos")))]`:

1. `device_init` — finds Vulkan device or skips gracefully
2. `add_one_kernel` — WGSL kernel, dispatch, readback
3. `vector_scale_1024` — 1024 elements, 4 workgroups of 256
4. `compile_error` — invalid WGSL → CompileFailed
5. `roundtrip_empty` — zero-init buffer roundtrip

Target hardware: Intel Arc (integrated), AMD Radeon, NVIDIA RTX 5080, Grace Blackwell DGX.
