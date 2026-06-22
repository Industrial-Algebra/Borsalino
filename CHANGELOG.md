# Changelog

All notable changes to Borsalino are documented in this file.

## [0.2.0] — 2026-06-11

### Added
- **Async dispatch** — `dispatch_async()` returns `Pulse` handle for non-blocking GPU execution. VkFence (Vulkan), MTLCommandBuffer (Metal). Implicit join on drop.
- **Persistent buffers** — `create_device_buffer()` / `create_device_buffer_uninit()` for GPU-resident data. VRAM on discrete GPUs, zero-copy on unified memory.
- **GPU timestamps** — `gpu.timestamp()` for profiling. Vulkan: vkCmdWriteTimestamp query pool. Metal: CPU epoch fallback.
- **2D/3D tiled dispatch** — matmul example with WGSL shared memory and workgroup barriers. RTX 5080: 1,120 GFLOPS; GB10: 1,097 GFLOPS.
- **Benchmark matrix** — SAXPY, batched SAXPY, tiled matmul, timestamp resolution across M3 Pro, GB10, RTX 5080, AMD iGPU.
- **ROADMAP.md** — current and future development plans.

### Changed
- **CI** — clippy enforces `--all-features -- -D warnings` (zero warnings required).
- **README** — badges (crates.io, docs.rs, CI, license), async dispatch section, profiling section.
- **BENCHMARKS.md** — updated with v0.2.0 numbers including tiled matmul.

### Removed
- HANDOFF.md and docs/metal-debug-handoff.md (internal agent scaffolding).

## [0.3.0] — 2026-06-12

### Added
- **Shader caching** — `compile_cached()` skips naga translation on repeat calls. SPIR-V/MSL cached to `~/.cache/borsalino/`.
- **IA kernel benchmark** — geometric product of 32-blade multivectors (5D GA). Batched 4096× dispatch: 23× speedup over CPU.
- **`dispatch_verified()`** — `DispatchConfig` + `WorkgroupProof` safety gate for workgroup divisibility.
- **Miri + Kani verification** — buffer lifecycle safety under Miri; alignment, divisibility, and overflow proofs under Kani. CI label-gated.

### Changed
- ROADMAP.md: v0.3.0 items marked complete, future items reorganised.
- README: shader caching, verified dispatch, Miri/Kani sections added.

## [0.2.1] — 2026-06-11

### Removed
- `candle_tropical_mask` example (pending pre-print publication).

## [0.2.0] — 2026-06-11

### Added
- **Vulkan backend** — full `GpuBackend` trait implementation via `ash` raw FFI. WGSL→SPIR-V translation via `naga`, synchronous dispatch, pre-allocated descriptor sets and command pool. Tested on AMD Radeon, NVIDIA RTX 5080, and NVIDIA GB10.
- **WGSL shader language** — kernels authored in WGSL with `@group(0) @binding(N)` buffer declarations. Naga translates to MSL (Metal) and SPIR-V (Vulkan).
- **Device-local memory strategy** — automatic detection of discrete vs. unified memory GPUs. VRAM allocation with staging transfers for discrete GPUs (RTX 5080: 15× throughput improvement). Explicit `MemoryStrategy` enum for power users.
- **Batched dispatch (`dispatch_many`)** — multi-kernel command buffers amortise alloc/submit/wait overhead. RTX 5080: per-dispatch latency drops from 37 µs to 0.5 µs (75×). Peak throughput: 577 GFLOPS SAXPY on RTX 5080, 408 GFLOPS on GB10.
- **`verify` feature** — karpal-verify 0.5.0 GPU obligation bundles for `add_one`, `scale`, and `saxpy` kernels. Export to SMT-LIB2, Lean 4, and Kani backends.
- **Benchmarks** — cross-platform GPU benchmark example (`examples/bench.rs`) measuring pipeline compilation, dispatch latency, throughput scaling, batched SAXPY, and buffer I/O. Tested on AMD integrated, NVIDIA RTX 5080, NVIDIA GB10, and Apple M3 Pro.
- **Dispatch profiler** — per-component cost profiling example (`examples/dispatch_profile.rs`) isolating command buffer alloc, bind, dispatch, and sync costs.
- **Examples** — `hello_compute` (simplest add_one kernel), `saxpy` (fused multiply-add on 1024 elements).
- **Dual licensing** — AGPL-3.0 + commercial license (Schubert model).
- **CI workflow** — GitHub Actions: format check, clippy, multi-feature test matrix, documentation build, and crates.io publish on tag.

### Changed
- **Shader language: MSL → WGSL** — `GpuBackend::compile()` now accepts WGSL source instead of MSL. Metal backend translates WGSL→MSL via naga; Vulkan backend translates WGSL→SPIR-V via naga.
- **FFI: raw objc_msgSend → objc crate** — Metal backend now uses the `objc` 0.2 crate for correct ARM64 calling convention on Apple Silicon.
- **Dispatch: fence → queue_wait_idle** — Vulkan dispatch now uses `queue_wait_idle()` instead of per-dispatch fence creation, reducing syscall overhead.
- **Buffer memory: added HOST_CACHED** — Vulkan buffers prefer cached host-visible memory when available, reducing PCIe round-trip latency on discrete GPUs.

### Fixed
- **Metal backend on Apple Silicon M3** — seven root causes resolved: Rust 2024 edition compliance, framework linking, naga MSL resource binding, sizes buffer for runtime arrays, objc ARM64 ABI, MTLComputePipelineDescriptor for pipeline creation, and test-thread cleanup.
- **Naga MSL compatibility** — added `naga_msl_fixup()` post-processor for Metal 3 compatibility on M3.
- **Dependency resolution** — removed path dependencies; all deps resolve from crates.io.

### Dependencies
- Added: `ash` 0.38 (Vulkan FFI), `naga` 27 (shader translation), `objc` 0.2 (macOS), `karpal-verify` 0.5 (optional)
- Removed: `vulkano`, `vulkano-shaders`
