# Changelog

All notable changes to Borsalino are documented in this file.

## [0.1.0] — 2026-06-03

### Added
- **Vulkan backend** — full `GpuBackend` trait implementation via `ash` raw FFI. WGSL→SPIR-V translation via `naga`, synchronous dispatch, pre-allocated descriptor sets and command pool. Tested on AMD Radeon, NVIDIA RTX 5080, and NVIDIA GB10.
- **WGSL shader language** — kernels authored in WGSL with `@group(0) @binding(N)` buffer declarations. Naga translates to MSL (Metal) and SPIR-V (Vulkan).
- **Device-local memory strategy** — automatic detection of discrete vs. unified memory GPUs. VRAM allocation with staging transfers for discrete GPUs (RTX 5080: 15× throughput improvement). Explicit `MemoryStrategy` enum for power users.
- **`verify` feature** — karpal-verify 0.5.0 GPU obligation bundles for `add_one`, `scale`, and `saxpy` kernels. Export to SMT-LIB2, Lean 4, and Kani backends.
- **Benchmarks** — cross-platform GPU benchmark example (`examples/bench.rs`) measuring pipeline compilation, dispatch latency, throughput scaling, and buffer I/O. Tested on AMD integrated, NVIDIA RTX 5080, NVIDIA GB10, and Apple M3 Pro.
- **Examples** — `hello_compute` (simplest add_one kernel), `saxpy` (fused multiply-add on 1024 elements).
- **Dual licensing** — AGPL-3.0 + commercial license (Schubert model).
- **CI workflow** — GitHub Actions: format check, clippy, multi-feature test matrix, and documentation build.

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
