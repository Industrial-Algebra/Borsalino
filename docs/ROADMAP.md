# Borsalino Roadmap

## v0.1.0 (Released 2026-06-03)

- [x] Metal backend on Apple Silicon (M3, M4, M5)
- [x] Vulkan backend via ash (Linux, Windows)
- [x] WGSL shader language via naga
- [x] Device-local memory auto-detection
- [x] Batched dispatch (`dispatch_many`)
- [x] GPU timestamp queries (`gpu.timestamp()`)
- [x] Dual AGPL-3.0 + commercial license
- [x] CI: format, clippy, test matrix, docs, crates.io publish
- [x] Cross-platform benchmarks (AMD, GB10, RTX 5080, M3 Pro)

## v0.2.0 (Released 2026-06-11)

- [x] Async dispatch (`dispatch_async` -> `Pulse`)
- [x] Persistent GPU buffer retention (`create_device_buffer`)
- [x] GPU timestamps (`gpu.timestamp()`)
- [x] 2D/3D dispatch patterns with shared memory (tiled matmul: 1.4 TFLOPS)
- [x] Candle integration pattern (custom element-wise WGSL kernels)

## v0.3.0 (Released 2026-06-12)

- [x] SPIR-V shader caching (`compile_cached`)
- [x] Real IA kernel benchmark (geometric product of 32-element multivectors)
- [x] `dispatch_verified()` with WorkgroupProof safety gate
- [x] Miri + Kani verification harnesses for buffer lifecycle

## Future (Speculative)

- [ ] Multi-GPU dispatch
- [ ] Zero-copy Candle tensor interop
- [ ] amari-flynn statistical verification (kernel determinism)
- [ ] Metal performance counters (occupancy, bandwidth metrics)
- [ ] WASM target (WebGPU compute)
- [ ] Tropical masking example (re-integration after pre-print)
