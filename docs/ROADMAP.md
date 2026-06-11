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

## v0.2.0 (In Progress)

- [x] Persisted GPU buffer retention (`create_device_buffer`)
- [x] 2D/3D dispatch patterns with shared memory (tiled matmul: 278 GFLOPS)
- [x] Async dispatch (`dispatch_async` → `Pulse`)
- [x] Candle integration benchmark (tropical masking)
- [ ] Real IA kernel test (geometric product of 32-element multivectors)
- [ ] Miri integration for buffer lifecycle safety
- [ ] Kani harnesses for buffer roundtrip and alignment

## v0.3.0+ (Speculative)

- [ ] `dispatch_verified()` with `Proven<>` gates (Phase 3 verification)
- [ ] amari-flynn statistical verification (kernel determinism)
- [ ] Metal performance counters (occupancy, bandwidth metrics)
- [ ] Multi-GPU dispatch
- [ ] WASM target (WebGPU compute)
- [ ] SPIR-V shader caching
- [ ] Zero-copy tensor interop with Candle

*This roadmap is speculative beyond v0.2.0. Items may be reprioritised or descoped.*
