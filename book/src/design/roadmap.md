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

## v0.4.0 (2026-06-29)

- [x] Relicense from AGPL-3.0 to Apache-2.0
- [x] CONTRIBUTING.md with CLA reference

## v0.8.0 (2026-10-23) — the Zunesha substrate release

- [x] Completed staged Zunesha migration (Phases 1–3): device/queue/buffer
      ownership on the substrate for BOTH backends (`zunesha = "0.2"`)
- [x] Phase 1 (v0.7-era): Vulkan device/queue swap — instance/loader/device
      move to `zunesha::vulkan::VulkanDevice`
- [x] Phase 2: buffer delegation — `create_device_buffer(_uninit)` /
      `read_buffer` delegate; ~330 lines of allocation/transfer machinery
      deleted; rides the substrate's seven-round protocol soundness pass
- [x] Phase 3: Metal delegation — `MetalBackend` on
      `zunesha::metal::MetalDevice`; `MemoryStrategy::DeviceLocal` no
      longer silently ignored (Private + staging)
- [x] naga `_mslBufferSizes` ABI: module-declaration-order layout, wide
      layouts (Metal `arrayLength` kernels)
- [x] Workspace sibling: `borsalino-tensor` 0.1.0 (typed op-set contract,
      candle replacement — versions independently)

## v0.7.0 (2026-10-03)

- [x] Review-hardening pass (quick-start bounds guard, drop-path fixes)
- [x] Verification pipeline expansion

## v0.6.0 (2026-08-04)

- [x] GC safety: `GpuEpochTracker` (in-flight dispatch tracking, WASM
      runtime quiescence gating) + `BufferPinHandle<'a>` RAII pinning

## v0.5.0 / v0.5.1 (2026-06-29 / 2026-07-20)

- [x] Numerical correctness verification (`numerical_check`,
      DeepReinforce exact-match protocol)
- [x] Vulkan init SIGSEGV fix under Mesa ICD setups (#34)

## Future (Speculative)

- [ ] Multi-GPU dispatch
- [ ] Tensor extension T1–T4 (CPU reference → WGSL kernels → L3 ops → golden vectors) on `borsalino-tensor` (the candle replacement path; candle itself is being replaced after the zip-8.x yank)
- [ ] amari-flynn statistical verification (kernel determinism)
- [ ] Metal performance counters (occupancy, bandwidth metrics)
- [ ] WASM target (WebGPU compute)
- [ ] Tropical masking example (re-integration after pre-print)
