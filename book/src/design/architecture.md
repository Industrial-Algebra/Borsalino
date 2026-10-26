# Architecture

One trait, four support modules, two backends:

```
GpuBackend trait
    │
    ├── MetalBackend     (metal.rs)
    │   ├── naga WGSL → MSL translation (+ _mslBufferSizes ABI)
    │   └── substrate: zunesha::metal::MetalDevice
    │       (consumer objc: pipelines + dispatch encoding)
    │
    ├── VulkanBackend    (vulkan.rs)
    │   ├── naga WGSL → SPIR-V translation
    │   ├── ash pipelines / descriptors / command buffers
    │   └── substrate: zunesha::vulkan::VulkanDevice
    │
    ├── epoch            (epoch.rs) — AtomicU64 dispatch tracking
    ├── determinism      (determinism.rs) — empirical determinism check
    ├── numerical_check  (numerical_check.rs) — exact-match protocol
    └── kani_harnesses   (kani_harnesses.rs) — bounded model checking
```

Opaque handle types (`ComputePipeline`, `GpuBuffer`) carry raw pointers and
backend-specific drop functions — no coupling between `lib.rs` and backend
modules.

## Substrate split (shipped in v0.8.0)

The device/queue/buffer substrate lives in
[Zunesha](https://github.com/Industrial-Algebra/Zunesha) (crates.io `0.2`);
Borsalino is its compute consumer. The staged migration completed in
v0.8.0: Phase 1 moved Vulkan device/queue ownership, Phase 2 delegated
buffer allocation and reads, Phase 3 mirrored both for Metal. Zunesha owns
selection, allocation, placement strategy, staging transfers, and teardown
quiescence (its submission protocol survived a seven-round review —
allocation-failure cleanup, quiesce-or-leak teardown, per-protocol
reentrancy). The seam is three raw surfaces per backend (device for
compile, queue for dispatch, buffer for binding). Cross-crate proof
agreement (quiescence proofs valid across the boundary) is specified as
Zunesha ADR 0003.

## Verification docs

- `docs/verification-integration.md` — how the verification pipeline wires
  into consumers
- `docs/VERIFICATION_ROADMAP_SUPPLEMENT.md` — the roadmap's verification
  annex
