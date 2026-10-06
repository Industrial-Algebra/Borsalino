# Architecture

One trait, four support modules, two backends:

```
GpuBackend trait
    │
    ├── MetalBackend     (metal.rs)
    │   ├── naga WGSL → MSL translation
    │   └── objc_msgSend FFI (19 selectors, 0 Metal crate deps)
    │
    ├── VulkanBackend    (vulkan.rs)
    │   ├── naga WGSL → SPIR-V translation
    │   └── ash FFI (Vulkan 1.3)
    │
    ├── epoch            (epoch.rs) — AtomicU64 dispatch tracking
    ├── determinism      (determinism.rs) — empirical determinism check
    ├── numerical_check  (numerical_check.rs) — exact-match protocol
    └── kani_harnesses   (kani_harnesses.rs) — bounded model checking
```

Opaque handle types (`ComputePipeline`, `GpuBuffer`) carry raw pointers and
backend-specific drop functions — no coupling between `lib.rs` and backend
modules.

## Substrate split (planned)

The device/buffer/queue substrate is *planned* to be extracted into
[Zunesha](https://github.com/Industrial-Algebra/Zunesha); Borsalino will
become its compute consumer. As of v0.7 this crate still owns its device,
buffers, and epoch tracker. Cross-crate proof agreement (quiescence proofs
valid across the boundary) is already specified as Zunesha ADR 0003.

## Verification docs

- `docs/verification-integration.md` — how the verification pipeline wires
  into consumers
- `docs/VERIFICATION_ROADMAP_SUPPLEMENT.md` — the roadmap's verification
  annex
