# One Trait, Two Backends

Everything Borsalino does flows through one trait:

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

## Why raw FFI

The Metal and Vulkan backends hand-roll their FFI deliberately: no `wgpu`, no
Metal crate wrappers. The cost is a small, fixed set of selectors/calls that
an auditor can read in an afternoon; the benefit is that ownership and
lifetime decisions stay visible instead of hidden behind a convenience layer.

## Backend selection

| Backend | Platform | Feature | Status |
|---|---|---|---|
| Metal | macOS (Apple Silicon) | `metal` | ✅ Active |
| Vulkan | Linux, Windows | `vulkan` | ✅ Active |
| Stub | Any | (none) | Returns `NoBackend` — safe fallback |
