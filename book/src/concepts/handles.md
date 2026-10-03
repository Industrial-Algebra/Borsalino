# Opaque Handles & Ownership

`ComputePipeline` and `GpuBuffer` are opaque handles: they carry raw pointers
to backend-specific inner state plus a backend drop function, so `lib.rs`
never couples to either backend module.

## Backend ownership discipline (Metal)

The objc ownership rule: release only what `new`/`copy`/`retain` handed you.
Convenience constructors and out-params are autoreleased (+0) and owned by
the autorelease pool — releasing them double-frees at pool drain.

Borsalino's Metal backend therefore:

- wraps dispatch and compilation in **scoped autorelease pools**, so
  autoreleased command buffers/encoders/strings are reclaimed deterministically
  even on plain Rust worker threads;
- takes an explicit **retain** on the one object that must escape a pool — the
  async `Pulse`'s command buffer — balanced by a release in its drop;
- releases every `new`-owned intermediate (library, function, pipeline
  descriptor) on **all** paths, including pipeline-creation failure.

## Shared lifetimes (Vulkan)

An async `Pulse` can outlive its backend, so the pulse shares ownership of the
epoch tracker (`Arc<GpuEpochTracker>`) and of the logical device
(`SharedDevice` — an `Arc` + `Deref` wrapper around `ash::Device`). The
device is destroyed only when the last holder drops, and the loader
(`Entry`) stays pinned for the whole chain. Dropping a backend with dispatches
still in flight idles the device first — destroying pools a pending
submission references is UB.
