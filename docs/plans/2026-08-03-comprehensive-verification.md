# Comprehensive Verification — Implementation Plan (v0.6.0)

**Date:** 2026-08-03
**Status:** Brainstormed, ready for implementation
**Branch:** `feature/v0.6.0-comprehensive-verification`
**Motivation:** Baedeker is consuming `dispatch_verified()` and moving to WASM 3.0 GC after its 0.1.0 release. GPU buffers referencing GC-managed memory need lifecycle safety. Full multi-tier verification makes GPU compute trustworthy for mathematical applications.

---

## 0. Goal

Make Borsalino's GPU layer fully verified across all five tiers, with
GC-safe buffer lifecycle tracking that enables zero-copy on unified memory
while preventing use-after-free from WASM GC compaction.

---

## 1. Pin-and-Track: Buffer Lifecycle Safety (GC)

### 1.1 Epoch Counter

Add an `AtomicU64` epoch counter to each backend. Incremented on dispatch,
decremented on readback completion. Baedeker's GC checks `is_quiescent()`
before compacting memory.

```rust
/// Tracks outstanding GPU dispatches for GC safety.
/// Zero means idle — safe to compact host memory.
pub trait GpuEpoch {
    /// Number of dispatches that have not yet completed readback.
    fn in_flight(&self) -> u64;

    /// True when no GPU operations are outstanding — GC may compact.
    fn is_quiescent(&self) -> bool {
        self.in_flight() == 0
    }
}
```

**Files:** `src/lib.rs` (+trait), `src/vulkan.rs` (+counter), `src/metal.rs` (+counter)

### 1.2 Zero-Copy Pinned Buffers

New buffer creation method that maps host memory directly (zero-copy) with
a pin handle that prevents reallocation:

```rust
/// RAII guard preventing host memory reallocation while a GPU buffer
/// references it. Drop un-pins the memory.
pub struct BufferPinHandle {
    unpin_fn: fn(*mut c_void),
    raw: *mut c_void,
}

impl Drop for BufferPinHandle {
    fn drop(&mut self) {
        (self.unpin_fn)(self.raw);
    }
}

/// Create a zero-copy GPU buffer backed by the host slice.
/// Returns the buffer and a pin handle that must outlive the buffer.
pub fn create_buffer_pinned<T: Pod>(
    &self,
    data: &[T],
) -> Result<(GpuBuffer, BufferPinHandle)>
```

- **Metal:** `newBufferWithBytesNoCopy:length:options:`
- **Vulkan:** host-visible `VkDeviceMemory` mapped to existing pointer
- **Discrete GPUs:** falls back to copy (VRAM allocation — pin is no-op)

**Files:** `src/lib.rs` (+trait method + types), `src/vulkan.rs`, `src/metal.rs`

### 1.3 Quiescence Proof

```rust
/// Compile-time proof that no GPU operations are outstanding.
/// Constructed only when `is_quiescent()` returns true.
pub struct QuiescenceProof { epoch: u64 }
```

Gated dispatch for GC-sensitive contexts:

```rust
fn dispatch_verified_gc(
    &self,
    pipeline: &ComputePipeline,
    buffers: &[&GpuBuffer],
    config: &DispatchConfig,
    workgroup_proof: &WorkgroupProof,
    quiescence_proof: &QuiescenceProof,
) -> Result<()>
```

**Files:** `src/lib.rs` (+method +proof type)

---

## 2. Expanded Structural Gates

### 2.1 Richer DispatchConfig::verify()

`DispatchConfig::verify()` currently checks only workgroup divisibility.
Expand to also verify:

- Buffer alignment: each buffer's offset divisible by `min_storage_buffer_offset_alignment`
- Dispatch limits: thread count within `max_compute_work_group_count`

```rust
pub struct DispatchProof {
    _workgroup: WorkgroupProof,  // phantom — evidence carried
}

impl DispatchConfig {
    pub fn verify_with_backend(
        self,
        min_alignment: u32,
        max_workgroups: u32,
    ) -> Result<DispatchProof> {
        // divisibility check (existing)
        // + alignment check (new)
        // + limit check (new)
    }
}
```

### 2.2 Obligation bundle updates

Add device-specific alignment and limit values to the obligation bundles in
`verify.rs`, sourced from the backend's queried properties.

**Files:** `src/lib.rs` (+DispatchProof), `src/verify.rs` (+richer bundles)

---

## 3. Statistical Determinism (Empirical)

### 3.1 Runtime Determinism Check

Concrete "dispatch twice, compare" check — no amari-flynn dependency required:

```rust
/// Result of a determinism check.
pub struct DeterminismResult {
    pub trials: u32,
    pub all_identical: bool,
    pub max_reproducibility_error: f32,
}

/// Dispatch the same inputs N times, compare outputs bit-for-bit.
/// If all N runs match, the kernel is deterministic within empirical bounds.
pub fn verify_deterministic(
    gpu: &impl GpuBackend,
    pipeline: &ComputePipeline,
    input_bytes: &[Vec<u8>],
    input_sizes: &[usize],
    output_len: usize,
    trials: u32,
) -> Result<DeterminismResult>
```

Catches nondeterministic atomic operations, race conditions, and
implementation-defined reductions. Simple, effective, no new dependencies.

### 3.2 Wire into obligation bundles

`IsMSLKernelDeterministic` obligation gets a `with_determinism()` builder
method (analogous to `with_numerical_correctness()`), recording the
empirical check as the verification mechanism.

**Files:** `src/determinism.rs` (new, ~150 LOC), `src/lib.rs` (+module),
`src/verify.rs` (+builder calls)

---

## 4. Kani/Miri Wiring

### 4.1 Fix dead kani_harnesses module

```rust
// src/lib.rs — add:
#[cfg(kani)]
mod kani_harnesses;
```

Two-line fix. Makes the existing 3 harnesses (`buffer_alignment_boundary`,
`workgroup_divisibility`, `buffer_size_no_overflow`) discoverable by Kani.

### 4.2 New Kani harnesses

- `epoch_counter_never_negative` — prove `fetch_sub` can't underflow
- `pin_handle_outlives_buffer` — prove pinned memory can't be reallocated
  while `BufferPinHandle` is live

### 4.3 Miri: epoch + pin

Miri test for the epoch counter's atomic operations and `BufferPinHandle`'s
RAII drop semantics under the buffer lifecycle.

**Files:** `src/lib.rs` (+module decl), `src/kani_harnesses.rs` (+2 harnesses),
`src/vulkan.rs` (+Miri test)

---

## 5. Numerical Check Expansion

### 5.1 GeometricProductReference

Add a `NumericalReference` impl for the geometric product kernel:

```rust
pub struct GeometricProductReference {
    pub blades: u32,  // 32 for 5D GA
}

impl NumericalReference for GeometricProductReference {
    // generate_inputs: binary multivectors (each blade ∈ {0,1})
    // compute_reference: sign-table weighted sum of products
}
```

Bilinear with binary inputs — exact-match applies. The sign table is
recomputed independently in the reference to catch sign-table bugs.

### 5.2 Wire into GPU CI

Add `verify_numerical` and `verify_deterministic` as `run-gpu`-gated CI jobs
on the Spark, alongside the existing `#[ignore]`d dispatch tests.

**Files:** `src/numerical_check.rs` (+GP reference), `.github/workflows/ci.yml`
(+verification jobs)

---

## 6. Implementation Order

| Step | Section | Depends on | LOC |
|---|---|---|---|
| 1 | 4.1 | — | 2 |
| 2 | 1.1 | — | ~80 |
| 3 | 1.2 | step 2 | ~250 |
| 4 | 1.3 | steps 2-3 | ~60 |
| 5 | 2.1 | — | ~80 |
| 6 | 3.1 | — | ~150 |
| 7 | 5.1 | — | ~100 |
| 8 | 4.2 | steps 2-4 | ~60 |
| 9 | 5.2 | steps 6-7 | ~30 |
| 10 | docs | all | ~100 |

**Total:** ~900 LOC

---

## 7. Success Criteria

- [ ] `is_quiescent()` works — epoch counter tracks dispatch/readback
- [ ] `create_buffer_pinned()` enables zero-copy on unified memory
- [ ] `dispatch_verified_gc()` gates on both workgroup proof and quiescence proof
- [ ] `DispatchConfig::verify_with_backend()` checks alignment + limits + divisibility
- [ ] `verify_deterministic()` dispatches twice, compares bit-for-bit
- [ ] `GeometricProductReference` passes exact-match protocol
- [ ] Kani harnesses discoverable (module declared)
- [ ] `run-gpu` CI job runs numerical + determinism checks on Spark
- [ ] Baedeker's GC can call `is_quiescent()` before compaction

---

## 8. What This Does NOT Cover

- **amari-flynn formal probability bounds** — the empirical determinism check
  catches practical nondeterminism without the full probabilistic contract
  machinery. Can layer on later.
- **Non-linear kernel numerical verification** — log/exp/tanh still cannot use
  exact-match (irrationals). Documented limitation.
- **Multi-GPU dispatch** — epoch tracking is per-backend, not cross-device.
- **Pin-and-track on discrete GPUs** — VRAM buffers are always copies; pin
  handle is a no-op.
