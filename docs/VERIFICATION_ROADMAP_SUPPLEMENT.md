# Borsalino Verification Roadmap Supplement

**Date:** 2026-06-18
**Status:** Gap analysis and implementation plan — bridging v0.2.1 (GPU works) to v0.3.0+ (GPU is provably safe)
**Based on:** Rabbit Hole analysis (2026-06-16), `verification-integration.md` (2026-05-19), current codebase (v0.2.1)

---

## 0. Executive Summary

Borsalino v0.2.1 is a working GPU compute abstraction. Metal and Vulkan backends dispatch kernels, benchmarks span four platforms, async dispatch works, tiled matmul hits 1.4 TFLOPS on GB10. The GPU compute is real.

The verification infrastructure — which was the *design intent* behind Borsalino's minimal surface area — is partially implemented and almost entirely not operational:

| Layer | Specified | Implemented | Operational |
|-------|-----------|-------------|-------------|
| Property types (`IsBufferAlignedTo16`, etc.) | ✅ Full | ✅ Re-exports from karpal-verify | ✅ Compiles |
| Obligation bundles (SMT/Lean/Kani export) | ✅ Full | ✅ 3 kernels (add_one, scale, saxpy) | ✅ Tests pass |
| `Proven<>` type-level gates on `dispatch()` | ✅ Spec'd | ❌ Not in trait | ❌ |
| Miri buffer lifecycle verification | ✅ Spec'd | ❌ Not in CI | ❌ |
| Kani harnesses (buffer roundtrip, alignment) | ✅ Harnesses written | ❌ Not wired to CI | ❌ |
| amari-flynn statistical verification | ✅ Spec'd | ❌ Not implemented | ❌ |
| Proptest harnesses | ✅ Spec'd | ❌ Not implemented | ❌ |
| Tiered CI pipeline (host → Miri → GPU → amari → Kani) | ✅ YAML templated | ❌ Not deployed | ❌ |

**The task:** Close these gaps in dependency order. Phase 1 can start immediately.

---

## 1. The Verification Debt Ledger

Ordered by criticality × implementation proximity. Each item includes what blocks it.

### 🔴 Critical — type safety gates that should exist before v0.3.0

| # | Item | What's Missing | Blocked By |
|---|------|---------------|------------|
| V1 | `dispatch_verified()` on `GpuBackend` trait | Method exists in spec but not in `lib.rs`. Needs `Proven<IsWorkgroupSizeDivisible, DispatchConfig>` parameter and a `DispatchConfig` struct. | Nothing — karpal-proof 0.5 is on crates.io |
| V2 | `AlignedBuffer` newtype | Spec'd to gate buffer creation on 16-byte alignment. Currently `create_buffer` accepts any `&[T: Pod]` with no alignment check. | Nothing — pure Rust |
| V3 | `Proven<IsDispatchWithinLimits, _>` gate | Dispatch config should be checked against device caps at construction time. Currently no device caps query exists. | Needs device caps query method on trait |

### 🟡 Medium — runtime verification that catches regressions

| # | Item | What's Missing | Blocked By |
|---|------|---------------|------------|
| V4 | Miri in CI | `cargo miri test` for buffer lifecycle. Miri finds use-after-free and pointer provenance bugs in the opaque handle drop implementations. | CI: Apple Silicon self-hosted runner for Metal path; x86 runner sufficient for Vulkan stubs |
| V5 | Proptest harnesses | Shrinking tests for MSL compilation determinism, buffer roundtrip, dispatch correctness. Currently only manual tests exist. | Nothing — proptest is a dev-dependency |
| V6 | Dispatch result correctness suite | More than just add_one/saxpy. Need: zero-copy roundtrip, multi-buffer dispatch, batched dispatch correctness, async dispatch result equivalence. | Nothing — pure Rust |

### 🟢 Strategic — mechanized proofs and ecosystem integration

| # | Item | What's Missing | Blocked By |
|---|------|---------------|------------|
| V7 | Kani harnesses in CI | `buffer_create_read_roundtrip`, `buffer_alignment_satisfies_16_byte_boundary`, `workgroup_divisibility_prevents_partial_threadgroups`. Code exists in spec. | Kani GitHub Action, runner with enough memory |
| V8 | amari-flynn Monte Carlo | 4096-dispatch determinism test, 10k-cycle memory leak test. Code exists in spec. | amari-flynn maturity, GPU CI runner |
| V9 | Cross-crate trust (Phase 17) | Amari GPU kernels reference Borsalino verification certificates. Schubert gates GPU access on proof validity. | All of the above + Karpal Phase 17 |
| V10 | `Certified<AmariStatistical, IsMSLKernelDeterministic, ComputePipeline>` | Statistical proof wrapper type | V8 + karpal-verify statistical support |

---

## 2. Phase 1: Type-Level Hardening (Now — zero blockers)

**Goal:** Make the type system prevent the GPU fault classes that Borsalino was designed to prevent. Every item can be implemented today with existing dependencies.

**Dependencies:** `karpal-proof` 0.5 (already in `Cargo.toml` under `verify` feature), `karpal-verify` 0.5 (same)

### 2.1 Implement `dispatch_verified()` — V1

Add to `GpuBackend` trait in `lib.rs`:

```rust
/// Dispatch with compile-time verification of dispatch configuration.
///
/// The `proof` parameter ensures the caller has validated:
/// - Thread count is divisible by workgroup size
/// - Dispatch is within device limits
///
/// Without a `Proven<>` gate, use `dispatch_ex` instead.
fn dispatch_verified(
    &self,
    pipeline: &ComputePipeline,
    buffers: &[&GpuBuffer],
    config: &DispatchConfig,
    proof: &Proven<IsWorkgroupSizeDivisible, DispatchConfig>,
) -> Result<()> {
    self.dispatch_ex(
        pipeline,
        buffers,
        config.workgroups,
        config.threads_per_group,
    )
}
```

Requires a new `DispatchConfig` struct:

```rust
/// Verified dispatch configuration.
pub struct DispatchConfig {
    pub workgroups: (u32, u32, u32),
    pub threads_per_group: (u32, u32, u32),
}
```

**Files changed:** `src/lib.rs` (+DispatchConfig struct, +dispatch_verified default method, +re-export Proven types from verify module)

**Verification:** `cargo build --features verify` compiles. The method exists on the trait and is callable with a `Proven<>` gate.

### 2.2 Implement `AlignedBuffer` newtype — V2

Currently `create_buffer` has no alignment enforcement. Metal requires 16-byte alignment; violating this causes GPU faults.

```rust
/// Buffer guaranteed to be 16-byte aligned for Metal compatibility.
///
/// Only constructable via `AlignedBuffer::new()` which checks alignment.
/// Wraps a `GpuBuffer` with a type-level alignment proof.
#[cfg(feature = "verify")]
pub struct AlignedBuffer {
    inner: GpuBuffer,
    _proof: Proven<IsBufferAlignedTo16, GpuBuffer>,
}

#[cfg(feature = "verify")]
impl AlignedBuffer {
    pub fn new(buffer: GpuBuffer) -> Option<Self> {
        if buffer.raw as usize % 16 == 0 {
            Some(Self {
                inner: buffer,
                _proof: Proven::axiom(), // SAFETY: alignment checked above
            })
        } else {
            None
        }
    }

    pub fn as_gpu_buffer(&self) -> &GpuBuffer {
        &self.inner
    }
}
```

**Files changed:** `src/verify.rs` (+AlignedBuffer type)
**Note:** This is gated behind `verify` feature. Without `verify`, alignment failures remain runtime errors (GpuError). With `verify`, the type system prevents unaligned buffers from reaching dispatch.

### 2.3 Add device caps query — V3 prerequisite

`IsDispatchWithinLimits` needs to be checked against actual device limits. Currently Borsalino has no way to query these.

```rust
/// Device capability limits.
#[derive(Debug, Clone, Copy)]
pub struct DeviceCaps {
    pub max_threads_per_threadgroup: u32,
    pub max_threadgroup_memory: u32,
    pub max_total_threads: u32,
}

/// Add to GpuBackend trait:
fn device_caps(&self) -> DeviceCaps;
```

Default implementation returns conservative Metal limits. Backends override with queried values.

**Files changed:** `src/lib.rs` (+DeviceCaps struct, +device_caps() trait method), `src/metal.rs` (+implementation), `src/vulkan.rs` (+implementation)

### 2.4 Phase 1 Completion Criteria

- [ ] `dispatch_verified()` compiles and is callable
- [ ] `DispatchConfig` struct exists
- [ ] `AlignedBuffer` newtype exists with alignment check
- [ ] `device_caps()` returns real or conservative limits
- [ ] All existing tests pass with `--features verify`

---

## 3. Phase 2: Runtime Verification Wiring (Short-term)

**Goal:** Catch regressions in buffer lifecycle, dispatch correctness, and shader compilation determinism automatically.

**Dependencies:** CI runner with GPU (self-hosted Apple Silicon for Metal, x86 for Vulkan), Miri nightly toolchain

### 3.1 Miri integration — V4

Miri detects undefined behavior in unsafe code. Borsalino's opaque handle types (`ComputePipeline`, `GpuBuffer`, `Pulse`) carry raw pointers and custom drop functions — Miri catches use-after-free, double-free, pointer provenance violations.

```toml
# .github/workflows/ci.yml addition
miri:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@nightly
      with:
        components: miri
    - run: cargo miri test -p borsalino --features stub --lib
```

**Note:** Miri can't run actual GPU code (no FFI to Metal/Vulkan drivers). It tests the lib.rs types and drop semantics. The stub backend provides a `NoBackendStub` that returns errors — Miri verifies the type machinery is sound even when no GPU is present.

### 3.2 Proptest harnesses — V5

Property-based tests that shrink to minimal counterexamples:

```rust
#[cfg(test)]
mod proptest {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn buffer_create_read_roundtrip(data in prop::collection::vec(any::<f32>(), 0..4096)) {
            let gpu = borsalino::init().unwrap();
            let buf = gpu.create_buffer(&data).unwrap();
            let result: Vec<f32> = gpu.read_buffer(&buf).unwrap();
            prop_assert_eq!(result, data);
        }

        #[test]
        fn buffer_alignment_is_16_byte(data in prop::collection::vec(any::<f32>(), 1..1024)) {
            let gpu = borsalino::init().unwrap();
            let buf = gpu.create_buffer(&data).unwrap();
            prop_assert_eq!(buf.raw as usize % 16, 0);
        }
    }
}
```

**Note:** Proptest requires a GPU. These tests run on the `metal-gpu` or `vulkan-gpu` CI tier, gated by labels.

### 3.3 CI tier structure

```
Tier 1 (every push):    cargo check, clippy, fmt, host tests
Tier 2 (every PR):      + Miri, proptest (stub), docs
Tier 3 (label-gated):   + Metal GPU tests, Vulkan GPU tests
Tier 4 (label-gated):   + Kani harnesses
Tier 5 (label-gated):   + amari-flynn statistical
```

### 3.4 Phase 2 Completion Criteria

- [ ] Miri passes on `lib.rs` types (stub backend) in CI
- [ ] Proptest harnesses exist for buffer roundtrip and alignment
- [ ] Tier 1-3 CI pipeline operational
- [ ] At least one self-hosted GPU runner registered

---

## 4. Phase 3: Mechanized Proofs (Medium-term)

**Goal:** Kani verifies bounded correctness of buffer operations and dispatch parameter validation.

**Dependencies:** Kani v31+ (model checking tool), runner with ≥16 GB RAM, Karpal Phase 12e complete (GPU proof obligations stable)

### 4.1 Kani harness activation — V7

The harnesses are already written in `verification-integration.md` §5. They need to be moved to `src/kani_harnesses.rs` (gated behind `#[cfg(kani)]`) and wired to CI.

Three proofs to activate:

1. **`buffer_create_read_roundtrip`** — For any buffer size ≤1024 elements, create → read produces identical data
2. **`buffer_alignment_satisfies_16_byte_boundary`** — For any allocation size ≤65536 bytes with 16-byte alignment request, result is 16-byte aligned
3. **`workgroup_divisibility_prevents_partial_threadgroups`** — For any valid thread count and workgroup size, if divisible, no partial threadgroup

### 4.2 Kani CI integration

```yaml
kani:
  if: contains(github.event.pull_request.labels.*.name, 'run-kani')
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: model-checking/kani-github-action@v31
    - run: |
        cargo kani -p borsalino \
          --features stub \
          --harness buffer_create_read_roundtrip \
          --harness buffer_alignment_satisfies_16_byte_boundary \
          --harness workgroup_divisibility_prevents_partial_threadgroups
```

### 4.3 Phase 3 Completion Criteria

- [ ] Three Kani proofs pass on every `run-kani` labeled PR
- [ ] Proofs are in `src/kani_harnesses.rs` (not just in docs)
- [ ] No `kani::assume` violations found in bounded checking
- [ ] Kani results visible in CI artifacts

---

## 5. Phase 4: Statistical Verification (Longer-term)

**Goal:** amari-flynn Monte Carlo verifies kernel determinism and memory stability over large dispatch counts.

**Dependencies:** amari-flynn crate published and stable, GPU CI runner with consistent performance characteristics

### 5.1 Kernel determinism test — V8

The test in `verification-integration.md` §3.4 is fully specified. Needs:
- amari-flynn `verify_rare_event` API
- Hoeffding-bound confidence intervals (ε = 0.001, confidence 0.99)
- 4096 dispatches on identical input → bit-identical output
- Test gate: `#[cfg(feature = "amari")]` + `#[ignore]` (long-running)

### 5.2 Memory leak detection

- 10,000 dispatch cycles
- Resident memory measured before/after
- amari-flynn bounds: ε = 0.01, confidence 0.99
- Catches: buffer leaks, command-buffer leaks, Metal/Vulkan resource leaks

### 5.3 Phase 4 Completion Criteria

- [ ] `verify_kernel_determinism_statistically` test passes
- [ ] `verify_no_memory_leaks_over_dispatch_cycles` test passes
- [ ] Tests gated behind `--ignored` and `run-metal` label
- [ ] Statistical verification results committed as CI artifacts

---

## 6. Phase 5: Ecosystem Integration (Vision)

**Goal:** Borsalino verification certificates become first-class inputs to Amari's GPU kernels and Schubert's access control.

**Dependencies:** All previous phases complete, Karpal Phase 17 (cross-crate trust), Schubert verification integration

### 6.1 Cross-crate trust — V9

When Amari calls Borsalino for geometric product computation:

```rust
// Amari GPU module
let verified_gpu = borsalino::init_verified()?; // returns Verified<GpuBackend>
let proof = verified_gpu.dispatch_verified(&pipeline, &buffers, &config)?;
// proof: Certified<IsMSLKernelDeterministic, ComputePipeline>
// Amari can pass this certificate to Schubert for access gating
```

### 6.2 Schubert integration

Schubert gates GPU access by proof validity:

```
Schubert policy: "allow GPU dispatch iff kernel is certified deterministic
                  AND buffer alignment is proven"
Intersection number: allow ∩ deterministic ∩ aligned
```

### 6.3 Phase 5 Completion Criteria

- [ ] Amari can call `borsalino::init_verified()`
- [ ] Borsalino dispatch produces `Certified<>` proof certificates
- [ ] Schubert can evaluate Borsalino certificates in access decisions
- [ ] End-to-end: Amari → Borsalino (verified) → Schubert (gated) → GPU

---

## 7. Dependency Map

```
Phase 1 (Type-Level)
  ├── karpal-proof 0.5 ✅ (on crates.io)
  └── karpal-verify 0.5 ✅ (on crates.io)

Phase 2 (Runtime)
  ├── Miri nightly ✅ (rustup component)
  ├── proptest ✅ (crates.io)
  └── GPU CI runner ⚠️ (needs self-hosted Apple Silicon or Vulkan-capable runner)

Phase 3 (Mechanized)
  ├── Kani v31+ ⚠️ (toolchain + GitHub Action)
  ├── Karpal Phase 12e ⚠️ (GpuObligationBundle stable)
  └── CI runner with 16+ GB RAM

Phase 4 (Statistical)
  ├── amari-flynn ⚠️ (not yet published? verify)
  └── GPU CI runner with consistent perf

Phase 5 (Ecosystem)
  ├── Karpal Phase 17 ⚠️ (cross-crate trust)
  ├── Schubert verification integration ⚠️
  └── Amari GPU kernel module ⚠️
```

**Key:** ✅ Available now | ⚠️ Needs work/investigation

---

## 8. The "Verification Debt" Philosophy

Borsalino's current state — GPU compute works, proofs don't — is an intentional strategy, not a failure. The question is how to manage it.

### What's defensible

- **Type-level properties are real proofs.** `IsBufferAlignedTo16` enforced by the Rust type system is a compile-time guarantee, even without Kani checking the harness. The `Proven<>` gate pattern is valid without mechanized verification — it's a contract, and the contract is enforced by construction.
- **Obligation bundles have value before execution.** Exporting to SMT/Lean/Kani is a capability. The tests in `verify.rs` prove the bundles are well-formed and export correctly. That's not nothing.
- **The opaque handle pattern *is* verification.** Carrying drop functions in function pointers rather than concrete types is a deliberate isolation strategy. `lib.rs` knows nothing about GPU hardware — it can be verified independently.

### What's not defensible

- **No Miri means no UB detection.** The opaque handle types carry raw pointers with custom drop semantics. Without Miri, use-after-free and double-free bugs in the drop implementations are invisible until they crash on real hardware.
- **No Kani means bounded properties are assumptions, not proofs.** The alignment and divisibility checks in the type system are correct *if the constructors enforce them*. Kani verifies that the constructors actually do what they claim.
- **No amari-flynn means kernel determinism is untested.** GPU scheduling is nondeterministic at the hardware level. Only statistical verification can bound the probability of nondeterministic output.

### Recommended strategy

1. **Phase 1 immediately** — Type-level gates on dispatch cost nothing and improve the API's safety story
2. **Phase 2 when GPU runner available** — Miri + proptest catch the bugs that slip through review
3. **Phase 3 when Karpal 12e stabilizes** — Kani proofs are the "real" verification, but they depend on upstream
4. **Phase 4 when amari-flynn ships** — Statistical verification is the capstone, not the foundation
5. **Phase 5 when the ecosystem catches up** — Cross-crate trust requires everything else to exist first

---

## 9. Immediate Actions (This Week)

These can be done now with no new dependencies:

1. **Implement `dispatch_verified()` with `DispatchConfig`** — 30 lines in `lib.rs`
2. **Implement `AlignedBuffer` newtype** — 25 lines in `verify.rs`
3. **Add `device_caps()` to trait** — 20 lines in `lib.rs` + backend implementations
4. **Move Kani harnesses from docs to `src/kani_harnesses.rs`** — Copy-paste from `verification-integration.md` §5
5. **Add proptest as dev-dependency** — `cargo add --dev proptest`

**Estimated effort:** 2-3 hours for items 1-3, 1 hour for items 4-5.

---

## 10. References

- [verification-integration.md](./verification-integration.md) — Full verification specification (440 lines)
- [ROADMAP.md](./ROADMAP.md) — Current development roadmap
- [Borsalino Rabbit Hole (2026-06-16)](../../IA-documents/RESEARCH_REPORTS/RABBIT_HOLE_2026-06-16_20260616_230000.md) — Philosophical analysis
- [src/verify.rs](../src/verify.rs) — Current verification module (185 lines)
- [src/lib.rs](../src/lib.rs) — GpuBackend trait and opaque handle types
- [Karpal Roadmap](../../karpal/ROADMAP.md) — Phase 12e (GPU obligations), Phase 17 (ecosystem)
- [Schubert verification integration](../../Schubert/docs/verification-integration.md)
