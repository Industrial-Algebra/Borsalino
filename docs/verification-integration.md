# Borsalino — Karpal Verification Integration

**Date:** 2026-05-19
**Status:** Initial integration design — Phase 12e + Phase 17
**Dependencies:** `karpal-verify` (Phase 12), `karpal-proof` (Phase 11), Borsalino Metal backend

---

## 1. Architecture

Borsalino is a thin GPU compute abstraction with raw Metal FFI. It operates
at the unsafe boundary between Rust and GPU hardware. Verification here is
about **safety properties** — buffer alignment, thread counts, dispatch limits,
kernel determinism — rather than the algebraic laws that Schubert and Amari
verify. The Karpal verification stack provides a three-tier framework for
these properties.

```
Borsalino trait (GpuBackend)
    ↓
Metal backend (raw objc_msgSend FFI, 568 unsafe blocks)
    ↓ type-level guards prevent invalid states
GPU hardware (MTLDevice, MTLCommandQueue, threadgroup execution)
    ↓ Kani verifies bounded correctness of buffer operations
    ↓ amari-flynn verifies statistical determinism of kernels
```

---

## 2. Verification Tiers Applied to GPU Compute

### 2.1 Type-level (Impossible)

Properties that the Rust type system prevents at compile time — no runtime
check needed. These are encoded as Karpal `Property` types that gate
unsafe operations.

| Property | What it prevents | Mechanism |
|---|---|---|
| `IsBufferAlignedTo16` | MTLBuffer requires 16-byte alignment. Unaligned buffers → GPU fault. | Newtype wrapper that only constructs via `AlignedBuffer::new()` with alignment check |
| `IsWorkgroupSizeDivisible` | `total_threads % workgroup_size == 0`. Mismatch → undefined behavior. | `Proven<IsWorkgroupSizeDivisible, DispatchConfig>` gate on `dispatch()` |
| `IsThreadCountWithinLimits` | `max_total_threads_per_threadgroup` per Metal spec. Excess → validation error. | `Proven<IsThreadCountWithinLimits, DispatchConfig>` constructed from device caps |
| `IsBufferSizeSufficient` | Buffer index < buffer length for every thread ID. OOB → GPU fault. | `Proven<IsBufferSizeSufficient, GpuBuffer>` checked at buffer creation |

### 2.2 Runtime (Emergent)

Properties checked at runtime through proptest and assertion, using
`karpal-proof-derive` to auto-generate test harnesses.

| Property | Test | Frequency |
|---|---|---|
| MSL compilation determinism | Same source → same metallib binary (byte-identical across compilations) | Proptest, CI |
| Buffer upload/download roundtrip | `create_buffer(data)` → `read_buffer()` → same data | Proptest, per-PR |
| Dispatch result correctness | Known kernel (add_one, saxpy) produces expected output | Integration test, per-PR |
| Buffer lifecycle safety | No use-after-free, no double-free of MTLBuffer handles | Miri, CI |

### 2.3 Statistical (Rare)

Properties where exhaustive checking is infeasible (infinite input space,
nondeterministic GPU scheduling). Verified via amari-flynn Monte Carlo with
Hoeffding-bound confidence intervals.

| Property | Method | Bound |
|---|---|---|
| Kernel determinism | Same inputs, 4096 dispatches → identical outputs (bit-exact) | ε = 0.001, confidence 0.99 |
| Dispatch latency bound | `dispatch()` completes within 100ms for N ≤ 10⁶ | ε = 0.05, confidence 0.95 |
| No memory leaks over repeated dispatches | 10,000 dispatch cycles, resident memory stays within bounds | ε = 0.01, confidence 0.99 |

---

## 3. Integration Plan — Phase 12e

### 3.1 New Property Types for GPU Compute

```rust
// borsalino/src/verify.rs (new module, gated behind karpal-verify feature)

use karpal_proof::Property;

/// Property: buffer is 16-byte aligned for Metal MTLBuffer compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IsBufferAlignedTo16;

impl Property for IsBufferAlignedTo16 {
    const NAME: &'static str = "buffer aligned to 16 bytes";
}

/// Property: total thread count is divisible by workgroup size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IsWorkgroupSizeDivisible;

impl Property for IsWorkgroupSizeDivisible {
    const NAME: &'static str = "thread count divisible by workgroup size";
}

/// Property: kernel produces deterministic output across dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IsMSLKernelDeterministic;

impl Property for IsMSLKernelDeterministic {
    const NAME: &'static str = "MSL kernel is deterministic";
}

/// Property: dispatch parameters are within Metal device limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IsDispatchWithinLimits;

impl Property for IsDispatchWithinLimits {
    const NAME: &'static str = "dispatch within Metal device limits";
}
```

### 3.2 Type-Level Guards on dispatch()

```rust
// borsalino/src/lib.rs — upgraded GpuBackend trait

/// Dispatch with compile-time verification that thread count is divisible
/// by workgroup size.
///
/// The `proof` parameter ensures the caller has validated the dispatch
/// configuration. Without this proof, use `dispatch_ex` instead.
pub fn dispatch_verified(
    &self,
    pipeline: &ComputePipeline,
    buffers: &[&GpuBuffer],
    workgroups: (u32, u32, u32),
    proof: &Proven<IsWorkgroupSizeDivisible, DispatchConfig>,
) -> Result<()> {
    // SAFETY: compile-time proof that threads_per_group * workgroups = total_threads
    self.dispatch_ex(
        pipeline,
        buffers,
        workgroups,
        proof.value().threads_per_group,
    )
}
```

### 3.3 Obligation Bundles for GPU Properties

```rust
// borsalino/src/verify.rs

use karpal_verify::{
    Obligation, ObligationBundle, Origin, Sort,
    VerificationTier, AlgebraicSignature,
};

/// Generate proof obligations for Metal buffer alignment.
pub fn buffer_alignment_obligations() -> ObligationBundle {
    ObligationBundle::new(
        "borsalino_buffer_alignment",
        Origin::new("borsalino", "GpuBuffer alignment"),
    )
    .with(Obligation::for_property::<IsBufferAlignedTo16>(
        "create_buffer_aligns_to_16",
        Origin::new("borsalino", "GpuBackend::create_buffer"),
        VerificationTier::External,
        // ∀buf created by create_buffer: buf.addr % 16 == 0
        Term::eq(
            Term::app("mod", vec![Term::var("buf_addr"), Term::int(16)]),
            Term::int(0),
        ),
    ))
}

/// Generate amari-flynn statistical obligations for kernel determinism.
#[cfg(feature = "amari")]
pub fn kernel_determinism_obligations() -> ObligationBundle {
    ObligationBundle::new(
        "borsalino_kernel_determinism",
        Origin::new("borsalino", "GpuBackend::dispatch"),
    )
    .with(Obligation::for_property::<IsMSLKernelDeterministic>(
        "add_one_deterministic",
        Origin::new("borsalino", "add_one kernel"),
        VerificationTier::Rare,
        // ∀ dispatches d1, d2 on same input: read_buffer(d1) == read_buffer(d2)
        Term::eq(
            Term::app("read", vec![Term::var("dispatch1")]),
            Term::app("read", vec![Term::var("dispatch2")]),
        ),
    ))
}
```

### 3.4 amari-flynn Statistical Verification

```rust
#[cfg(feature = "amari")]
#[test]
fn verify_kernel_determinism_statistically() {
    use karpal_verify::{verify_rare_event, StatisticalBound, VerificationTier};
    use std::sync::Arc;

    let gpu = Arc::new(borsalino::init().expect("Metal device required"));

    let msl = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void add_one(device const float* input  [[buffer(0)]],
                            device float*       output [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
            output[id] = input[id] + 1.0;
        }
    "#;

    let pipeline = gpu.compile("add_one", msl).unwrap();

    // Monte Carlo: run 4096 dispatches, check all produce identical output
    let verification = verify_rare_event(
        &Obligation::for_property::<IsMSLKernelDeterministic>(
            "add_one_deterministic",
            Origin::new("borsalino", "add_one kernel"),
            VerificationTier::Rare,
            Term::bool(true), // simplified — actual comparison done in closure
        ),
        &StatisticalBound::new(0.001).with_samples(4096),
        || {
            let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let expected: Vec<f32> = input_data.iter().map(|x| x + 1.0).collect();

            let input = gpu.create_buffer(&input_data).unwrap();
            let output = gpu.create_buffer_uninit::<f32>(8).unwrap();
            gpu.dispatch(&pipeline, &[&input, &output], (1, 1, 1)).unwrap();
            let result: Vec<f32> = gpu.read_buffer(&output).unwrap();

            result != expected // rare event: non-deterministic output
        },
    );

    assert_eq!(verification.tier(), VerificationTier::Rare);
    assert!(verification.is_verified(), "kernel determinism verification failed");
}
```

---

## 4. CI Integration

### 4.1 Workflow

```yaml
# .github/workflows/borsalino-verify.yml
name: Borsalino Verification

on:
  push:
    paths:
      - 'Borsalino/**'
  pull_request:

jobs:
  # Tier 1: Host tests (runs everywhere)
  host:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo check -p borsalino --features metal

  # Tier 2: Miri (per-PR, no GPU needed)
  miri:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
        with:
          components: miri
      - run: cargo miri test -p borsalino --features metal

  # Tier 3: Metal GPU tests (label-gated, requires Apple Silicon runner)
  metal-gpu:
    if: contains(github.event.pull_request.labels.*.name, 'run-metal')
    runs-on: [self-hosted, macos, apple-silicon]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test -p borsalino --features metal -- --test-threads=1

  # Tier 4: Statistical verification (label-gated, runs on metal)
  amari-verify:
    if: contains(github.event.pull_request.labels.*.name, 'run-metal')
    runs-on: [self-hosted, macos, apple-silicon]
    needs: metal-gpu
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test -p borsalino --features metal,amari -- --ignored

  # Tier 5: Kani verification (label-gated)
  kani:
    if: contains(github.event.pull_request.labels.*.name, 'run-kani')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: model-checking/kani-github-action@v31
      - run: cargo kani -p borsalino --features metal --harness buffer_roundtrip
```

---

## 5. Kani Verification Examples

### 5.1 Buffer Roundtrip Correctness

```rust
// borsalino/src/kani_harnesses.rs

#[cfg(kani)]
mod harnesses {
    use kani::proof;

    #[proof]
    fn buffer_create_read_roundtrip() {
        // Allocate a buffer of any size up to 4096 bytes
        let len: usize = kani::any();
        kani::assume(len > 0 && len <= 1024);

        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();

        // Create buffer (simulated — no real GPU in Kani)
        let buf = create_test_buffer(&data);

        // Read buffer
        let result = read_test_buffer::<f32>(&buf, len);

        // Verify roundtrip
        assert_eq!(result.len(), data.len());
        for i in 0..len {
            assert_eq!(result[i], data[i]);
        }
    }

    #[proof]
    fn buffer_alignment_satisfies_16_byte_boundary() {
        let size: usize = kani::any();
        kani::assume(size > 0 && size <= 65536);

        // MTLBuffer always returns 16-byte aligned allocation
        let ptr = allocate_aligned(size, 16);
        assert!(ptr as usize % 16 == 0);
    }
}
```

### 5.2 Workgroup Divisibility Check

```rust
#[cfg(kani)]
#[proof]
fn workgroup_divisibility_prevents_partial_threadgroups() {
    let total_threads: u32 = kani::any();
    let workgroup_size: u32 = kani::any();
    kani::assume(workgroup_size > 0 && workgroup_size <= 1024);
    kani::assume(total_threads <= 1_048_576); // Metal max

    let workgroups = total_threads.div_ceil(workgroup_size);

    // Property: workgroups * workgroup_size >= total_threads
    assert!(workgroups * workgroup_size >= total_threads);

    // Property: no partial threadgroup if divisible
    if total_threads % workgroup_size == 0 {
        assert_eq!(workgroups * workgroup_size, total_threads);
    }
}
```

---

## 6. Verification Coverage Matrix

| What | Type-level | Proptest | Kani | amari-flynn | Miri | Status |
|---|---|---|---|---|---|---|
| Buffer 16-byte alignment | ✅ | ✅ | ✅ | — | — | Phase 2 |
| Buffer upload/download roundtrip | — | ✅ | ✅ | — | ✅ | Phase 2 |
| Workgroup size divisible | ✅ | — | ✅ | — | — | Phase 2 |
| MSL compilation determinism | — | ✅ | — | — | — | Phase 2 |
| Kernel output determinism | — | — | — | ✅ | — | Phase 3 |
| Buffer lifecycle (no UAF) | — | — | — | — | ✅ | Phase 2 |
| Dispatch within device limits | ✅ | — | — | — | — | Phase 2 |
| No memory leaks over dispatch cycles | — | — | — | ✅ | — | Phase 3 |
| objc_msgSend selector validity | — | ✅ | — | — | — | Phase 2 |

---

## 7. Migration Path

### Phase 1 (now) — Test hardening
- Miri integration for buffer lifecycle safety
- Proptest harnesses for Metal API surface (compile, dispatch, readback)
- No karpal-verify dependency yet

### Phase 2 (Phase 12e) — Kani + type-level proofs
- `IsBufferAlignedTo16`, `IsWorkgroupSizeDivisible`, `IsDispatchWithinLimits`
- Kani harnesses for buffer roundtrip and alignment
- `dispatch_verified()` method with `Proven` gate

### Phase 3 (Phase 12e) — Statistical verification
- amari-flynn Monte Carlo for kernel determinism
- amari-flynn for memory leak detection over dispatch cycles
- `Certified<AmariStatistical, IsMSLKernelDeterministic, ComputePipeline>`

### Phase 4 (Phase 17) — Ecosystem integration
- Borsalino verification certificates referenced by Amari GPU kernels
- Cross-crate trust: Amari trusts Borsalino's alignment/determinism proofs
- CI artifact registry for GPU verification results

---

## 8. Relationship to Quanta's Verification

Borsalino was designed after studying Quanta's approach. The key
difference in verification strategy:

| Aspect | Quanta | Borsalino (planned) |
|---|---|---|
| Formal spec language | Lean 4 (external) | karpal-verify Obligation IR (Rust-native) |
| Code-to-spec proof | Verus (modified compiler) | Kani (bounded checking, stable Rust) |
| GPU property coverage | IR semantics, cross-emitter agreement | Buffer safety, dispatch correctness, kernel determinism |
| Statistical verification | None | amari-flynn (Monte Carlo + Hoeffding) |
| Trust boundary | 35 GPU hardware axioms in Lean | `unsafe { Proven::axiom }` with Certificate audit trail |

Quanta proves the compiler is correct. Borsalino verifies the *runtime* is
safe — buffer alignment, thread counts, dispatch correctness. The approaches
are complementary: Quanta's Lean axioms (A1-A13) are the mathematical
foundation that Borsalino's type-level properties operationalize in Rust.

---

## 9. References

- [Karpal Roadmap](../../karpal/ROADMAP.md) — Phase 12e (GPU compute obligations), Phase 17 (ecosystem integration)
- [Schubert verification integration](../../Schubert/docs/verification-integration.md) — Schubert calculus verification
- [Borsalino HANDOFF.md](../HANDOFF.md) — Architecture overview and current state
- [amari-flynn documentation](https://github.com/Industrial-Algebra/amari-flynn) — Statistical verification backend
- Apple Metal Shading Language Specification (v3.2) — MTLBuffer alignment, threadgroup execution model
- Quanta `specs/machine_model.md` — GPU hardware axioms A1-A13 for reference
