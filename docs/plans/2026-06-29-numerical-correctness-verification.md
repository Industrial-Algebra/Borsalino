# Numerical Correctness Verification — Implementation Plan

**Date:** 2026-06-29
**Status:** Scoped, ready for implementation (v0.5.0)
**Closes:** #31
**Based on:** [DeepReinforce — Towards a Reliable Kernel Correctness Check in Matrix Multiplication](https://deep-reinforce.com/correctness_check.html)

---

## 0. Executive Summary

Borsalino's `src/verify.rs` verifies **structural safety** (buffer alignment, workgroup divisibility, dispatch limits, determinism phantom types). It does **not** verify **numerical correctness** — that a kernel produces the right answer.

DeepReinforce's paper exposes this gap precisely: the standard `torch.allclose` tolerance approach is unreliable because floating-point associativity does not hold. Different GPU thread orderings produce different accumulation sequences, so two correct kernels produce different outputs. No universal tolerance works across matrix sizes or precision formats.

Their solution — **exact match with binary inputs** — is directly applicable to every kernel Borsalino ships, because every shipped kernel is linear. This plan adds a numerical correctness verification layer alongside the existing structural safety layer.

**Scope:** v0.5.0. Three phases. Pure Rust (no new dependencies beyond optional `rand` for trial input generation). All work gated behind the existing `verify` feature flag.

---

## 1. The DeepReinforce Protocol

For a linear kernel `K(A, B) → C`:

1. **Sample binary inputs** `A, B ∈ {0, 1}^{...}` with a zero-biased distribution (e.g., `p_zero = 0.7`) to keep accumulated sums below the FP16 exact-representation ceiling (2048)
2. **Compute reference** `C_ref = K_fp32_cpu(A, B)` using FP32 on CPU — exact integer arithmetic
3. **Require bit-exact equality** at every output position `(i, j)` where `C_ref[i,j] ≤ 2048`:
   `C_custom[i,j] == C_ref[i,j]`
4. **Ignore** positions where `C_ref[i,j] > 2048` (those have lost exactness)
5. **Repeat** across N random trials (default 16); any single failure = incorrect kernel

**Why binary inputs?** They guarantee non-negative, monotonically non-decreasing partial sums, which keeps all intermediate values in the FP16 exact-integer range `[0, 2048]`. Within that range, floating-point associativity holds.

**Why it works for Borsalino:**

| Kernel | Linearity | Reference Function | Notes |
|---|---|---|---|
| `add_one` | Linear | `x + 1` | Trivially exact |
| `scale` | Linear | `α · x` | α ∈ {0, 1} keeps products exact |
| `saxpy` | Linear | `α·x + y` | Two-term sum, exact in range |
| `tiled_matmul` | Linear | `A @ B` | Canonical case from the paper |
| `geometric_product` | Bilinear | Sign-table weighted sum of `a[i]·b[j]` | Integer products if inputs binary |

Non-linear kernels (`log`, `exp`, `tanh`) are out of scope — those produce irrationals and cannot be checked with exact match.

---

## 2. Phase 1: karpal-verify — `IsNumericallyCorrect` property

**Goal:** Add the property type and obligation builder method to karpal-verify so Borsalino (and other IA crates) can declare numerical correctness obligations.

**Repository:** `karpal/karpal-verify`

### 2.1 New property type

In `karpal/karpal-verify/src/gpu.rs`:

```rust
/// A linear kernel's output matches a CPU FP32 reference under the
/// exact-match protocol (binary inputs, threshold-gated bit-exact compare).
///
/// This is a *runtime* verification property, not a compile-time phantom.
/// The obligation records that a numerical check has been specified; the
/// actual check runs in the consumer crate (e.g., Borsalino's
/// `verify_numerical()`).
pub struct IsNumericallyCorrect;
impl Property for IsNumericallyCorrect {
    const NAME: &'static str = "IsNumericallyCorrect";
}
```

### 2.2 Obligation builder method

Add to `GpuObligationBundle`:

```rust
/// Declare that this kernel has a numerical correctness check specified.
///
/// `threshold` is the exact-match ceiling (2048 for FP16, configurable for
/// other formats). `trials` is the number of random binary-input trials.
pub fn with_numerical_correctness(
    mut self,
    kernel: impl Into<String>,
    threshold: i64,
    trials: u32,
) -> Self {
    let kernel = kernel.into();
    self.bundle.push(Obligation {
        name: format_obligation_name(&kernel, "numerical_correctness"),
        property: IsNumericallyCorrect::NAME,
        declarations: vec![
            Declaration::new("threshold".into(), Sort::Int),
            Declaration::new("trials".into(), Sort::Int),
        ],
        assumptions: vec![
            Term::app("le", [Term::int(0), Term::var("threshold")]),
            Term::app("le", [Term::int(1), Term::var("trials")]),
        ],
        conclusion: Term::app(
            "numerically_correct_under_exact_match",
            [Term::var(kernel.clone()), Term::var("threshold"), Term::var("trials")],
        ),
        origin: self.bundle.origin.clone(),
        tier: VerificationTier::Runtime,
    });
    self
}
```

This requires adding `VerificationTier::Runtime` to the `VerificationTier` enum in `karpal-verify/src/lib.rs` (distinct from `External` which is for SMT/Kani/Lean proof backends).

### 2.3 Tests

- `gpu_bundle_contains_numerical_correctness` — builder produces an obligation with the right property name
- `numerical_correctness_exports_through_all_backends` — SMT/Lean/Kani exports include the obligation
- `runtime_tier_distinct_from_external` — `VerificationTier::Runtime` is a separate variant

**Files changed:** `karpal/karpal-verify/src/gpu.rs` (+property, +builder method), `karpal/karpal-verify/src/lib.rs` (+`VerificationTier::Runtime`)

**Verification:** `cargo test -p karpal-verify` passes. Version bump karpal-verify to 0.6.0 (new enum variant is a breaking change).

---

## 3. Phase 2: Borsalino — `verify_numerical()` runtime check

**Goal:** Implement the actual exact-match protocol as a runtime method on `GpuBackend`.

**Repository:** `Borsalino`

### 3.1 The protocol module

New file `src/numerical_check.rs`:

```rust
//! DeepReinforce exact-match numerical correctness protocol.
//!
//! Restricts kernel inputs to binary {0, 1} values so that floating-point
//! associativity holds exactly within the FP16 integer range [0, 2048].
//! Compares GPU output against an FP32 CPU reference with bit-exact equality
//! at every position below the threshold.
//!
//! Reference: https://deep-reinforce.com/correctness_check.html

use crate::{GpuBackend, ComputePipeline, GpuBuffer, Result};

/// Result of a numerical correctness check.
#[derive(Debug, Clone)]
pub struct NumericalCheckResult {
    pub trials: u32,
    pub positions_checked: usize,
    pub positions_exact: usize,
    pub max_diff_below_threshold: f32,
    pub passed: bool,
}

/// Configuration for the exact-match protocol.
#[derive(Debug, Clone)]
pub struct ExactMatchConfig {
    /// FP16 exact-integer ceiling (default 2048).
    pub threshold: f32,
    /// Number of random binary-input trials (default 16).
    pub trials: u32,
    /// Probability of sampling 0 vs 1 (default 0.7 = 70% zeros).
    pub p_zero: f32,
}

impl Default for ExactMatchConfig {
    fn default() -> Self {
        Self {
            threshold: 2048.0,
            trials: 16,
            p_zero: 0.7,
        }
    }
}

/// A kernel's reference implementation (CPU FP32).
///
/// Implement for each kernel type. The reference must compute the same
/// mathematical function as the GPU kernel.
pub trait NumericalReference {
    /// Generate binary inputs for trial `i`, return as flat byte buffers
    /// ready for GPU upload.
    fn generate_inputs(&self, trial: u32, cfg: &ExactMatchConfig) -> (Vec<Vec<u8>>, Vec<usize>);

    /// Compute the FP32 CPU reference output from the same inputs.
    fn compute_reference(&self, inputs: &[Vec<u8>], sizes: &[usize]) -> Vec<f32>;
}

/// Run the exact-match protocol against a GPU kernel.
///
/// Returns `NumericalCheckResult::passed = true` only if every trial achieved
/// bit-exact equality at every output position where the reference value was
/// ≤ `cfg.threshold`.
pub fn verify_numerical(
    gpu: &dyn GpuBackend,
    pipeline: &ComputePipeline,
    reference: &dyn NumericalReference,
    cfg: &ExactMatchConfig,
) -> Result<NumericalCheckResult> {
    // For each trial:
    //   1. Generate binary inputs
    //   2. Compute CPU reference
    //   3. Dispatch on GPU
    //   4. Read back output
    //   5. Compare bit-exact at positions where ref <= threshold
    //   6. Accumulate stats
    todo!("Phase 2 implementation")
}
```

### 3.2 Reference implementations

Implement `NumericalReference` for each shipped kernel:

- `AddOneReference` — input `[u8; N]` → reference `x as f32 + 1.0`
- `ScaleReference { alpha: f32 }` — `alpha * x`
- `SaxpyReference { alpha: f32 }` — `alpha * x + y`
- `MatmulReference { m, k, n }` — `A @ B` in FP32
- `GeometricProductReference { blades: u32 }` — sign-table weighted sum

These live in `src/numerical_check.rs` or a `src/references.rs` submodule.

### 3.3 Trait method (optional, behind `verify` feature)

Add to `GpuBackend` trait:

```rust
#[cfg(feature = "verify")]
fn verify_numerical(
    &self,
    pipeline: &ComputePipeline,
    reference: &dyn numerical_check::NumericalReference,
    cfg: &numerical_check::ExactMatchConfig,
) -> Result<numerical_check::NumericalCheckResult> {
    numerical_check::verify_numerical(self, pipeline, reference, cfg)
}
```

This provides a default implementation that backends inherit — no per-backend code needed because it uses the existing `dispatch` + `read_buffer` methods.

### 3.4 Obligation bundles updated

In `src/verify.rs`, add `.with_numerical_correctness("kernel_name", 2048, 16)` to each obligation bundle:

```rust
pub fn add_one_obligations() -> ObligationBundle {
    GpuObligationBundle::metal_kernel("borsalino_add_one", Origin::new("borsalino", "kernels::add_one"))
        .with_buffer_alignment("input_buffer", 16)
        .with_buffer_alignment("output_buffer", 16)
        .with_workgroup_divisibility("thread_count", 256)
        .with_dispatch_limit("workgroup_count", 65_535)
        .with_kernel_determinism("add_one_kernel")
        .with_numerical_correctness("add_one_kernel", 2048, 16)  // NEW
        .into_bundle()
}
```

Same for `scale_obligations()`, `saxpy_obligations()`, and new bundles for `matmul` and `geometric_product`.

### 3.5 Tests

- `add_one_numerical_check_passes` — run the protocol on `add_one`, expect pass
- `matmul_numerical_check_passes` — run on tiled matmul with small binary matrices (8×8)
- `incorrect_kernel_detected` — run against a deliberately broken kernel (e.g., adds 2 instead of 1), expect failure
- `threshold_gating_works` — outputs above 2048 are ignored, not flagged as failures
- `geometric_product_numerical_check` — run on the IA geometric product kernel

**Files changed:** `src/numerical_check.rs` (new, ~300 LOC), `src/lib.rs` (+module decl, +trait method), `src/verify.rs` (+obligation bundles), `Cargo.toml` (+`rand` dev-dependency for trial input generation)

**Verification:** `cargo test --features vulkan,verify` passes. `cargo test --features metal,verify` passes on Apple Silicon.

---

## 4. Phase 3: Documentation and honest scoping

**Goal:** Make the verification claims in docs match what's actually verified.

### 4.1 `src/verify.rs` module docs

Update the properties table:

```markdown
| Property | What it prevents | Mechanism | Tier |
|---|---|---|---|
| Buffer aligned to 16 bytes | Unaligned buffer → GPU fault | Type-level guard | Structural |
| Workgroup size divides thread count | Mismatch → undefined behavior | Type-level guard | Structural |
| Dispatch within device limits | Excess threads → validation error | Type-level guard | Structural |
| Kernel output is deterministic | Nondeterministic GPU behaviour | Statistical (amari-flynn) | Structural |
| Kernel output is numerically correct | Wrong answer in FP16/BF16 | Exact-match protocol (runtime) | Numerical |
```

Add a section distinguishing the two tiers:

```markdown
## Verification Tiers

Borsalino verifies two distinct properties:

### Structural Safety (Phase 12e)

Type-level guards that prevent GPU faults, undefined behavior, and validation
errors. These are checked at compile time via `Proven<>` phantom types and
exported to SMT/Lean/Kani proof backends.

### Numerical Correctness (v0.5.0)

Runtime verification that a kernel produces the correct output, using the
DeepReinforce exact-match protocol. Restricted to linear kernels with binary
inputs. Does not apply to non-linear operations (log, exp, tanh).
```

### 4.2 README

Update the "Verification" section to mention both tiers and link to the DeepReinforce paper.

### 4.3 Blog post

Publish a blog post on industrialalgebra.com: "Borsalino v0.5.0: Numerical Correctness Verification" — explaining the gap, the protocol, and what it covers (linear kernels) vs. what it doesn't (non-linear ops).

**Files changed:** `src/verify.rs` (docs), `README.md`, new blog post

---

## 5. Implementation Order

| Step | Phase | Repository | Depends On | Estimated LOC |
|---|---|---|---|---|
| 1 | 1 | karpal-verify | — | ~60 |
| 2 | 1 | karpal-verify | step 1 | ~40 (tests) |
| 3 | 2 | Borsalino | step 2 (new karpal-verify release) | ~300 |
| 4 | 2 | Borsalino | step 3 | ~200 (references + tests) |
| 5 | 2 | Borsalino | step 4 | ~30 (obligation bundles) |
| 6 | 3 | Borsalino | step 5 | ~100 (docs) |
| 7 | 3 | IA-home | step 6 | blog post |

**Total:** ~730 LOC across two repos.

---

## 6. What This Does NOT Cover

- **Non-linear kernels** — `log`, `exp`, `tanh`, sigmoid, etc. produce irrationals that cannot be checked with exact match. These remain unchecked. A future approach (statistical bounds, interval arithmetic) is noted in the DeepReinforce paper as a long-term goal.
- **Mixed precision** — the protocol assumes a single precision format per kernel. Mixed-precision accumulation (FP16 inputs, FP32 accumulate) needs a separate analysis.
- **Race conditions** — the exact-match check runs single-threaded CPU reference vs. GPU. It catches wrong answers but not data races that produce nondeterministic output. That's the domain of the existing `IsMSLKernelDeterministic` property + amari-flynn statistical verification.
- **Proof of correctness** — passing the protocol does not *prove* a kernel is correct. It provides strong evidence (high probability of catching bugs) but is not a mathematical proof. The docs must state this honestly.

---

## 7. Open Questions

1. **`rand` dependency** — the protocol needs RNG for trial input generation. Add as dev-dependency only (tests + examples), or behind the `verify` feature? Recommendation: dev-dependency, since `verify_numerical()` is a testing tool, not a production dispatch path.

2. **FP16 vs FP32 GPU output** — the protocol compares against FP16's 2048 threshold. If Borsalino kernels run in FP32 on GPU, the threshold should be higher (2^24 for FP32 exact integers). Configurable via `ExactMatchConfig::threshold`. Default 2048 assumes FP16 GPU compute.

3. **Geometric product sign table** — the GP kernel uses a precomputed sign table. For the reference implementation, should we recompute it independently or reuse the same table? Recommendation: recompute independently in the reference to catch sign-table bugs too.

---

## 8. Success Criteria

- [ ] `cargo test --features vulkan,verify` runs numerical checks on all linear kernels
- [ ] A deliberately broken kernel is detected and reported as incorrect
- [ ] `IsNumericallyCorrect` property appears in SMT/Lean/Kani exports
- [ ] `src/verify.rs` docs clearly distinguish structural safety from numerical correctness
- [ ] Blog post published explaining the approach and its limitations
- [ ] DeepReinforce paper cited in docs and README
