# Changelog

All notable changes to this crate are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com), and the crate
follows [Semantic Versioning](https://semver.org).

## [Unreleased]

### Fixed — T2c review round 1

- Overflow-class shapes panicked before the zero could matter (r1 P1):
  `[MAX,MAX,MAX,0] x [MAX,MAX,0,0]` overflowed u64 computing `B*H*S`
  BEFORE multiplying by the trailing zero — the "widened" product still
  overflowed. `dim_product` now scans for a zero dimension FIRST (any
  zero → empty output, exact), then checked-multiplies; all-nonzero
  products that exceed u64 reject with a named error (the gate rejects
  them as > u32::MAX anyway).
- Public `BatchedMatMul::params` panicked on overflowing u32 stride
  products (r1 P2 — the method is public with no documented
  precondition): stride products are now zero-aware (a zero factor →
  stride 0, unused) and checked — overflow rejects. The entry path
  cannot reach the overflow (span checks bound S*D <= u32::MAX); the
  defense is for direct calls.
- Coverage added: H=0 and S=0 empty-output arms, the two overflow
  regressions (CPU-only, ErrBackend), and a second-call cache+correct
  assertion.

### Added — T2c: GEMM kernels (feature `gpu`)

- `matmul` (`[M,K] x [K,N] -> [M,N]`) and `batched_matmul`
  (`[B,H,S,D] x [B,H,D,T] -> [B,H,S,T]`) as FixedOrderGemm kernels:
  one invocation per output element, K contraction strictly ascending,
  one `fma` request per term — the CPU reference's accumulation order
  and term granularity. The batch/head walk and the output-element
  parallelism add no accumulation. Guards before indexing (the #74 r1
  lesson applied from the start): missing-shape and wrong-rank calls
  reject without panicking (CPU-only `ErrBackend` regressions);
  u64-widened output counts.
- Empty-output contracts: `M`/`N`/`B`/`H`/`S`/`T` zero writes nothing;
  `K`/`D` zero produces an all-zero output (an empty contraction
  accumulates to `0.0`, matching the CPU loop).
- CPU gemm term granularity switched to one `mul_add` request per term
  (the same explicit-fusion request as T2b's reductions; existing T1
  fixtures are exact-product cases — unaffected).
- 11-test `gpu_l2` suite: independent triple-loop oracles, order
  sensitivity (ascending-vs-descending K differs in bits — pinned),
  empty/zero-dimension contracts, mismatch rejections, determinism,
  pipeline cache. Mutation-verified: reversed K order and swapped
  batched slice strides each fail their oracle.
- GEMM has no architecture-precision tail op: outputs compare
  bit-exact (empirically, for tested fixtures, on stacks that pass the
  `fma` semantics probe).

### Fixed — T2b review round 2

- The "same rounding sequence by construction" claim was false (r2 P1,
  reopened with spec evidence): WGSL's `fma` accuracy is "inherited
  from e1*e2+e3" — a backend may LEGALLY evaluate it unfused
  (discriminator: x=1+2⁻²³, y=1-2⁻²³, z=-1 → fused exactly -2⁻⁴⁶,
  unfused exactly 0). The docs now state there is NO spec-level
  accumulator guarantee; instead the `gpu_fma_backend_semantics_probe`
  test compiles the discriminator kernel through the backend and
  REQUIRES the single-rounded answer — a stack with unfused `fma`
  fails the probe. The 5080 passes (single-rounded). The probe checks
  a STANDALONE kernel; the reduction oracle tests compare final
  outputs (ulp-bounded where division/sqrt tails could mask sub-ulp
  accumulator deviations) — accumulator equality is claimed for the
  tested fixtures, not verified as a full rounding sequence.
- bytemuck re-export moved out of the determinism module's doc block
  (r2 P3).

### Fixed — T2b review round 1

- Reduction entries indexed shapes before validating them (r1 P1):
  empty shape lists and short-rank shapes PANICKED at `dims()` before
  the gate could reject. Arity + rank guards now run first, the
  masked-mean `B*D` product is u64-widened (u32 multiply could overflow
  before the gate compared it), and CPU-only `ErrBackend` regression
  tests pin all three reproductions (no GPU needed — the reviewer's
  technique, made permanent).
- Documentation no longer claims WGSL forbids reassociation/fusion (it
  promises nothing of the kind — r1 P1): both sides request the fused
  operation explicitly, and the evidence is empirical — the standalone
  semantics probe plus fixture oracles (r3/r4 scoped the wording
  further).
- Precision docs now include BroadcastDiv's finite results in the
  architecture-precision seam, and mark the ulp bounds as observed on
  this stack rather than portable guarantees (r1 P2).
- `borsalino-tensor` now depends on the workspace `borsalino` by path
  (version+path — publish-compatible): the version-only dep resolved
  ambiguously and could compile against the published crate instead of
  the local tree.
- `borsalino` re-exports `bytemuck` so external `GpuBackend`
  implementors (tests, spies) can name the `Pod` bound.

### Added — T2b: broadcast execution + fixed-order reductions (feature `gpu`)

- Broadcast execution: `Add`/`Mul`/`BroadcastDiv` with unequal shapes
  run stride-mapped kernels (params: `[rank, out_dims, lhs_strides,
  rhs_strides]`; de-linearization from the fastest axis down). Equal
  shapes keep the T2a elementwise pipeline. The r2 "broadcasting not
  yet supported" rejection is gone — incompatible shapes still reject
  at the shape rule. `execute_binary` is now bounded by the new
  `BroadcastKernel` trait (compile-time: only 2-input all-f32
  Broadcast-rule ops fit).
- Reductions, preserving the CPU reference's exact parenthesization
  (one invocation per output element, sequential within): `sum` (empty
  input sums to `0.0`), `max` (strictly-`>` earliest-wins; empty input
  rejects with `EmptyReduction` before dispatch), `rowwise_l2_norm`
  (`[R,C]→[R]`, parallel across rows), `masked_mean`
  (`[B,S,D]+[B,S]→[B,D]`, all-zero mask rows are `0.0`, zero-width
  outputs skip dispatch).
- **Term granularity is now one fused multiply-add** in
  `masked_mean`/`rowwise_l2_norm` accumulations — WGSL `fma`, CPU
  `mul_add`. WGSL guarantees nothing about contraction or `fma`
  rounding (accuracy "inherited from e1*e2+e3"); both sides REQUEST
  the fused operation so a conforming single-rounding backend computes
  the same values, and the backend semantics probe checks that
  property per stack. Accumulator equality is empirical (probe +
  fixture oracles), not a spec guarantee. Order unchanged; CPU T1
  tests unaffected (their fixtures are exact-product cases).
- **New GPU equivalence seam, documented in the `gpu` module docs**:
  f32 division and `sqrt` are architecture-precision on this stack
  (observed ≤1 and ≤2 ulp vs correctly-rounded — observed bounds, not
  portable guarantees; the same oracle tests measure any other stack).
  Affected: `masked_mean`, `rowwise_l2_norm`, and finite `div` results
  whose exact quotient is not representable. Determinism stays
  unconditional. NaN payloads are platform-defined (`0/0` emits
  `0x7FFFFFFF` on GPU vs canonical quiet NaN on CPU) — NaNs compare by
  class.
- 16-test `gpu_l1` suite: independent broadcast oracle (right-aligned
  stride reference), reduction oracles, determinism, pipeline-cache,
  empty/edge cases. Mutation-verified: broadcast order and reversed
  sum order both fail their oracles.

### Added — T2a: GPU execution through Borsalino (feature `gpu`)

- `gpu` feature + `gpu` module: `TensorGpu<B: GpuBackend>` executes
  contracted ops as WGSL kernels (compile-once-per-op cache, operands
  in port order with output last and op-params buffers after it);
  `GpuKernel` binds an op type to its source/entry/params.
- Kernels for the BitExact op class: `relu`, `add`/`mul` (same-shape),
  `gather`, `transpose_2d` (shape carried in a trailing u32 params
  buffer — not recoverable from binding lengths).
- Equivalence gates: every kernel runs against the T1 CPU oracle with
  bit-identical comparison on the normal range, plus repeat-dispatch
  determinism. Mutation-checked (a passthrough relu mutant fails).
- **Subnormal policy (FTZ/DAZ), the GPU equivalence seam:** dispatch
  determinism is unconditional; oracle bit-equality holds on the normal
  range. Subnormals flush to a sign-preserving zero (inputs AND
  results), and comparisons read them as zero — Vulkan/Metal permit
  both, WGSL cannot disable them, and the hardware (RTX 5080)
  exercises both. Pinned as contract by
  `gpu_subnormal_flush_rule_is_sign_preserving_and_daz`. Katakuri's
  TrustPolicy keeps subnormal weights — consumers moving them to GPU
  accept the flush.
- New `TensorError::GpuDispatch` taxonomy variant (op-named backend
  failures).

### Added — issue #71 requirements (from Amari Cohort 8, the first T1 consumer)

- `ops::l3` module with `Relu` — the first activation op, so
  `katakuri::model::Activation::Relu` maps onto a contracted kernel.
  Elementwise, `DeterminismClass::BitExact`, shape-identity. The
  contract pins the comparison form `if x > 0.0 { x } else { 0.0 }`
  (NOT `f32::max`, whose `-0.0` vs `+0.0` result is unspecified):
  positives and subnormals pass through bit-identically, everything
  else maps to `+0.0`.
- `ops::l2::Transpose2D` — `[R, C] -> [C, R]` with the new
  `ShapeRule::Transpose2D` (rank-2 enforced). Removes the per-consumer
  weight-permute step: Katakuri's canonical `[out, in]` weights
  transpose once onto `matmul`'s `[K, N]` right operand. Pure
  permutation, `BitExact` by construction. Index arithmetic widened to
  `u64` before multiplying (valid `[65536, 65537]` overflowed `u32` at
  `c = 65536`); empty shapes early-return after the full gate
  (proportional-to-rows work on `[n, 0]` eliminated).
- `Shape::squeeze` / `Shape::flatten` — pure layout reinterpretations
  for head-code rank juggling (`[rows, 1]` -> `[rows]`; any rank -> one
  axis). No data moves. `flatten` is fallible: a count above `u32::MAX`
  has no rank-1 representation and rejects as `FlattenOverflow` rather
  than silently changing the element count.
- `BroadcastDiv` naming note: broadcasting is symmetric NumPy-style;
  the prefix is a usage convention (divisor is the broadcast operand at
  every real call site), not a direction claim.

### Deferred (with reasoning)

- `tanh` / `gelu_erf` / `gelu_tanh`: transcendental tolerance classes
  need the tolerance machinery (handover §5) — next slice, with `Relu`
  mapping first.
- SmallVec / const-generic shapes: breaking to the public API and
  perf-motivated; measure before changing.
- Katakuri-side companion (`MlpSpec::from_parts` validated constructor):
  filed as Katakuri #21.
