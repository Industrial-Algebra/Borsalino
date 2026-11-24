# Changelog

All notable changes to this crate are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com), and the crate
follows [Semantic Versioning](https://semver.org).

## [Unreleased]

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
