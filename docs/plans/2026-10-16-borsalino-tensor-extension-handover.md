# Borsalino — Tensor Extension Handover (Filling In for Candle, Compute Side)

**Date:** 2026-10-16
**From:** Amari core-matrix session (zip-yank incident response)
**Companion document:** `Shandora/Katakuri/docs/plans/2026-10-16-katakuri-architecture-handover.md`
(Katakuri = the ecosystem half: weights, tokenization, model specs, hub access)
**Status:** no tensor code exists in Borsalino today; this document defines the
extension's scope, boundaries, and slice plan.

---

## 1. Provenance — why Borsalino grows a tensor layer

On 2026-10-16 zip's entire 8.x line was yanked from crates.io, making
`candle-core 0.11.0`'s `zip ^8.6.0` requirement permanently unsatisfiable and
breaking every fresh `--all-features` resolution that reached candle. The
operator's strategic response is an IA-owned replacement of the candle
surface, split along the seam that already exists in the Shandora stack:

- **Katakuri** (new) — the *non-tensor-math* half: safetensors loading,
  WordPiece tokenization, model specs as typed data, hub access. Never
  computes.
- **Borsalino tensor extension** (this document) — the *compute* half: the
  tensor operations candle performed, as kernels dispatched through
  Borsalino.

**Operator doctrine (2026-10-16, verbatim intent):** *Borsalino is the
workhorse; Amari is the precision.* f32 GPU compute and exact/verified CPU
math are separate layers. The tensor extension is workhorse-grade: f32 is the
contract, not a compromise.

## 2. What Borsalino already brings (build on this — do not reinvent)

Measured against `main` @ 9c28ef6 (v0.7.0 just merged):

| Asset | What it gives the tensor layer |
|---|---|
| `GpuBackend` dispatch (`compile`, `dispatch`, `dispatch_ex`, buffers) | The execution substrate; kernels are WGSL `@compute` (workgroups x/y/z, 256 threads/group default) |
| `numerical_check.rs` — DeepReinforce exact-match protocol for **linear kernels** | The GEMM correctness gate already exists; matmul-class ops adopt it directly |
| `determinism.rs` — statistical bit-for-bit determinism verification | Every tensor kernel must pass repeat-dispatch determinism (catches atomics/races in reductions) |
| `kani_harnesses.rs` — workgroup divisibility, buffer alignment proofs | Shape/stride/tile contracts get the same formal treatment |
| Zunesha substrate (device, queues, `zunesha::Buffer`, epoch tracker, `NoDevice`) | Weight buffers from Katakuri flow through zero-copy; quiescence/GC safety is inherited, not restated |

## 3. Measured op inventory (scope ground truth)

From the measured consumer surface (Ijima `embeddings_candle.rs`, 287 lines —
the fleet's entire inference-class candle usage) plus the one custom kernel
already identified in the fleet (Quantizon):

**Level 0 — elementwise/buffer:**
copy, cast (f32 ↔ f16/bf16), add, mul, `broadcast_div`, **gather**
(embedding-table row lookup — the `Tensor::new` + vocab-table access pattern).

**Level 1 — reductions:**
sum, **attention-masked mean** (pooling over the token axis), L2 norm
(rowwise), max. *Reductions are the determinism-sensitive class — every one
needs a fixed-order or tree-reduction strategy proven by `determinism.rs`.*

**Level 2 — GEMM:**
matmul (2D), **batched matmul** `[B,H,S,D]` (attention projections). Gate:
DeepReinforce exact-match vs CPU oracle, including the adversarial shape
classes this fleet has already learned to test: subnormal-magnitude inputs,
non-power-of-2 dims, alignment-boundary buffer offsets (the kani harness
classes), empty batch/head dims.

**Level 3 — composition/activation:**
layernorm, GELU, rowwise softmax, scaled dot-product attention with additive
mask — composed from L0–L2 first, then the flagship fused kernel:
**fused 4D bidirectional attention mask** (Quantizon's identified need: mask
creation fused into attention, avoiding the zeros-tensor memory overhead).

**Explicitly NOT in scope:** autodiff, training, optimizers, gradient
anything (fleet-wide candle-nn replacement is a separate future decision;
Quantizon training stays put for now); model specs/loading/tokenization
(Katakuri); f64 exactness (Amari is the precision layer); graphics.

## 4. Architecture decisions to settle first

1. **Crate shape: `borsalino-tensor` as a new crate over Borsalino, vs a
   module inside Borsalino.** Session recommendation: **separate crate**.
   Borsalino's value is its thinness (the Zunesha precedent: substrate
   libraries stay substrate); tensor ops are policy built *on* dispatch. A
   crate also versions independently — Katakuri and consumers can pin it
   without forcing Borsalino releases.
2. **CPU reference path: first-class, not fallback.** Ijima runs on GPU-less
   hardware — a `NoDevice` CPU reference implementation (scalar or tiled
   CPU) is simultaneously (a) Ijima's execution path and (b) the equivalence
   oracle for every WGSL/Metal kernel. This mirrors amari's exact-Fraction
   oracle methodology (W03: 73,000 exact comparisons) adapted to f32
   tolerance classes.
3. **Dtype policy: f32 is the contract.** Katakuri delivers f32 weights;
   f16/bf16 exist only as explicit casts at the load boundary, with
   cast-roundtrip tests. Never silent.
4. **Layout contract with Katakuri S1:** weights arrive in
   `zunesha::Buffer`, row-major, with a typed shape/stride/dtype manifest.
   Kernel-side consumption is zero-copy; the manifest is the shared type —
   co-design it with the Katakuri session before either side freezes an API.

## 5. Verification doctrine (load-bearing)

Every op ships as a triple: **CPU reference + WGSL kernel + equivalence
tests**. Gates per op: DeepReinforce exact-match (L2 class) or
tolerance-class equivalence (transcendental class: GELU/softmax) vs CPU
oracle on adversarial shapes; `determinism.rs` repeat-dispatch bit-for-bit;
kani workgroup/alignment proofs for tiling contracts. The composed embedding
path (gather → GEMMs → layernorm → GELU → attention → masked pool → L2)
gets an end-to-end golden-vector test against published BERT outputs —
shared fixtures with Katakuri's tokenizer golden vectors so both halves
prove against the same corpus.

## 6. Slice plan

- **T0** — op-set contract (this document, hardened) + crate scaffold +
  buffer/manifest interop spec co-signed with Katakuri S1.
- **T1** — CPU reference for L0–L2 + property tests (shapes, strides,
  boundary offsets, subnormal-magnitude spreads).
- **T2** — WGSL kernels for L0–L2 behind DeepReinforce/determinism gates.
- **T3** — L3 composition ops + the fused 4D attention-mask kernel.
- **T4** — end-to-end embedding-path golden vectors + perf regression gates
  (no cliff vs candle on the same hardware, where comparable).

## 7. Consumers

- **Ijima** (first): GPU-less → CPU reference path; weights via Katakuri;
  forward implemented against the borsalino-tensor op set.
- **Quantizon:** the fused attention kernel is its identified win; training
  unaffected (stays wherever it is).
- **Minuet:** surface not yet measured (repo not on the authoring machine);
  measure before promising scope.
- **amari-gpu relationship:** the standing fleet question (wgpu vs Borsalino
  as the IA GPU story — both currently alive) is *referenced, not settled*
  here; the tensor extension must not prejudge it.

## 8. Conventions

ia-rust-stack checklist, ia-coding-standards, Apache-2.0 + SPDX, gitflow via
PRs (never push develop/main directly), review loops with inline PR posting
(12-round cap), evidence hygiene (`$SHANDORA`/`$HOME` scrub), and the new
fleet rule born of this incident: **no pre-declared dependencies** — every
dep carries a written justification at introduction.

## 9. Paths

- This repo: `/home/lucien/working/industrial-algebra/Shandora/Borsalino`
  (main @ 9c28ef6, v0.7.0; SSH remote; docs/plans convention).
- Katakuri: `../Katakuri` — handover at `docs/plans/2026-10-16-katakuri-architecture-handover.md`;
  repo `Industrial-Algebra/Katakuri` (private, main @ 320029d).
- Zunesha: `../Zunesha` — read `src/lib.rs` module docs for the buffer/
  epoch/NoDevice surface before T0.
- Measured consumer: `~/working/industrial-algebra/Anima/Ijima/ijima-server/src/embeddings_candle.rs`.
- Incident record: Amari PR #313 (candle removal + full diagnosis); upstream
  huggingface/candle PR #4041.
