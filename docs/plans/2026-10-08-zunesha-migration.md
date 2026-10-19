# Borsalino → Zunesha Migration Plan

- **Date:** 2026-10-08
- **Status:** approved and in execution — Phase 0 (kernel guard, PR #58)
  and Phase 1 (the swap, PR #59) filed from this plan's session, awaiting
  review; Phases 2–4 pending
- **Decision (Justin, 2026-10-08):** **staged migration — Vulkan first.**
  Borsalino's Vulkan backend swaps onto `zunesha::Device` now; the Metal
  backend keeps its internal device behind the same `GpuBackend` trait until
  Zunesha ships Metal (requirements drafted in
  [Zunesha `docs/plans/2026-10-08-metal-backend-requirements.md`](https://github.com/Industrial-Algebra/Zunesha/blob/develop/docs/plans/2026-10-08-metal-backend-requirements.md)
  for a parallel session).
- **Predecessor:** [`2026-10-06-zunesha-migration-handoff.md`](./2026-10-06-zunesha-migration-handoff.md)
- **Contract:** [Zunesha ADR 0003 — cross-crate proof agreement](https://github.com/Industrial-Algebra/Zunesha/blob/develop/docs/adr/0003-cross-crate-proof-agreement.md)

## 1. Goal

Replace Borsalino's internal Vulkan device plumbing (its own instance/physical
device/logical device creation, queue selection, memory-type negotiation,
buffer allocation, staging transfers) with `zunesha::vulkan::VulkanDevice`,
**keeping Borsalino's own layer**: WGSL→SPIR-V compilation (+ disk cache),
pipeline/descriptor plumbing, dispatch, dispatch_many, dispatch_async
(`Pulse`), timestamps, epoch gating, and numerical verification. Zunesha by
design refuses pipelines and verification — those stay Borsalino's.

### Why staged (not wait-for-Metal)

1. Zunesha Metal is an unchecked roadmap item with no timeline — waiting
   blocks the migration indefinitely.
2. Borsalino is the *harder* consumer (Zunesha's own critique: "habits to
   unlearn"). Migrating now surfaces trait gaps while Zunesha is 0.1 with one
   backend to fix them against. The survey already found one gap (§5.1).
3. The `GpuBackend` trait surface is unchanged — Metal keeps its internals,
   macOS CI (hantaro) is untouched, no user-facing asymmetry.
4. IA sequencing doctrine: substrate-first (Zunesha-Metal precedes
   Goldenweek-Metal; Borsalino-Vulkan-over-Zunesha precedes both).

## 2. What moves vs. what stays

| Concern | Owner after Phase 1 | Note |
|---|---|---|
| Instance/entry/physical-device selection + scoring | **Zunesha** | `pick_physical_device`, `device_type_score`, `negotiate_api_version`, `create_device` deleted from `src/vulkan.rs` |
| Queue family selection | **Zunesha** | Borsalino reads `queues().compute` (`Queue::raw` + `family_index`) |
| Memory strategy negotiation, `detect_device_local` | **Zunesha** | `MemoryStrategy` is the same enum, lifted from this design — 1:1 |
| `find_memory_type_index`, `align_up` | *(was: Borsalino until Phase 2)* | **deleted in Phase 2** — `create_device_buffer` delegates to Zunesha (§5.1 resolved) |
| Buffer create/read (strategy-respecting) | **Zunesha** | staging design is identical (same code lineage) |
| `create_device_buffer` (GPU-resident profile) | **Borsalino, until Phase 2** | Zunesha 0.1.0's default delegates to `create_buffer`; Borsalino's temp-staging profile is a memory-vs-read-latency tradeoff (§5.1) — kept on `raw_device()` + `memory_properties()` until the profile alignment is an explicit decision |
| WGSL→SPIR-V (naga), `compile_cached`, disk cache | **Borsalino** | unchanged |
| Pipeline layout / descriptor sets / shader modules | **Borsalino** | built on `raw_device()` |
| Command pools, dispatch, `dispatch_many`, fences/`Pulse` | **Borsalino** | built on `raw_device()` + compute queue |
| Timestamp query pool | **Borsalino** | period queried via `entry()`/`raw_instance()`/`physical_device()` |
| Epoch tracker (`Arc<GpuEpochTracker>`) | **Borsalino** | honest state: Zunesha's tracker has no consumer dispatch accounting yet (ADR 0003 transfers ownership only when it does) |
| Backend drop epoch-gated `device_wait_idle` | **Borsalino** | via `raw_device()`; do **not** regress to blanket waits (§7) |

## 3. Ownership design — `Arc<zunesha::vulkan::VulkanDevice>`

Borsalino's public contract is stronger than Goldenweek's: `GpuBuffer`,
`ComputePipeline`, and `Pulse` may **outlive the backend** (P1 review
findings; the `SharedDevice` wrapper exists for exactly this). Goldenweek
solves sharing by borrowing `&'a VulkanDevice` — insufficient here, because
Zunesha's buffer inners hold raw `ash::Device` handle clones, not device
references; a buffer outliving the device is UB.

**Design:** `VulkanBackend` holds `z: Arc<zunesha::vulkan::VulkanDevice>`.
Every inner that can outlive the backend (`VulkanBufferInner` wrapper,
`VulkanPipelineInner`, `VulkanPulseInner`) holds an `Arc` clone of the same.
The device is destroyed when the *last* holder drops — the exact semantics
`SharedDevice`/`SharedInstance` provided, now with one owner (Zunesha) and
one sharing mechanism (Borsalino's `Arc`).

Drop ordering preserved: backend `Drop` destroys **only its own** pools
(descriptor/command/timestamp) after the epoch-gated idle wait; device and
instance destruction happen in Zunesha's `VulkanDevice::Drop` at final
`Arc` release.

New constructors this enables (the ADR 0001 interop story):

- `VulkanBackend::from_zunesha(z: Arc<zunesha::vulkan::VulkanDevice>)` —
  wrap a device shared with e.g. Goldenweek;
- `VulkanBackend::into_zunesha(self) -> Arc<…>` — hand the device onward.
- `init()` / `init_with_strategy()` remain and delegate to
  `VulkanDevice::init()` / `init_with_strategy()`.

## 4. Feature unification

```toml
zunesha = { version = "0.1", optional = true }
[features]
vulkan = ["dep:ash", "dep:zunesha", "zunesha/vulkan"]
```

- `ash` stays a direct dependency — Borsalino's pipeline/dispatch plumbing
  calls it directly on Zunesha's raw handles.
- **No path dependencies** — Borsalino CI (GitHub-hosted Linux + self-hosted
  macOS) cannot see `../Zunesha`; a path dep makes CI red.
- **Exception taken in Phase 1 (review-noted):** the swap needs Zunesha's
  `VulkanDevice: Send + Sync` (soundness of the `Arc` sharing — §3), which
  was unpublished at Phase-1 time. The dependency is temporarily a **git
  dep on Zunesha PR #9's branch** (public repo, CI-fetchable), with the
  merge-order contract: **Zunesha #9 merged → 0.1.1 published → flip to
  `version = "0.1"` → merge Borsalino Phase 1.** Registry 0.1.0 alone is
  sufficient only for the buffer/read delegation, NOT for the `Arc`
  ownership soundness.
- `MemoryStrategy` re-export: keep Borsalino's own enum as the public type
  (semver-stable for consumers) and convert at the boundary — the mapping is
  1:1 (`Auto`/`Unified`/`DeviceLocal`).

## 5. Zunesha-side companion work

### 5.1 Resolved — `create_device_buffer` placement semantics (Phase 2, aligned 2026-10-10)

> **Corrected 2026-10-08 after deeper reading** — the first draft of this
> section overstated the gap. The accurate picture:

Borsalino's `create_device_buffer(_uninit)` override is a *memory-profile*
optimization, not a forced-placement one: under a device-local strategy it
allocates device-local with **temp** staging (allocate → upload → free;
 readback allocates staging on demand), and under forced `Unified` it simply
takes the host-visible path (diverging from the trait's own "device-local"
doc wording). Zunesha 0.1.0's trait default also delegates to
`create_buffer`, and Zunesha's PR #9 now implements the documented contract:
`create_device_buffer` **forces** device-local + persistent staging
regardless of strategy.

Consequence for this migration: **delegating `create_device_buffer` in
Phase 1 would change behavior twice** (forced placement under
`Unified`, persistent vs temp staging) — so Borsalino keeps its own raw
implementation until Phase 2 makes the profile alignment an *explicit,
tested* decision (align to Zunesha's documented contract, or carry the
write-optimized profile). Zunesha 0.1.0 suffices for the buffer/read
delegation; the Phase-1 git-dep exception (§4) exists for the `Send + Sync`
soundness, not for this gap.

### 5.2 Resolved — blanket `device_wait_idle` in `VulkanDevice::Drop` (fixed in the companion PR)

Zunesha's `Drop` *originally* called `device_wait_idle()` unconditionally — the
behavior that stamped driver-internal locks under parallel load on the 5080
(Borsalino had moved away from blanket waits for that reason). The companion
branch (Zunesha PR #9, review round 1) now **gates the final-drop wait on
`epoch.in_flight() > 0`**, matching Borsalino's discipline: today the tracker
is never incremented in Zunesha 0.1, so the wait is a no-op and the consumer
contract is "submissions retired before the last handle drops"; full
consumer-side dispatch accounting (the backstop that re-arms the wait) is
ADR 0004 territory. Borsalino's own backend drop remains epoch-gated.

### 5.3 Optional nicety — `timestamp_period` in `DeviceLimits`

Borsalino currently queries `timestamp_period` itself via the exposed
`entry()`/`raw_instance()`/`physical_device()` handles. Adding it to
`DeviceLimits` would simplify consumers; not required.

## 6. Phases

### Phase 0 — warm-up (independent of the migration)

`GEOMETRIC_PRODUCT_BATCHED` bounds guard (handoff §3.1): early-out
`if (idx >= arrayLength(&out_batch)) { return; }` mirroring the five verified
kernels in `src/numerical_check.rs`; non-multiple-of-8 batch case in the
example test. Small standalone PR.

### Phase 1 — the swap (this session)

One atomic increment (the struct cannot hold two device sources; the field
collapse and the init-delegation are one change):

1. **RED:** new `#[serial]` GPU tests — `init_creates_zunesha_backed_device`
   (init → trait round-trip still passes) and `from_zunesha_shared_device`
   (construct `zunesha::vulkan::VulkanDevice` directly, `from_zunesha`,
   compile/dispatch/read `add_one`, then `into_zunesha` and use the device
   again — proves sharing).
2. Field collapse in `VulkanBackend`: `_entry`, `instance`, `device`,
   `queue`, `queue_family_index`, `min_storage_buffer_offset_alignment`,
   `memory_properties`, `memory_strategy`, `uses_device_local` →
   `z: Arc<zunesha::vulkan::VulkanDevice>` + `queue_family_index: u32`,
   `min_storage_buffer_offset_alignment`, `memory_properties`,
   `uses_device_local`, timestamp fields (derived from `z` at
   construction). **No cached queue** — the queue handle lives in the
   substrate; every submission runs inside
   `z.with_compute_queue(|queue| …)` (the externally-synchronized
   queue/pool protocol; added during review round 1).
3. `SharedDevice`/`SharedInstance`/`InstanceInner`/`DeviceInner` **deleted**;
   inners switch to `Arc<VulkanDevice>`.
4. `create_buffer` / `create_buffer_uninit` / `read_buffer` delegate to the
   `Device` trait on `z`. `GpuBuffer` wraps `zunesha::Buffer` (boxed, with
   the `Arc` keeping the device alive).
5. `create_device_buffer(_uninit)` **stay Borsalino's** raw implementation
   (on `z.raw_device()` + `z.memory_properties()`) until §5.1 ships.
6. `init`/`init_with_strategy` delegate to Zunesha; `from_zunesha`/
   `into_zunesha` added; `dispatch*`/`compile*`/`timestamp` switch handle
   source to `z.raw_device()` (mechanical).
7. **GREEN bar:** the full existing GPU suite (16 `#[serial]` tests +
   numerical_check + examples) passes unchanged on the 5080 — the trait
   surface did not move, so no caller changes.

### Phase 2 — finish the collapse ✅ DONE (2026-10-10, feat/phase2-buffer-delegation)

Decision taken (operator, 2026-10-10): **align to Zunesha's documented
contract** — `create_device_buffer(_uninit)` and `read_buffer` delegate to
the substrate (forced device-local + persistent staging, including under
forced `Unified` on discrete hardware). Borsalino's remaining allocation
code is deleted (`find_memory_type_index`, `align_up`, both allocate
paths, `one_shot_transfer`, `VulkanBufferInner`, the transfer command
pool, and the `memory_properties` / `uses_device_local` /
`min_storage_buffer_offset_alignment` fields). Pinned by delegation tests
(substrate tag + drop identity) and round-trips under both strategies;
see §5.1 for the changelog-noted behavior changes. The temp-staging
write-optimized profile remains recoverable later via an additive
Zunesha buffer-profile API if the persistent-staging memory cost ever
bites.

### Phase 3 — Metal migration (after Zunesha Metal; parallel track)

Mirror Phase 1 for `MetalBackend` per the
[Metal backend requirements doc](https://github.com/Industrial-Algebra/Zunesha/blob/develop/docs/plans/2026-10-08-metal-backend-requirements.md)
(Zunesha repo). Requires Zunesha to expose `raw_device()` (MTLDevice),
`raw_buffer()` (MTLBuffer), and queue escape hatches. Until then Metal is
untouched.

### Phase 4 — release

Version bump (0.8.0 — the public API gains `from_zunesha`/`into_zunesha` and
the `zunesha` dependency), CHANGELOG, book ownership-chapter rewrite
(`SharedDevice` story → `Arc<VulkanDevice>` story), ia-release-polish, tag
`v0.8.0`, backmerge per ia-gitflow.

## 7. Regression guardrails (each bit us once — do not reintroduce)

- GPU tests stay `#[serial]`; new ones too (driver-lock contention, RTX 5080
  deadlock reproduced 3-in-5 in parallel).
- Epoch-gated `device_wait_idle` at backend drop **only**; never blanket.
- Backend drop destroys only its own pools; device destruction is Zunesha's
  at final `Arc` release.
- Every `Pulse` path waits its fence **before** releasing its `Arc`
  (single-wait discipline; no double `end_dispatch` — keep the
  `epoch_completed` AtomicBool).
- macOS CI: if the hantaro job goes quiet, check Holmberg `github-runner`
  state before assuming a code failure.
- `#[ignore]`-less `-- --ignored` runs zero tests — hardware jobs assert they
  touched a device (`BORSALINO_REQUIRE_METAL=1`).

## 8. Open design questions (not Phase 1 scope)

1. **`Proven<NumericallyCorrect, Pipeline>`** — handoff §2.2. Lives in the
   verify roadmap (`docs/VERIFICATION_ROADMAP_SUPPLEMENT.md` is the running
   checklist); the marker-vs-recompile question gets a short design note in
   the PR that introduces it, not here.
2. **`compare_outputs` dedupe** (handoff §3.3) and **karpal-verify
   0.6.1→0.9.1 port** (handoff §3.2) — independent of the migration; schedule
   separately.
