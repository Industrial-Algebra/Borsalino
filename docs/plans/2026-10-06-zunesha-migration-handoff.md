# Borsalino → Zunesha Migration — Session Handoff

- **Date:** 2026-10-06
- **Status:** handoff — the migration itself has NOT started; no code in this
  document has been written
- **For:** the agent session that picks up Borsalino development next
- **Written after:** Borsalino 0.7.0 and Zunesha 0.1.0 published to crates.io;
  Goldenweek switched to registry zunesha; backmerges merged

## 1. Where everything stands

| Artifact | State |
|---|---|
| Zunesha 0.1.0 | crates.io live; tag `v0.1.0`; book + GH Release live |
| Borsalino 0.7.0 | crates.io live; tag `v0.7.0`; book + GH Release live |
| `develop` history | main and develop realigned (the v0.5.0–v0.6.0 squash damage is reconciled — future release PRs should merge clean; if one ever conflicts again, see §5 "lineage") |
| Goldenweek | consumes registry `zunesha = "0.1"`; PR #6 (path-dep → version-dep) awaiting merge as of writing |
| macOS CI | self-hosted hantaro runner; `BORSALINO_REQUIRE_METAL=1` fails hard if Metal is unreachable |
| Holmberg | hantaro power policy merged (#110) |

Announcement posts for both releases are live on industrialalgebra.com
(published via admin API; content recorded in IA-home PR #24). Auto-announce
workflow was proposed and **deliberately not wired** — Justin held on it.

## 2. The migration itself

**Goal:** replace Borsalino's internal device plumbing (its own
`VulkanDevice`/`MetalDevice`, buffers, queue selection) with `zunesha::Device`,
keeping Borsalino's own layer (pipelines/compile, dispatch, memory strategy
semantics, verification). Zunesha by design refuses pipelines and
verification — those stay Borsalino's. The contract between the crates is
[Zunesha ADR 0003 — cross-crate proof agreement](https://github.com/Industrial-Algebra/Zunesha/blob/develop/docs/adr/0003-cross-crate-proof-agreement.md).

### What Zunesha 0.1.0 offers today

- `Device` trait: enumerate, resolve queues, create/copy/read/write buffers,
  limits. Complete **Vulkan** backend, `NoDeviceStub` for hardware-free tests.
  Metal does not exist yet (post-0.1) — **so a full Borsalino-over-Zunesha
  Metal path is blocked until Zunesha ships Metal.** Decide up front whether
  the migration lands Vulkan-only first with the Metal backend keeping its
  internal device behind the same trait (staged), or waits for Zunesha Metal.
- Capability-driven queues (ADR 0002): compute always; graphics/transfer
  optional. Borsalino only needs compute — this should be a non-issue.
- **Epoch tracker exists but consumer dispatch accounting is NOT wired** in
  Zunesha 0.1.0 (no production begin/end_dispatch). The docs say so
  explicitly; do not assume `QuiescenceProof` certifies Borsalino's work yet.
  Borsalino currently has its own epoch gating (see §4) — during migration the
  honest state is that gating lives on Borsalino's side.

### The existing `SharedDevice`/`SharedInstance` overlap

Borsalino's `src/vulkan.rs` already has Arc+Deref `SharedDevice`/`SharedInstance`
and an `Arc<Entry>` loader pin — the exact pattern Zunesha 0.1.0 shipped. The
migration should **collapse Borsalino's copies onto Zunesha's**, not keep both.
Relevant design decisions are in the Borsalino book
(borsalino.industrial-algebra.com, ownership chapters) and Zunesha
architecture docs.

### Design questions to settle early (before writing code)

1. **Dispatch-scope guard.** What prevents a `Pulse`/command buffer from
   outliving the device after the device moves under Zunesha's ownership
   model? Borsalino's answer today: `Arc<GpuEpochTracker>` + epoch-gated
   `device_wait_idle` at backend drop (blanket wait_idle stampedes driver
   locks — do not regress to that). Decide how this maps onto Zunesha's
   (currently unwired) tracker, and whether a planned ADR 0004 (cross-queue /
  interop) is needed before or after the swap.
2. **`Proven<NumericallyCorrect, Pipeline>`** — type-level gate linking
   numerical verification to a compiled pipeline, sketched in
   `docs/VERIFICATION_ROADMAP_SUPPLEMENT.md` (phase 3 in `src/verify.rs`).
   Open design question: does the marker survive re-`compile` of the same
   source? At what layer does it live relative to Zunesha pipelines? Worth a
   short design note in the PR that introduces it.
3. **Feature unification.** `borsalino/vulkan` would forward to
   `zunesha/vulkan`; confirm `MemoryStrategy` mapping is 1:1 (it was lifted
   from the same design).

## 3. Flagged-not-fixed items (from 0.7.0 review rounds)

Each verified against `develop` at the time of writing:

1. **`GEOMETRIC_PRODUCT_BATCHED` bounds guard** — `src/lib.rs` kernel (the
   `gp_batched` const): 256-thread workgroup, 8 multivectors per workgroup;
   the example dispatches `div_ceil` workgroups, so tail invocations index
   `out_batch` OOB unless the batch is a multiple of 8. Fix: early-out
   `if (idx >= arrayLength(&out_batch)) { return; }` at kernel top (the five
   verified kernels in `src/numerical_check.rs` already have exactly this
   guard — mirror them). Add a non-multiple-of-8 batch case to the example
   test. Small standalone PR.
2. **`karpal-verify` 0.6.1 → 0.9.1 port** — `Cargo.toml` pins 0.6.1; 0.9.1
   is current. The verify feature compiles against 0.6-era API surface; port
   + adapt + re-run the verify CI job. Not release-blocking.
3. **`compare_outputs` dedupe** — `src/numerical_check.rs:505` vs
   `baedeker-core`'s GPU compare routine (the Baedeker `baedeker-borsalino`
   crate wraps Borsalino for karpal). Same semantics in two places; decide
   which crate owns it. Coordinate with whoever owns Baedeker state.
4. **`docs/VERIFICATION_ROADMAP_SUPPLEMENT.md`** tracks the remaining
   phase-3 items (`dispatch_verified()` on the trait, `DispatchConfig`,
   `Proven<>` gates) — that document is the running checklist; update it as
   items land rather than duplicating here.

## 4. Hard-won gotchas (regressions to avoid — each bit us once)

- **GPU tests must be `#[serial]`.** Parallel GPU tests on real hardware
  contend for driver-internal locks (deadlock reproduced 3-in-5 on RTX 5080;
  `wchan rt_mutex_schedule`). All 16 GPU tests are serialized now; keep any
  new one serialized.
- **Metal:** every path that mints an objc object runs inside
  `autoreleasepool` — including `compile()`/`compile_msl()` and the dispatch
  paths; `desc` must be released on pipeline-null failure paths; async
  `Pulse` command buffers escape the pool via explicit retain (the
  `async_pulse_survives_pool_drain` regression pins this).
- **objc ownership rule:** release only new/copy/retain products — never
  borrowed returns.
- **Epoch-gated `device_wait_idle` at drop only** — never a blanket wait on
  every drop path.
- **`#[ignore]`-less `-- --ignored` runs zero tests** — verify hardware jobs
  assert they touched a device (`BORSALINO_REQUIRE_METAL=1` on macOS) instead
  of silently passing.
- **CI:** macOS job runs on hantaro (self-hosted). If it goes quiet, check
  Holmberg `github-runner` state before assuming a code failure.
- **Lineage:** main and develop are now realigned. If a release PR ever
  conflicts with main again, the cause is a squash merge or missed backmerge —
  reconcile on the release branch toward develop (see `release/v0.7.0`
  history) and backmerge after release, per ia-gitflow.

## 5. Environment notes (rindler seat)

- `source /etc/set-environment` before cargo/GPU work.
- GPU: RTX 5080; GPU tests run serialized, ~seconds each.
- crates.io publishes go through `.github/workflows/release.yml` on `v*` tags
  (tag version must equal Cargo.toml version; changelog extract between
  `## [` headers). CARGO_REGISTRY_TOKEN + NETLIFY secrets are set on both
  repos.
- IA website announcements: `ADMIN_API_KEY` via the ia-website skill
  (manual publishing is the current mode; auto-announce deliberately on
  hold per Justin).

## 6. Suggested first steps for the next session

1. Read Zunesha ADR 0001–0003 + book, this doc, and
   `docs/VERIFICATION_ROADMAP_SUPPLEMENT.md`.
2. Decide the staged-vs-wait question (§2 — Zunesha Metal dependency).
3. Write the migration plan as a dated doc in `docs/plans/` per convention,
   TDD per ia-coding-standards (RED→GREEN per increment).
4. Optionally knock out the §3 flagged items first — #1 is a clean
   standalone warm-up PR.
