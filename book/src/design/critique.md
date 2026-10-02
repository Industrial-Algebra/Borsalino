# Critique of the **Borsalino** Project

> **v0.2.0 update (2026-06-03):** Since this critique was written (v0.1.0 pre-release),
> benchmarks have been published (4 platforms), documentation has been
> expanded (README, BENCHMARKS, CHANGELOG, ROADMAP), a dual commercial license
> model is in place, and CI covers format/clippy/test/docs/publish. 
> license is intentional by design, not a gap. See CHANGELOG.md for full history.

## TL;DR — Quick takeaways
- **Purpose** - A minimal, synchronous GPU compute abstraction that lets you write WGSL kernels and run them on either Metal (macOS) or Vulkan (Linux/Windows).  It aims for *zero-ceremony* - just a few function calls.
- **License** - **Apache-2.0** (was AGPL-3.0-only, relicensed v0.4.0). Permissive: shipping a binary that includes Borsalino carries no copyleft obligation beyond the license notice.
- **Maturity** - 0.7.0 pre-release.  The crate compiles, has a modest API surface, and includes a `verify` feature that pulls in `karpal-verify`/`karpal-proof` for GPU-safety checks, but many parts are still experimental.
- **Target audience** - Researchers or hobbyists needing a *thin* cross-platform GPU compute layer and who are comfortable with a pre-1.0 API.

---

## Strengths (Why it could be useful)
| Area | Details |
|------|---------|
| **Very small surface** | Only four public types (`ComputePipeline`, `GpuBuffer`, `GpuError`, `Result`) and a single trait (`GpuBackend`).  Minimal boilerplate for dispatching compute kernels. |
| **Cross-platform backends** | Metal on macOS (via `objc`) and Vulkan on Linux/Windows (via `ash`).  The same WGSL source works on both, thanks to `naga` for translation. |
| **Synchronous default, async available** | `dispatch` blocks until the GPU is finished (simple mental model); `dispatch_async` returns a `Pulse` for overlap. |
| **Safety wrapper** | The public API is safe; all `unsafe` is confined to the backend modules.  Buffer bounds are NOT checked — pipelines compile with `Unchecked` policies, so bounding accesses is the kernel's responsibility (`if (i >= N) { return; }`). Shader compilation errors surface as `GpuError`. |
| **Verification feature** | Optional `verify` feature brings in `karpal-verify`/`karpal-proof`, offering formal GPU-safety checks for those who need stronger guarantees. |
| **No heavy abstractions** | No bind-group layout gymnastics, no descriptor-set management - you just pass buffers in order.  This matches the "zero-ceremony" promise. |
| **Rust-first design** | Uses `bytemuck` for POD data, `naga` for shader translation, and follows idiomatic error handling (`Result`). |

---

## Weaknesses / Red Flags (What limits its adoption)
| Issue | Impact |
|-------|--------|
| **Apache-2.0 (was AGPL-3.0-only, relicensed v0.4.0) license** | Resolved in v0.4.0: relicensed to Apache-2.0. |
| **Very early stage (0.1.0)** | API is still stabilising; breaking changes are likely.  The crate has limited documentation and few examples. |
| **Async is minimal** | `dispatch_async` exists (returns a `Pulse`), but there is no future/stream composition — just wait-or-drop. Richer async pipelines remain unbuilt. |
| **Thin feature set** | Only compute pipelines; no support for graphics, ray-tracing, or compute-shader pipelines with multiple dispatches. |
| **Backend complexity** | The Metal backend uses raw `objc` FFI; the Vulkan backend uses `ash`.  Users must have the appropriate SDKs installed (Xcode for Metal, Vulkan SDK for Linux/Windows). |
| **Sparse documentation** | The README gives a quick start, but module-level docs are minimal.  Users need to read source to understand lifetime rules, buffer alignment, and error handling. |
| **Testing concentrated on Linux + one Mac** | CI runs 13 jobs including a self-hosted Apple Silicon runner and a GPU job on real hardware (RTX 5080), but Windows is untested. |
| **Benchmarks are self-published** | Numbers exist (RTX 5080, GB10, and others in README/BENCHMARKS), but no third-party comparison against raw Metal/Vulkan usage. |
| **Verification optional** | The `verify` feature is powerful but optional; without it you lose the formal safety guarantees that the `karpal` ecosystem provides. |
| **`verify_numerical` could not verify (found 2026-09-29, fixed in 0.7.0)** | The v0.6.0 driver never allocated an output buffer (read back the last *input*), uploaded binary inputs as u8 bytes against f32-reading kernels (denormals), omitted the GP sign table, and could not fail CI (`|| true` + no exit code + exit-0 no-GPU). Three stacked silence layers hid all of it. Fixed with the reference-as-metadata redesign, a recording-fake driver test, mutation tests, exit-code gating, and real `#[ignore]`d GPU tests — see CHANGELOG 0.7.0. |
| **Verdicts lived in no type** | Even post-fix, a numerical verdict is a runtime `NumericalCheckResult` value — printed by examples, gated by exit codes. The doctrine-shaped next step (`Proven<NumericallyCorrect, Pipeline>` carried by dispatch itself) is unbuilt anywhere in the ecosystem (2026-09-29 dive §4). |
| **karpal-verify pinned at 0.6.1** | The registry is at 0.9.1 (2026-09-13); the obligation-bundle API has moved. Any verification-stack work starts with a port. |
| **`compare_outputs` duplicated in Baedeker** | The exact-match core now exists in both `borsalino::numerical_check` and `baedeker_core::runtime::verify`. Where the canonical core should live is unresolved. |
| **No multi-GPU support** | The design assumes a single device; scaling to multi-GPU would require substantial changes. |

---

## Who would actually benefit?
- **Academic or hobbyist developers** experimenting with GPU compute kernels and wanting a uniform API across Metal and Vulkan.
- **Prototype engineers** who need a quick "write-once-run-anywhere" WGSL compute pipeline without dealing with bind-group boilerplate.
- **Rust-first teams** that already use `naga` and `bytemuck` and are comfortable with low-level GPU FFI.

*Not a good fit* for:
- Production-grade services that need a permissive license and guaranteed API stability.
- Applications requiring asynchronous pipelines, multi-GPU scaling, or advanced graphics features.
- Teams that lack the required platform SDKs (Xcode, Vulkan SDK) or need a stable, semver-committed 1.0 API today.

---

## Recommendations for Improvement (If you control the project)
1. ~~**Offer a dual-license**~~ - Resolved in v0.4.0: relicensed to Apache-2.0.
2. **Stabilise the API** - Move to a 1.0 release or at least a clear deprecation policy; publish a changelog.
3. **Expand documentation** - Add a "Getting Started" guide with a full end-to-end example (buffer creation → compile → dispatch → read).  Document safety guarantees, alignment requirements, and error codes.
4. ~~**Add async support**~~ - Resolved: `dispatch_async` + `Pulse` (wait-or-drop semantics; richer composition still open).
5. **Publish benchmarks** - Show latency and throughput for typical workloads (e.g., vector addition) on Metal vs. Vulkan.
6. **Increase test coverage** - Add integration tests for both backends, stress tests for large buffers, and CI that runs on macOS, Linux, and Windows.
7. **Make verification default** - Consider enabling `karpal-verify` by default (or at least warn users when it's disabled) to promote safer GPU code.
8. **Provide a higher-level abstraction** - Optional helper for bind-group layout generation or automatic buffer alignment could attract users who want a bit more convenience without losing the "thin" philosophy.

---

*Prepared by the coding-agent on 2026-06-03; corrected to v0.7.0 reality on 2026-10-01.*