// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! GPU verification obligations for Borsalino kernels.
//!
//! This module integrates with the Karpal verification stack
//! (Phase 12e) to encode the safety properties that each GPU
//! kernel must satisfy. Properties are collected into
//! [`ObligationBundle`] instances that can be exported to SMT,
//! Lean 4, and Kani verification backends.
//!
//! # Properties verified
//!
//! | Property | What it prevents | Mechanism |
//! |---|---|---|
//! | Buffer aligned to 16 bytes | Unaligned buffer → GPU fault | Type-level guard |
//! | Workgroup size divides thread count | Mismatch → undefined behavior | Type-level guard |
//! | Dispatch within device limits | Excess threads → validation error | Type-level guard |
//! | Kernel output is deterministic | Nondeterministic GPU behaviour | Statistical (amari-flynn) |
//!
//! # Phase
//!
//! Phase 2 of the verification migration path: obligation bundles
//! and property types. Phase 3 adds `Proven<>` gates on dispatch.
//! Phase 4 adds Kani harnesses and statistical verification.

use karpal_verify::gpu::GpuObligationBundle;
use karpal_verify::{ObligationBundle, Origin};

// ── Re-exports ────────────────────────────────────────────────────

pub use karpal_proof::{Property, Proven};
pub use karpal_verify::gpu::{
    IsBufferAlignedTo16, IsDispatchWithinLimits, IsMSLKernelDeterministic, IsWorkgroupSizeDivisible,
};

// ── Obligation bundles ────────────────────────────────────────────

/// Build verification obligations for the `add_one` kernel.
///
/// Properties asserted:
/// - Input and output buffers are 16-byte aligned
/// - Thread count (4 × 256) is divisible by workgroup size (256)
/// - Dispatch fits within Metal's 1D grid limit
/// - Kernel is deterministic across dispatches
pub fn add_one_obligations() -> ObligationBundle {
    GpuObligationBundle::metal_kernel(
        "borsalino_add_one",
        Origin::new("borsalino", "kernels::add_one"),
    )
    .with_buffer_alignment("input_buffer", 16)
    .with_buffer_alignment("output_buffer", 16)
    .with_workgroup_divisibility("thread_count", 256)
    .with_dispatch_limit("workgroup_count", 65_535)
    .with_kernel_determinism("add_one_kernel")
    .into_bundle()
}

/// Build verification obligations for the `scale` kernel.
pub fn scale_obligations() -> ObligationBundle {
    GpuObligationBundle::metal_kernel(
        "borsalino_scale",
        Origin::new("borsalino", "kernels::scale"),
    )
    .with_buffer_alignment("input_buffer", 16)
    .with_buffer_alignment("output_buffer", 16)
    .with_workgroup_divisibility("thread_count", 256)
    .with_dispatch_limit("workgroup_count", 65_535)
    .with_kernel_determinism("scale_kernel")
    .into_bundle()
}

/// Build verification obligations for the `saxpy` kernel.
pub fn saxpy_obligations() -> ObligationBundle {
    GpuObligationBundle::metal_kernel(
        "borsalino_saxpy",
        Origin::new("borsalino", "kernels::saxpy"),
    )
    .with_buffer_alignment("x_buffer", 16)
    .with_buffer_alignment("y_buffer", 16)
    .with_buffer_alignment("out_buffer", 16)
    .with_workgroup_divisibility("thread_count", 256)
    .with_dispatch_limit("workgroup_count", 65_535)
    .with_kernel_determinism("saxpy_kernel")
    .into_bundle()
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use karpal_verify::{export_kani_bundle, export_lean_bundle, export_smt_bundle};

    #[test]
    fn add_one_bundle_contains_all_properties() {
        let bundle = add_one_obligations();
        let obligations = bundle.obligations();

        assert!(
            obligations
                .iter()
                .any(|o| o.property == IsBufferAlignedTo16::NAME)
        );
        assert!(
            obligations
                .iter()
                .any(|o| o.property == IsWorkgroupSizeDivisible::NAME)
        );
        assert!(
            obligations
                .iter()
                .any(|o| o.property == IsDispatchWithinLimits::NAME)
        );
        assert!(
            obligations
                .iter()
                .any(|o| o.property == IsMSLKernelDeterministic::NAME)
        );
    }

    #[test]
    fn add_one_bundle_exports_through_all_backends() {
        let bundle = add_one_obligations();

        let smt = export_smt_bundle(&bundle);
        let lean = export_lean_bundle("BorsalinoVerify", &bundle);
        let kani = export_kani_bundle(&bundle);

        assert!(!smt.is_empty(), "SMT export should not be empty");
        assert!(
            lean.contains("deterministic_kernel"),
            "Lean export should contain deterministic_kernel"
        );
        assert!(!kani.is_empty(), "Kani export should not be empty");
        assert!(
            kani.iter().any(|h| h.source.contains("kani::assert")),
            "Kani harness should contain assertions"
        );
    }

    #[test]
    fn scale_bundle_contains_all_properties() {
        let bundle = scale_obligations();
        let obligations = bundle.obligations();

        assert!(
            obligations
                .iter()
                .any(|o| o.property == IsBufferAlignedTo16::NAME)
        );
        assert!(
            obligations
                .iter()
                .any(|o| o.property == IsWorkgroupSizeDivisible::NAME)
        );
        assert!(
            obligations
                .iter()
                .any(|o| o.property == IsDispatchWithinLimits::NAME)
        );
        assert!(
            obligations
                .iter()
                .any(|o| o.property == IsMSLKernelDeterministic::NAME)
        );
    }

    #[test]
    fn saxpy_bundle_contains_three_buffer_alignments() {
        let bundle = saxpy_obligations();
        let obligations = bundle.obligations();

        let alignment_count = obligations
            .iter()
            .filter(|o| o.property == IsBufferAlignedTo16::NAME)
            .count();
        assert_eq!(
            alignment_count, 3,
            "saxpy should have 3 buffer alignment obligations (x, y, out)"
        );
    }
}
