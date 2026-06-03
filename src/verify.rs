// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! GPU verification obligations for Borsalino kernels.
//!
//! This module encodes the safety properties that each GPU kernel must
//! satisfy: buffer alignment, workgroup divisibility, dispatch limits,
//! and kernel determinism.
//!
//! These property types and obligation bundles are designed to integrate
//! with the Karpal verification stack (karpal-verify 0.5+). Until 0.5.0
//! is published to crates.io, the property types and bundle builder are
//! vendored locally. When karpal-verify 0.5.0 is available, switch to
//! `pub use karpal_verify::gpu::{...}`.
//!
//! # Properties
//!
//! | Property | What it prevents | Mechanism |
//! |---|---|---|
//! | Buffer aligned to 16 bytes | Unaligned buffer → GPU fault | Type-level guard |
//! | Workgroup size divides thread count | Mismatch → undefined behavior | Type-level guard |
//! | Dispatch within device limits | Excess threads → validation error | Type-level guard |
//! | Kernel outputs are deterministic | Nondeterministic GPU behaviour | Statistical (amari-flynn) |
//!
//! # Phase
//!
//! Phase 2 of the verification migration path: obligation bundles
//! and property types. Phase 3 adds `Proven<>` gates on dispatch.
//! Phase 4 adds Kani harnesses and statistical verification.

// ── Property types ────────────────────────────────────────────────
//
// Vendored from karpal-verify 0.5.0 (gpu.rs).
// Replace with `pub use karpal_verify::gpu::{...}` when published.

/// Property: buffer is 16-byte aligned for MTLBuffer / storage buffer compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IsBufferAlignedTo16;

/// Property: total thread count is divisible by workgroup size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IsWorkgroupSizeDivisible;

/// Property: dispatch parameters are within device limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IsDispatchWithinLimits;

/// Property: kernel produces deterministic output across dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IsMSLKernelDeterministic;

// ── Obligation bundle ─────────────────────────────────────────────
//
// Minimal obligation representation — a named container with
// typed property assertions. Designed to match karpal-verify's
// Obligation / ObligationBundle API for future drop-in replacement.

/// A single verification obligation: a property that must hold for a symbol.
#[derive(Debug, Clone)]
pub struct Obligation {
    /// Human-readable obligation name.
    pub name: String,
    /// The property being asserted (e.g. "IsBufferAlignedTo16").
    pub property: String,
    /// The symbol or entity this obligation applies to.
    pub target: String,
    /// Origin: (crate, module) where this obligation originates.
    pub origin: (String, String),
}

/// A collection of verification obligations for a named entity.
#[derive(Debug, Clone)]
pub struct ObligationBundle {
    name: String,
    origin: (String, String),
    obligations: Vec<Obligation>,
}

impl ObligationBundle {
    /// Create a new bundle for a named kernel.
    pub fn new(name: impl Into<String>, origin: (String, String)) -> Self {
        Self {
            name: name.into(),
            origin,
            obligations: Vec::new(),
        }
    }

    /// Add an obligation to this bundle.
    pub fn push(&mut self, obligation: Obligation) {
        self.obligations.push(obligation);
    }

    /// Iterate over all obligations.
    pub fn obligations(&self) -> &[Obligation] {
        &self.obligations
    }

    /// Bundle name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ── Builder helpers ───────────────────────────────────────────────

/// Create a bundle for a Metal/WGSL compute kernel.
pub fn kernel_bundle(name: &str, crate_origin: &str, module: &str) -> ObligationBundle {
    ObligationBundle::new(name, (crate_origin.into(), module.into()))
}

/// Add a buffer alignment obligation.
pub fn require_buffer_alignment(bundle: &mut ObligationBundle, buffer: &str, alignment: u32) {
    let name = format!("{buffer}_aligned_to_{alignment}");
    bundle.push(Obligation {
        name,
        property: "IsBufferAlignedTo16".into(),
        target: buffer.into(),
        origin: bundle.origin.clone(),
    });
}

/// Add a workgroup divisibility obligation.
pub fn require_workgroup_divisibility(bundle: &mut ObligationBundle, symbol: &str, divisor: u32) {
    let name = format!("{symbol}_divisible_by_{divisor}");
    bundle.push(Obligation {
        name,
        property: "IsWorkgroupSizeDivisible".into(),
        target: symbol.into(),
        origin: bundle.origin.clone(),
    });
}

/// Add a dispatch limit obligation.
pub fn require_dispatch_limit(bundle: &mut ObligationBundle, symbol: &str, limit: u32) {
    let name = format!("{symbol}_within_{limit}");
    bundle.push(Obligation {
        name,
        property: "IsDispatchWithinLimits".into(),
        target: symbol.into(),
        origin: bundle.origin.clone(),
    });
}

/// Add a kernel determinism obligation.
pub fn require_kernel_determinism(bundle: &mut ObligationBundle, kernel: &str) {
    let name = format!("{kernel}_deterministic");
    bundle.push(Obligation {
        name,
        property: "IsMSLKernelDeterministic".into(),
        target: kernel.into(),
        origin: bundle.origin.clone(),
    });
}

// ── Kernel obligation bundles ─────────────────────────────────────

/// Build verification obligations for the `add_one` kernel.
pub fn add_one_obligations() -> ObligationBundle {
    let mut b = kernel_bundle("borsalino_add_one", "borsalino", "kernels::add_one");
    require_buffer_alignment(&mut b, "input_buffer", 16);
    require_buffer_alignment(&mut b, "output_buffer", 16);
    require_workgroup_divisibility(&mut b, "thread_count", 256);
    require_dispatch_limit(&mut b, "workgroup_count", 65_535);
    require_kernel_determinism(&mut b, "add_one_kernel");
    b
}

/// Build verification obligations for the `scale` kernel.
pub fn scale_obligations() -> ObligationBundle {
    let mut b = kernel_bundle("borsalino_scale", "borsalino", "kernels::scale");
    require_buffer_alignment(&mut b, "input_buffer", 16);
    require_buffer_alignment(&mut b, "output_buffer", 16);
    require_workgroup_divisibility(&mut b, "thread_count", 256);
    require_dispatch_limit(&mut b, "workgroup_count", 65_535);
    require_kernel_determinism(&mut b, "scale_kernel");
    b
}

/// Build verification obligations for the `saxpy` kernel.
pub fn saxpy_obligations() -> ObligationBundle {
    let mut b = kernel_bundle("borsalino_saxpy", "borsalino", "kernels::saxpy");
    require_buffer_alignment(&mut b, "x_buffer", 16);
    require_buffer_alignment(&mut b, "y_buffer", 16);
    require_buffer_alignment(&mut b, "out_buffer", 16);
    require_workgroup_divisibility(&mut b, "thread_count", 256);
    require_dispatch_limit(&mut b, "workgroup_count", 65_535);
    require_kernel_determinism(&mut b, "saxpy_kernel");
    b
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_one_bundle_contains_all_properties() {
        let bundle = add_one_obligations();

        assert!(
            bundle
                .obligations()
                .iter()
                .any(|o| o.property == "IsBufferAlignedTo16")
        );
        assert!(
            bundle
                .obligations()
                .iter()
                .any(|o| o.property == "IsWorkgroupSizeDivisible")
        );
        assert!(
            bundle
                .obligations()
                .iter()
                .any(|o| o.property == "IsDispatchWithinLimits")
        );
        assert!(
            bundle
                .obligations()
                .iter()
                .any(|o| o.property == "IsMSLKernelDeterministic")
        );
    }

    #[test]
    fn add_one_has_five_obligations() {
        let bundle = add_one_obligations();
        assert_eq!(
            bundle.obligations().len(),
            5,
            "add_one: 2 buffer alignments + workgroup + dispatch + determinism"
        );
    }

    #[test]
    fn scale_bundle_contains_all_properties() {
        let bundle = scale_obligations();

        assert!(
            bundle
                .obligations()
                .iter()
                .any(|o| o.property == "IsBufferAlignedTo16")
        );
        assert!(
            bundle
                .obligations()
                .iter()
                .any(|o| o.property == "IsWorkgroupSizeDivisible")
        );
        assert!(
            bundle
                .obligations()
                .iter()
                .any(|o| o.property == "IsDispatchWithinLimits")
        );
        assert!(
            bundle
                .obligations()
                .iter()
                .any(|o| o.property == "IsMSLKernelDeterministic")
        );
    }

    #[test]
    fn saxpy_bundle_has_three_buffer_alignments() {
        let bundle = saxpy_obligations();

        let alignment_count = bundle
            .obligations()
            .iter()
            .filter(|o| o.property == "IsBufferAlignedTo16")
            .count();
        assert_eq!(alignment_count, 3, "saxpy: x, y, out buffers");
    }

    #[test]
    fn bundles_have_origin_info() {
        for (bundle, expected_name) in &[
            (add_one_obligations(), "borsalino_add_one"),
            (scale_obligations(), "borsalino_scale"),
            (saxpy_obligations(), "borsalino_saxpy"),
        ] {
            assert_eq!(bundle.name(), *expected_name);
            assert!(!bundle.obligations().is_empty());
        }
    }
}
