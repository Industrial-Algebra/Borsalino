// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Katakuri manifest interop — the T0 co-sign gate (handover §4.4, §6).
//!
//! Katakuri **emits** a [`TensorManifest`]; this crate **consumes** it. The
//! manifest is **data**: it carries typed shape, byte span, dtype, layout,
//! and determinism metadata, and it names no buffer. Binding an entry to a
//! `zunesha::Buffer` handle is the consumer's concern, which is exactly why
//! this module re-exports Katakuri's types instead of defining its own — a
//! local copy would be a second contract that could drift.
//!
//! Two functions make up the interop:
//!
//! - [`validate_manifest`] checks a manifest against itself and the data
//!   section: shape × dtype footprint per entry, no overlapping or
//!   overflowing byte spans, a determinism pin that agrees with the declared
//!   dtype.
//! - [`bind_to_op`] checks a manifest against an op's ports: one entry per
//!   port, dtype per port, and the op's shape rule over the entry shapes.
//!   Because the entries are row-major contiguous, the implied strides are
//!   the layout's strides — nothing is materialized, so T1 can bind bytes
//!   zero-copy.
//!
//! # The co-sign status
//!
//! Katakuri 0.2.0 ships `manifest` as a **proposal** that is explicitly not
//! frozen (`katakuri::manifest`'s module docs say so). This module is the
//! Borsalino half of that review: the types are consumed as-is, and the
//! tests in `tests/manifest_interop.rs` pin the fields this layer relies on.
//! A `0.2.x` change to those fields is a breaking change *for this crate*,
//! which is why the dependency is version-pinned to `0.2`.

use crate::dtype::{Dtype, require_contract};
use crate::error::TensorError;
use crate::layout::{Shape, TensorLayout};
use crate::op::TensorOp;

pub use katakuri::manifest::{
    DeterminismPolicy, FastPathPolicy, ManifestDtype, ManifestEntry, ManifestLayout, TensorManifest,
};

/// One manifest entry, bound to a checked tensor layout.
///
/// The layout is derived from the entry's shape and the manifest's dtype:
/// row-major and contiguous by construction, so the value is what a T1
/// consumer needs to reinterpret the entry's byte span in place. The byte
/// span is carried through unchanged — binding it to a buffer is the
/// consumer's step, not this module's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestBinding {
    /// The entry's canonical tensor name.
    pub name: String,
    /// The entry's byte offset into the data section.
    pub byte_offset: u64,
    /// The entry's byte length.
    pub byte_len: u64,
    /// The dtype, shape, and implied strides.
    pub layout: TensorLayout,
}

/// Map a manifest dtype to the tensor layer's dtype.
///
/// v1 manifests are f32-only, so this is total and cannot fail; it exists so
/// the mapping lives in one place.
fn dtype_from_manifest(dtype: ManifestDtype) -> Dtype {
    match dtype {
        ManifestDtype::F32 => Dtype::F32,
    }
}

/// Validate a manifest against itself and its data section.
///
/// Checks, in order, and for every entry:
///
/// 1. the manifest's dtype is the f32 contract dtype,
/// 2. the determinism pin names the same dtype as the manifest itself (a pin
///    that disagrees cannot be honoured),
/// 3. the entry's shape is representable, its byte footprint fits in a `u64`
///    (no saturating arithmetic: `[u32::MAX, u32::MAX]` f32 needs
///    73,786,976,260,478,468,100 bytes), and its `byte_len` is exactly
///    `element count × dtype size` — a manifest may not under- or
///    over-declare a tensor's footprint,
/// 4. the entry's `byte_offset..byte_offset + byte_len` span fits inside the
///    `available`-byte data section,
/// 5. no two entries' spans intersect. Entries name distinct parameters, so
///    aliasing a byte range would make two tensors the same memory; a
///    checkpoint that means to tie weights names one entry twice. Zero-length
///    entries (a zero dimension) cover nothing and never conflict.
///
/// Layout is checked by construction: [`ManifestLayout`] has one variant
/// (row-major contiguous), which is the only layout T0 binds.
///
/// # Errors
///
/// [`TensorError::NonContractDtype`], [`TensorError::DeterminismDtypeMismatch`],
/// [`TensorError::ShapeOverflow`], [`TensorError::ByteLenOverflow`],
/// [`TensorError::ByteLenMismatch`], [`TensorError::OutOfBounds`], or
/// [`TensorError::OverlappingEntries`].
pub fn validate_manifest(manifest: &TensorManifest, available: u64) -> Result<(), TensorError> {
    let dtype = dtype_from_manifest(manifest.dtype);
    require_contract(dtype)?;

    let pinned = dtype_from_manifest(manifest.determinism.dtype_pinned);
    if pinned != dtype {
        return Err(TensorError::DeterminismDtypeMismatch {
            pinned,
            manifest: dtype,
        });
    }
    // One policy in v1: no fast path may change a bit pattern. The
    // irrefutable binding is the guard that fires the day a second policy
    // appears without a determinism check here.
    let FastPathPolicy::Forbidden = manifest.determinism.fast_paths;
    let ManifestLayout::RowMajorContiguous = manifest.layout;

    // (end offset, start offset, entry index), for the overlap sweep below.
    let mut spans: Vec<(u64, u64, usize)> = Vec::with_capacity(manifest.entries.len());

    for (index, entry) in manifest.entries.iter().enumerate() {
        let shape = Shape::new(&entry.shape)?;
        let layout = TensorLayout::row_major(dtype, shape)?;
        layout.check_byte_len_named(&entry.name, entry.byte_len)?;

        let end = entry
            .byte_offset
            .checked_add(entry.byte_len)
            .ok_or_else(|| TensorError::OutOfBounds {
                name: entry.name.clone(),
                offset: entry.byte_offset,
                // A span that overflows u64 cannot fit in any u64 section.
                end: u64::MAX,
                available,
            })?;
        if end > available {
            return Err(TensorError::OutOfBounds {
                name: entry.name.clone(),
                offset: entry.byte_offset,
                end,
                available,
            });
        }

        // Zero-length entries cover no bytes, so they never conflict.
        if entry.byte_len > 0 {
            spans.push((entry.byte_offset, end, index));
        }
    }

    check_no_overlap(&manifest.entries, &mut spans)?;

    Ok(())
}

/// Reject intersecting byte spans.
///
/// Sorts the non-empty spans by offset and sweeps them, remembering the
/// furthest end seen so far: a span that starts before it intersects the
/// entry that reached it. Adjacent spans (`0..4`, `4..8`) do not intersect.
fn check_no_overlap(
    entries: &[ManifestEntry],
    spans: &mut [(u64, u64, usize)],
) -> Result<(), TensorError> {
    spans.sort_unstable();

    let mut furthest: Option<(u64, usize)> = None;
    for &(start, end, index) in spans.iter() {
        if let Some((reached, reached_index)) = furthest {
            if start < reached {
                return Err(TensorError::OverlappingEntries {
                    first: entries[reached_index].name.clone(),
                    second: entries[index].name.clone(),
                    offset: start,
                    end: end.min(reached),
                });
            }
            if end > reached {
                furthest = Some((end, index));
            }
        } else {
            furthest = Some((end, index));
        }
    }

    Ok(())
}

/// Bind a manifest's entries to an op's ports.
///
/// Runs [`validate_manifest`] first, then requires the entry count to match
/// the op's port count and the resulting layouts to satisfy the op's dtype
/// policy and shape rule. Entries are bound in manifest order, which is the
/// canonical parameter order, so port *i* receives entry *i*.
///
/// A manifest can never satisfy an [`Indices`](crate::op::PortRole::Indices)
/// port: entries are f32 data, indices are `u32` computed at run time. That
/// is a deliberate refusal, not an omission — indices never come from a
/// checkpoint.
///
/// # Errors
///
/// [`TensorError::ManifestArityMismatch`], plus everything
/// [`validate_manifest`] reports, plus the dtype and shape errors the op's
/// [`check_inputs`](TensorOp::check_inputs) raises.
pub fn bind_to_op<O: TensorOp>(
    manifest: &TensorManifest,
    available: u64,
) -> Result<Vec<ManifestBinding>, TensorError> {
    if manifest.entries.len() != O::INPUTS.len() {
        return Err(TensorError::ManifestArityMismatch {
            op: O::NAME,
            expected: O::INPUTS.len(),
            got: manifest.entries.len(),
        });
    }

    validate_manifest(manifest, available)?;

    let dtype = dtype_from_manifest(manifest.dtype);
    let mut bindings = Vec::with_capacity(manifest.entries.len());
    let mut layouts = Vec::with_capacity(manifest.entries.len());
    for entry in &manifest.entries {
        // Shapes and byte lengths were validated above; the footprint cannot
        // overflow here, and the byte length is already pinned.
        let shape = Shape::new(&entry.shape)?;
        let layout = TensorLayout::row_major(dtype, shape)?;
        layouts.push(layout.clone());
        bindings.push(ManifestBinding {
            name: entry.name.clone(),
            byte_offset: entry.byte_offset,
            byte_len: entry.byte_len,
            layout,
        });
    }

    O::check_inputs(&layouts)?;

    Ok(bindings)
}
