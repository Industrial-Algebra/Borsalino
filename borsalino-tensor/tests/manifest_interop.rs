// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Katakuri manifest interop tests (T0 co-sign gate).
//!
//! The manifest is **data**: these tests construct Katakuri's own types,
//! hand them to the tensor layer, and pin what it accepts and what it
//! rejects. No buffer type appears anywhere — binding entries to
//! `zunesha::Buffer` handles is the consumer's concern (handover §4.4).

use borsalino_tensor::dtype::{Dtype, F16, F32};
use borsalino_tensor::error::TensorError;
use borsalino_tensor::layout::Shape;
use borsalino_tensor::manifest::{ManifestBinding, bind_to_op, validate_manifest};
use borsalino_tensor::op::{PortRole, TensorOp};
use borsalino_tensor::ops::l0::Copy as CopyOp;
use borsalino_tensor::ops::l0::{Add, Cast, Copy, Gather};
use borsalino_tensor::ops::l1::Sum;
use borsalino_tensor::ops::l2::MatMul;

// Constructed through Katakuri's own path: if the re-export in
// `borsalino_tensor::manifest` were a local copy instead of the co-signed
// type, this import pair would not be one type.
use borsalino_tensor::manifest::{
    DeterminismPolicy, FastPathPolicy, ManifestDtype, ManifestEntry, ManifestLayout, TensorManifest,
};

fn entry(name: &str, shape: &[u32], byte_offset: u64, byte_len: u64) -> ManifestEntry {
    ManifestEntry {
        name: name.to_string(),
        shape: shape.to_vec(),
        byte_offset,
        byte_len,
    }
}

fn manifest(entries: Vec<ManifestEntry>) -> TensorManifest {
    TensorManifest {
        entries,
        dtype: ManifestDtype::F32,
        layout: ManifestLayout::RowMajorContiguous,
        determinism: DeterminismPolicy::pinned(),
    }
}

#[test]
fn katakuri_manifest_is_the_co_signed_type() {
    // The manifest types are re-exports of `katakuri::manifest`, not a
    // second definition: the same value must typecheck as Katakuri's type.
    let ours: TensorManifest = manifest(vec![entry("head.bias", &[1], 0, 4)]);
    let theirs: katakuri::manifest::TensorManifest = ours.clone();
    assert_eq!(theirs, ours);
}

#[test]
fn matmul_manifest_binds_to_the_op_input_contract() {
    // Two projection weights: [4,8] then [8,16], 128 + 512 bytes.
    let m = manifest(vec![
        entry("encoder.weight", &[4, 8], 0, 128),
        entry("projection.weight", &[8, 16], 128, 512),
    ]);

    assert_eq!(validate_manifest(&m, 640), Ok(()));

    let bindings = bind_to_op::<MatMul>(&m, 640).unwrap();
    assert_eq!(bindings.len(), 2);

    let ManifestBinding {
        name,
        byte_offset,
        byte_len,
        layout,
    } = bindings[0].clone();
    assert_eq!(name, "encoder.weight");
    assert_eq!(byte_offset, 0);
    assert_eq!(byte_len, 128);
    assert_eq!(layout.dtype(), Dtype::F32);
    assert_eq!(layout.shape(), &Shape::new(&[4, 8]).unwrap());
    // Row-major contiguous: the manifest implies the strides, zero-copy.
    assert_eq!(layout.strides().as_slice(), &[8u64, 1]);
    assert_eq!(layout.byte_len(), 128);
    assert!(layout.is_row_major_contiguous());

    assert_eq!(bindings[1].layout.strides().as_slice(), &[16u64, 1]);
    assert_eq!(bindings[1].layout.byte_len(), 512);
}

#[test]
fn manifest_arity_must_match_the_op_ports() {
    let m = manifest(vec![entry("encoder.weight", &[4, 8], 0, 128)]);

    assert_eq!(
        bind_to_op::<MatMul>(&m, 128),
        Err(TensorError::ManifestArityMismatch {
            op: "matmul",
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn manifest_dtype_must_match_the_port_dtype() {
    // A manifest is always f32; a narrowing cast's *input* port is f16, so
    // the binder must refuse to feed f32 weights into it.
    let m = manifest(vec![entry("weights", &[4], 0, 16)]);

    assert!(matches!(
        bind_to_op::<Cast<F16, F32>>(&m, 16),
        Err(TensorError::DtypeMismatch {
            port: 0,
            expected: Dtype::F16,
            got: Dtype::F32,
            ..
        })
    ));
}

#[test]
fn gather_indices_cannot_come_from_a_manifest() {
    // Indices are computed at run time, never loaded from a checkpoint:
    // the second port is U32 and the binder rejects the f32 entry.
    let m = manifest(vec![
        entry("vocab.table", &[10, 4], 0, 160),
        entry("not.indices", &[3], 160, 12),
    ]);

    assert!(matches!(
        bind_to_op::<Gather>(&m, 172),
        Err(TensorError::DtypeMismatch {
            port: 1,
            expected: Dtype::U32,
            ..
        })
    ));
    // The port is still declared as an index port.
    assert_eq!(<Gather as TensorOp>::INPUTS[1].role, PortRole::Indices);
}

#[test]
fn manifest_byte_len_must_match_shape_times_dtype() {
    // [4,8] f32 is 128 bytes, not 100.
    let m = manifest(vec![entry("bad.weight", &[4, 8], 0, 100)]);

    assert_eq!(
        validate_manifest(&m, 1024),
        Err(TensorError::ByteLenMismatch {
            name: "bad.weight".to_string(),
            shape: vec![4, 8],
            dtype: Dtype::F32,
            expected: 128,
            got: 100,
        })
    );
}

/// Review r1 P1: an F32 entry whose byte_offset is not 4-byte aligned
/// must be rejected — WGSL/Vulkan SSBO f32 loads require scalar
/// alignment, and manifest conformance must imply kernel-readiness.
/// (The Katakuri co-review asked for this invariant; the public fields
/// do not guarantee upstream loader validation, so it lives HERE.)
#[test]
fn manifest_f32_offsets_must_be_element_aligned() {
    for bad in [1u64, 2, 3, 5, 6, 7, 9] {
        let m = manifest(vec![entry("w", &[1], bad, 4)]);
        match validate_manifest(&m, bad + 4) {
            Err(TensorError::UnalignedOffset {
                name,
                offset,
                alignment,
            }) => {
                assert_eq!(name, "w");
                assert_eq!(offset, bad);
                assert_eq!(alignment, 4);
            }
            other => panic!("offset {bad} must be rejected as unaligned, got {other:?}"),
        }
    }
}

/// Aligned offsets of the same shapes pass validation and bind.
#[test]
fn manifest_aligned_f32_offsets_pass() {
    for good in [0u64, 4, 8, 12] {
        let m = manifest(vec![entry("w", &[2], good, 8)]);
        validate_manifest(&m, good + 8).expect("aligned offset must validate");
        bind_to_op::<CopyOp>(&m, good + 8).expect("aligned offset must bind");
    }
}

#[test]
fn manifest_spans_must_fit_the_data_section() {
    // 128..640 exceeds a 600-byte data section.
    let m = manifest(vec![
        entry("a.weight", &[4, 8], 0, 128),
        entry("b.weight", &[8, 16], 128, 512),
    ]);

    assert!(matches!(
        validate_manifest(&m, 600),
        Err(TensorError::OutOfBounds {
            offset: 128,
            end: 640,
            available: 600,
            ..
        })
    ));
    assert!(validate_manifest(&m, 640).is_ok());
}

#[test]
fn manifest_determinism_pin_is_honoured_by_every_t0_op() {
    // v1's only policy: f32 pinned, bit-pattern-changing fast paths
    // forbidden. Every T0 op must accept it — none selects a fast path.
    let m = manifest(vec![entry("x", &[8], 0, 32)]);
    assert_eq!(m.determinism.dtype_pinned, ManifestDtype::F32);
    assert_eq!(m.determinism.fast_paths, FastPathPolicy::Forbidden);

    assert!(bind_to_op::<Copy>(&m, 32).is_ok());
    assert!(bind_to_op::<Sum>(&m, 32).is_ok());

    let pair = manifest(vec![entry("a", &[8], 0, 32), entry("b", &[8], 32, 32)]);
    assert!(bind_to_op::<Add>(&pair, 64).is_ok());
}

#[test]
fn manifest_entries_must_not_overlap() {
    // Reviewer repro (PR #65 round 1): two entries claiming the same span
    // both passed validation, though the module documents that spans do not
    // overlap. Aliasing weights is expressed by naming one entry twice, not
    // by overlapping byte ranges.
    let aliased = manifest(vec![
        entry("a.weight", &[1], 0, 4),
        entry("b.weight", &[1], 0, 4),
    ]);
    assert!(matches!(
        validate_manifest(&aliased, 4),
        Err(TensorError::OverlappingEntries {
            ref first,
            ref second,
            offset: 0,
            end: 4,
        }) if first == "a.weight" && second == "b.weight"
    ));

    // Partial overlap, with the entries out of offset order in the list.
    let straddling = manifest(vec![
        entry("b.weight", &[2], 4, 8),
        entry("a.weight", &[2], 0, 8),
    ]);
    assert!(matches!(
        validate_manifest(&straddling, 12),
        Err(TensorError::OverlappingEntries {
            offset: 4,
            end: 8,
            ..
        })
    ));

    // Adjacent spans do not overlap, and a zero-length entry aliases nothing.
    let adjacent = manifest(vec![entry("a", &[1], 0, 4), entry("b", &[1], 4, 4)]);
    assert!(validate_manifest(&adjacent, 8).is_ok());
    let empty = manifest(vec![entry("empty", &[0], 0, 0), entry("x", &[1], 0, 4)]);
    assert!(validate_manifest(&empty, 4).is_ok());
}

#[test]
fn manifest_byte_footprints_that_overflow_u64_are_rejected() {
    // Reviewer repro (PR #65 round 1): [u32::MAX, u32::MAX] f32 needs
    // 73_786_976_260_478_468_100 bytes. A u64 byte_len cannot express that,
    // so validation must reject the shape rather than compare it against a
    // saturated footprint.
    let m = manifest(vec![ManifestEntry {
        name: "huge.weight".to_string(),
        shape: vec![u32::MAX, u32::MAX],
        byte_offset: 0,
        byte_len: u64::MAX,
    }]);

    assert!(matches!(
        validate_manifest(&m, u64::MAX),
        Err(TensorError::ByteLenOverflow {
            dtype: Dtype::F32,
            ..
        })
    ));

    // The binder rejects it too — through the same footprint check, once the
    // arity gate has passed (two entries for matmul's two ports).
    let paired = manifest(vec![
        ManifestEntry {
            name: "a.weight".to_string(),
            shape: vec![u32::MAX, u32::MAX],
            byte_offset: 0,
            byte_len: u64::MAX,
        },
        ManifestEntry {
            name: "b.weight".to_string(),
            shape: vec![u32::MAX, u32::MAX],
            byte_offset: 0,
            byte_len: u64::MAX,
        },
    ]);
    assert!(matches!(
        bind_to_op::<MatMul>(&paired, u64::MAX),
        Err(TensorError::ByteLenOverflow { .. })
    ));
}

#[test]
fn manifest_shapes_are_checked_against_the_op_shape_rule() {
    // Two [4,8] weights cannot be multiplied: the inner dims disagree.
    let m = manifest(vec![
        entry("a.weight", &[4, 8], 0, 128),
        entry("b.weight", &[4, 8], 128, 128),
    ]);

    assert!(matches!(
        bind_to_op::<MatMul>(&m, 256),
        Err(TensorError::InnerDimensionMismatch { .. })
    ));

    // Wrong rank for a matmul operand.
    let m = manifest(vec![
        entry("a.weight", &[4], 0, 16),
        entry("b.weight", &[4], 16, 16),
    ]);
    assert!(matches!(
        bind_to_op::<MatMul>(&m, 32),
        Err(TensorError::RankMismatch { .. })
    ));
}
