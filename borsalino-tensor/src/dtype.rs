// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Dtype policy: **f32 is the contract**; f16/bf16 are explicit casts.
//!
//! Handover §4.3 draws the line this module enforces:
//!
//! - f32 is the type every kernel computes in. Weights arrive as f32 from
//!   Katakuri, buffers are bound as f32, and no backend may substitute a
//!   narrower or faster representation (no TF32, no f16 accumulation).
//! - f16 and bf16 exist **only** as the target or source of an explicit
//!   [`Cast`](crate::ops::l0::Cast), with the roundtrip semantics documented
//!   in [`CastContract`]. Nothing narrows silently.
//! - `u32` is the index type ([`Gather`](crate::ops::l0::Gather)'s second
//!   port). It is not a compute dtype and not a cast target.
//! - f64 is not here at all: f64 exactness is Amari's layer, not
//!   Borsalino's (handover §3, NOT-scope).

use core::fmt;

use crate::error::TensorError;

/// An element type the tensor op-set can name.
///
/// The inventory is closed by design: [`Dtype::ALL`] is the contract's
/// complete dtype surface, so a new element type is a deliberate contract
/// change rather than an accident of a dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// 32-bit IEEE-754 float — the one compute dtype (handover §4.3).
    F32,
    /// IEEE-754 binary16 — a cast source or target only.
    F16,
    /// bfloat16 — f32's exponent range, 8-bit significand. Cast only.
    BF16,
    /// Unsigned 32-bit integer — index ports only, never computed on.
    U32,
}

impl Dtype {
    /// Every dtype the contract knows, in canonical order.
    pub const ALL: &'static [Dtype] = &[Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::U32];

    /// The dtype's lowercase name, as it appears in WGSL and manifests.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::BF16 => "bf16",
            Dtype::U32 => "u32",
        }
    }

    /// The dtype's size in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> u64 {
        match self {
            Dtype::F32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }

    /// Whether this dtype is the compute contract dtype.
    ///
    /// Only [`Dtype::F32`] is: every arithmetic op's data ports declare it,
    /// and any other float must be cast to f32 first.
    #[must_use]
    pub const fn is_contract(self) -> bool {
        matches!(self, Dtype::F32)
    }

    /// Whether this dtype may appear on an index port.
    #[must_use]
    pub const fn is_index(self) -> bool {
        matches!(self, Dtype::U32)
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Reject a dtype that is not the f32 contract dtype.
///
/// v1 manifests are f32-only, so this guard cannot fire today; it exists so
/// that the load boundary has one place to enforce
/// [`Dtype::is_contract`] when a manifest type grows a second dtype.
///
/// # Errors
///
/// [`TensorError::NonContractDtype`] when `dtype` is not f32.
pub fn require_contract(dtype: Dtype) -> Result<(), TensorError> {
    if dtype.is_contract() {
        Ok(())
    } else {
        Err(TensorError::NonContractDtype { got: dtype })
    }
}

/// A dtype that can be named as a compile-time operand marker.
///
/// Markers let an op carry its dtype in its type: `Cast<F32, F16>` *is* a
/// narrowing cast, checked by the type system before any shape is seen.
/// Only the three floating types have markers — indices are never cast.
pub trait DtypeMarker: Copy + fmt::Debug + 'static {
    /// The runtime dtype this marker denotes.
    const DTYPE: Dtype;
}

/// The f32 marker — the contract dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32;

impl DtypeMarker for F32 {
    const DTYPE: Dtype = Dtype::F32;
}

/// The f16 marker — cast source or target only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F16;

impl DtypeMarker for F16 {
    const DTYPE: Dtype = Dtype::F16;
}

/// The bf16 marker — cast source or target only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BF16;

impl DtypeMarker for BF16 {
    const DTYPE: Dtype = Dtype::BF16;
}

/// What a cast does to the bit pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Roundtrip {
    /// The cast is lossless: every source value is representable.
    Exact,
    /// The cast rounds to nearest-even, so a downcast roundtrip is inexact
    /// by up to half an ulp.
    HalfUlpInexact,
}

/// What a cast does to values beyond the target's range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overflow {
    /// Values beyond the target's maximum round to infinity.
    ToInfinity,
    /// The target's range contains the source's, so overflow cannot happen.
    CannotOverflow,
}

/// What a cast does to subnormal source values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubnormalPolicy {
    /// The target represents every source subnormal.
    Preserved,
    /// Source subnormals below the target's minimum subnormal become zero.
    FlushesToZero,
}

/// The documented semantics of one cast pair.
///
/// This is the whole of T0's cast contract: dtype policy is data, so a
/// kernel author reads the semantics from the type instead of guessing at
/// the hardware's rounding mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastContract {
    /// The source dtype.
    pub from: Dtype,
    /// The target dtype.
    pub to: Dtype,
    /// Whether the value survives a roundtrip.
    pub roundtrip: Roundtrip,
    /// What happens beyond the target's range.
    pub overflow: Overflow,
    /// What happens to subnormal source values.
    pub subnormals: SubnormalPolicy,
}

impl CastContract {
    /// The contract for a cast pair.
    ///
    /// Defined pairs (handover §4.3):
    ///
    /// - `f32 -> f16`: round-to-nearest-even, half-ulp inexact; values above
    ///   f16's maximum become infinity; f32 subnormals below f16's minimum
    ///   subnormal become zero.
    /// - `f32 -> bf16`: round-to-nearest-even, half-ulp inexact; bf16 shares
    ///   f32's exponent range so it cannot overflow; f32 subnormals below
    ///   bf16's minimum subnormal become zero.
    /// - `f16 -> f32`, `bf16 -> f32`: exact, no overflow, subnormals
    ///   preserved.
    ///
    /// # Errors
    ///
    /// [`TensorError::CastNotDefined`] for any other pair — including the
    /// identity cast (that is `copy`) and `f16 <-> bf16` (two explicit steps
    /// through f32), and for index dtypes.
    pub fn for_dtypes(from: Dtype, to: Dtype) -> Result<Self, TensorError> {
        use Dtype::{BF16 as B, F16 as H, F32 as S};
        let (roundtrip, overflow, subnormals) = match (from, to) {
            (S, H) => (
                Roundtrip::HalfUlpInexact,
                Overflow::ToInfinity,
                SubnormalPolicy::FlushesToZero,
            ),
            (S, B) => (
                Roundtrip::HalfUlpInexact,
                Overflow::CannotOverflow,
                SubnormalPolicy::FlushesToZero,
            ),
            (H, S) | (B, S) => (
                Roundtrip::Exact,
                Overflow::CannotOverflow,
                SubnormalPolicy::Preserved,
            ),
            _ => return Err(TensorError::CastNotDefined { from, to }),
        };
        Ok(Self {
            from,
            to,
            roundtrip,
            overflow,
            subnormals,
        })
    }
}
