// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! The L0–L2 inventory, grouped by handover §3 level.
//!
//! One module per level, one type per op. Level 3 (activation and
//! composition ops, including the fused 4-D attention mask) is a later
//! slice: it is composed from these, not declared alongside them.

pub mod l0;
pub mod l1;
pub mod l2;
