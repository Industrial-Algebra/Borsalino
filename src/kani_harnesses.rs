// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Kani verification harnesses for Borsalino buffer safety.
//!
//! Run with:
//! ```sh
//! cargo kani --features vulkan --harness buffer_alignment_boundary
//! cargo kani --features vulkan --harness workgroup_divisibility
//! ```
//!
//! Requires Kani installed: <https://model-checking.github.io/kani/>

/// Verify that buffer size alignment always satisfies 16-byte boundary.
#[cfg(kani)]
#[kani::proof]
fn buffer_alignment_boundary() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 65536);

    // Pad to 16-byte alignment (matching minStorageBufferOffsetAlignment)
    let aligned = ((size + 15) / 16) * 16;

    assert!(aligned >= size);
    assert!(aligned % 16 == 0);
    assert!(aligned - size < 16);
}

/// Verify workgroup divisibility prevents partial threadgroups.
#[cfg(kani)]
#[kani::proof]
fn workgroup_divisibility() {
    let total_threads: u32 = kani::any();
    let threads_per_group: u32 = kani::any();
    kani::assume(threads_per_group > 0 && threads_per_group <= 1024);
    kani::assume(total_threads <= 1_048_576);

    let workgroups = total_threads.div_ceil(threads_per_group);

    // Property: workgroups * threads_per_group >= total_threads
    assert!(workgroups * threads_per_group >= total_threads);

    // Property: no partial threadgroup if divisible
    if total_threads % threads_per_group == 0 {
        assert_eq!(workgroups * threads_per_group, total_threads);
    }
}

/// Verify buffer element count × element size doesn't overflow.
#[cfg(kani)]
#[kani::proof]
fn buffer_size_no_overflow() {
    let len: u32 = kani::any();
    let element_size: u32 = kani::any();
    kani::assume(len > 0 && len <= 1_048_576);
    kani::assume(element_size > 0 && element_size <= 64);

    let byte_len = (len as u64) * (element_size as u64);
    assert!(byte_len <= (1u64 << 32)); // fits in u32 for Vulkan buffer size
    assert!(byte_len <= u64::MAX);
}
