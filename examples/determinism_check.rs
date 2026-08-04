// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Determinism verification example — dispatches the same kernel
//! multiple times and checks for bit-identical output.
//!
//! Run with:
//! ```sh
//! cargo run --features vulkan,verify --example determinism_check
//! ```

use borsalino::determinism::{DeterminismResult, verify_deterministic};
use borsalino::{GpuBackend, init};

fn main() {
    let gpu = match init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU backend available: {e}");
            return;
        }
    };

    // Use the add_one kernel for determinism testing
    let wgsl = r#"
        @group(0) @binding(0) var<storage, read> input: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i = gid.x;
            if (i < arrayLength(&input)) {
                output[i] = input[i] + 1u;
            }
        }
    "#;

    let pipeline = gpu.compile("main", wgsl).expect("compile kernel");

    let input_data: Vec<u8> = vec![1u8; 256];
    let output_len = 256;
    let trials = 5;

    println!("Running determinism check ({trials} trials)...");
    let result: DeterminismResult =
        verify_deterministic(&gpu, &pipeline, &[input_data], output_len, trials)
            .expect("determinism dispatch");

    println!("{result:#?}");

    if result.is_deterministic() {
        println!("✅ PASS: kernel is deterministic ({} identical runs)", result.trials);
    } else {
        println!(
            "❌ FAIL: kernel is nondeterministic ({}% byte disagreement)",
            (result.disagreement_fraction * 100.0) as u32
        );
    }
}
