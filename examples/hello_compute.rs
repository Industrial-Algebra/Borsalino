// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Borsalino hello compute — simplest possible GPU kernel.
//!
//! Compiles an `add_one` WGSL kernel, dispatches it on 4 elements,
//! and prints the result.
//!
//! # Run
//!
//! ```sh
//! cargo run --example hello_compute --features vulkan   # Linux / Windows
//! cargo run --example hello_compute --features metal     # macOS
//! ```

use borsalino::GpuBackend;

fn main() -> Result<(), borsalino::GpuError> {
    let gpu = borsalino::init()?;

    let wgsl = r#"
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(256)
        fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i = gid.x;
            output[i] = input[i] + 1.0;
        }
    "#;

    let pipeline = gpu.compile("add_one", wgsl)?;
    let input = gpu.create_buffer(&[1.0f32, 2.0, 3.0, 4.0])?;
    let output = gpu.create_buffer_uninit::<f32>(4)?;

    gpu.dispatch(&pipeline, &[&input, &output], (1, 1, 1))?;

    let result: Vec<f32> = gpu.read_buffer(&output)?;
    println!("{result:?}");
    assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
    println!("✅ add_one kernel: all correct");

    Ok(())
}
