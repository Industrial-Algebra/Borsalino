// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Borsalino SAXPY example — fused multiply-add on the GPU.
//!
//! Computes `out[i] = a * x[i] + y[i]` for 1024 elements,
//! using 4 workgroups of 256 threads each.
//!
//! # Run
//!
//! ```sh
//! cargo run --example saxpy --features vulkan   # Linux / Windows
//! cargo run --example saxpy --features metal     # macOS
//! ```

use borsalino::GpuBackend;

fn main() -> Result<(), borsalino::GpuError> {
    let gpu = borsalino::init()?;

    let wgsl = r#"
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read> y: array<f32>;
        @group(0) @binding(2) var<storage, read_write> out: array<f32>;

        @compute @workgroup_size(256)
        fn saxpy(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i = gid.x;
            out[i] = 2.5 * x[i] + y[i];
        }
    "#;

    const N: usize = 1024;
    let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.125).collect();
    let y: Vec<f32> = (0..N).map(|i| 1.0 - i as f32 * 0.0625).collect();
    let expected: Vec<f32> = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| 2.5 * xi + yi)
        .collect();

    let pipeline = gpu.compile("saxpy", wgsl)?;
    let buf_x = gpu.create_buffer(&x)?;
    let buf_y = gpu.create_buffer(&y)?;
    let buf_out = gpu.create_buffer_uninit::<f32>(N)?;

    gpu.dispatch(&pipeline, &[&buf_x, &buf_y, &buf_out], (4, 1, 1))?;

    let result: Vec<f32> = gpu.read_buffer(&buf_out)?;

    let mut mismatches = 0usize;
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        if (r - e).abs() > 1e-6 {
            eprintln!("mismatch at {i}: got {r:.6}, expected {e:.6}");
            mismatches += 1;
        }
    }

    if mismatches == 0 {
        println!("SAXPY: {N} elements, all correct ✅");
    } else {
        eprintln!("SAXPY: {mismatches} mismatches ❌");
    }

    Ok(())
}
