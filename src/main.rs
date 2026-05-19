// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Borsalino — GPU compute smoke test.
//!
//! Run with:
//! ```sh
//! cargo run --features metal    # macOS
//! cargo run --features vulkan   # Linux/Windows
//! ```

use borsalino::GpuBackend;

fn main() -> Result<(), borsalino::GpuError> {
    let gpu = borsalino::init()?;

    let wgsl = r#"
        @compute @workgroup_size(256)
        fn saxpy(@builtin(global_invocation_id) gid: vec3<u32>,
                 @storage(0) x: array<f32>,
                 @storage(1) y: array<f32>,
                 @storage(2) out: array<f32>) {
            let i = gid.x;
            out[i] = 2.5 * x[i] + y[i];
        }
    "#;

    let n = 1024usize;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.125).collect();
    let y: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 * 0.0625).collect();
    let expected: Vec<f32> = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| 2.5 * xi + yi)
        .collect();

    let pipeline = gpu.compile("saxpy", wgsl)?;
    let buf_x = gpu.create_buffer(&x)?;
    let buf_y = gpu.create_buffer(&y)?;
    let buf_out = gpu.create_buffer_uninit::<f32>(n)?;

    gpu.dispatch(&pipeline, &[&buf_x, &buf_y, &buf_out], (4, 1, 1))?;

    let result: Vec<f32> = gpu.read_buffer(&buf_out)?;

    let mut ok = true;
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        if (r - e).abs() > 1e-6 {
            eprintln!("mismatch at {i}: got {r}, expected {e}");
            ok = false;
        }
    }

    if ok {
        println!("SAXPY: {n} elements, all correct ✅");
    } else {
        eprintln!("SAXPY: FAILED ❌");
    }

    Ok(())
}
