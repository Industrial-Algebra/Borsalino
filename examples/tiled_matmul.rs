// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Borsalino 2D tiled matrix multiplication example.
//!
//! Demonstrates 2D workgroup dispatch with shared memory (workgroup tile)
//! and workgroup barriers. Compares GPU tiled matmul against CPU na\
//! ive matmul
//! for correctness and measures performance.
//!
//! # Architecture
//!
//! C = A × B  where A is M×K, B is K×N, C is M×N
//!
//! Each workgroup computes a 16×16 tile of C. Threads cooperatively load
//! tiles of A and B into shared memory, synchronise with barriers, compute
//! a partial dot product, synchronise again, and advance to the next
//! K-dimension tile.
//!
//! # Run
//!
//! ```sh
//! cargo run --example tiled_matmul --features vulkan --release
//! cargo run --example tiled_matmul --features metal --release
//! ```

use std::time::Instant;

use borsalino::GpuBackend;

const TILE_SIZE: u32 = 16;
const MATRIX_SIZE: u32 = 1024;

/// Tiled 2D matrix multiply kernel using shared memory.
const KERNEL_MATMUL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

const TILE: u32 = 16u;
const N: u32 = 1024u;
const K: u32 = 1024u;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn matmul(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = wg_id.y * TILE + local_id.y;
    let col = wg_id.x * TILE + local_id.x;
    let local_idx = local_id.y * TILE + local_id.x;

    var sum = 0.0;

    for (var k_tile = 0u; k_tile < K; k_tile += TILE) {
        // Cooperative load: each thread loads one element from A and B
        let a_idx = row * K + k_tile + local_id.x;
        let b_idx = (k_tile + local_id.y) * N + col;
        tile_a[local_idx] = a[a_idx];
        tile_b[local_idx] = b[b_idx];

        workgroupBarrier();

        // Compute partial dot product over shared tile
        for (var i = 0u; i < TILE; i++) {
            let a_val = tile_a[local_id.y * TILE + i];
            let b_val = tile_b[i * TILE + local_id.x];
            sum += a_val * b_val;
        }

        workgroupBarrier();
    }

    c[row * N + col] = sum;
}
"#;

fn main() -> Result<(), borsalino::GpuError> {
    let gpu = borsalino::init()?;
    let n = MATRIX_SIZE as usize;
    let total = n * n;

    println!("Borsalino 2D Tiled Matrix Multiply");
    println!("  matrix: {n}×{n}  tile: {}×{}", TILE_SIZE, TILE_SIZE);
    println!("  workgroups: {0}×{0}  threads/wg: {1}×{1} ({2})",
        n as u32 / TILE_SIZE, TILE_SIZE, TILE_SIZE * TILE_SIZE);
    println!();

    // Generate matrices
    let a: Vec<f32> = (0..total).map(|i| (i % 997) as f32 * 0.001).collect();
    let b: Vec<f32> = (0..total).map(|i| ((i * 3 + 1) % 997) as f32 * 0.001).collect();

    // CPU reference
    println!("--- CPU Reference ---");
    let cpu_start = Instant::now();
    let expected = cpu_matmul(&a, &b, n);
    let cpu_time = cpu_start.elapsed();
    println!("  cpu naive: {:>8.1} ms", cpu_time.as_secs_f64() * 1e3);

    // GPU compile
    println!("\n--- GPU Compile + Dispatch ---");
    let compile_start = Instant::now();
    let pipeline = gpu.compile("matmul", KERNEL_MATMUL)?;
    let compile_time = compile_start.elapsed();
    println!("  compile:   {:>8.1} ms", compile_time.as_secs_f64() * 1e3);

    // Upload
    let buf_a = gpu.create_buffer(&a)?;
    let buf_b = gpu.create_buffer(&b)?;
    let buf_c = gpu.create_buffer_uninit::<f32>(total)?;

    // Dispatch: 2D workgroup grid
    let wgs = n as u32 / TILE_SIZE;
    let dispatch_start = Instant::now();
    gpu.dispatch_ex(
        &pipeline,
        &[&buf_a, &buf_b, &buf_c],
        (wgs, wgs, 1),          // 2D workgroup grid
        (TILE_SIZE, TILE_SIZE, 1), // 2D thread layout
    )?;
    let dispatch_time = dispatch_start.elapsed();
    let gpu_time_us = dispatch_time.as_micros();
    println!("  dispatch:  {:>8} µs", gpu_time_us);

    // Read result
    let result: Vec<f32> = gpu.read_buffer(&buf_c)?;
    let total_time = compile_start.elapsed();
    println!("  total:     {:>8.1} ms", total_time.as_secs_f64() * 1e3);

    // Verify
    println!("\n--- Verification ---");
    let mut max_err: f32 = 0.0;
    let mut mismatches = 0usize;
    for i in 0..total {
        let err = (result[i] - expected[i]).abs();
        if err > 1e-3 {
            mismatches += 1;
            if err > max_err { max_err = err; }
        }
    }

    if mismatches == 0 {
        println!("  ✅ all {total} elements correct (max err < 0.001)");
    } else {
        println!("  ⚠ {mismatches} mismatches (max err: {max_err:.6})");
    }

    // Throughput
    let flops = 2.0 * (n as f64).powi(3);
    let gflops = flops / dispatch_time.as_secs_f64() / 1e9;
    println!("\n--- Throughput ---");
    println!("  {:.2} GFLOPS (GPU compute only)", gflops);

    Ok(())
}

/// Naive CPU matrix multiply (row-major, triple loop).
fn cpu_matmul(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}
