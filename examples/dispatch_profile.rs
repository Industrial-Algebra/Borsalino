// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Micro-benchmark to profile dispatch overhead components on Metal.
//!
//! # Run
//!
//! ```sh
//! cargo run --example dispatch_profile --features metal --release
//! ```

use borsalino::GpuBackend;
use std::time::Instant;

const KERNEL_NOOP: &str = r#"
    @group(0) @binding(0) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(1)
    fn noop(@builtin(global_invocation_id) gid: vec3<u32>) {
        out[gid.x] = 1.0;
    }
"#;

fn time_it<F: FnMut()>(label: &str, iters: u32, mut f: F) {
    // Warm-up
    for _ in 0..3 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per = elapsed.as_secs_f64() / iters as f64;

    println!(
        "  {:<45} {:>8.1} µs/op  ({iters} iters, {:.1} ms total)",
        label,
        per * 1e6,
        elapsed.as_secs_f64() * 1e3
    );
}

fn main() -> Result<(), borsalino::GpuError> {
    let gpu = borsalino::init()?;
    println!("Dispatch Overhead Profile (M3 Pro)\n");

    // ── Phase 1: Pipeline compilation (one-time cost) ──────────
    println!("--- Pipeline Compilation ---");
    let t0 = Instant::now();
    let pipeline = gpu.compile("noop", KERNEL_NOOP)?;
    println!("  compile: {:>8.1} µs", t0.elapsed().as_secs_f64() * 1e6);

    // ── Phase 2: Buffer creation ──────────────────────────────
    println!("\n--- Buffer Creation ---");
    time_it("create_buffer 256 f32", 100, || {
        let _ = gpu.create_buffer(&vec![1.0f32; 256]).unwrap();
    });
    time_it("create_buffer_uninit 256 f32", 100, || {
        let _ = gpu.create_buffer_uninit::<f32>(256).unwrap();
    });

    let out = gpu.create_buffer_uninit::<f32>(256)?;

    // ── Phase 3: Full dispatch (the baseline) ─────────────────
    println!("\n--- Full Dispatch (baseline) ---");
    time_it("full dispatch (1×1 thread)", 200, || {
        gpu.dispatch_ex(&pipeline, &[&out], (1, 1, 1), (1, 1, 1))
            .unwrap();
    });
    time_it("full dispatch (1×256 threads)", 200, || {
        gpu.dispatch(&pipeline, &[&out], (1, 1, 1)).unwrap();
    });

    // ── Phase 4: Batch dispatch vs individual ───────────────
    println!("\n--- Batch Dispatch (dispatch_many) ---");

    // Warm up
    for _ in 0..3 {
        gpu.dispatch(&pipeline, &[&out], (1, 1, 1)).unwrap();
    }

    let rounds: &[(u32, u32)] = &[(1, 200), (4, 100), (16, 50), (64, 20), (256, 10)];

    println!("  Individual dispatch (separate command buffers):");
    for &(ndispatches, niters) in rounds {
        let start = Instant::now();
        for _ in 0..niters {
            for _ in 0..ndispatches {
                gpu.dispatch(&pipeline, &[&out], (1, 1, 1)).unwrap();
            }
        }
        let total = start.elapsed();
        let per = total.as_secs_f64() / (ndispatches * niters) as f64;
        let total_dispatches = ndispatches * niters;
        println!(
            "    {ndispatches:>3}× dispatch ({total_dispatches:>4} total): {:>8.1} µs/dispatch",
            per * 1e6,
        );
    }

    println!("  Batched dispatch (single command buffer):");
    let buffer_slice: &[&borsalino::GpuBuffer] = &[&out];
    for &(ndispatches, niters) in rounds {
        let specs: Vec<_> = (0..ndispatches as usize)
            .map(|_| borsalino::DispatchSpec {
                pipeline: &pipeline,
                buffers: buffer_slice,
                workgroups: (1, 1, 1),
                threads_per_group: (256, 1, 1),
            })
            .collect();

        let start = Instant::now();
        for _ in 0..niters {
            gpu.dispatch_many(&specs).unwrap();
        }
        let total = start.elapsed();
        let per = total.as_secs_f64() / (ndispatches * niters) as f64;
        let total_dispatches = ndispatches * niters;
        println!(
            "    {ndispatches:>3}× batched ({total_dispatches:>4} total): {:>8.1} µs/dispatch",
            per * 1e6,
        );
    }

    // ── Phase 5: Thread scaling ───────────────────────────────
    println!("\n--- Thread Scaling (1 workgroup, varying threads) ---");
    for threads in [1u32, 32, 64, 128, 256, 512, 1024] {
        let big_out = gpu.create_buffer_uninit::<f32>(threads as usize)?;
        time_it(&format!("1 wg × {threads:>4} threads"), 100, || {
            gpu.dispatch_ex(&pipeline, &[&big_out], (1, 1, 1), (threads, 1, 1))
                .unwrap();
        });
    }

    // ── Phase 6: Workgroup scaling (fixed total threads) ──────
    println!("\n--- Workgroup Scaling (1M threads total, varying wg size) ---");
    let big_out = gpu.create_buffer_uninit::<f32>(1_048_576)?;
    for wg_size in [64u32, 128, 256, 512, 1024] {
        let wg_count = 1_048_576 / wg_size;
        time_it(
            &format!("{wg_count:>5} wg × {wg_size:>4} threads"),
            20,
            || {
                gpu.dispatch_ex(&pipeline, &[&big_out], (wg_count, 1, 1), (wg_size, 1, 1))
                    .unwrap();
            },
        );
    }

    // ── Phase 7: Buffer count scaling ─────────────────────────
    println!("\n--- Buffer Count Scaling (1 wg × 256 threads) ---");
    let buffers_2 = [
        gpu.create_buffer_uninit::<f32>(256)?,
        gpu.create_buffer_uninit::<f32>(256)?,
    ];
    let buffers_4 = [
        gpu.create_buffer_uninit::<f32>(256)?,
        gpu.create_buffer_uninit::<f32>(256)?,
        gpu.create_buffer_uninit::<f32>(256)?,
        gpu.create_buffer_uninit::<f32>(256)?,
    ];
    let _buffers_8 = (0..8)
        .map(|_| gpu.create_buffer_uninit::<f32>(256).unwrap())
        .collect::<Vec<_>>();

    time_it("1 buffer", 200, || {
        gpu.dispatch(&pipeline, &[&out], (1, 1, 1)).unwrap();
    });
    time_it("2 buffers", 200, || {
        gpu.dispatch(&pipeline, &[&buffers_2[0], &buffers_2[1]], (1, 1, 1))
            .unwrap();
    });
    time_it("4 buffers", 200, || {
        gpu.dispatch(
            &pipeline,
            &[&buffers_4[0], &buffers_4[1], &buffers_4[2], &buffers_4[3]],
            (1, 1, 1),
        )
        .unwrap();
    });

    // ── Phase 8: Readback timing ──────────────────────────────
    println!("\n--- Readback Latency ---");
    for size in [256usize, 1024, 4096, 16384, 65536, 262144, 1_048_576] {
        let buf = gpu.create_buffer_uninit::<f32>(size)?;
        gpu.dispatch(&pipeline, &[&buf], (1, 1, 1)).unwrap(); // write something
        time_it(&format!("read_buffer {size:>8} f32"), 100, || {
            let _: Vec<f32> = gpu.read_buffer(&buf).unwrap();
        });
    }

    println!("\nDone.");
    Ok(())
}
