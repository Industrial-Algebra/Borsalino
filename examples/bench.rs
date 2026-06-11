// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Borsalino GPU backend benchmarks.
//!
//! # Run
//!
//! ```sh
//! cargo run --example bench --features metal --release    # macOS
//! cargo run --example bench --features vulkan --release   # Linux / Windows
//! ```
//!
//! Measures: pipeline compilation, dispatch latency, throughput scaling,
//! and buffer I/O overhead on the active GPU backend.

use std::time::Instant;

use borsalino::GpuBackend;

// ── Kernel sources ────────────────────────────────────────────────

const KERNEL_NOOP: &str = r#"
    @group(0) @binding(0) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(1)
    fn noop(@builtin(global_invocation_id) gid: vec3<u32>) {
        out[gid.x] = 1.0;
    }
"#;

const KERNEL_VADD: &str = r#"
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(256)
    fn vadd(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        out[i] = a[i] + b[i];
    }
"#;

const KERNEL_SAXPY: &str = r#"
    @group(0) @binding(0) var<storage, read> x: array<f32>;
    @group(0) @binding(1) var<storage, read> y: array<f32>;
    @group(0) @binding(2) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(256)
    fn saxpy(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        out[i] = 2.5 * x[i] + y[i];
    }
"#;

// ── Helpers ───────────────────────────────────────────────────────

fn workgroups_for(n: u32, threads_per_group: u32) -> u32 {
    n.div_ceil(threads_per_group)
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn stddev(data: &[f64], mean_val: f64) -> f64 {
    let variance = data.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

struct BenchResult {
    name: String,
    value: f64,
    unit: String,
    iters: usize,
    stddev: f64,
}

fn run_bench<T>(name: &str, unit: &str, iters: usize, mut f: impl FnMut() -> T) -> BenchResult {
    // Warm-up
    for _ in 0..3 {
        f();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let _ = f();
        times.push(start.elapsed().as_secs_f64());
    }

    let avg = mean(&times);
    let sd = stddev(&times, avg);

    BenchResult {
        name: name.to_string(),
        value: avg,
        unit: unit.to_string(),
        iters,
        stddev: sd,
    }
}

fn print_results(results: &[BenchResult]) {
    println!();
    println!(
        "{:<48} {:>10} {:>8} {:>8} {:>8}",
        "Benchmark", "Time", "Unit", "Iters", "±%"
    );
    println!("{}", "-".repeat(85));

    for r in results {
        println!(
            "{:<48} {:>10} {:>8} {:>8} {:>7.1}%",
            r.name,
            format_time(r),
            r.unit,
            r.iters,
            if r.value > 0.0 {
                (r.stddev / r.value) * 100.0
            } else {
                0.0
            }
        );
    }
    println!();
}

fn scale_unit(n: u32) -> (&'static str, f64) {
    if n >= 1_048_576 {
        ("ms", 1e3)
    } else {
        ("µs", 1e6)
    }
}

fn format_time(result: &BenchResult) -> String {
    let scale = match result.unit.as_str() {
        "ms" => 1e3,
        "µs" => 1e6,
        _ => 1.0,
    };
    format!("{:.3} {}", result.value * scale, result.unit)
}

// ── Main ──────────────────────────────────────────────────────────

fn main() -> Result<(), borsalino::GpuError> {
    let gpu = borsalino::init()?;
    println!("Borsalino GPU Benchmarks");
    println!("==========================");
    println!();

    let mut results: Vec<BenchResult> = Vec::new();

    // ── Pipeline compilation ──────────────────────────────────

    println!("--- Pipeline Compilation ---");

    let bench = run_bench("compile (noop kernel)", "ms", 10, || {
        gpu.compile("noop", KERNEL_NOOP).unwrap()
    });
    println!(
        "  compile: {:>8.3} ms ±{:.1}%",
        bench.value * 1e3,
        (bench.stddev / bench.value) * 100.0
    );
    results.push(bench);

    // ── Dispatch latency ──────────────────────────────────────

    println!("--- Dispatch Latency ---");

    let pipeline_noop = gpu.compile("noop", KERNEL_NOOP)?;
    let out_buf = gpu.create_buffer_uninit::<f32>(256)?;

    // GPU timestamp dispersion
    println!("--- GPU Timestamp Accuracy ---");
    let t0 = gpu.timestamp()?;
    let t1 = gpu.timestamp()?;
    let ns_per_tick = t1 as f64 - t0 as f64;
    println!("  timestamp resolution: ~{:.1} ns", ns_per_tick.max(1.0));

    // Measure GPU-side dispatch time vs CPU wall time
    let bench = run_bench("GPU timestamp dispatch (1 wg × 256)", "µs", 200, || {
        let t0 = gpu.timestamp().unwrap();
        gpu.dispatch(&pipeline_noop, &[&out_buf], (1, 1, 1))
            .unwrap();
        let t1 = gpu.timestamp().unwrap();
        let elapsed = t1.saturating_sub(t0);
        // Return elapsed in seconds for the bench framework
        elapsed
    });
    let gpu_ns = bench.value;
    println!(
        "  GPU timestamp:  {:>8.1} ns ±{:.1}%",
        gpu_ns,
        (bench.stddev / bench.value.max(0.001)) * 100.0
    );
    results.push(BenchResult {
        name: "GPU timestamp (1 wg dispatch)".into(),
        value: gpu_ns / 1e3,
        unit: "µs".into(),
        iters: bench.iters,
        stddev: bench.stddev / 1e3,
    });

    // Single dispatch overhead (1 workgroup × 1 thread)
    let bench = run_bench("dispatch (1 workgroup × 1 thread)", "µs", 200, || {
        gpu.dispatch_ex(&pipeline_noop, &[&out_buf], (1, 1, 1), (1, 1, 1))
            .unwrap();
    });
    println!(
        "  dispatch 1×1:    {:>8.1} µs ±{:.1}%",
        bench.value * 1e6,
        (bench.stddev / bench.value) * 100.0
    );
    results.push(bench);

    // Single workgroup × 256 threads
    let bench = run_bench("dispatch (1 workgroup × 256 threads)", "µs", 200, || {
        gpu.dispatch(&pipeline_noop, &[&out_buf], (1, 1, 1))
            .unwrap();
    });
    println!(
        "  dispatch 1×256:  {:>8.1} µs ±{:.1}%",
        bench.value * 1e6,
        (bench.stddev / bench.value) * 100.0
    );
    results.push(bench);

    // ── Throughput scaling (vadd) ─────────────────────────────

    println!("--- Throughput Scaling (vadd) ---");

    let pipeline_vadd = gpu.compile("vadd", KERNEL_VADD)?;
    let sizes: &[u32] = &[1024, 16_384, 262_144, 1_048_576, 16_777_216];

    for &n in sizes {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let buf_a = gpu.create_buffer(&a)?;
        let buf_b = gpu.create_buffer(&b)?;
        let buf_out = gpu.create_buffer_uninit::<f32>(n as usize)?;
        let wgs = workgroups_for(n, 256);
        let iters = if n <= 262_144 { 50 } else { 10 };
        let (unit, scale) = scale_unit(n);

        let bench = run_bench(&format!("vadd {n:>9} el ({wgs} wgs)"), unit, iters, || {
            gpu.dispatch(&pipeline_vadd, &[&buf_a, &buf_b, &buf_out], (wgs, 1, 1))
                .unwrap();
        });

        let elem_sec = n as f64 / bench.value;
        let gflops = (n as f64 * 2.0) / bench.value / 1e9;
        println!(
            "  {n:>9} el  {:>8.3} {unit}  {:>8.2} Gelem/s  {:>8.2} GFLOPS  ±{:.1}%",
            bench.value * scale,
            elem_sec / 1e9,
            gflops,
            (bench.stddev / bench.value) * 100.0
        );
        results.push(bench);
    }

    // ── Buffer I/O ────────────────────────────────────────────

    println!("--- Buffer I/O ---");

    let buf_sizes: &[usize] = &[1024, 16_384, 262_144, 1_048_576];

    for &n in buf_sizes {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let bench = run_bench(&format!("create_buffer {n} f32"), "µs", 50, || {
            gpu.create_buffer(&data).unwrap()
        });
        let gb_s = (n as f64 * 4.0) / bench.value / 1e9;
        println!(
            "  create {:>8}: {:>8.1} µs  ({:.1} GB/s)  ±{:.1}%",
            n,
            bench.value * 1e6,
            gb_s,
            (bench.stddev / bench.value) * 100.0
        );
        results.push(bench);

        let buf = gpu.create_buffer(&data)?;
        let bench = run_bench(&format!("read_buffer {n} f32"), "µs", 50, || {
            let _: Vec<f32> = gpu.read_buffer(&buf).unwrap();
        });
        let gb_s = (n as f64 * 4.0) / bench.value / 1e9;
        println!(
            "  read   {:>8}: {:>8.1} µs  ({:.1} GB/s)  ±{:.1}%",
            n,
            bench.value * 1e6,
            gb_s,
            (bench.stddev / bench.value) * 100.0
        );
        results.push(bench);
    }

    // ── SAXPY throughput ──────────────────────────────────────

    println!("--- SAXPY Throughput ---");

    let pipeline_saxpy = gpu.compile("saxpy", KERNEL_SAXPY)?;

    for &n in sizes {
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let buf_x = gpu.create_buffer(&x)?;
        let buf_y = gpu.create_buffer(&y)?;
        let buf_out = gpu.create_buffer_uninit::<f32>(n as usize)?;
        let wgs = workgroups_for(n, 256);
        let iters = if n <= 262_144 { 50 } else { 10 };
        let (unit, scale) = scale_unit(n);

        let bench = run_bench(&format!("saxpy {n:>9} el ({wgs} wgs)"), unit, iters, || {
            gpu.dispatch(&pipeline_saxpy, &[&buf_x, &buf_y, &buf_out], (wgs, 1, 1))
                .unwrap();
        });

        let elem_sec = n as f64 / bench.value;
        let gflops = (n as f64 * 3.0) / bench.value / 1e9;
        println!(
            "  {n:>9} el  {:>8.3} {unit}  {:>8.2} Gelem/s  {:>8.2} GFLOPS  ±{:.1}%",
            bench.value * scale,
            elem_sec / 1e9,
            gflops,
            (bench.stddev / bench.value) * 100.0
        );
        results.push(bench);
    }

    // ── Batched SAXPY (amortise dispatch overhead) ────────────

    println!("--- Batched SAXPY (256 dispatches per command buffer) ---");

    let batch_size: u32 = 256;

    for &n in sizes {
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let buf_x = gpu.create_buffer(&x)?;
        let buf_y = gpu.create_buffer(&y)?;
        let buf_out = gpu.create_buffer_uninit::<f32>(n as usize)?;
        let wgs = workgroups_for(n, 256);
        let iters = if n <= 262_144 { 50 } else { 10 };
        let (unit, scale) = scale_unit(n);

        let buffers: &[&borsalino::GpuBuffer] = &[&buf_x, &buf_y, &buf_out];
        let spec = borsalino::DispatchSpec {
            pipeline: &pipeline_saxpy,
            buffers,
            workgroups: (wgs, 1, 1),
            threads_per_group: (256, 1, 1),
        };
        let specs: Vec<_> = (0..batch_size as usize).map(|_| spec).collect();

        let bench = run_bench(
            &format!("saxpy {n:>9} el ×{batch_size} batched"),
            unit,
            iters,
            || {
                gpu.dispatch_many(&specs).unwrap();
            },
        );

        let total_el = n as f64 * batch_size as f64;
        let total_ops = total_el * 3.0;
        let _elem_sec = total_el / bench.value;
        let gflops = total_ops / bench.value / 1e9;
        let per_dispatch = bench.value / batch_size as f64;
        println!(
            "  {n:>9} el  {:>8.3} {unit} total  {:>8.1} µs/dispatch  {:>8.2} GFLOPS  ±{:.1}%",
            bench.value * scale,
            per_dispatch * 1e6,
            gflops,
            (bench.stddev / bench.value) * 100.0
        );
        results.push(bench);
    }

    // ── Tiled Matmul (2D workgroup, shared memory) ────────────

    println!("--- Tiled Matmul (2D, shared memory) ---");

    let kernel_matmul = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

const T: u32 = 16u;
const N: u32 = MATRIX_Nu;
const K: u32 = MATRIX_Ku;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn matmul(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = wg_id.y * T + local_id.y;
    let col = wg_id.x * T + local_id.x;
    let idx = local_id.y * T + local_id.x;
    var sum = 0.0;
    for (var kt = 0u; kt < K; kt += T) {
        tile_a[idx] = a[row * K + kt + local_id.x];
        tile_b[idx] = b[(kt + local_id.y) * N + col];
        workgroupBarrier();
        for (var i = 0u; i < T; i++) {
            sum += tile_a[local_id.y * T + i] * tile_b[i * T + local_id.x];
        }
        workgroupBarrier();
    }
    c[row * N + col] = sum;
}
"#;

    for &n in &[512u32, 1024u32] {
        let total = (n * n) as usize;
        let a: Vec<f32> = (0..total).map(|i| (i % 997) as f32 * 0.001).collect();
        let b: Vec<f32> = (0..total)
            .map(|i| ((i * 3 + 1) % 997) as f32 * 0.001)
            .collect();

        let ksrc = kernel_matmul
            .replace("MATRIX_Nu", &n.to_string())
            .replace("MATRIX_Ku", &n.to_string());
        let pipeline = gpu.compile("matmul", &ksrc)?;
        let buf_a = gpu.create_buffer(&a)?;
        let buf_b = gpu.create_buffer(&b)?;
        let buf_c = gpu.create_buffer_uninit::<f32>(total)?;
        let wgs = n / 16;
        let iters = if n <= 512 { 10 } else { 5 };

        let bench = run_bench(
            &format!("matmul {n}×{n} ({wgs}×{wgs} wgs)"),
            "ms",
            iters,
            || {
                gpu.dispatch_ex(
                    &pipeline,
                    &[&buf_a, &buf_b, &buf_c],
                    (wgs, wgs, 1),
                    (16, 16, 1),
                )
                .unwrap();
            },
        );

        let flops = 2.0 * (n as f64).powi(3);
        let gflops = flops / bench.value / 1e9;
        println!(
            "  {n}×{n}  {:>8.3} ms  ({:>8.2} GFLOPS)  ±{:.1}%",
            bench.value * 1e3,
            gflops,
            (bench.stddev / bench.value) * 100.0
        );
        results.push(bench);
    }

    // ── Summary table ─────────────────────────────────────────

    print_results(&results);

    Ok(())
}
