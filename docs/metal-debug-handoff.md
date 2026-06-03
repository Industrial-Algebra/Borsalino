# Metal Backend Debugging Handoff

**Date:** 2026-05-19
**Status:** Metal backend compiles on M3 but SIGSEGV at pipeline creation
**Branch:** `develop`, commit `279c53a`

---

## What Works

- `metal::tests::device_init` — ✅ M3 Metal device found, command queue created
- WGSL → MSL translation via naga — ✅ produces valid Metal Shading Language
- Framework linking — ✅ Metal + Foundation frameworks linked correctly
- Vulkan backend — ✅ 5/5 tests pass on Linux, NVIDIA, GB10

## Where It Crashes

SIGSEGV in `newComputePipelineStateWithFunction:error:` at `src/metal.rs:325`:

```
let pipeline: *mut c_void = msg_send![
    obj(dev),
    newComputePipelineStateWithFunction: func
    error: &mut perr
];
```

All preceding steps succeed:
1. ✅ naga WGSL→MSL generation (valid MSL, 499 bytes)
2. ✅ `newLibraryWithSource:options:error:` (valid MTLLibrary created)
3. ✅ `newFunctionWithName:` (valid MTLFunction found)

## Fixes Already Applied

1. **Rust 2024 compliance:** `unsafe extern "C"`, `Send+Sync` for Selectors, explicit `unsafe {}` blocks
2. **Framework linking:** `#[link(name = "Metal")]` + `#[link(name = "Foundation")]`
3. **Naga MSL resource binding:** `EntryPointResources` with `BindingMap`, `fake_missing_bindings: false`
4. **Sizes buffer:** `sizes_buffer: Some(30)`, sizes buffer created at dispatch
5. **Bounds checking:** `BoundsCheckPolicy::Unchecked` for index/buffer/image_load
6. **objc crate:** Replaced raw `objc_msgSend` transmute with `msg_send!` macro
7. **obj() cast helper:** All receivers cast through `obj(ptr)` for `Message` trait

## Debug Output Already In Place

`eprintln!` guards at every step in `compile()`:
- `[borsalino::metal] compile: entry_point=...`
- `[borsalino::metal] MSL generated (N bytes)`
- `[borsalino::metal] creating MTLLibrary...`
- `[borsalino::metal] nsstring created`
- `[borsalino::metal] library created`
- `[borsalino::metal] looking up function 'add_one'...`
- `[borsalino::metal] entry nsstring created`
- `[borsalino::metal] func=0x...`
- `[borsalino::metal] creating compute pipeline state...`
- Then SIGSEGV

## Commands to Run

```bash
git clone git@github.com:Industrial-Algebra/Borsalino.git
cd Borsalino
git checkout develop
cargo clean && cargo test --features metal -- --nocapture
```

## Suspected Areas

1. **`func` might not be a valid MTLFunction** despite non-null pointer — try printing the Objective-C class name: `let cls: *const i8 = msg_send![obj(func), className];`
2. **`dev` might not support the selector** — try using `newComputePipelineStateWithDescriptor:error:` instead
3. **`perr` double-pointer ABI** — try without the error parameter: use `respondsToSelector:` first
4. **Metal 3 API changes on M3** — need `MTLComputePipelineDescriptor` on newer macOS?
5. **Thread safety** — tests run in parallel, both hitting Metal simultaneously

## Things to Try

### 1. Verify the function object is valid
```rust
// After newFunctionWithName:
let cls_name: *const std::ffi::c_char = msg_send![obj(func), className];
eprintln!("func class: {:?}", std::ffi::CStr::from_ptr(cls_name));
```

### 2. Try without error pointer
```rust
// Try method that doesn't take error:
let pipeline2: *mut c_void = msg_send![
    obj(dev),
    newComputePipelineStateWithFunction: func
    // no error: parameter
];
```

### 3. Try with MTLComputePipelineDescriptor
```rust
let desc: *mut c_void = msg_send![class!(MTLComputePipelineDescriptor), new];
let _: () = msg_send![obj(desc), setComputeFunction: func];
let pipeline: *mut c_void = msg_send![
    obj(dev),
    newComputePipelineStateWithDescriptor: desc
    options: 0u64
    reflection: std::ptr::null_mut::<c_void>()
    error: &mut perr
];
```

### 4. Disable parallel tests
```bash
cargo test --features metal -- --test-threads=1 --nocapture
```

### 5. Test with hand-written MSL (bypass naga)
```rust
let msl = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void add_one(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
        output[id] = input[id] + 1.0;
    }
"#;
// Use this instead of naga-generated MSL in compile()
```

## Dependencies

```toml
objc = "0.2"        # macOS only, msg_send! macro
naga = "27"         # WGSL → MSL translation
thiserror = "2"
bytemuck = "1"
```
