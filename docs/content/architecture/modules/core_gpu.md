+++
title = "core/gpu"
weight = 25
+++

# core/gpu — the GPU seam, WebGPU backend & WGSL Toolchain

`core/gpu` owns everything GPU. Since the *gpu-abstraction-layer* refactor it has two faces:

- the **agnostic seam** — `gpu_device.h` declares the virtual `gpu_device` façade, the opaque handle types (`buffer_id` / `kernel_id` / `plan_id`), and the `step_plan` POD. This header carries **no SDK types** and is the surface the simulation core (`env`, the `io_buffer` base, the GPU entities) talks to. See [Heterogeneous Execution]({{% relref "/key-abstractions/heterogeneous-execution" %}}).
- the **WebGPU backend** — `backend_webgpu.cpp` is the *single* translation unit that includes `<webgpu/webgpu.h>`. It implements `gpu_device`, owns the `gpu_context`, compiles `step_plan`s into cached pipelines + pre-built per-generation bind groups, and re-records the per-step encoder.

Underneath sit the original utilities it still wraps: the **wgpu-native** device lifecycle (`gpu_context`), the WGSL shader-module loader, and a shared PRNG snippet. The agnostic core no longer mentions any `WGPU*` type — that is the canary the refactor enforces (zero `<webgpu>` outside `backend_webgpu.cpp`).

The GPU TUs here are compiled when the CMake option `COAST_GPU_BACKEND` is not `none` (default `webgpu`; the other values are `null` and `none`). With `=none`, COAST builds and runs CPU-only with no GPU dependency.

## Directory Layout

```
include/core/gpu/                    # AGNOSTIC path — no WebGPU header lives or is included here
├── gpu_device.h       # the seam: gpu_device façade + buffer_id/kernel_id/plan_id + step_plan POD + make_gpu_device()
└── shared_prng.h      # accessor for the embedded shared PRNG WGSL (backend-neutral WGSL string source)

src/core/gpu/
├── shared_prng.cpp    # #embed of shared_prng.wgsl
├── shared_prng.wgsl   # PCG-1D32 stateless hash PRNG (WGSL source)
└── backends/                        # exactly one of these is compiled (selected by COAST_GPU_BACKEND)
    ├── webgpu/        # WebGPU backend — the ONLY place <webgpu/webgpu.h> is included
    │   ├── backend_webgpu.cpp # implements gpu_device, owns gpu_context, defines make_gpu_device()
    │   ├── gpu_context.{h,cpp} # instance/adapter/device/queue lifecycle + fault state
    │   ├── wgpu_util.h         # pump_until() and small create helpers
    │   └── wgsl_module.{h,cpp} # WGSL string → WGPUShaderModule loader
    └── null/
        └── backend_null.cpp   # headless backend: host-array buffers, no-op compute; defines make_gpu_device()
```

The parameterless `make_gpu_device()` factory (declared in the agnostic `gpu_device.h`, defined by exactly the one backend TU that CMake compiles) is the only seam-construction entry point — the backend builds its own context internally. Backend selection is the compile switch `COAST_GPU_BACKEND` (`none`|`webgpu`|`null`); the concrete backend type stays private to its TU. `gpu_fwd.h` (which used to forward-declare opaque `WGPU*` typedefs) was deleted in this cycle: the agnostic core no longer names any `WGPU*` type.

## gpu_context

The single owner of all wgpu handles, now **private to the WebGPU backend** (it lives under `backends/webgpu/`, not on the agnostic path). It is constructed by the backend itself inside `make_gpu_device()`, lazily, only when a configuration contains a GPU entity. `env` no longer names or owns a `gpu_context`: it holds just a `std::unique_ptr<gpu_device>` and drives faults through the agnostic `gpu_device::check_or_throw()` at phase boundaries.

```cpp
struct gpu_context {
    WGPUInstance instance = nullptr;
    WGPUAdapter  adapter  = nullptr;
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;
    WGPULimits   limits   = {};
    std::string  adapter_name;

    std::atomic<bool> faulted_{false};
    std::mutex        fault_msg_mutex_;
    std::string       fault_message_;

    gpu_context();    // boots instance → adapter → device → queue (throws on failure)
    ~gpu_context();   // releases in reverse order, never throws

    void check_or_throw();  // re-raise any async fault at a phase boundary
    void wait_idle();       // wgpuDevicePoll(device, /*wait=*/true)

    // non-copyable, non-movable
};
```

### Construction

The constructor walks the wgpu bring-up synchronously: request instance → adapter → query limits → device → queue. The wgpu create-* calls are asynchronous; they are driven to completion by `pump_until()` (in `wgpu_util.h`), which polls the instance until the callback fires. Any failure throws a `std::runtime_error` carrying the verbatim wgpu diagnostic.

**Backend selection** is a compile-time decision:

```cpp
#ifdef __APPLE__
    constexpr WGPUBackendType kBackend = WGPUBackendType_Metal;
#else
    constexpr WGPUBackendType kBackend = WGPUBackendType_Vulkan;
#endif
```

The adapter is requested with this explicit backend and `WGPUPowerPreference_HighPerformance`. The device is created with the adapter's reported limits echoed back as required limits, and with both fault callbacks wired into its descriptor.

### Fault Handling

Two callbacks are installed on the device:

- **uncaptured-error** — any async validation/runtime error; stores `"uncaptured error: …"`.
- **device-lost** — stores `"device lost: …"` (filtering out the normal `Destroyed` / `CallbackCancelled` reasons that fire during teardown).

Both run on arbitrary wgpu-internal threads, so they do nothing but record the message under `fault_msg_mutex_` and set `faulted_` (release). They never call back into wgpu (re-entrancy is forbidden).

The simulation thread surfaces faults at each phase boundary by calling `check_or_throw()` at the top of `env::process()` and `env::swap_io()`: it loads `faulted_` (acquire) and, if set, throws `std::runtime_error("gpu_context: " + message)`. The exception propagates out of the sim thread and terminates the run — the correct outcome for an unrecoverable GPU error. There is no recovery path in this cycle.

### wait_idle

`wait_idle()` is `wgpuDevicePoll(device, /*wait=*/true, nullptr)` — a synchronous block until the queue drains. The backend uses it to realize `gpu_device::rendezvous_step()` — the one mandatory per-step coherence wait (BARRIER 1) — so the sim thread never races ahead of the GPU. (The contract is specified as *coherence*, not literally a drain, so a future backend could realize it with an event instead; this cycle's backend uses the poll.)

## wgsl_module

A single function turns a WGSL string into a compiled shader module:

```cpp
WGPUShaderModule load_wgsl_module(WGPUInstance instance, WGPUDevice device,
                                  std::string_view source,
                                  std::string_view label = "wgsl");
```

It wraps creation in a validation error scope (`wgpuDevicePushErrorScope` / `wgpuDevicePopErrorScope`) and, on any compile error, **throws a `std::runtime_error` carrying the full verbatim wgpu (naga) diagnostic** — line, column, and expected token. This message fidelity *is* the deliverable: it is the day-to-day kernel-author iteration loop. Callers concatenate the shared PRNG snippet ahead of their entity WGSL before calling this (`std::string(shared_prng_wgsl()) + entity_src`); the snippet declares only functions, so it never clashes with entity bindings.

## gpu_fwd.h (deleted)

Historically `gpu_fwd.h` forward-declared the opaque wgpu handle typedefs (`WGPUBuffer`, `WGPUDevice`, `WGPUInstance`, …), and before the *gpu-abstraction-layer* refactor those typedefs leaked into `io_buffer.h` and `gpu_io_entity.h`. The *fixing-gpu-abstraction* cycle completed the boundary by moving every WebGPU header under `backends/webgpu/` and **deleting `gpu_fwd.h` entirely** — the agnostic core no longer names any `WGPU*` type, so there is nothing left to forward-declare on the agnostic path. Backend TUs include the real `<webgpu/webgpu.h>` directly.

## shared_prng.wgsl — the GPU PRNG

GPU kernels do **not** carry per-cell RNG state. Instead they use a *stateless* hash PRNG — PCG-1D32 (Jarzynski & Olano, "Hash Functions for GPU Rendering"):

```wgsl
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn pcg_uniform(input: u32) -> f32 {
    return f32(pcg_hash(input)) * (1.0 / 4294967296.0);   // [0, 1)
}
```

About six ALU ops, no platform intrinsics. Callers form the key from entity coordinates and time — e.g. `pcg_uniform((neuron ^ dendrite ^ rep ^ step ^ slot) …)` — yielding a value that is deterministic per `(seed, step)` (run-to-run reproducible) yet decorrelated across cells. This is **intentionally not bit-equivalent** to the CPU `mt19937_simd` stream, which is why CPU/GPU parity is statistical rather than exact (see [Architecture Decisions]({{% relref "/architecture/decisions" %}})).

The `.wgsl` source is compiled into the binary with C++23 `#embed`:

```cpp
constexpr unsigned char kSharedPrngWgsl[] = {
#embed "shared_prng.wgsl"
};
```

`shared_prng_wgsl()` returns a `std::string_view` over it. The same `#embed` approach embeds `mnist_io_gpu.wgsl` and `hd_ngm3_gpu.wgsl`. (A CMake byte-array codegen fallback is documented should a toolchain lack `#embed`; GCC 15 and Clang support it, so it is not currently needed.)

## Build Integration

`COAST_GPU_BACKEND` (string option, `none`|`webgpu`|`null`, default `webgpu`) selects the single backend. For any value ≠ `none`, CMake derives and defines the `COAST_WITH_GPU` compile guard `PUBLIC` on `coast_lib` (gating the agnostic GPU TUs `io_buffer_gpu`, `gpu_io_entity`, `mnist_io_gpu`, `hd_ngm3_gpu_neuron_group`) and compiles exactly one backend TU. `gpu_device.h` itself is agnostic and compiles unconditionally.

When `COAST_GPU_BACKEND=webgpu`, CMake additionally:

- Pulls **wgpu-native v29.0.0.0** as a prebuilt release `.zip` via `FetchContent_Declare`, selected per platform (`linux-x86_64`, `linux-aarch64`, `macos-x86_64`, `macos-aarch64`) with a **pinned SHA256 per asset**. The prebuilt has no CMake project — it is populate-and-extract only, into each build tree's own `_deps/`. The zip itself is taken from a machine-wide dependency cache when it holds a verified copy (`COAST_DEPS_CACHE`, default `~/.cache/coast/deps`, see `cmake/coast_dep_cache.cmake`); otherwise it is downloaded into the cache once. GitHub is asked only when the cache has no verified copy, and a cache that is off (`-DCOAST_DEPS_CACHE=`), not writable or failing to download falls back to the direct download.
- Imports it as the shared `wgpu_native::wgpu_native` target (with `IMPORTED_NO_SONAME` and a build-RPATH entry so every executable resolves the `.so`/`.dylib` at launch with no install step).
- Links it to `coast_lib` as `PRIVATE $<LINK_ONLY:wgpu_native::wgpu_native>`: the `.so` reaches every executable's link line, but none of its usage requirements do, and neither does the `backends/webgpu/` directory. Only the three backend TUs (`backend_webgpu.cpp`, `gpu_context.cpp`, `wgsl_module.cpp`) get the SDK include root, per source. A core, GUI or test TU that includes `<webgpu/webgpu.h>` or `gpu_context.h` therefore fails to compile: the seam is enforced by the build, not by convention.
- Defines the INTERFACE target `coast_webgpu_backend_private` (backend header dir + SDK). The raw drivers that operate `gpu_context` directly (`coast_gpu_smoke`, `coast_gpu_spike`, `coast_gpu_min_kernel`, `coast_mnist_io_gpu_spike`) link it, which is their explicit declaration that they bypass the seam.

When `COAST_GPU_BACKEND=null`, CMake compiles `backends/null/backend_null.cpp` instead and links **no** wgpu dependency — a GPU-free build that still exercises the full agnostic seam. `=none` compiles no backend and no GPU TUs at all.
- Suppresses only the `#embed` extension warning on the three `.cpp` files that use it (the rest of `coast_lib` keeps `-Wall -Wpedantic -Wextra`).

A set of standalone GPU probe executables (built only with the option on) exercises each layer in isolation: `coast_gpu_smoke`, `coast_gpu_min_kernel`, `coast_dispatcher_probe`, `coast_mnist_io_gpu_probe`, `coast_ngm3_gpu_probe`, plus the `*_spike` bandwidth/latency probes. (The former `coast_io_buffer_gpu_probe` was retired with the refactor — the GPU `io_buffer` variants no longer carry their own WGPU pages.)
