+++
title = "IO Buffer"
weight = 25
+++

# IO Buffer — The Device-Boundary Seam

The `io_buffer` is the fundamental data-transfer mechanism between entities. It implements classic double buffering to eliminate read-write races without locks, and — since the *ngm3-gpu-port* cycle — it is also the **sole seam at which data crosses between CPU and GPU**. See [Heterogeneous Execution]({{% relref "/key-abstractions/heterogeneous-execution" %}}) for the bigger picture.

`io_buffer` is an **abstract virtual base** with four concrete variants. The base carries only the minimum contract; each variant carries exactly the fields it needs.

## The Base Contract

```cpp
class io_buffer {
public:
    virtual ~io_buffer() = default;

    virtual void swap_buffer() = 0;                          // index flip at step boundary
    [[nodiscard]] virtual std::size_t size() const = 0;
    [[nodiscard]] virtual std::span<float>       cur_write_buffer() = 0;   // CPU-visible page
    [[nodiscard]] virtual std::span<const float> cur_read_buffer()  = 0;   // CPU-visible page

    // --- State snapshot / restore (full-state save / load) ---
    [[nodiscard]] virtual bool snapshot(std::span<float> dst) = 0;         // read page → dst
    [[nodiscard]] virtual bool restore(std::span<const float> src) = 0;    // src → read page, no swap

    // --- Opaque GPU seam (default no-ops, overridden by GPU variants) ---
    [[nodiscard]] virtual bool read_back_sync(std::span<float> dst);       // on-demand pull (READ side)
    virtual void device_to_cpu_readback() {}                               // GPU→CPU mirror at swap

    // Opaque GPU identity, registered once by env at build time.
    void gpu_register(coast::gpu::gpu_device& dev,
                      coast::gpu::buffer_id page, coast::gpu::topology topo) noexcept;
    [[nodiscard]] coast::gpu::buffer_id gpu_page_id()  const noexcept;     // single ping-pong page id
    [[nodiscard]] coast::gpu::topology  gpu_topology() const noexcept;
    [[nodiscard]] bool                  gpu_backed()   const noexcept;     // true once registered

protected:
    coast::gpu::gpu_device* device_    = nullptr;   // opaque device facade
    coast::gpu::buffer_id   pp_buffer_ = coast::gpu::invalid_buffer;  // ping-pong page id
    coast::gpu::topology    topo_      = coast::gpu::topology::gpu_resident;
};
```

The base includes only `core/gpu/gpu_device.h` (opaque ids), so this header never pulls in the full `<webgpu/webgpu.h>` — it is fully backend-agnostic. `pp_buffer_` is a *single* ping-pong `buffer_id`: the device keeps one physical page per generation and resolves the current page itself, so the buffer needs no write/read page pair.

> **Refactored away (S0).** The old `io_buffer` carried per-buffer statistics (`stats`, `read_buffer_stats`, `update_stats()`) and handed entities **closures** (`outp_buffer_func()`, `inp_buffer_func()`, the `inp_buf_t` typedef). Both are gone. Statistics are now a *consumer* concern — the neuron group computes input stats once per step via `ngm2::compute_input_stats()` and passes them down (see [hd_ngm2]({{% relref "/architecture/modules/hd_ngm2" %}})). Entities now hold an `io_buffer&` directly; the loose coupling the closures provided is now provided by the virtual base. `swap_buffer()` is a **pure index flip** with no side effects.

## The Four Variants

The variant of each buffer is selected automatically by `sim::env` from the devices of its writer and readers (see [sim_core]({{% relref "/architecture/modules/sim_core" %}})). All four flip the same `write_idx`/`read_idx` pair in `swap_buffer()`.

### `io_buffer_cpu` — CPU writer, CPU readers

The classic double buffer, behaviorally identical to pre-GPU COAST.

```
io_buffer_cpu
├── buffer[0]: vector<float>
├── buffer[1]: vector<float>
├── write_idx: 0 or 1
└── read_idx:  0 or 1   (always != write_idx)
```

`cur_write_buffer()` / `cur_read_buffer()` return spans into the two CPU vectors. The GPU seam members are inherited defaults (no opaque page registered). `io_buffer_cpu` lives in `io_buffer.h`; the three GPU variants below live in `io_buffer_gpu.{h,cpp}`, gated by `COAST_WITH_GPU`. The actual GPU pages no longer live in these CPU-side classes — they reside in the WebGPU backend (`backend_webgpu.cpp`) and are resolved via the opaque `pp_buffer_` id.

### `io_buffer_cpu_to_gpu` — CPU writer, GPU reader

The producer writes the CPU page as usual; at swap, env drives the cross-device move through the `gpu_device` facade: `gpu_device::upload(page_id, cur_write_buffer())` pushes the just-written CPU page to the device **before** the generation flip. GPU consumers then read the device-resident page resolved from `pp_buffer_`.

### `io_buffer_gpu_to_cpu` — GPU writer, CPU reader

Holds a CPU read mirror. After `advance_generation()` (the device READ side is then this step's fresh output) and before `swap_buffer()`, env calls `device_to_cpu_readback()`, which pulls the device output into the CPU read mirror via the `gpu_device` facade. `cur_read_buffer()` then returns the CPU page; `cur_write_buffer()` is empty (the GPU writes).

### `io_buffer_gpu` — GPU writer, GPU readers

No standing CPU mirror; both `cur_write_buffer()` and `cur_read_buffer()` return **empty spans**. CPU-side consumers that occasionally need to look call `read_back_sync()` (the on-demand pull used by logging and time-series visualization — see below), which delegates to `device_->readback(pp_buffer_, …)` of the READ side, preserving the variant's memory efficiency. `read_back_sync()` never throws: it returns `false` on any error so logging/vis hooks degrade gracefully.

## The One-Step Delay

This is the most important mental model for data flow, and it is identical for all four variants:

```
Step 1: Dataset writes "cat image" to its write page
        Neuron group reads "nothing" (initial state)
Step 1 → swap: "cat image" moves to the read page
Step 2: Dataset writes "dog image"; Neuron group reads "cat image"
Step 2 → swap: "dog image" moves to the read page
Step 3: Neuron group reads "dog image"
```

Every consumer sees its producer's output from **one step ago**. For GPU variants the cross-device movement runs inside `swap_io()` and is driven by env through the `gpu_device` facade: a `cpu_to_gpu` buffer's page is uploaded (`gpu_device::upload`) *before* `advance_generation()` (the device page flip that carries the one-step lag); a `gpu_to_cpu` buffer's output is pulled (`device_to_cpu_readback()`) *after* that flip; then `swap_buffer()` performs the pure index flip on all buffers. So the one-step contract holds across the device boundary too.

## State Snapshot / Restore (`snapshot` / `restore`)

State save/load targets the **read page** — the only buffer state that is coherent at the step boundary; the write page is overwritten by its producer on every step anyway. The pair is symmetric: save copies the read page out via `snapshot(dst)`, and load writes the read page **directly** via `restore(src)`, with no swap to perform or forget.

The direct-restore contract matters: an earlier load path restored into the *write* page and relied on "the next swap" to publish it — but `process()` runs before the swap, so the restored data was overwritten and the stale read page consumed. Restoring the read page eliminates that bug class (see `design/quiesced-state-access.md` §5.3).

Each variant reads/writes whatever physically backs its read page: the CPU page (`io_buffer_cpu`, `io_buffer_gpu_to_cpu`), the CPU mirror *and* the GPU read page (`io_buffer_cpu_to_gpu` — the GPU consumer reads the latter), or the GPU read page alone (`io_buffer_gpu` — readback for snapshot, upload for restore). Because the GPU variants issue wgpu-queue operations, both calls run on the sim thread inside a quiesced task (see [Threading]({{% relref "/threading" %}})). Like `read_back_sync`, both return `false` rather than throw on a size mismatch or failed transfer, so save/load degrades to a warning.

## On-Demand Readback (`read_back_sync`)

A pure `io_buffer_gpu` has no CPU-visible data, which is a problem for CPU-side consumers that occasionally need to *look* — the buffer logger and the Time Series visualization. Rather than maintain a standing mirror, these consumers call `read_back_sync(dst)`, which performs a self-contained `CopyBufferToBuffer` → submit → map → `memcpy` into `dst`. The buffer logger calls it from `capture_frame()` when `cur_read_buffer()` is empty; the Time Series vis calls it on the sim thread during the post-swap barrier (see [Threading]({{% relref "/threading" %}})). This closed the one conditional gap flagged during Demonstrate.
