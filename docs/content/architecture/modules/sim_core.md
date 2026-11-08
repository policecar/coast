+++
title = "sim_core"
weight = 20
+++

# sim_core — Simulation Engine

The `sim_core` module provides the simulation container (`env`), the `io_entity` interface, double-buffered IO, and the threading controller.

## Directory Layout

```
include/sim_core/
├── sim_env.h                  # Simulation environment (entity + buffer container + gpu_context)
├── io_entity.h                # Abstract entity interface
├── io_buffer.h                # Abstract io_buffer base + io_buffer_cpu
├── io_buffer_gpu.h            # 3 GPU io_buffer variants (COAST_WITH_GPU)
├── gpu_io_entity.h            # Intermediate base for GPU entities (COAST_WITH_GPU)
├── sim_thread_controller.h    # Dedicated simulation thread with mode control
└── sim_instance.h             # RAII container wrapping env + builder + controller + logging_mgr

src/sim_core/
├── sim_env.cpp                # Entity mgmt, hook processing, topology pre-pass, CPU/GPU dispatch via gpu_device
├── io_buffer.cpp              # io_buffer_cpu (pure index-flip swap)
├── io_buffer_gpu.cpp          # GPU variant implementations (COAST_WITH_GPU)
├── sim_thread_controller.cpp  # Thread main loop, SPS measurement
└── sim_instance.cpp           # Config loading, entity construction, move semantics
```

## sim_instance — Runtime Container

`coast::sim_instance` is the RAII container for everything needed to run a simulation from a JSON config. Owns, in construction order (which determines destruction order):

1. `json_loader` — parses the config file
2. `sim::env` — empty environment, populated by the builder
3. `sim::sim_builder` — registers handlers, builds entities from config
4. `sim::sim_thread_controller` — starts the dedicated sim thread in paused mode
5. `logging::logging_manager` — creates the logging coordinator

Destruction runs in reverse: logging stops, the thread joins, then the env is destroyed.

`sim_instance` lives in `sim_core/` (not `gui_core/`) so that unit tests and headless tools can construct a fully-loaded simulation without pulling in the GUI's SDL/OpenGL dependencies.

## run_manifest datasets mirror

`coast_runtime.json` (written at the end of every headless run) contains a `datasets` object mirroring the resolved active dataset configs — data paths, sequencing parameters, and any `index_manifest`; `frozen` records the run's top-level `frozen`, and `run_id` the run's id as it ran (its checkpoints carry it; SPEC S.5.14). A downstream harness audits from this what a run actually presented, rather than trusting the config it generated is the one that ran. A sibling `retina` object does the same for the stimulus front end: the configs in use by [retina]({{% relref "/architecture/modules/retina" %}}) nodes, keyed by config name, with each seed as the config's seed before offsets, as its 32-bit value — the effective seed of each node is in `node_seeds` (empty when the run has none).

It also records a `node_seeds` array: the effective RNG seed — the config seed with the global `--seed-offset` and per-node offsets folded in — of each active dataset, retina and model node, read from the built entity (`io_entity::effective_seed()`) so a consumer needn't reconstruct coast's seed composition. A group with a `random_index` inhibitor also has an `inhibition_seed` entry (`io_entity::inhibition_seed()`): the seed of its inhibition matrix, composed with the same offsets, or drawn (`ri_seed_from_time`), or restored from a loaded checkpoint.

### Path fields — the resolution contract

Every path the manifest records (`savestate_outputs[].path`, `observe.output_dir`) follows one rule, so a consumer resolves them all the same way:

- a path **at or below the manifest's own directory** is stored **relative**, marked with a leading `"./"` (e.g. `"./snapshots/state.h5"`) — resolve it against the manifest's location;
- a path **outside** that directory is stored **absolute** — it is never reached with a `"../.."` climb.

So *where you write the manifest chooses the outcome*: emit it into the run's output directory and you get a relocatable `"./…"` bundle (move or archive the folder, the links still resolve, no machine-specific absolute paths); emit it elsewhere and outside paths come out absolute and unambiguous. **Consumer contract:** a path beginning with `"./"` is relative to the manifest file's directory; any other path is absolute.

## sim::env — The Simulation Environment

Central container for all entities and their IO buffers. Owns:

- **`entities`** — `unordered_map<type_index, unique_ptr<entity_vec>>`: Type-erased storage. Each concrete entity type gets its own contiguous `vector<T>` for cache efficiency, wrapped in a `typed_entity_vec<T>`.
- **`io_buffers`** — `unordered_map<size_t, unique_ptr<io_buffer>>`: One buffer per entity, keyed by `output_id_t`. Now holds `unique_ptr` because `io_buffer` became an abstract base with concrete variants (see below); pointers into the mapped values stay address-stable across rehash.
- **`gpu_dev_`** — `unique_ptr<coast::gpu::gpu_device>` (only under `COAST_WITH_GPU`), with a `bool gpu_enabled_` flag: null in CPU-only runs, lazily constructed via the agnostic `make_gpu_device()` when a config contains a GPU entity. `env` holds only this agnostic handle — the concrete backend owns its own context privately. Declared so it is destroyed *last* — GPU entities and buffers release their device handles against it in their destructors.
- **Hook system** — Five hook kinds (`pre_process`, `post_process`, `pre_swap`, `post_swap`, `post_swap_barrier`) with deferred registration and synchronous removal (see [Hook System]({{% relref "/key-abstractions/hook-system" %}})).

### Key Methods

| Method | Purpose |
|--------|---------|
| `emplace_back<T>(args...)` | Create an entity of type T |
| `get_entities<T>()` | Get the typed vector (optional) |
| `iterate_entities()` | C++23 generator yielding all entities |
| `get_entity(output_id_t)` | Find entity by ID (linear scan) |
| `init_io_buffers()` | Classify buffer variants (topology pre-pass), construct, and wire writers/readers |
| `process()` | Run one simulation step: hooks + GPU batch submit + CPU dispatch + the per-step coherence rendezvous |
| `swap_io()` | Cross-device sync (upload / advance_generation / readback) + swap all buffers + hooks |

### Topology Pre-Pass (`init_io_buffers`)

Before wiring, a three-pass classification chooses each buffer's concrete variant as a pure function of its writer's device and its readers' devices (a buffer's writer device is read via `dynamic_cast<gpu_io_entity*>`):

1. Record each buffer's writer + writer device (throws on duplicate writer id — the *single-writer* invariant, enforced here).
2. Record whether each buffer has any GPU reader and any CPU reader.
3. Construct the matching variant (`io_buffer_cpu` / `io_buffer_cpu_to_gpu` / `io_buffer_gpu_to_cpu` / `io_buffer_gpu`); throw if a GPU variant is needed but no `gpu_context` is installed.

Then writers are bound via `set_output_buffer(buf&)` and readers via `set_input_buffer(id, buf&)`. CPU-only configs select `io_buffer_cpu` for everything and never touch the GPU path. See [Heterogeneous Execution]({{% relref "/key-abstractions/heterogeneous-execution" %}}).

### Execution Order (CPU/GPU overlap via the `gpu_device` facade)

`env` no longer touches WebGPU directly: it drives the GPU through a virtual `gpu_device` facade (held as `gpu_dev_`, with `gpu_ctx_` borrowing the context that was moved into the backend). Each GPU entity builds a **static `step_plan` once at construction** (`build_step_plan`); per step only its uniforms change, and the dispatcher enqueues the entity's `plan()` by opaque id.

```
process():
    process_pending_hooks()
    for each pre_process hook:  hook()
    if (gpu_dev_):
        gpu_ctx_->check_or_throw()
        gpu_dev_->begin_step()
        for each gpu entity g:
            g.process()                    // refresh uniforms (gpu_device::upload)
            gpu_dev_->enqueue(g.plan())     // opaque plan id
        gpu_dev_->submit_step()             // submit, NO wait — GPU batch in flight
    dispatch_cpu_entities()                 // TBB parallel_for, overlaps the GPU batch
    if (gpu_dev_):
        gpu_dev_->rendezvous_step()         // the ONE mandatory per-step coherence wait
        for each gpu entity g: g.read_probes()   // gated readback
    for each post_process hook: hook()

swap_io():
    process_pending_hooks()
    for each pre_swap hook:     hook()
    if (gpu_dev_):
        gpu_ctx_->check_or_throw()
        for each cpu_to_gpu buffer: gpu_dev_->upload(buf.gpu_page_id(), buf.cur_write_buffer())  // before the flip
        gpu_dev_->advance_generation()      // device ping-pong page flip (one-step lag)
        for each gpu_to_cpu buffer: buf.device_to_cpu_readback()                                 // after the flip
    for each io_buffer:         buffer.swap_buffer()           // pure index flip
    for each post_swap hook:    hook()
    for each barrier hook:      hook()    // same locked phase as post_swap
```

Each `for each … hook` line is one hook phase, run under `active_hooks_mutex_` so a concurrent `remove_*_hook()` waits until the phase is over (#235).

There is exactly **one coherence wait per step** — `rendezvous_step()` in `process()`, after the GPU batch was submitted without waiting so the CPU entities overlap it. `swap_io()` does direct `upload` / `advance_generation` / `device_to_cpu_readback`, not a second recorded-encoder submit. With no `gpu_dev_`, every `if (gpu_dev_)` branch is skipped and the behaviour is identical to a pure-CPU step.

## io_entity — The Entity Interface

All simulation entities (neuron groups, datasets) implement `io_entity`, which inherits from `core::entity_element`.

### Contract

| Method | Required | Description |
|--------|----------|-------------|
| `process()` | **Pure virtual** | Execute one simulation step |
| `get_outp_id()` | **Pure virtual** | Unique entity ID |
| `get_outp_size()` | **Pure virtual** | Output buffer size (floats) |
| `get_inp_ids()` | **Pure virtual** | Input entity IDs (empty for sources) |
| `set_output_buffer(io_buffer&)` | Virtual | Receive a reference to the write buffer |
| `set_input_buffer(id, io_buffer&)` | Virtual | Receive a reference to an input buffer |
| `save_state(writer)` | Virtual | Serialize state for checkpointing |
| `load_state(reader)` | Virtual | Restore state from checkpoint |

The binding methods take `io_buffer&` references directly (they replaced the closure-returning `set_outp_func` / `set_inp_func` in the S0 refactor). Each entity also owns a `structure_registry` (for probe path lookup) and an `entity_probe_set` (for thread-safe probe management).

### gpu_io_entity (COAST_WITH_GPU)

GPU entities derive from the intermediate base `gpu_io_entity : public io_entity`. It adds `build_step_plan(gpu_device&)` (called once at construction — registers buffers + kernels and emits the op list via `register_plan` / `build_plan`), `plan_id plan()` and `graph_epoch()` accessors, and a `read_probes()` hook (post-rendezvous readback). Per step the entity's `process()` only refreshes its uniforms; the dispatcher enqueues `plan()` by opaque id. It identifies these entities with `dynamic_cast` — there is no runtime device tag. None of these hooks name WebGPU; the entity talks only to the agnostic `gpu_device` facade. See [io_entity]({{% relref "/key-abstractions/io-entity" %}}).

## io_buffer — The Device Seam

`io_buffer` is an **abstract virtual base** with four concrete variants (`io_buffer_cpu` plus three GPU variants in `io_buffer_gpu.{h,cpp}`). The base declares `swap_buffer()` / `size()` / `cur_write_buffer()` / `cur_read_buffer()`, the state-IO pair `snapshot()` / `restore()` (read-page save/load — see [IO Buffer]({{% relref "/key-abstractions/io-buffer" %}})), plus opaque, backend-agnostic GPU hooks (`gpu_register`, `gpu_page_id`, `gpu_topology`, `gpu_backed`, `read_back_sync`, `device_to_cpu_readback`) that default to no-ops. These hooks deal only in opaque `gpu_device` handles — none of them mention WebGPU. `swap_buffer()` is now a **pure index flip** — the statistics and the closure-returning accessors (`outp_buffer_func`/`inp_buffer_func`/`inp_buf_t`) were removed in the S0 refactor; stats are computed by the consuming entity instead.

This still creates the **one-step delay**: consumers always read the *previous* step's output, eliminating read-write races without locks — and the same mechanism carries data across the CPU↔GPU boundary. The full contract and per-variant layouts are in [IO Buffer]({{% relref "/key-abstractions/io-buffer" %}}).

## sim_thread_controller — Threading

Manages a dedicated `std::jthread` for the simulation, with three modes:

| Mode | Behavior |
|------|----------|
| `paused` | Waits on condition variable, supports single-step |
| `throttled` | Runs at configurable steps/second with sleep-based pacing |
| `unlimited` | Runs as fast as possible (yields every 1000 steps) |

All mode control is via lock-free atomics (`memory_order_release`/`acquire`). The condition variable is used to wake the thread from paused mode and to signal quiesced tasks (below).

**Quiesced tasks (`run_quiesced`)**: `set_mode(paused)` only stores an atomic and returns — the sim thread may still be mid-step. Code that must touch a *quiescent* sim (state save/load, checkpointing, `set_step_count`) calls `run_quiesced(fn)`: the closure is enqueued (MPSC, FIFO), the **sim thread** runs it at the next quiescent boundary — right after a `do_step()` completes, or while parked in the paused wait — and the caller blocks for the result (exceptions rethrow). Running on the sim thread is the point: the task may touch io_buffer read pages, the wgpu queue, and entity `save_state`/`load_state`, all of which the sim thread owns. The mode is never modified, and a stop-escape makes callers throw instead of deadlocking at teardown. Used by the GUI logging window and snapshot resume (state save/load) and headless checkpointing/final save. See [Threading]({{% relref "/threading" %}}) for the full contract and how it relates to the hook system.

**SPS measurement**: Exponential moving average over 500ms windows (`0.7 * measured + 0.3 * previous`).

**Step limit**: When `step_limit_ > 0` and `step_count_ >= step_limit_`, auto-transitions to `paused`. Used by headless mode.

### do_step()

```cpp
void do_step() {
    probe::set_current_step(step_count_);  // Global step for probe timestamps
    sim_env_.process();                     // Run all entities
    sim_env_.swap_io();                     // Swap all buffers
    step_count_.fetch_add(1);
}
```
