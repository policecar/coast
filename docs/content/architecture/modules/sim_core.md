+++
title = "sim_core"
weight = 20
+++

# sim_core — Simulation Engine

The `sim_core` module provides the simulation container (`env`), the `io_entity` interface, double-buffered IO, and the threading controller.

## Directory Layout

```
include/sim_core/
├── sim_env.h                  # Simulation environment (entity + buffer container)
├── io_entity.h                # Abstract entity interface
├── io_buffer.h                # Double-buffered float data
└── sim_thread_controller.h    # Dedicated simulation thread with mode control

src/sim_core/
├── sim_env.cpp                # Entity management, hook processing, buffer wiring
├── io_buffer.cpp              # Buffer swap and statistics
└── sim_thread_controller.cpp  # Thread main loop, SPS measurement
```

## sim::env — The Simulation Environment

Central container for all entities and their IO buffers. Owns:

- **`entities`** — `unordered_map<type_index, unique_ptr<entity_vec>>`: Type-erased storage. Each concrete entity type gets its own contiguous `vector<T>` for cache efficiency, wrapped in a `typed_entity_vec<T>`.
- **`io_buffers`** — `unordered_map<size_t, io_buffer>`: One double-buffered buffer per entity, keyed by `output_id_t`.
- **Hook system** — Four hook points with deferred registration (see [Hook System]({{% relref "/key-abstractions/hook-system" %}})).

### Key Methods

| Method | Purpose |
|--------|---------|
| `emplace_back<T>(args...)` | Create an entity of type T |
| `get_entities<T>()` | Get the typed vector (optional) |
| `iterate_entities()` | C++23 generator yielding all entities |
| `get_entity(output_id_t)` | Find entity by ID (linear scan) |
| `init_io_buffers()` | Wire up all read/write buffer accessors |
| `process()` | Run one simulation step: hooks + all entity process() |
| `swap_io()` | Swap all buffers: hooks + all buffer swaps |

### Execution Order

```
process():
    process_pending_hooks()       // apply deferred hook adds/removes
    for each pre_process hook:  hook()
    for each entity:            entity.process()
    for each post_process hook: hook()

swap_io():
    process_pending_hooks()
    for each pre_swap hook:     hook()
    for each io_buffer:         buffer.swap_buffer()
    for each post_swap hook:    hook()
```

## io_entity — The Entity Interface

All simulation entities (neuron groups, datasets) implement `io_entity`, which inherits from `core::entity_element`.

### Contract

| Method | Required | Description |
|--------|----------|-------------|
| `process()` | **Pure virtual** | Execute one simulation step |
| `get_outp_id()` | **Pure virtual** | Unique entity ID |
| `get_outp_size()` | **Pure virtual** | Output buffer size (floats) |
| `get_inp_ids()` | **Pure virtual** | Input entity IDs (empty for sources) |
| `set_outp_func(func)` | Virtual | Receive write buffer accessor |
| `set_inp_func(id, func)` | Virtual | Receive read buffer accessor |
| `save_state(writer)` | Virtual | Serialize state for checkpointing |
| `load_state(reader)` | Virtual | Restore state from checkpoint |

Each entity also owns a `structure_registry` (for probe path lookup) and an `entity_probe_set` (for thread-safe probe management).

## io_buffer — Double Buffering

Each buffer holds two `vector<float>` and a read/write index pair.

**Swap** flips the indices: the write buffer becomes the new read buffer, and statistics (sum, avg, min, max, normalized Shannon entropy) are computed on the new read buffer.

This creates a **one-step delay**: consumers always read the *previous* step's output. This eliminates read-write races without locks.

**Accessors** are closure-based:
- `outp_buffer_func()` → returns a `function<span<float>()>` for the write buffer
- `inp_buffer_func()` → returns a `function<tuple<span<const float>, stats>()>` for the read buffer

## sim_thread_controller — Threading

Manages a dedicated `std::jthread` for the simulation, with three modes:

| Mode | Behavior |
|------|----------|
| `paused` | Waits on condition variable, supports single-step |
| `throttled` | Runs at configurable steps/second with sleep-based pacing |
| `unlimited` | Runs as fast as possible (yields every 1000 steps) |

All mode control is via lock-free atomics (`memory_order_release`/`acquire`). The condition variable is only used to wake the thread from paused mode.

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
