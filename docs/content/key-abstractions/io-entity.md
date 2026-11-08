+++
title = "IO Entity Contract"
weight = 20
+++

# IO Entity Contract

`io_entity` is the interface that every simulation participant must implement. It defines how an entity produces output, consumes input, and participates in the simulation loop.

## The Contract

Every entity must implement:

| Method | What You Must Do |
|--------|-----------------|
| `process()` | Read your inputs (previous step), compute, write your output |
| `get_outp_id()` | Return your unique ID |
| `get_outp_size()` | Return how many floats you produce |
| `get_inp_ids()` | Return the IDs of entities you read from |

And may optionally implement:

| Method | When |
|--------|------|
| `set_output_buffer(io_buffer&)` | Store a reference to your write buffer |
| `set_input_buffer(output_id_t, io_buffer&)` | Store a reference to each input buffer |
| `save_state(writer)` | Checkpoint your internal state |
| `load_state(reader)` | Restore from a checkpoint |

> The binding methods take an `io_buffer&` directly. They replaced the old closure setters `set_outp_func` / `set_inp_func` in the S0 refactor — entities now hold buffer references, and the loose coupling the closures provided comes from the virtual `io_buffer` base instead. See [IO Buffer]({{% relref "/key-abstractions/io-buffer" %}}).

## Two Kinds of Entities

**Source entities** (datasets): No inputs, `get_inp_ids()` returns `{}`. They produce data from external sources (files). Example: `mnist_io`.

**Processing entities** (neuron groups): Have inputs, read from other entities' buffers. Example: `neuron_group_t`.

## GPU Entities: `gpu_io_entity`

An entity that runs on the GPU derives not from `io_entity` directly but from the intermediate base `gpu_io_entity` (declared in `sim_core/gpu_io_entity.h`). Since the *gpu-abstraction-layer* refactor this base is fully backend-agnostic — it crosses the GPU seam only through the opaque `gpu_device` façade, never a WebGPU type:

```cpp
class gpu_io_entity : public io_entity {
public:
    virtual void           build_step_plan(coast::gpu::gpu_device&) {}  // build the static plan ONCE
    virtual coast::gpu::plan_id plan() const { return coast::gpu::invalid_plan; }  // its plan_id
    virtual std::uint64_t  graph_epoch() const { return 0; }            // structural epoch (dirty/rebuild)
    virtual void           read_probes() {}                             // post-rendezvous readback (default no-op)
};
```

The device is carried by the **type**, not by a runtime flag — the dispatcher finds GPU work with `dynamic_cast<gpu_io_entity*>(&entity)`. A GPU entity splits its per-step work:

- **`build_step_plan()`** runs **once** at construction: it registers the entity's buffers + kernels on the device and emits its per-step work as a static `step_plan` (a backend-agnostic POD), registered under a `plan_id`.
- **`process()`** runs every step, but only does *host-side bookkeeping* — advancing counters and refreshing small uniform buffers via `gpu_device::upload()`. The plan itself does not change; the dispatcher replays it each step.
- **`read_probes()`** runs after the phase-1 submit completes; it fans probe/OBSERVE values back to the CPU, gating the GPU→CPU readback on whether any probe is active.

Concrete examples: `mnist_io_gpu` ([io]({{% relref "/architecture/modules/io" %}})) and `neuron_group_gpu_t` ([hd_ngm3]({{% relref "/architecture/modules/hd_ngm3" %}})). The full mechanism is in [Heterogeneous Execution]({{% relref "/key-abstractions/heterogeneous-execution" %}}).

## The Process/Swap Cycle

```
Step N:
  process():
    - Source entity A writes sample data to its write buffer
    - Processing entity B reads A's READ buffer (step N-1 data)
    - B writes its output to its write buffer

  swap_io():
    - All write buffers become read buffers
    - A's step N output is now available as B's input for step N+1
```

The one-step delay is fundamental — it's what makes the system race-free without locks.

## Type Erasure

`entity_vec` is the abstract base for storing heterogeneous entity vectors:

```cpp
struct entity_vec {
    virtual io_entity& as_base(size_t idx) = 0;
    virtual size_t get_size() = 0;
};

template<typename T>
struct typed_entity_vec : entity_vec, std::vector<T> {
    // Override as_base() to cast vector element to io_entity&
};
```

`sim_env` stores `unordered_map<type_index, unique_ptr<entity_vec>>`. This means:
- Each concrete type gets its own contiguous `vector<T>` (cache-friendly)
- But `sim_env` can iterate all entities polymorphically via `as_base()`
- New entity types can be added without modifying `sim_env`
