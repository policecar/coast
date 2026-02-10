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
| `set_outp_func(func)` | Store the write buffer accessor |
| `set_inp_func(id, func)` | Store read buffer accessors for each input |
| `save_state(writer)` | Checkpoint your internal state |
| `load_state(reader)` | Restore from a checkpoint |

## Two Kinds of Entities

**Source entities** (datasets): No inputs, `get_inp_ids()` returns `{}`. They produce data from external sources (files). Example: `mnist_io`.

**Processing entities** (neuron groups): Have inputs, read from other entities' buffers. Example: `neuron_group_t`.

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
