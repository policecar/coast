+++
title = "IO Buffer"
weight = 25
+++

# IO Buffer — Double Buffering

The `io_buffer` is the fundamental data transfer mechanism between entities. It implements classic double buffering to eliminate read-write races without locks.

## How It Works

```
io_buffer
├── buffer[0]: vector<float>
├── buffer[1]: vector<float>
├── write_idx: 0 or 1
├── read_idx:  0 or 1  (always != write_idx)
└── read_buffer_stats: {sum, avg, min, max, nse}
```

**During `process()`**:
- The producing entity writes to `buffer[write_idx]`
- Consuming entities read from `buffer[read_idx]` (previous step's data)
- No overlap — write and read indices are always different

**During `swap_io()`**:
- `read_idx = write_idx` (fresh data becomes readable)
- `write_idx = (write_idx + 1) & 1` (old read buffer recycled for writing)
- Statistics are computed on the new read buffer

## Closure-Based Access

Entities don't receive raw buffer pointers. Instead, they receive **closures**:

```cpp
// Producer gets a write accessor
entity.set_outp_func(buffer.outp_buffer_func());
// Returns: function<span<float>()>

// Consumer gets a read accessor
entity.set_inp_func(source_id, buffer.inp_buffer_func());
// Returns: function<tuple<span<const float>, stats>()>
```

This encapsulation ensures:
- Entities can't accidentally swap buffers themselves
- The buffer size is fixed — no reallocation bugs
- Statistics are always available alongside data

## Wiring

`sim_env::init_io_buffers()` performs a two-pass wiring:

1. **Pass 1**: For each entity, create an `io_buffer` of size `entity.get_outp_size()` and give the entity its write accessor
2. **Pass 2**: For each entity, look up its inputs (`get_inp_ids()`) and give it read accessors to the corresponding buffers

## The One-Step Delay

This is the most important mental model for understanding data flow:

```
Step 1: Dataset writes "cat image" to its write buffer
        Neuron group reads "nothing" (initial state)

Step 1 → swap: "cat image" moves to read buffer

Step 2: Dataset writes "dog image" to its write buffer
        Neuron group reads "cat image" from read buffer

Step 2 → swap: "dog image" moves to read buffer

Step 3: Neuron group reads "dog image"
```

Every consumer sees its producer's output from **one step ago**. This is the price of lock-free operation, and it's negligible at thousands of steps per second.
