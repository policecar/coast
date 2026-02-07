+++
title = "probe"
weight = 60
+++

# probe — Real-Time Observation System

The `probe` module enables real-time observation of simulation values with near-zero overhead when inactive.

## Directory Layout

```
include/probe/
├── entity_probe_set.h         # Per-entity probe management (TBB concurrent_hash_map)
├── probe_instance.h           # Per-probe state with SPSC queue
├── probe_observer.h           # Abstract observer interface
├── spsc_queue.h               # Lock-free single-producer single-consumer queue
└── observers/
    ├── scalar_observer.h      # Current value + running statistics
    ├── time_series_observer.h # Line chart with optional EMA
    ├── histogram_observer.h   # Value distribution (auto-ranging)
    └── csv_file_observer.h    # Direct CSV file output

src/probe/
├── entity_probe_set.cpp       # TBB hash map operations
├── probe_instance.cpp         # Record, activate/deactivate, observer management
└── observers/
    ├── scalar_observer.cpp
    ├── time_series_observer.cpp
    ├── histogram_observer.cpp
    └── csv_file_observer.cpp
```

## Core Components

### spsc_queue\<T\>

Lock-free ring buffer (default capacity 4096). Uses `memory_order_release` on producer writes and `memory_order_acquire` on consumer reads.

- `try_push(value)` — Producer: returns false if full (drops value)
- `try_pop(out)` — Consumer: returns false if empty
- `drain(fn)` — Pops all available values

### probe_instance

Per-probe state at a specific (element, probe_name) location:

- `active` — `atomic<bool>`, checked on every probe call
- `decimation` — `atomic<uint32_t>`, record every Nth call
- `samples` — `spsc_queue<probe_sample>` (step + value pairs)
- `observers` — `vector<unique_ptr<probe_observer>>` (GUI thread only)

**Key methods**:
- `record(value)` — Applies decimation, pushes to SPSC queue (sim thread)
- `update_observers()` — Drains queue, feeds observers (GUI thread)
- `activate()` / `deactivate()` — Manages active_probe_count on the element

### entity_probe_set

Per-entity probe management using TBB `concurrent_hash_map` for thread-safe access:
- **Hot path** (`record`): `const_accessor` (read lock) for lookup, then `probe_instance::record()`
- **Registration** (`discover`, `activate`): `accessor` (write lock) for insert
- Hash key: `hash(element_ptr) ^ (hash(name) << 1)`

### probe_observer (Abstract)

```cpp
void push(uint64_t step, double value);  // Receive new data
void render();                            // Render ImGui UI
string name();                            // Display name
```

## Observer Implementations

| Observer | Display | Notes |
|----------|---------|-------|
| `scalar_observer` | Current, Min, Max, Mean, Count | Reset button |
| `time_series_observer` | ImPlot line chart | Circular buffer (1000), optional EMA |
| `histogram_observer` | ImPlot bar chart | Auto-ranging (first 100 samples) or fixed range |
| `csv_file_observer` | Status indicator | Writes `step,value` lines, flushes every 100 samples |

## Zero-Cost Design

When no probes are active, the overhead is a single `atomic<uint32_t>` load (relaxed) per `probe()` call. The check gates are:

1. **`active_probe_count_ == 0`** → return (inline in entity_element)
2. **TBB map lookup** → if probe inactive, return
3. **Decimation check** → skip if `count % dec != 0`
4. **SPSC push** → single cache line write
