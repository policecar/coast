+++
title = "logging"
weight = 65
+++

# logging — Data Persistence

The `logging` module provides three persistence capabilities: buffer logging, probe logging, and state checkpointing. All use HDF5 as the primary format.

## Directory Layout

```
include/logging/
├── logging.h                  # Convenience include
├── logging_common.h           # Shared types: log_result, compression_config, etc.
├── logging_manager.h          # Central coordinator
├── buffer_logger.h            # Per-step IO buffer capture
├── probe_logger.h             # Probe value streaming (HDF5 or CSV)
├── state_logger.h             # Full state save/load
├── state_io.h                 # state_writer/state_reader interfaces
├── hdf5_utils.h               # RAII wrappers for HDF5 C API
└── file_observer.h            # probe_observer that delegates to probe_logger

src/logging/
├── logging_manager.cpp
├── buffer_logger.cpp          # Hook-based capture, chunked writes
├── probe_logger.cpp           # Buffered writing, HDF5/CSV formats
├── state_logger.cpp           # hdf5_state_writer/reader implementations
├── hdf5_utils.cpp
└── file_observer.cpp
```

## HDF5 Utilities (hdf5_utils.h)

RAII wrappers for all HDF5 handle types:

| Wrapper | C API | Purpose |
|---------|-------|---------|
| `hdf5::file` | `H5Fopen/create/close` | File access (read, readwrite, create) |
| `hdf5::group` | `H5Gcreate/open/close` | Hierarchical groups (recursive creation) |
| `hdf5::dataspace` | `H5Screate*` | Shape descriptors (scalar, simple, extensible) |
| `hdf5::plist` | `H5Pcreate` | Property lists (chunking, compression) |
| `hdf5::dataset` | `H5Dcreate/open/close` | Data containers with extend/write/read |

Plus free functions: `write_attr()`, `read_attr_*()`, `link_exists()`.

## Buffer Logger

Captures all IO buffer contents at each simulation step.

**Mechanism**: Registers a `post_swap_hook` that runs on the simulation thread after each `swap_io()`. Copies buffer data into a frame buffer; when `chunk_size` frames accumulate (default 1000), batch-writes to HDF5.

**HDF5 Schema**:
```
/metadata/
    @schema_version, @start_step, @config, @created
    topology/
        buffer_ids [N], buffer_sizes [N], buffer_names [N]
        connectivity [E x 2], vis_hints [N x 3]
/steps/
    step_numbers [T], timestamps [T]
    buffers/<id>/
        data [T x S]    (float32, chunked + compressed)
        stats [T x 5]   (sum, avg, min, max, nse)
```

## Probe Logger

Streams selected probe values to HDF5 or CSV files.

Creates `file_observer` instances (a `probe_observer` subclass) and attaches them to probe instances. When the SPSC queue is drained during `update_observers()`, the file observer calls `probe_logger::record_value()`, which buffers data and flushes periodically.

**HDF5 Schema**:
```
/probes/<sanitized_path>/
    @probe_path = <original path>
    steps [N] uint64
    values [N] float64
```

**CSV Format**: `step,probe1,probe2,...` with one row per time step.

## State Logger

Full simulation state save/load using abstract `state_writer` / `state_reader` interfaces.

### Interfaces

**`state_writer`**: `begin_group/end_group`, `write(name, scalar/vector)`, `write_2d()`, `write_attr()`

**`state_reader`**: `enter_group/leave_group`, `has_field()`, `read_*(name, default)` with graceful fallbacks

Both have HDF5 implementations (`hdf5_state_writer`, `hdf5_state_reader`) using a group stack for nesting.

**HDF5 Schema**:
```
/metadata/
    @schema_version, @type, @config, @timestamp
    step_count
/io_buffers/<id>/data [S]
/entities/<id>/ ... (entity-specific via save_state/load_state)
```

## Logging Manager

Central coordinator owning the three loggers:

- `start/stop_buffer_logging(config)` — creates/destroys `buffer_logger`
- `start/stop_probe_logging(config)` — creates/destroys `probe_logger`
- `save/load_state(...)` — delegates to `state_logger`
- `flush_all()` / `stop_all()` — lifecycle management
