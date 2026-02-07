# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COAST (Coherent Active State Dynamics in Self-Organizing Systems) is a C++23 neural simulation framework with an interactive SDL2/OpenGL3 GUI for visualizing the HD-NGM2 (Hierarchical Dendritic - Neuron Group Model 2) neural model.

## Build Commands

**IMPORTANT:** Always use `-DCMAKE_BUILD_TYPE=Release` to enable `-O3` optimization. Debug builds are ~10x slower.

```bash
# Linux
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# macOS (requires GCC 15 - default Clang lacks C++23 features)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$(brew --prefix gcc)/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=$(brew --prefix gcc)/bin/g++-15
cmake --build build -j

# Run (use System > Open Configuration menu to load a config)
./build/coast

# Headless benchmark (500k steps)
./build/coast -n 500000 configs/mnist.json
```

## Architecture

```
┌─────────────────────────────────┐
│   GUI Layer (gui_core)          │  SDL2 + ImGui event loop, dialogs, browsers
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Visualization (gui_vis)       │  OpenGL renderers, vis_registry factory pattern
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Simulation Core (sim_core)    │  env container, sim_thread_controller
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Neural Model (hd_ngm2)        │  neuron_group, dendrite components
└─────────────────────────────────┘
```

**Key modules:**
- `gui_core/` - SDL2 app framework, GUI components (entity/buffer/dataset browsers), config editor
- `gui_vis/` - Visualization system with factory pattern (vis_registry), colormaps, GL textures
- `sim_core/` - Simulation environment (env), IO buffers, threading controller with lock-free atomics
- `hd_ngm2/` - Neural model implementation (neurons, dendrites, neuron groups)
- `io/` - Dataset loaders (MNIST, CIFAR, dSprites, etc.) implementing io_entity interface
- `json_cfg/` - JSON configuration loading for reproducible experiments
- `tools/` - sim_builder for constructing simulations from config

## Threading Model

- **Main thread**: GUI event loop (ImGui, SDL input)
- **Simulation thread**: Dedicated thread for `process()` and `swap_io()` calls
- **Synchronization**: Lock-free atomics for mode control; hook callbacks for data updates

## Key Design Patterns

- **RAII**: Resources managed via `unique_ptr`; GPU resources need explicit `free_resources()` before context destruction
- **Factory pattern**: `vis_registry` for creating visualization instances
- **Hook system**: Deferred hook registration in `sim::env` for thread-safe GUI-sim communication
- **Type erasure**: `entity_vec` base for heterogeneous `io_entity` storage

## Live Probe System

The probe system enables real-time observation of simulation values with minimal overhead.

**Core components** (`include/core/`, `include/probe/`):
- `entity_element`: Base class providing hierarchical structure (neurons, dendrites)
- `element_type`: Type-safe identifier using `typeid`
- `structure_registry`: O(1) lookup from element pointer to structural path
- `entity_probe_set`: Thread-safe probe registration (TBB concurrent_hash_map)
- `probe_instance`: Per-probe state with SPSC queue and decimation

**Usage in simulation code:**
```cpp
// In entity_element subclass (neuron, dendrite, etc.)
probe("activity", activity_value);  // Records value if probe is active
```

**GUI** (`include/gui_probe/`):
- Probe Browser: Discover and activate probes via Introspection menu
- Observer Panel: View active probe visualizations (scalar, time series, histogram)

**Threading**: Each probe instance has a single producer (one sim thread) and single consumer (GUI thread). Thread-safe registration handled by TBB concurrent_hash_map.

## Style Conventions

- All naming uses `snake_case`: classes, methods, members
- Type aliases use `_t` suffix (e.g., `neuron_group_t`)
- Header guards: `COAST_FILENAME_H`
- Headers in `include/<module>/`, implementation in `src/<module>/`
- Use forward declarations in headers to reduce compile times
- Mark intentionally unused members/params with `[[maybe_unused]]`

## Documentation

The `docs/` directory contains a Hugo static site (Relearn theme) that documents the project's architecture, data flow, threading model, key abstractions, and a full file reference. Its purpose is to support the formation of a detailed mental model of the project and its code structure in the reader's mind.

**When making changes to the codebase, keep the documentation in sync.** If you add, rename, or remove modules, classes, data flows, or threading patterns, update the corresponding pages under `docs/content/`. Build with `cd docs && hugo` and verify zero warnings.

## Datasets

Download datasets using `scripts/download_datasets.sh` or manually into `data/`.
Configuration files in `configs/` reference these datasets.
