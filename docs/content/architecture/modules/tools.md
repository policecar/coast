+++
title = "tools"
weight = 55
+++

# tools — Simulation Builder

The `tools` module contains `sim_builder`, which constructs a complete simulation from a parsed configuration.

## Directory Layout

```
include/tools/
├── sim_builder.h              # Builder class
├── sim_builder_cfg.h          # Node types, entity type enum
├── sim_builder_interface.h    # Entity registration and factory functions
└── inc_on_exit.h              # RAII scope guard (increments counter on exit)

src/tools/
└── sim_builder.cpp            # build_local() implementation
```

## sim_builder

Maintains two registries of factory functions:
- **`create_func`** — Creates an entity in `sim_env` from a config name
- **`init_func`** — Initializes an entity's IO wiring from an input map

And a node map tracking the resolved simulation graph.

### build_local() — The 5-Phase Build

1. **Create direct nodes** — Copy `sim_node_t` entries from config
2. **Build derived nodes** — Expand `sim_node_set_t` templates with ID offsets and deltas
3. **Create entities** — Call `create_func` for each node, emplacing entities into `sim_env`
4. **Initialize IO** — Call `init_func` with the resolved input map for each entity
5. **Wire buffers** — Call `sim_env.init_io_buffers()` to create double buffers and connect accessors

### Entity Registration (sim_builder_interface.h)

`register_entity_handlers(sim_builder&, json_loader&)` registers factory pairs for all entity types:

**neuron_group**: Looks up `ngm2::cfg_store`, creates `neuron_group_t`, then calls `init_io()` with input map.

**Datasets** (MNIST, CIFAR, etc.): Looks up the dataset-specific config map, creates the IO entity, then calls `init_structure()`.

## inc_on_exit

Simple RAII scope guard: increments a `size_t&` counter in the destructor. Non-copyable, non-movable. Used in visualization code to ensure `step_count_` advances on any function exit path.
