+++
title = "gui_vis"
weight = 75
+++

# gui_vis — Visualization System

The `gui_vis` module provides the visualization framework: a factory registry, core rendering primitives, and concrete visualization implementations.

## Directory Layout

```
include/gui_vis/
├── vis_registry.h             # Factory pattern: register + create visualizations
├── core/
│   ├── color_types.h          # RGBA color struct
│   ├── colormap.h             # Colormap enum + mapping functions
│   ├── gl_texture.h           # RAII OpenGL texture wrapper
│   ├── gradient.h             # N-stop linear color gradient
│   └── vis_hints.h            # Visualization config from JSON
├── primitives/
│   ├── vec_vis.h              # Single float vector → 2D pixel grid
│   └── vec_group_vis.h        # Multiple vec_vis in layout
├── io_buffer/
│   └── vec_ring_buffer.h      # Scrolling time-series visualization
├── layout/
│   └── fbgd.h                 # Force-based graph drawing (original)
└── sim_entity/neuron_group/
    ├── ngm_flat_vis.h         # "Dendritic Representations" grid view
    └── ngm_tree_vis.h         # "Neuron Tree" force-directed tree

src/gui_vis/
├── vis_registry.cpp           # Registry singleton + registration logic
├── core/
│   ├── gl_texture.cpp
│   └── colormap.cpp           # Viridis, plasma, inferno, etc.
├── primitives/
│   ├── vec_vis.cpp
│   └── vec_group_vis.cpp
├── io_buffer/
│   └── vec_ring_buffer.cpp    # Ring buffer rendering
├── layout/
│   └── fbgd.cpp               # Grid-based force simulation
└── sim_entity/neuron_group/
    ├── ngm_flat_vis.cpp       # 545 lines: texture grid + zoom/pan
    └── ngm_tree_vis.cpp       # 1001 lines: tree layout + textures
```

## Visualization Registry

**Factory pattern** for creating visualization instances. Uses Meyer's singleton (`get_vis_registry()`).

### Core Types

- **`vis_info`** — Metadata: id, display_name, description, target (entity/buffer), supported entity types
- **`vis_instance`** — Abstract base: `update()`, `render_imgui()`, `free_resources()`, visibility control
- **`entity_vis_factory`** / **`buffer_vis_factory`** — Callables that create `vis_instance` from entity/buffer + hints

### Registration Flow

Each visualization type has a file-scope static registrar that calls `register_entity_vis(info, factory)` at program startup (before `main()`).

### Query API

- `get_entity_visualizations(type)` — Returns matching vis_info entries for an entity type
- `get_buffer_visualizations()` — Returns all buffer visualization options
- `create_entity_vis(id, entity, env, hints, key, label)` — Factory invocation

## Visualization Hints (vis_hints_t)

Hierarchical config from JSON, matching entity names to rendering parameters:

```
vis_hints_t
├── datasets: dataset_hints_t
│   ├── mnist: map<string, mnist_hint_t>    (default: 28x28x1)
│   ├── cifar10: map<string, cifar10_hint_t> (default: 32x32x3)
│   └── ...
├── ngm2: ngm2_hints_t
│   └── neuron_groups: map<string, neuron_group_hints_t>
│       ├── ngm_flat_vis: {width, height, num_channels, update_interval}
│       └── ngm_tree_vis: {rep_width, rep_height, colors, radii, ...}
└── buffer_vis: {update_interval, ring_length}
```

Each map has a `"default"` fallback entry.

## Concrete Visualizations

### ngm_flat_vis — "Dendritic Representations"

Flat 2D grid of all neuron dendritic representations. Each representation is a colormap-mapped texture showing synapse permanence values.

- Zoomable/pannable canvas with smooth interpolation
- Colormap selector (grayscale, viridis, plasma, inferno, etc.)
- Thread-safe update via 3-state atomic FSM: `idle → requested → updating → idle`

### ngm_tree_vis — "Neuron Tree"

Force-directed tree showing the dendritic binary tree structure per neuron:
- Root nodes (orange circles) = neurons
- Branch nodes (blue circles) = internal tree segments
- Leaf textures = synapse permanence patterns
- Adaptive layout: larger trees get stronger repulsion
- Click to select a neuron and see its leaf textures

### vec_ring_buffer — "Time Series"

Scrolling 2D texture: columns = time steps, rows = buffer elements. Auto-normalized, colormap-applied. Configurable update interval.

## Core Primitives

| Class | Purpose |
|-------|---------|
| `gl_texture` | RAII OpenGL texture (move-only, `as_imgui_texture()`) |
| `colormap` | 6 colormaps: grayscale, green, viridis, plasma, inferno, cool |
| `gradient<N>` | N-stop linear color interpolation |
| `vec_vis` | Float vector → 2D pixel grid (grayscale/colormap or RGB) |
| `vec_group_vis` | Multiple vec_vis in vertical/horizontal/grid layout |

## fbgd — Original Force Layout

Grid-based spatial decomposition for the ngm_tree_vis. Fixed-region force simulation with:
- Cell-based neighbor queries
- Configurable repulsion/attraction forces
- Node target position animation for smooth tree repositioning
