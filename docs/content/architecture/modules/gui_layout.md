+++
title = "gui_layout"
weight = 85
+++

# gui_layout — Graph Layout Engine

The `gui_layout` module is a general-purpose force-directed graph layout system with Barnes-Hut optimization. Used by the buffer replay system.

## Directory Layout

```
include/gui_layout/
├── layout_graph.h             # Graph data structure (nodes, edges)
├── force_layout.h             # Barnes-Hut O(n log n) force simulation
└── render/
    ├── node_renderer.h        # Abstract + debug node renderer
    └── graph_canvas.h         # ImGui widget with zoom/pan/interaction

src/gui_layout/
├── layout_graph.cpp
├── force_layout.cpp           # Quadtree construction + force computation
└── render/
    ├── node_renderer.cpp
    └── graph_canvas.cpp
```

## layout_graph

Graph data structure with `node_t` (id, position, velocity, radius, mass, fixed flag, user_data as `std::any`) and `edge_t` (from, to, strength, ideal_length). Maintains an `id_to_index_` hash map for O(1) node lookup.

## force_layout (Barnes-Hut)

Sophisticated force-directed layout using a quadtree for O(n log n) repulsion:

**Per step**:
1. Rebuild quadtree (bounds → root quad_node → insert all nodes → compute mass centers)
2. Compute repulsion (tree traversal with `theta = 0.8` approximation threshold)
3. Compute attraction (spring forces along edges)
4. Compute gravity (optional centering force)
5. Enforce velocity limits, apply damping

**Parameters**: `repulsion=5000`, `attraction=0.01`, `damping=0.85`, `theta=0.8`, `max_velocity=50`.

**`settle(energy_threshold, max_iterations)`**: Runs until kinetic energy drops below threshold — useful for initial layout convergence.

## graph_canvas

ImGui widget for interactive graph visualization:

- **Rendering**: Grid background, edges (optional arrows), nodes via `node_renderer`
- **Interaction**: Mouse wheel zoom (exponential), drag pan (middle/right button), node selection (left click), node dragging
- **Coordinate system**: `world_to_screen(x,y)` = `canvas_pos + (world_pos + pan) * zoom`
- **Smooth animation**: Optional linear interpolation for zoom/pan

## node_renderer

Abstract base with `render()` and `get_size()`. The `debug_node_renderer` draws filled circles with configurable colors and shows labels when zoomed in enough.
