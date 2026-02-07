+++
title = "gui_replay"
weight = 90
+++

# gui_replay — Buffer Replay

The `gui_replay` module enables offline playback of recorded buffer activity from HDF5 log files, displaying the simulation as an interactive force-directed graph.

## Directory Layout

```
include/gui_replay/
├── buffer_log_reader.h        # HDF5 reader for buffer_log files
├── buffer_replay_dialog.h     # Main replay dialog with playback controls
└── replay_node_renderer.h     # Node renderer with 3 visualization modes

src/gui_replay/
├── buffer_log_reader.cpp      # Frame-by-frame HDF5 reading
├── buffer_replay_dialog.cpp   # Full replay UI
└── replay_node_renderer.cpp   # Image, stats, and delta rendering
```

## buffer_log_reader

Reads HDF5 files produced by `buffer_logger`. Provides random access by frame index:

- `open(path)` — Reads metadata and topology
- `read_frame_data(buffer_id, frame_idx, out)` — Single frame via hyperslab selection
- `read_frame_stats(buffer_id, frame_idx)` — Frame statistics (sum, avg, min, max, nse)
- `get_step(frame_idx)` / `get_timestamp(frame_idx)` — Lazy-loaded time data

Topology includes: buffer IDs, sizes, names, connectivity edges, optional vis hints.

## replay_node_renderer

Extends `node_renderer` with three visualization modes:

| Mode | Rendering |
|------|-----------|
| `image_2d` | Reshape buffer to 2D grid, apply colormap (grayscale) or direct RGB |
| `stats_color` | Single color based on aggregate stat (avg/sum/min/max/nse) |
| `delta_flow` | Blue→red gradient showing change from previous frame |

Each buffer node stores its data in `buffer_node_data` (inside `node_t::user_data` as `shared_ptr`), holding the GL texture, current/previous frame data, and per-buffer visualization config.

## buffer_replay_dialog

Main dialog composing all components:

**UI sections**:
- **Menu bar**: File (Open/Close), View (panels, fit graph), Playback (transport)
- **Toolbar**: Rewind/prev/play-pause/next/fast-forward, speed slider, frame counter
- **Graph view**: `graph_canvas` with force-directed layout of buffer nodes
- **Config panel**: Vis mode, colormap, normalization, layout physics controls
- **Info panel**: File metadata, selected buffer details with editable dimensions
- **Timeline**: Frame slider at bottom

**Graph construction**: Distributes buffer nodes in a grid, creates edges from connectivity, settles the force layout.

**Playback**: Timer-based frame advancement at configurable speed (1-120 fps). Reads all buffer data per frame from HDF5.
