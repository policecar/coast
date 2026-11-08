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
- `read_frame_range(buffer_id, start_frame, count, out)` — Consecutive frames of one buffer
- `get_step(frame_idx)` / `get_timestamp(frame_idx)` — Lazy-loaded time data

Topology includes: buffer IDs, sizes, names, connectivity edges, optional vis hints. `buffer_logger` writes the vis hints guessed from each buffer's size (square → image, RGB if divisible by 3, else common widths); the config's `vis_hints` are not consulted, since they are keyed by config entry and the logger sees only entity types.

Every dataset is shape-checked against what `buffer_logger` writes (1-D ids, sizes,
names, steps and timestamps; `[n, 2]` connectivity; `[n, 3]` vis hints;
`[frames, size]` data) before a buffer is sized from it — see "Reading files COAST
did not write" in the logging module. Every check runs once, in `open()`: a mismatch
refuses the file, with the object named in the error log. The exception is the
optional vis hints, which only steer the display: hints of another shape or a
non-numeric type are skipped with a warning and the buffers get computed defaults.

## replay_node_renderer

Extends `node_renderer` with three visualization modes:

| Mode | Rendering |
|------|-----------|
| `image_2d` | Reshape buffer to 2D grid, apply colormap (grayscale) or direct RGB |
| `stats_color` | Single color based on aggregate stat (avg/sum/min/max) |
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
