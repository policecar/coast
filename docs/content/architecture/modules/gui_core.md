+++
title = "gui_core"
weight = 70
+++

# gui_core — Application Framework

The `gui_core` module is the SDL2/ImGui application framework: the main window, event loop, menu system, browser windows, and editors.

## Directory Layout

```
include/gui_core/
├── sdl_app.h                  # SDL2 + OpenGL3 + ImGui lifecycle
├── gui_manager.h              # Main GUI orchestration (menus, dialogs)
├── sim_instance.h             # Wraps a complete simulation
├── entity_browser.h           # Entity table + visualization panel
├── buffer_browser.h           # IO buffer table + live stats
├── dataset_browser.h          # Thumbnail grid for dataset samples
├── config_editor.h            # JSON config editing GUI
├── network_editor.h           # Node-graph editor for simulation topology
├── logging_window.h           # Logging controls (buffer, probe, state)
└── imgui_scaling.h            # HiDPI scaling helper: scaled(base_value)

src/gui_core/
├── sdl_app.cpp                # SDL init, ImGui init, main loop, shutdown
├── gui_manager.cpp            # Menu bar, control dialog, status bar
├── sim_instance.cpp           # Config loading, entity construction
├── entity_browser.cpp         # Sortable/filterable table, vis management
├── buffer_browser.cpp         # Buffer table with live statistics
├── dataset_browser.cpp        # Virtualized thumbnail rendering
├── config_editor.cpp          # Full config editing UI (2300+ lines)
├── network_editor.cpp         # ImNodeFlow integration
└── logging_window.cpp         # Buffer/probe/state logging controls
```

## sdl_app

Manages the SDL2 + OpenGL lifecycle:

1. **Construction**: Init SDL2, create OpenGL context (3.0/3.2), create window (1280x800, maximized), init ImGui + ImPlot, load font with HiDPI scaling
2. **run()**: Main loop — poll events, state update callbacks, ImGui frame, GUI callbacks, render, swap
3. **Destruction**: Cleanup ImGui, destroy GL context, destroy window, SDL_Quit

Provides callback registration: `register_state_func()`, `register_gui_func()`, `register_shutdown_func()` — each returns an ID for later deregistration.

## gui_manager

Orchestrates all GUI windows and dialogs. Holds:

**Browsers**: `entity_browser`, `buffer_browser`, `dataset_browser` (heap-allocated, recreated on sim load)

**Panels**: `logging_window`, `probe_browser`, `observer_panel`, `buffer_replay_dialog` (value members)

**Editors**: `config_editor` (heap-allocated)

### Menu Structure

| Menu | Items |
|------|-------|
| **System** | Open Configuration (Ctrl+O), Config Editor, Save/Load State (Ctrl+S/L), Exit |
| **Simulation** | Control, Status |
| **Introspection** | Sim Entities, IO Buffers, Dataset, Logging, Performance, Probe Browser |
| **Analysis** | Buffer Replay |

### set_sim_instance()

Called when a new config is loaded. Frees old browser GL resources, creates new browsers, initializes probe system, creates config editor with reload callback, checks for existing snapshots and offers resume.

### Auto-save

Detects running→paused transitions. Saves state to `<snapshot_dir>/<config>_<timestamp>_<step>.h5`. Snapshot directory is configurable per-config JSON.

## sim_instance

Wraps a complete running simulation in construction order (determines destruction order):

1. `json_loader` — Parse config file
2. `sim_env` — Empty environment
3. `sim_builder` — Register handlers, build from config
4. `thread_controller` — Start sim thread (paused)
5. `logging_manager` — Create logging coordinator

Destruction is reverse: logging stops, thread joins, then env is destroyed.

## Entity Browser

Sortable, filterable table of all simulation entities. Features:
- Column sorting (ID, Name, Type, Config, Output Size, Input Count)
- Substring filter bar
- **Detail panel** with draggable splitter (ratio 0.2–0.9)
- Visualization checkboxes from `vis_registry` — toggle creates/destroys `vis_instance`
- Active visualizations update every frame, even when browser is hidden

## Buffer Browser

Same pattern as entity browser but for IO buffers. Shows live statistics (avg, min, max) refreshed every frame. Detail panel with buffer visualization checkboxes.

## Dataset Browser

Thumbnail grid with virtualized scrolling — only visible rows (+ 2-row buffer) are rendered. Features:
- Dataset selector combo (auto-detects entity types)
- 64x64 thumbnails via nearest-neighbor resampling
- Lazy texture cache
- Hover tooltip with sample index

## Config Editor

Full editing UI for `global_cfg_t` with sections for datasets, NGM2, simulation graph, logging, and vis hints. Features dirty tracking, save/restore, and a reload callback.

## Network Editor

Graphical node-graph editor built on ImNodeFlow. Config editor is the single source of truth — user edits are validated through callbacks before being applied.

## Logging Window

See [logging module]({{% relref "logging" %}}) for the underlying system. The window provides:
- Buffer logging controls (path, start/stop, status)
- Probe logging controls (path, format selector, start/stop)
- State save/load with file browsers
- Auto-checkpoint with configurable interval

## imgui_scaling

`scaled(base_value)` = `base_value * (ImGui::GetFontSize() / 16.0f)`. Ensures all pixel sizes scale with font size for HiDPI support.
