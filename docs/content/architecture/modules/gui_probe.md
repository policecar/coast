+++
title = "gui_probe"
weight = 80
+++

# gui_probe — Probe GUI

The `gui_probe` module provides the GUI for the probe system: a tree browser for discovering probes and a panel for viewing active observers.

## Directory Layout

```
include/gui_probe/
├── probe_browser.h            # Tree view for probe discovery + activation
└── observer_panel.h           # Single-window view of all active observers

src/gui_probe/
├── probe_browser.cpp          # Tree rendering, observer window management
└── observer_panel.cpp         # Collapsible headers for active probes
```

## Probe Browser

Tree view browser that mirrors the entity hierarchy. For each entity → element → probe point:

1. **Discovery**: Collects probe names from two sources:
   - Static: `structure_entry::probe_names` (from `ELEMENT_PROBES` macro)
   - Dynamic: `probe_name_registry` (from `PROBE` macro, populated on first sim step)

2. **Controls per probe**:
   - Checkbox: activate/deactivate the probe
   - Decimation: record every Nth call (reduces data volume)
   - "+ Obs" button: add an observer (Scalar, Time Series, Histogram, CSV File)

3. **Observer windows**: Each observer opens in its own ImGui window with type-appropriate initial size (scaled for HiDPI). Windows are tracked in `observer_windows_` vector. Closing a window removes the observer; when the last observer is removed, the probe is deactivated.

4. **CSV integration**: "CSV File" observer opens a file browser to select the output path.

### update()

Called every GUI frame. Drains all entity probe SPSC queues by calling `entity.probes().update_observers()`.

## Observer Panel

Alternative view: shows all active probe observers in a single window with collapsible headers. Each section shows the probe path, decimation info, and its observers with remove buttons. Deactivates probes when the last observer is removed.
