+++
title = "gui_observe"
weight = 82
+++

# gui_observe — Observe GUI

The `gui_observe` module provides the GUI front-end for the [observe]({{% relref "/architecture/modules/observe" %}}) system: a panel for inspecting and controlling capture groups at runtime.

## Directory Layout

```
include/gui_observe/
└── observe_panel.h        # ImGui panel for observe group control

src/gui_observe/
└── observe_panel.cpp      # Panel render: state badge, range editing, trigger buttons
```

## Observer Capture Panel

Opened from `Introspection → Observer Capture`. It lists every configured group with:

- **Live state badge** — `ACTIVE` / `IDLE`. This reads the group's single `active_this_step` flag — the truthful "capturing right now" signal, recomputed each pass from ranges, combine, and triggers.
- **Site count** and **output path**.
- **Editable inclusive ranges** — add / remove / modify `[start, stop]` intervals.
- **`Combine` mode selector** — `OR` / `AND`.
- **Manual trigger buttons** — `START` / `STOP` / `SINGLE_CAPTURE`, routed through `observe_registry::on_trigger`.
- **Add-group field** — create a new group at runtime.

## Thread Safety

All mutations to ranges and combine mode are serialised through `observe_registry::with_group_locked()`, so the GUI thread does not race with the sim thread's pre-/post-process hooks. The trigger buttons call `observe_registry::on_trigger`, which is itself thread-safe and may be invoked from any thread.
