+++
title = "gui_tools"
weight = 72
+++

# gui_tools — Analysis Tools

The `gui_tools` module hosts sim-independent analysis panels: the
**Metric Explorer** (Tools menu), built on a reusable zoomable-matrix
renderer, and the **HDF5 Browser** panel (Analysis menu), a thin ImGui
renderer over the GUI-free [`h5browse`]({{% relref "h5browse" %}}) domain
module.

## Directory Layout

```
include/gui_tools/
├── matrix_canvas.h            # Zoomable colored-grid renderer (zoom/pan + headers + legend)
├── dataset_source.h           # Interchangeable source model (imported / synthetic)
├── metric_explorer.h          # "Tools > Metric Explorer" panel
└── h5_browser_window.h        # "Analysis > HDF5 Browser" panel (renders h5browse snapshots)

src/gui_tools/
├── matrix_canvas.cpp          # Cursor-centred zoom, cell fill, scaled text, pinned headers
├── dataset_source.cpp         # imported_source (wraps dataset_loader) + synthetic_source
├── metric_explorer.cpp        # Source mgmt, distortion + generator, matrix compute, histogram + sort, picker
└── h5_browser_window.cpp      # Tree+search, stats table, virtualized table, graph, histograms, pairwise, log panel
```

The HDF5 Browser panel follows the `buffer_replay_dialog` conventions
(constructed with the shared `hdf5::executor`, `render(bool*)` behind a
`gui_manager`-owned toggle) and never blocks: it consumes engine snapshots
per frame and enqueues commands. Its virtualized table is the codebase's
first `ImGuiListClipper` use; plots use ImPlot.

It reuses the shared GUI helpers from `gui_core` (`dataset_access`,
`thumbnail_grid`) and the pure tools (`tools/metric.h`, `tools/distortion.h`,
`tools/ensemble_code.h`, `tools/dataset_loader.h`) rather than duplicating dataset,
metric, or grid logic.

## matrix_canvas

A self-contained renderer for a colored cell grid. It is **stateless with respect
to the data** (the caller owns the matrix) but **stateful with respect to the
view** (zoom/pan persist across frames).

`render(const inputs&)` takes:

| Field | Meaning |
|-------|---------|
| `rows`, `cols` | matrix dimensions |
| `values` | row-major `rows*cols` floats |
| `vmin`, `vmax` | whole-matrix normalization range |
| `palette` | `colormap_t` applied live at render time |
| `row_thumbs`, `col_thumbs` | `std::span<const gl_texture>` header thumbnails (size == rows / cols) |
| `row_order`, `col_order` | optional display→underlying permutations (`null` = identity), applied to **both** cells and headers |
| `highlight_row`, `highlight_col` | display index of the selected header to outline (`-1` = none) |
| `sort_dir` | direction glyph on the selected header: `+1` ▲ ascending, `−1` ▼ descending, `0` none |

Per frame it:
- handles **cursor-centred scroll-wheel zoom** plus drag-pan;
- reads each cell and its header thumbnail through the optional `row_order` / `col_order` permutations, so a selected header reorders cells and gutters **in lockstep** without touching the caller's value buffer;
- culls to visible cells; for each, fills `AddRectFilled` with `apply_colormap(palette, (v−vmin)/(vmax−vmin))` and draws the value as **scaled text that hides below a legibility threshold**;
- draws the column-header thumbnails in the top gutter (body's X-transform, pinned Y, clipped) and the row-header thumbnails in the left gutter (body's Y-transform, pinned X, clipped) — they **scale in lockstep with the body**;
- **outlines the highlighted header** and draws its `sort_dir` ▲/▼ glyph;
- draws the corner and a min/max legend.

`render` returns an `interaction { clicked_row, clicked_col }` (display indices,
`-1` = none): a **left press+release with no past-threshold pan drag**, hit-tested
against the top / left gutters, is reported as a header click — the caller
(`metric_explorer`) maps it back to an underlying index to drive sorting. A drag
past `io.MouseDragThreshold` pans instead and reports no click.

`reset_view()` re-fits the matrix (called after a fresh compute).

## dataset_source

The interchangeable "source of datasets" abstraction behind the Metric Explorer's
combos. A `dataset_source` exposes a `label()`, a `datasets()` list of ready-named
[`dataset_handle`]({{% relref "gui_core" %}})s, and a stable `id` (the explorer
derives each source's memory footprint from its handles). It is **non-copyable** —
enumerated handles capture pointers into storage the source owns, so the source must
outlive every handle (the explorer holds sources by `unique_ptr`, giving each a
stable address).

- **`imported_source`** wraps a [`dataset_loader`]({{% relref "tools" %}}) over one
  config; it stem-prefixes handle names and remembers its canonical path (used to
  block duplicate imports).
- **`synthetic_source`** owns an `ensemble_codes` buffer and exposes exactly one
  handle whose geometry follows the explorer's global 1D/2D toggle (`S×S` vs
  `S²×1`). The handle's `get_sample` closure captures the buffer's stable base
  pointer + stride, matching the same lifetime contract.

## metric_explorer

The panel that builds and displays a distance/similarity matrix
`metric(row_i, col_j)` over two user-chosen sample sets.

### Interchangeable source model

The Metric Explorer is **fully decoupled from `sim_instance`** — it owns its
datasets, which come from interchangeable [`dataset_source`](#dataset_source)s:

- `sources_` — a `std::vector<std::unique_ptr<dataset_source>>`, each an
  `imported_source` (wraps a [`dataset_loader`]({{% relref "tools" %}}) over one
  config) or a `synthetic_source` (an in-memory set of generated ensemble codes).
  Every source is assigned a **stable `id`** when added.
- `datasets_` — the **flat concatenation** of every source's `datasets()`, where
  each entry records its `(source_id, local_idx)`. Imported handles are prefixed
  with the config stem (e.g. `mnist: MNIST`) so duplicates stay distinguishable.

The **Import datasets from config… / Generate ensemble codes… / Remove / Clear
all** controls manage this list. Importing a config whose **canonical path is
already present is blocked** with a notice (no duplicate datasets). The panel shows
a per-source memory **estimate** (and a total, highlighted red above a ~2 GB soft
threshold — labeled an estimate, never blocking), plus a "(no CPU datasets)" note
for a source that imported cleanly but yielded nothing matrix-comparable.

### Stable axis references

Each axis (`rows_`, `cols_`) holds a **stable `(source_id, local_idx)` reference**
(not a flat index), a `std::set` of selected sample indices, and the header
thumbnail textures. `resolve_axis_flat()` maps that reference to the current flat
index, or `-1` if its source is gone. On every `rebuild_dataset_list()` (after any
add/remove) each axis re-resolves: an axis whose source survives **keeps its
selection**, and only an axis pointing at the *removed* source is cleared. This is
the v2 fix to v1's reset-everything-on-removal behavior.

### Metric & compute

**Compute** copies the selected samples into owned buffers (`get_sample` may return
a span into shared scratch, so it must not alias), then fills
`values[r*cols+c] = compute_metric(kind, params, row_r, col_c)` using
[`tools/metric.h`]({{% relref "tools" %}}) (cosine similarity or Minkowski-`p`) and
scans for the whole-matrix `vmin/vmax`. Changing palette never recomputes (applied
live at render); changing metric/`p`/selection recomputes; zoom/pan never
recompute. Cross-dataset rows-vs-columns is allowed **iff** the feature dimensions
`w·h·c` match, otherwise the build is blocked with a message.

### Value histogram & header sorting

A band above the matrix pairs the setup strip (left, stretch) with a **value
histogram** (right, fixed width). After every compute / re-bin it shows the
distribution of the matrix values via [`tools/histogram.h`]({{% relref "tools" %}}):
an `nbins` slider (`[10, 100]`, default 32) sets the bucket count, and the bars are
**palette-matched to the matrix cells** (each bin colored by its center value over
the observed `[vmin, vmax]`). The x-axis is the metric's *theoretical* domain from
`metric_domain` — fixed `[-1, 1]` for cosine, `[0, max]` for Minkowski — so it never
jumps when a slice is selected.

Clicking a **row or column header** selects a single **focus** (row XOR column):

- a **column** click filters the histogram to that column and reorders the **rows**
  by their value in it; a **row** click does the mirror (filters to that row,
  reorders the **columns**);
- the default direction is **most-similar-first** (cosine descending, Minkowski
  ascending); **re-clicking** the same header reverses it;
- the focused header is outlined with a **▲/▼** glyph, and **Clear selection**
  restores the identity order + full-matrix histogram.

The sort is a pure display-order permutation (`row_order_` / `col_order_` built by
`order_by_key`) handed to `matrix_canvas`; the underlying `values_` buffer is never
reordered. A focus set on a matrix of **unchanged dimensions survives** a metric/`p`
change (the order is re-derived against the new values); a full recompute that may
change dimensions drops it.

### Distortion mode (Layer 4)

A toggle replaces the columns axis with **distorted copies of the row selection**:
the matrix becomes `R×R` with `col_j = apply_distortion(row_j)` via
[`tools/distortion.h`]({{% relref "tools" %}}) (Gaussian noise or dropout, with a
magnitude slider and seed). The column picker is hidden; column header thumbnails
show the *distorted* vectors, so the diagonal reads `metric(sampleᵢ,
distort(sampleᵢ))`. The same seeded perturbation realization is applied to every
column. (Note: distorting a non-square-`D` synthetic code with Gaussian noise lets
a little noise into the zero-padding region — bounded and harmless; square `D`,
including the default `D=256`, has no padding.)

### Synthetic ensemble codes (Layer 5)

The **Generate ensemble codes…** dialog feeds `ensemble_code_params` to
[`tools/ensemble_code.h`]({{% relref "tools" %}}) and wraps the result in a
`synthetic_source` that appears in the dataset combos like any import. A global
**Synthetic 2D** toggle (`show_synth_2d_`) switches every synthetic source's display
geometry between an `S²×1` strip and an `S×S` square; because `feature_dim = S²`
either way, the toggle rebuilds only thumbnails and **never** changes which datasets
are metric-comparable or the computed values.

### Sample picker

A modal popup backed by the shared `thumbnail_grid`: seeded **random-fill N** (via
bounded rejection sampling for distinct indices) and/or **click-to-toggle**
individual samples, with the selection set highlighted.

### Lifecycle

Because it is sim-independent, `metric_explorer` is constructed **once** in the
`gui_manager` constructor (it takes the process `hdf5::executor&`, needed by
HDF5-backed datasets such as dSprites) and **persists across sim load/unload** — it
is deliberately *not* recreated in `set_sim_instance()`. Its GPU textures (header
thumbnails, picker grid) are freed at application shutdown via
`gui_manager::free_resources()`, called from `run_gui`'s shutdown hook while the GL
context is still current.
