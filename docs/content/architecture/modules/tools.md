+++
title = "tools"
weight = 55
+++

# tools — Simulation Builder

The `tools` module contains `sim_builder` (constructs a complete simulation from a
parsed configuration), `dataset_loader` (constructs *only* a config's datasets),
and the header-only `metric`, `metric_registry`, `distortion`, `ensemble_code`, and
`histogram` functions that back the Metric Explorer and the
[HDF5 Browser]({{% relref "/architecture/modules/h5browse" %}})'s pairwise view.

## Directory Layout

```
include/tools/
├── sim_builder.h              # Builder class
├── sim_builder_cfg.h          # Node types, entity type enum, is_*_entity_type predicates
├── sim_builder_interface.h    # Entity registration and factory functions
├── dataset_loader.h           # Dataset-only build from a config (no sim)
├── metric.h                   # Header-only metric functions (cosine, Minkowski)
├── metric_registry.h          # Enumerable metric_def metadata over metric.h (D9b)
├── distortion.h               # Header-only seeded distortions (Gaussian noise, dropout)
├── ensemble_code.h            # Header-only synthetic ensemble-code generator
├── histogram.h                # Header-only value binning + index ordering (Metric Explorer)
└── inc_on_exit.h              # RAII scope guard (increments counter on exit)

src/tools/
├── sim_builder.cpp            # build_local() implementation
└── dataset_loader.cpp         # Dataset-only config filter + build
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
4. **Initialize IO** — Call `init_func` with the resolved input map for each entity. The map is keyed by channel name, so a node listing the same channel name twice is refused (`std::runtime_error`) rather than silently keeping only the last link. This includes node sets: to rewire a template input in a set, give its delta a `remove` of the old link next to the `add` of the new one; an `add` alone leaves both links and fails the build. The config editor refuses a second link on a wired channel and flags repeated names
5. **Wire buffers** — Call `sim_env.init_io_buffers()`, which classifies each buffer's variant from entity devices and connects writers/readers

> When any node is a GPU entity, the GPU context must be installed (`env::enable_gpu()`) **before** step 5 — `init_io_buffers()`'s topology pre-pass needs it to construct GPU buffer variants and throws if a GPU variant is required but no context is present. GPU entity factories receive the `gpu_context&` so they can allocate their device buffers.

### Entity Registration (sim_builder_interface.h)

`register_entity_handlers(sim_builder&, json_loader&, hdf5::executor&, seed_offset_t seed_offset = {})` registers a create/init pair for every entity type, one line per type over small helpers:

- **create** — `register_group<Group>` (ngm2, ngm3, ngm25, and the GPU groups) builds the group from its model's whole config store; a GPU group shares its CPU sibling's config block. `register_params<Entity>` (datasets, retina) looks up the node's named parameter set in its family's map, adds the run's and the node's seed offsets to its seed and constructs the entity; an unknown name throws `Unknown <family> configuration: <name>`, the family taken from the entity-type table. dSprites additionally receives the `hdf5::executor&`.
- **init** — `register_init_io<Entity>` calls the entity's `init_io()` with the input map (groups, retina); `register_init_structure` only builds the structure registry (datasets).
- `register_routed_group` and `register_dataset` pair the two for the common cases.

`sim_builder::handles(type)` reports whether both functions are registered; `unit.tools.entity_registration` checks it for every enumerator, so a missing registration fails a test rather than only a run whose config uses the type.

### The entity-type table (sim_builder_cfg.h)

`sim::entity_types` holds one row per `io_entity_type_t` — config name, display name, kind (neuron group / dataset / retina), whether it runs on the GPU, and a dataset's config family. A `static_assert` keeps rows and enumerators in step. Derived from it: `is_gpu_entity_type()`, `is_dataset_entity_type()`, `is_neuron_group_entity_type()`, `entity_type_display_name()`, the JSON names (`COAST_JSON_STRICT_ENUM_RANGE`), the run manifest's active datasets, and the GUI's type checks. Adding a type is an enumerator, a row, a registration line and — for a dataset — its family in `dataset::cfg_store` (`for_each_family`; `unit.tools.entity_types` checks the families agree).

The GPU entity factories are registered only under `COAST_WITH_GPU`.

## dataset_loader

Loads **only** the dataset (`io_entity`) objects described by a JSON config — no
neuron groups, no sim thread, no logging, no GPU context. It reuses the normal
pipeline (`json_loader` → `sim_builder` → `register_entity_handlers` →
`build_local`) but feeds `build_local` a config **filtered to CPU dataset nodes
only**:

- neuron-group nodes are dropped (so no ngm config is required);
- GPU dataset nodes are dropped (so no `gpu_context` is installed — they are not
  matrix-resolvable anyway);
- `node_set`s referencing any dropped node are dropped (so `build_local` cannot
  throw on a missing template node).

Registering *all* handlers is harmless because `build_local` only invokes the
factory for a node that survives the filter. The filter uses
`is_dataset_entity_type()` / `is_gpu_entity_type()` from `sim_builder_cfg.h`. The
constructed entities live in the loader's own `sim::env` for its lifetime, so
`dataset_handle`s enumerated from `env()`/`builder()` (via
[`enumerate_datasets`]({{% relref "gui_core" %}})) stay valid exactly as long as
the loader does. Its intended owner is the Metric Explorer (one loader per imported
config). `dataset_loader` is non-copyable and non-movable — handle accessors
capture pointers into the owned env, so its address must be stable.

## metric

Header-only (`inline`, no TU) pure functions in `namespace coast`, so the GUI panel
and the unit test both just `#include` it:

```cpp
enum class metric_kind { cosine_similarity, minkowski_distance };
struct metric_params { float p = 2.0f; };               // p used by Minkowski only
float compute_metric(metric_kind, const metric_params&,
                     std::span<const float> a, std::span<const float> b);

struct value_range { float lo, hi; };
value_range metric_domain(metric_kind, float observed_max);   // histogram x-domain
```

Preconditions: `a.size() == b.size()`, `p > 0`. As a cold-path GUI utility it
**throws** `std::invalid_argument` on violation (rather than the sim hot-path
`COAST_REQUIRE`/`std::expected` channel). Cosine guards a zero-norm vector
(returns 0); Minkowski is `(Σ|aᵢ−bᵢ|^p)^(1/p)`. `metric_domain` returns the metric's
*theoretical* value range for use as a fixed histogram x-axis — cosine is the
bounded `[-1, 1]`, Minkowski is `[0, observed_max]` (falling back to `[0, 1]` when
the max is non-positive). Pinned by the `unit.tools.metric` and
`unit.tools.histogram` tests.

## metric_registry

Header-only (`namespace coast`) table of `metric_def` records — one entry per
`metric_kind` — giving GUI panels enumerable metadata for building metric
dropdowns and parameter widgets without hard-coding `metric_kind` knowledge
per caller:

```cpp
struct metric_def {
    const char* name;                     // UI display name
    metric_kind kind;
    std::optional<value_range> fixed_domain;  // e.g. cosine: [-1, 1]; nullopt = adaptive
    bool has_param; const char* param_name;
    float param_default, param_min, param_max;  // Minkowski's p
};
std::span<const metric_def> metric_registry();
```

Computation stays in `metric.h` (`compute_metric`) — this header only
*describes* the available metrics, so adding one means adding a `metric_def`
entry here plus the `metric_kind` case in `metric.h`, and every consumer picks
it up automatically. First consumer: the HDF5 Browser's pairwise metric
dropdown; the Metric Explorer's own metric combo is a candidate for a future
refactor onto this same table (currently it still switches on `metric_kind`
directly). Pinned by `unit.tools.metric_registry`.

## distortion

Header-only (`inline`, `namespace coast`) seeded distortions that back the Metric
Explorer's **distortion mode** (columns = the row selection passed through a
controlled corruption):

```cpp
enum class distortion_kind { gaussian_noise, dropout };
struct distortion_params { float magnitude = 0.1f; std::uint32_t seed = 0; };
void apply_distortion(distortion_kind, const distortion_params&,
                      std::span<const float> in, std::vector<float>& out);
```

- **Gaussian noise** adds zero-mean noise (`σ = magnitude`) per element, clamped to
  `[0,1]`; `σ = 0` is the identity.
- **Dropout** zeroes exactly `round(p·n)` distinct elements (`p = magnitude`) via a
  partial Fisher–Yates; `p = 0` is the identity, `p = 1` zeroes everything.

`apply_distortion` seeds a fresh `std::mt19937` from `params.seed`, so the result is
reproducible and (in distortion mode) the *same* perturbation realization is applied
to every column. Cold-path: precondition violations **throw** `std::invalid_argument`.
Pinned by `unit.tools.distortion`.

## ensemble_code

Header-only (`inline`, `namespace coast`) generator for the Metric Explorer's
**synthetic ensemble codes** (Layer 5): vectors shaped to look like neuron-group
population outputs, with a *controllable* ground-truth cluster structure.

```cpp
struct ensemble_code_params { int D, N, M, k; float swap_frac, amp_var,
                              inhibition, noise, recognition; std::uint32_t seed; };
struct ensemble_codes { std::vector<float> data; int side; int count; };
ensemble_codes generate_ensemble_codes(const ensemble_code_params&);
```

Each of the `N` codes is sparse — `k` high-amplitude "winners" near 1, the rest
suppressed toward a bounded (`≤ 3·noise`) background. Codes are drawn from `M`
distinct prototypes, so a cosine matrix over the result shows `M` separable blocks
(the unit test pins *mean same-prototype cosine > mean cross-prototype cosine*).
`recognition = 0` yields a winner-free low-amplitude noise code. Geometry: `D` real
entries zero-padded to `S² = ceil(√D)²`, so a code can be shown as an `S×S` square
or an `S²×1` strip — the padding is metric-neutral for cosine/Minkowski (shared
trailing zeros add nothing). Cold-path: precondition violations **throw**. Pinned by
`unit.tools.ensemble_code`.

## histogram

Header-only (`inline`, `namespace coast`) pure helpers that back the Metric
Explorer's **value-distribution histogram** and **clickable-header sorting**:

```cpp
std::vector<int> histogram_bins(std::span<const float> vals, int nbins,
                                float lo, float hi);
std::vector<int> order_by_key(std::span<const float> keys, bool descending);
```

- **`histogram_bins`** buckets `vals` into `nbins` equal-width bins over the closed
  domain `[lo, hi]` (`bin(v) = clamp(floor((v−lo)/(hi−lo)·nbins), 0, nbins−1)`); the
  counts **sum to `vals.size()`** — values outside the domain clamp into the end
  bins rather than being dropped, and a degenerate `hi == lo` puts everything in
  bin 0. The domain comes from [`metric_domain`](#metric).
- **`order_by_key`** returns a **stable** display→underlying index permutation that
  sorts `[0, n)` by `keys` (ascending or descending). The Metric Explorer uses it to
  reorder the matrix rows / columns against a selected header's value slice.

Cold-path: `histogram_bins` **throws** `std::invalid_argument` on `nbins < 1` or
`hi < lo`. Pinned by `unit.tools.histogram`.

## inc_on_exit

Simple RAII scope guard: increments a `size_t&` counter in the destructor. Non-copyable, non-movable. Used in visualization code to ensure `step_count_` advances on any function exit path.
