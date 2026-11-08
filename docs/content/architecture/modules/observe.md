+++
title = "observe"
weight = 62
+++

# observe — Offline Data Capture

The `observe` module records user-instrumented values from simulation entity code to per-group HDF5 files, gated by configurable ranges and programmatic triggers. It is COAST's fourth introspection mechanism, alongside the io-buffer logger and `state_logger` (both in [logging]({{% relref "/architecture/modules/logging" %}})) and the live [probe]({{% relref "/architecture/modules/probe" %}}) system — and it is the one dedicated to **offline analysis** of an experimental run.

## observe vs. probe

| | observe | probe |
|---|---|---|
| Purpose | offline analysis | live GUI inspection |
| Output | per-group HDF5 files | in-memory ring buffers |
| Activation | config ranges + programmatic triggers | user-toggled in GUI |
| Capture mechanism | post-process `perform_observations` pass | inline `probe()` in entity code |
| Hot-path cost when off | **nothing in entity code** — the pass early-returns when no group is configured | one relaxed atomic check + queue head |
| Multi-instance | namespaced via `entity_element` structural path | hierarchical browser |

The two are orthogonal and can run simultaneously. See the [Observe Pipeline]({{% relref "/data-flow/observe-pipeline" %}}) for the end-to-end data flow.

## Capture model: the post-process observation pass

Unlike `probe()` (which is called inline from hot `process()` code), observe runs as a **single post-process pass** on the sim thread, once per step, after every entity's `process()` has finished and — on the GPU path — after the device has gone idle. The pass walks the entity tree and lets each element emit its observed values by reading the state it has already computed for the step.

This removes the old inline `OBSERVE(...)` macro entirely. The mechanism is one virtual on `core::entity_element`:

```cpp
// core::entity_element — default recurses into children
virtual void perform_observations(observe::observe_accessor& acc) {
    for (const auto& g : children())
        for (std::size_t i = 0; i < g.count; ++i)
            if (auto* c = child_impl(g.type.info(), i))
                c->perform_observations(acc);
}
```

An entity that has something to observe overrides this, emits its own values, then chains to the base to recurse:

```cpp
void neuron_group_t::perform_observations(observe::observe_accessor& acc) {
    if (acc.group_active("population"))
        acc.observe(*this, "population", "out", std::span<const float>(out));
    entity_element::perform_observations(acc);   // recurse: neurons -> dendrites
}
```

Because emission reads end-of-step state, an override only needs to *store* a value that `process()` would otherwise discard. In HD-NGM2 exactly two dendrite values are transient (`segment_activity`, `attenuation`) and are kept as members for the pass to read; every other site reads retained state directly. The dataset loaders similarly stash `observed_index_` before `cur_idx` advances.

## The `observe_accessor`

The pass hands each override an `observe_accessor&` — a thin handle over the registry. Its full surface:

```cpp
// scalar (float/double/(u)intN_t)
template<class T> void observe(entity_element& e, string_view group, string_view id, const T& value);
// 1-D span -> per-row shape [N]
template<class T> void observe(entity_element& e, string_view group, string_view id, span<const T> values);
// N-D span + explicit shape -> per-row shape [d0, d1, ...]  (product(shape) == values.size())
template<class T> void observe(entity_element& e, string_view group, string_view id,
                               span<const T> values, span<const size_t> shape);

bool group_active(string_view group) const;                 // cheap; skip expensive work when false
void trigger(string_view group, trigger_mode_t mode, bool cond);
```

`observe(...)` captures `e` as the owning element, used for the structural-path prefix and per-owner site resolution. `group_active(group)` is a lock-free read of the group's per-step commit decision — call it before any expensive value computation (e.g. a GPU→CPU readback) so dormant groups cost nothing beyond the check.

### Supported value types

Defined in `observe_dtype.h`:

| C++ type | HDF5 dtype | Per-row shape |
|---|---|---|
| `float` | `H5T_NATIVE_FLOAT` | scalar |
| `double` | `H5T_NATIVE_DOUBLE` | scalar |
| `(u)int8/16/32/64_t` | `H5T_NATIVE_(U)INT(N)` | scalar |
| `std::span<T>` (1-D overload) | as `T` above | `(values.size(),)` — runtime, locked on first call |
| `std::span<T>` + shape (N-D overload) | as `T` above | `(d0, d1, …)` — explicit, up to rank 8 |

Any other type fails to compile cleanly — no silent misinterpretation. Add a `dtype_traits<T>` specialisation to support a new type.

**Shape lock**: the first call against a `(group, full_id)` pair locks the dtype, shape, and byte count. A later mismatching call asserts in debug, aborts with a clear message in release.

### Built-in HD-NGM2 groups

The model ships pre-instrumented sites in three granularity tiers (one file per enabled group):

| group | sites | cost / step |
|---|---|---|
| `population` | `in` (the group's input channels concatenated in their declared `input_channels` order), `out` (post-inhibition group output vector) | two vectors per neuron group — cheap |
| `neuron` | `activity`, `dendrite_count` | one scalar per neuron |
| `dendrite` | `segment_activity`, `synapse_count`, `attenuation`, `representation_count` | per dendrite per neuron — the heavy tier |

For a frozen evaluation that only needs the population code, enable just `population`; reach for `dendrite` when debugging per-branch behavior. The hd_ngm3 GPU group guards its population emission with `acc.group_active("population")` so the GPU→CPU readback is skipped entirely when the group is dormant. (The GPU `read_probes()` path retains only the PROBE scalar readback; the population vector readback moved into `perform_observations`.)

### Additional ngm25 and ngm3 CPU sites

| Site | Dtype | Meaning |
|---|---|---|
| `population/out_pre_inhibition` | `float32`, vector of neuron count | Neuron activity before inhibition. The contract test asserts `out == inhibit(out_pre_inhibition)` using the configured inhibitor. |
| `population_structure/representation_count` | `uint64`, scalar | Group-total distinct segment IDs across dendrites (including configured dendrite banks). This is an observed structural count, not retained learning capacity. |

Neither site mutates state. `out_pre_inhibition` is a span over an existing member vector and costs nothing extra. `representation_count` is expensive: `get_total_representation_count()` is a serial, allocation-free scan of every segment id, recomputed on every active step. On a 24-core Linux box it added 0.8 ms per step to `configs/mnist_ngm3.json` and 0.19 ms to `configs/mnist_ngm25.json`, about +50–60% of the whole step; the share varies by machine (about +10–20% for ngm25 on an Apple Silicon laptop). It therefore has its own group, `population_structure`, so that a run observing `population` does not pay for it. Enable it only where the count can change or is needed: it cannot change in a frozen evaluation, so one sample of a frozen run is enough.

The CPU contract is defined in `tests/integration_observe/observe_contract.json`; real ngm25/ngm3 runs check emitted names, dtypes, shapes, the inhibition relationship and the count's value against a recount. GPU parity for these two sites is outside this contract. Frozen-state tests also include an unfrozen negative control.

## Programmatic triggers

Triggers are driven through `observe_accessor::trigger()` from inside a `perform_observations` override (the old `OBSERVE_TRIGGER` macro and registry `on_trigger()` are gone). The model is a single per-group boolean plus a one-shot:

```cpp
acc.trigger(group, trigger_mode_t::START | STOP | SINGLE_CAPTURE, cond);
```

`cond` is evaluated on every call (keep it cheap); the trigger acts only when true.

- **`START`** — `prog_active = true`. The group stays active until `STOP` or end of run. State-driven capture.
- **`STOP`** — `prog_active = false`. With no configured range the group goes dormant immediately.
- **`SINGLE_CAPTURE`** — arms `prog_single_pending`; the current pass captures and the flag is consumed at end of pass. Event-driven, one step.

Because the pass recomputes each group's `active_this_step` at its start *and* `trigger()` re-evaluates it live, an `observe()` call later in the same pass sees a trigger fired earlier in that pass. There is no sticky-gate / scratch machinery: emission is single-threaded and reads retained state, so the mid-step-race defenses that the old inline path needed no longer exist.

> **Note**: `trigger()` is re-exposed for completeness but has **no production caller** today — range-based capture is the only live activation path. Its state machine is covered by `tests/unit/observe/test_trigger.cpp`.

## JSON config schema

Parsed via nlohmann `from_json` into the `observe_cfg_t` structs of `observe_cfg.h`; `apply_config()` (in `observe_config.cpp`) seeds the registry groups.

```json
{
    "observe": {
        "output_dir": "",                   // optional LEAF; "" -> app default "observe"
        "groups": {
            "<group-name>": {
                "ranges":  [[start, stop], ...],  // optional; inclusive endpoints
                "combine": "or" | "and"           // optional; default "or"
            }
        }
    }
}
```

- `combine = "or"` (default): captured ⇔ `range_hit OR programmatic_active`. `"and"`: captured ⇔ `range_hit AND programmatic_active` (with no range, AND degrades to programmatic-only).
- Unknown keys are ignored. Malformed entries emit a `observe: config: ` stderr warning and are skipped — the simulation always proceeds.
- `output_dir` is an output **leaf** resolved under the application config's `config_base` (default `<config_dir>/<config>/`), so a config's captures land next to it — `configs/mnist.json` → `configs/mnist/observe/`. An empty leaf defers to the app-config default (`observe`); a non-empty relative leaf is placed under `config_base`; an **absolute** leaf wins outright. `sim_instance` resolves this itself — from the `app_config` reference it is handed plus its own parsed `observe.output_dir` leaf — via `coast::paths::resolve_config_output`, avoiding a second re-parse of the config file in the app layer. See [Application config]({{% relref "/architecture/modules/app_config" %}}).
- The output directory is created lazily on first commit. Files are truncated (`H5F_ACC_TRUNC`) on creation, so re-running with the same resolved directory overwrites prior captures — except when the run **continues** from a checkpoint of its own (equal `run_id`, SPEC S.5.14; a coastd resume): then an existing file of the same run is reopened, its rows for the steps from the checkpoint on are dropped (those steps run again), and the new rows are appended after the rest, so a resumed run's files equal those of one run straight through. A file of another run, a missing one, or one that cannot be read (a crash during a write can leave it so) is created afresh, and the writer logs why; the rows before the checkpoint are then lost, which a reader that checks its step coverage (the eval does) detects. Within a continued file, a site whose datasets hold another element type or row shape, or cannot be reopened, starts afresh on its own: its old datasets are unlinked and recreated, and the writer logs why.

The config round-trips through the run manifest (`src/sim_core/run_manifest.cpp`), so saved snapshots record the observe configuration that was active when they were taken.

## Output file format

One HDF5 file per group: `<output_dir>/<group-name>.observe.h5`. On creation the
file gets a `/metadata` group stamped with `type = "observe_log"`,
`schema_version = "1.0"`, a `created` timestamp and the `run_id` it was written for (`observe_writer_t::start()`).
This mirrors the buffer logger: every COAST log carries the same
`/metadata@type` marker, so readers — notably the
[HDF5 Browser]({{% relref "/architecture/modules/h5browse" %}}) — identify the
family uniformly without inspecting dataset layout.

Apart from `/metadata`, datasets are **flat at file root** (no nested HDF5 groups):

- `/<full_id_flat>` — value dataset, dtype per `dtype_traits<T>`, extensible along axis 0, chunked `[chunk_rows, shape...]`. `chunk_rows` comes from `pick_chunk_rows()` (`observe_chunk.h`): a ~1 MiB byte budget clamped to ≤1024 rows, so scalar/small sites keep 1024 rows while wide-row sites shrink to keep each chunk cache-resident (paired with a per-dataset `H5Pset_chunk_cache`).
- `/<full_id_flat>__step_index` — sibling `uint64` dataset, shape `(n_rows,)`, chunked `[1024]`, recording the simulation step for each row.

`<full_id_flat>` is the site's structural path with `/` replaced by `__`, e.g. `neuron_group/neuron_t[3]/dendrite_t[7]/activity` → `neuron_group__neuron_t[3]__dendrite_t[7]__activity`.

## Lifecycle

- `observe_registry::attach(env, cfg, output_dir, h5_exec)` — called once per `sim_instance` construction, after the env is built (`src/sim_core/sim_instance.cpp`). `output_dir` is the **already-resolved** directory: `sim_instance` computes it from its parsed `observe.output_dir` leaf placed under `config_base` (an absolute override wins), so `attach` sets it verbatim and `apply_config` only seeds the groups. Installs the **single post-process hook** (`run_observation_pass`), seeds groups from config, and eagerly calls `H5open()` on the executor thread to avoid library-init re-entry from a worker later.
- Each step, the post-process hook: returns immediately if no group is configured; otherwise recomputes each group's `active_this_step` under the registry mutex, walks `env.iterate_entities()` calling `root.perform_observations(acc)`, then consumes any `prog_single_pending` pulses.
- `observe_registry::detach()` — called in the `sim_instance` destructor after the sim thread stops. Flushes and closes all writers, removes the hook, clears groups. Re-attaching to a fresh `sim_instance` starts clean.

## Python loader

`scripts/load_observe.py` lists sites (needs `h5py + numpy`) or plots them to PNG (`--plot`, additionally needs `matplotlib`): scalar sites become value-vs-step line plots, 1-D vectors become heatmaps. For real analysis, open the file directly with `h5py` and read the value and `__step_index` datasets.

## Files

| Path | Role |
|---|---|
| `include/observe/observe_accessor.h` | `observe_accessor` — `observe()` overloads, `group_active()`, `trigger()` |
| `include/observe/observe_dtype.h` | `dtype_traits<T>` specialisations |
| `include/observe/observe_trigger.h` | `trigger_mode_t`, `combine_mode_t`, `step_range_t` |
| `include/observe/observe_site.h` | per-site state (HDF5 handles, shape lock, rows written) |
| `include/observe/observe_group.h` | per-group state (ranges, combine, trigger/active atomics, site map) |
| `include/observe/observe_cfg.h` | typed config structs + `from_json` parsing |
| `include/observe/observe_registry.h` | singleton registry, attach/detach, the observation pass |
| `include/observe/observe_writer.h` | per-group HDF5 writer (submits to `hdf5::executor`) |
| `include/observe/observe_chunk.h` | `pick_chunk_rows()` — byte-budget chunk-geometry policy (pure, HDF5-free) |
| `src/observe/observe_accessor.cpp` | accessor → registry emit glue |
| `src/observe/observe_config.cpp` | `apply_config()` — seed groups from parsed config |
| `src/observe/observe_registry.cpp` | core: post-process pass, site resolution, trigger evaluation |
| `src/observe/observe_writer.cpp` | HDF5 file/dataset creation, row append |
| `scripts/load_observe.py` | Python loader / plotter |
