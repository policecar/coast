+++
title = "json_cfg"
weight = 50
+++

# json_cfg — Configuration System

The `json_cfg` module parses JSON configuration files into typed C++ structs using nlohmann/json's macro-based serialization.

## Directory Layout

```
include/json_cfg/
├── json_loader.h             # Top-level loader class
├── dataset_json.h            # Dataset config parsing
├── ngm2_json.h               # Neural model config parsing
├── sim_json.h                # Simulation graph parsing
└── vis_hints_json.h          # Visualization hints parsing (structs: core/vis_hints.h)

src/json_cfg/
└── json_loader.cpp           # File I/O, top-level deserialization
```

## Config Structure

The complete configuration lives in `global_cfg_t`:

```
global_cfg_t
├── datasets: dataset::cfg_store
│   ├── mnist: map<string, mnist_param_t>
│   ├── dsprites: map<string, dsprites_param_t>
│   ├── cifar10: map<string, cifar10_param_t>
│   └── ... (one map per dataset type)
├── ngm2: ngm2::cfg_store
│   ├── dendrites: map<string, dendrite_params_t>
│   ├── neurons: map<string, neuron_params_t>
│   └── neuron_groups: map<string, neuron_group_params_t>
├── sim: sim::sim_cfg_t
│   ├── nodes: vector<sim_node_t>
│   └── node_sets: vector<sim_node_set_t>
├── vis_hints: vis_hints_t
│   ├── datasets: dataset_hints_t
│   ├── ngm2: ngm2_hints_t
│   └── buffer_vis: buffer_vis_hint_t
└── snapshot_dir: string  (optional output LEAF; "" = defer to the app config
                           default, then resolved under config_base — see below)
```

## Serialization Approach

All structs use nlohmann/json macros:

- `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Type, field1, ...)` — all fields required
- `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Type, field1, ...)` — missing fields use defaults
- `COAST_JSON_STRICT_ENUM(EnumType, ...)` (`json_cfg/strict_enum.h`) — enum ↔ string mapping
  with the same table syntax as `NLOHMANN_JSON_SERIALIZE_ENUM`, but **strict**: an unknown
  value throws `std::invalid_argument` naming the enum, the value and the accepted values
  (the nlohmann macro silently maps it to the first listed entry, so a typo such as
  `"max_sotfmax"` would run as the first enum value). Every config enum uses it. JSON `null` is
  rejected too: no table maps it to a `TYPE_COUNT`/`ENTITY_COUNT` sentinel, and serialising
  a sentinel throws.
- Seeds and seed offsets (`seed_t`, `seed_offset_t`) bind through the hooks in
  `core/seed.h`: an integer in [0, 2³² − 1], anything else refused (see *Random
  seeds* in [core]({{% relref "/architecture/modules/core" %}})).

This means `global_cfg = json;` recursively deserializes the entire tree. `json_loader`
prefixes any load error with `config '<file>': ` (a strict-enum rejection stays
`std::invalid_argument`; nlohmann's own errors are rethrown as `std::runtime_error`).
Before binding, `json_loader` walks the parsed tree for settings that no longer
exist (it warns) and for seed keys (`random_seed`, `rnd_seed`, `ri_seed`,
`random_seed_offset`) whose value breaks the seed rule (it throws
`std::invalid_argument` naming each by its JSON path, `~` and `/` in a name
escaped as in RFC 6901). Keys match by name anywhere in the tree; an object or
array under one (a neuron group called `random_seed`) is a named entry, not a
seed, and is left to its own binding.

## Entity Type Enum

```cpp
enum class io_entity_type_t {
    ngm2_neuron_group, mnist_input, dsprites_input,
    emnist_input, kmnist_input, cifar10_input,
    cifar100_input, chairs3d_input, color_mnist_input,
    retina   // centre-surround ON/OFF front end (see retina module)
};
```

Serialized as strings: `"ngm2_neuron_group"`, `"mnist_input"`, etc.

## Simulation Graph Topology

### sim_node_t

Direct node definition: `{id, label, entity_type, entity_cfg, inputs: [(source_id, channel_name)]}`.

### sim_node_set_t

Template-based replication for creating repeated sub-graphs:
- `base_id` — offset added to template node IDs
- `nodes` — template node IDs to replicate
- `node_deltas` — per-template-node wiring modifications (add/remove inputs)

This enables configs to define a pattern once and replicate it with slight variations.

Node ids must be unique across direct nodes and every expanded `base_id + id`:
`sim_builder::build_local` refuses a repeated id rather than letting the later
node overwrite the earlier one.

**Copies share their template's randomness, by design.** A copy takes the template
node's `entity_cfg` and `random_seed_offset` unchanged, and a node set has no offset
of its own. Every copy of a template node therefore composes the same effective seed
(config seed + `--seed-offset` + the template's `random_seed_offset`) and runs the
same random streams — the same noise and, for a group with a `random_index`
inhibitor, the same inhibition matrix; the same initial synapses too, unless a
`node_deltas` entry changes a copy's input size. Copies are replicas that differ
only in their wiring (`node_deltas`). One exception: with `ri_seed_from_time: true`
each copy's `random_index` inhibitor draws its own seed at construction, so copies
share their group seed but not their inhibition matrix. To give replicas independent
randomness, list them as separate direct nodes with distinct `random_seed_offset`s
instead of as a node set.

## json_loader

Constructor takes a file path, parses the JSON, and populates `global_cfg_t`.

nlohmann binds a number by a plain cast and ignores keys it does not know, so the
loader then serialises `global_cfg_t` again and compares it with the input (#199):

- a **number that comes back different** did not fit its C++ type — `2.7` for an
  integer (truncated to 2), `-1` for a `size_t` (wrapped), `259` for a `uint8_t`
  (wrapped to 3), `1e300` for a `float` (inf). The load is refused, naming every
  such value by its JSON path, e.g.
  `/ngm3/neuron_groups/x/max_branch_level: 259 -> 3`. Floats compare to float
  precision, so `0.1` for a `float` field is fine; a whole float (`4.0`) for an
  integer is fine. A float written for an integer field must also lie in the
  64-bit range of the field's signedness: `2^64` for a `size_t` binds (by an
  out-of-range cast, UB) to `18446744073709551615`, which converts back to `2^64`,
  so equality alone would pass it.
- a map with non-string keys (`node_deltas`) is written back as an array of
  `[key, value]` pairs in the map's own order, so its entries are matched **by
  key**, not by position (`k_keyed_arrays` in `json_loader.cpp`). An entry with
  no match — a key that does not fit, or a repeated key, of which the map keeps
  the first — is refused.
- a **key that does not come back** is bound to nothing. It is warned about by its
  JSON path (`unknown key(s) ignored: …`) and the load goes on; an unknown object is
  reported once, not key by key. The removed activity learning window keys keep
  their own message.
- a **renamed key** (`k_renamed_keys`) is refused, naming its new name:
  `dendrite_branch_depth` (ngm3) and `max_branch_depth` (ngm25) are
  `max_branch_level`, ngm2's and SPEC.md's name for the tree depth L (#206).
  Warned-and-defaulted like an unknown key, an old config would silently run with
  the default depth. `scripts/migrate_max_branch_level.py` converts old configs
  and ngm3 checkpoints.

The check relies on every binding writing back the keys it reads and the value it
stored. A hook that deliberately rewrites a value (as seeds once did, taking `-1`
as 4294967295) would need an exemption in `diff_round_trip`.

Provides `get_*_cfg()` accessors for each config section.

## Path resolution

COAST resolves paths in **two tiers**. The base rule is standard filesystem
semantics; per-config *output* locations add one explicit, configurable layer on
top (the app config's `config_base` — see the [Application config]({{% relref "/architecture/modules/app_config" %}}) page).

### Tier 1 — standard filesystem semantics (inputs & CLI paths, verbatim)

Every path that is **not** a per-config output leaf — dataset `image_file` /
`data_dir` / `hdf5_file`, the config path itself, and every CLI path argument
(`--save-state`, `--load-state`, `--log-buffers`, `--emit-run-manifest`,
`--log-file`, `--app-config`) — follows standard filesystem semantics, uniformly:

- a **relative** value (`./x` or bare `x`) resolves against the **process working
  directory** (CWD);
- an **absolute** value (`/x`) is used **as-is**.

Such a path is never rebased against the config directory or any implicit base;
COAST does not assume whether it is relative or absolute. A path a user *types on
the command line* is therefore **never** placed under `config_base` — that is the
top invariant of the location system. Configs meant to run from a different
directory (a staged run dir, another machine) should reference **datasets** with
absolute paths.

### Tier 2 — per-config outputs under `config_base`

The config's output *leaves* — `snapshot_dir` and `observe.output_dir` — are
**resolved under the application config's `config_base` template** (default
`<config_dir>/<config>/`), so a config's outputs land next to it, isolated per
config: `configs/mnist.json` writes snapshots to `configs/mnist/snapshots/`. The
leaf value still obeys Tier-1 semantics *after* expansion: an **absolute** leaf
(e.g. `"/data/snaps"`) wins outright and bypasses `config_base`; an empty leaf
(`""`) defers to the app-config default leaf. Both the write and the resume/read
paths route through one resolver, so they always agree. See the
[Application config]({{% relref "/architecture/modules/app_config" %}}) page for
the `config_base` tokens (`<config_dir>`, `<config>`), the discovery precedence,
and the full field list.
