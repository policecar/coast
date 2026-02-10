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
└── gui_vis_json.h            # Visualization hints parsing

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
└── snapshot_dir: string  (optional, default "snapshots")
```

## Serialization Approach

All structs use nlohmann/json macros:

- `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Type, field1, ...)` — all fields required
- `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Type, field1, ...)` — missing fields use defaults
- `NLOHMANN_JSON_SERIALIZE_ENUM(EnumType, ...)` — enum ↔ string mapping

This means `global_cfg = json;` recursively deserializes the entire tree.

## Entity Type Enum

```cpp
enum class io_entity_type_t {
    ngm2_neuron_group, mnist_input, dsprites_input,
    emnist_input, kmnist_input, cifar10_input,
    cifar100_input, chairs3d_input, color_mnist_input
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

## json_loader

Constructor takes a file path, parses the JSON, and populates `global_cfg_t`. Stores the config directory path for resolving relative dataset paths.

Provides `get_*_cfg()` accessors for each config section.
