+++
title = "Entity Element Hierarchy"
weight = 10
+++

# Entity Element Hierarchy

The `entity_element` class is the base for all structural components in COAST. It creates a **tree of elements** that mirrors the biological hierarchy: groups contain neurons, neurons contain dendrites.

## The Tree

```
io_entity (root)
├── neuron_group_t
│   ├── neuron_t [0]
│   │   ├── dendrite_t [0]
│   │   └── dendrite_t [1]
│   ├── neuron_t [1]
│   │   └── dendrite_t [0]
│   └── ...
```

Each element knows its children (via `ELEMENT_CHILDREN_*` macros) and its type (via `ELEMENT_TYPE` macro). The `structure_registry` walks this tree once at initialization and builds:

1. An O(1) map: `element pointer → structure_entry`
2. Precomputed path strings: `"neuron_group_t/neuron_t[42]/dendrite_t[3]"`
3. Cached probe names per element

## Why It Exists

The hierarchy serves three purposes:

1. **Probe addressing**: The probe browser needs to show a navigable tree of all elements and their probe points. The structure_registry provides the paths and names.

2. **State persistence**: When saving/loading state, each element can participate via `save_state()`/`load_state()`.

3. **Type-safe traversal**: GUI code can walk the tree with `child<neuron_t>(i)` to access specific elements by type and index.

## Declaring Elements

```cpp
class neuron_t : public core::entity_element {
    ELEMENT_TYPE(neuron_t, "neuron")
    ELEMENT_CHILDREN_1(dendrites, dendrite_t)
    // ...
};
```

This generates:
- `type_id()` → returns `element_type` for "neuron"
- `children()` → reports dendrites
- `child<dendrite_t>(i)` → returns `&dendrites[i]`
- `child_count<dendrite_t>()` → returns `dendrites.size()`

## Key Invariant

`init_structure()` must be called after the entity is in its **final memory location**. Because `structure_registry` stores raw pointers, calling it before `std::vector` reallocation would create dangling pointers. This is why entity initialization happens in two phases:

1. `sim_builder::build_local()` phase 3: create and emplace all entities
2. `sim_builder::build_local()` phase 4: call `init_io()` (which calls `init_structure()`)
