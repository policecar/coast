+++
title = "core"
weight = 10
+++

# core — Foundation Types

The `core` module provides the base type system for the entity hierarchy, probe registration infrastructure, and the SIMD-optimized random number generator.

## Directory Layout

```
include/core/
├── entity_element.h          # Base class for all hierarchical elements
├── entity_element_macros.h   # ELEMENT_TYPE, ELEMENT_CHILDREN, PROBE macros
├── element_type.h            # Type-safe identifier using typeid
├── structure_registry.h      # O(1) lookup: element pointer → structural path
├── probe_name_registry.h     # Global registry of probe names per type
└── rng/
    ├── fast_rng.h            # Type aliases: fast_rng_t, standard_rng_t
    ├── mt19937_simd.h        # SIMD Mersenne Twister (AVX-512/AVX2/NEON)
    └── rng_wrapper.h         # Generic wrapper template

src/core/
├── entity_element.cpp        # probe_slow_path implementation
├── structure_registry.cpp    # Tree traversal and path computation
└── probe_name_registry.cpp   # Mutex-guarded global registry
```

## Key Classes

### `entity_element`

Base class for all hierarchical elements (neuron groups, neurons, dendrites). Provides:

- **Hierarchy traversal**: `children()`, `child<T>(index)`, `child_count<T>()`
- **Type identity**: `type_id()` returns an `element_type`
- **Probe fast-path**: `probe(name, value)` — single atomic load, returns immediately when no probes are active
- **Root entity access**: `root_entity()` — navigates up to the owning `io_entity`
- **Active probe count**: `atomic<uint32_t> active_probe_count_` — incremented/decremented by probe activation/deactivation

The hierarchy is implemented via virtual `child_impl()` and `child_count_impl()`, made convenient by the `ELEMENT_CHILDREN_*` macros.

### `element_type`

Wraps `std::type_info` with a human-readable display name. Created via `ELEMENT_TYPE(cls, "display_name")`. Provides `type_info()`, `name()`, and comparison operators.

### `structure_registry`

Built once per `io_entity` during `init_structure()`. Recursively traverses the entity element hierarchy and creates a `structure_entry` tree with:
- Precomputed string paths (e.g., `"neuron_group_t/neuron_t[42]/dendrite_t[3]"`)
- Cached probe names from `probe_points()`
- An `element_map_` for O(1) pointer-to-entry lookup

### `probe_name_registry`

A global `static` registry mapping `type_index → set<string>`. Populated lazily by the `PROBE` macro on first invocation at each call site. Thread-safe via mutex (low contention — each call site registers only once).

### SIMD RNG (`mt19937_simd`)

A full reimplementation of MT19937 with SIMD acceleration. Produces **identical sequences** to `std::mt19937` for the same seed.

**SIMD dispatch** (compile-time):

| ISA | Batch Size | Platform |
|-----|-----------|----------|
| AVX-512 | 16 | x86-64 with AVX-512 |
| AVX2 | 8 | x86-64 with AVX2 |
| NEON | 4 | Apple Silicon / ARM64 |
| Scalar | 1 | Fallback |

The algorithm runs in two phases:
1. **Twist**: The standard MT19937 recurrence on the 624-element state array, vectorized in SIMD-width chunks
2. **Temper**: All 624 values are tempered in bulk into a pre-allocated output buffer

`operator()` returns from the pre-tempered buffer; when exhausted (every 624 calls), regenerates the full buffer. This amortizes the cost and enables SIMD vectorization. Achieves ~2.5x speedup over `std::mt19937`.

**Type aliases**:
- `fast_rng_t` = `rng_t<mt19937_simd>` — used throughout the codebase
- `standard_rng_t` = `rng_t<std::mt19937>` — available for comparison

## Key Macros

| Macro | Purpose |
|-------|---------|
| `ELEMENT_TYPE(cls, name)` | Generates `type_id()` override |
| `ELEMENT_CHILDREN_1(member, Type)` | Generates hierarchy for one child vector |
| `ELEMENT_CHILDREN_2(m1, T1, m2, T2)` | Generates hierarchy for two child vectors |
| `ELEMENT_PROBES(...)` | Generates `probe_points()` returning a static list |
| `PROBE(name, value)` | Auto-registers probe name, calls `entity_element::probe()` |
