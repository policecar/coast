+++
title = "core"
weight = 10
+++

# core — Foundation Types

The `core` module provides the base type system for the entity hierarchy, probe registration infrastructure, and the SIMD-optimized random number generator.

## Directory Layout

```
include/core/
├── app_config.h              # App-level config (where outputs/assets go), distinct from the experiment config
├── demand_published.h        # Single-slot, demand-driven sim → GUI publisher
├── element_type.h            # Type-safe identifier using typeid
├── entity_element.h          # Base class for all hierarchical elements
├── entity_element_macros.h   # ELEMENT_TYPE, ELEMENT_CHILDREN, PROBE macros
├── local_time.h              # Thread-safe local wall-clock time as text (file stamps, "created")
├── path_resolver.h           # Expands config_base templates (<config_dir>) for config outputs
├── probe_name_registry.h     # Global registry of probe names per type
├── require.h                 # COAST_REQUIRE: always-compiled hot-path contract check
├── run_id.h                  # A run's identity for checkpoint continuation ("auto" → fresh id)
├── seed.h                    # seed_t, seed_offset_t: strong 32-bit seed types
├── step_error.h              # Hot-path error payload for the std::expected error channel
├── structure_registry.h      # O(1) lookup: element pointer → structural path
├── vis_hints.h               # vis_hints_t: GUI-free visualization-hint config POD
├── gpu/                      # GPU device seam — see the core/gpu page
│   ├── gpu_device.h          # The agnostic seam between GPU entities and the backend
│   └── shared_prng.h         # Embedded PCG-1D32 WGSL hash-PRNG snippet
└── rng/
    ├── fast_rng.h            # Type aliases: fast_rng_t, standard_rng_t
    ├── mt11213b_simd.h       # SIMD MT11213b, drop-in alternative to mt19937_simd
    ├── mt19937_simd.h        # SIMD Mersenne Twister (AVX-512/AVX2/NEON)
    ├── philox_4x32.h         # Counter-based Philox-4x32-10 PRNG
    ├── philox_4x32_avx512.h  # SIMD-buffered Philox-4x32-10 (16 lanes, AVX-512)
    ├── rng_wrapper.h         # Generic wrapper template
    └── xor_table_prng.h      # XOR of five prime-length read-only tables

src/core/
├── app_config.cpp            # app_config loading and defaults
├── entity_element.cpp        # probe_slow_path implementation
├── path_resolver.cpp         # config_base token expansion
├── probe_name_registry.cpp   # Mutex-guarded global registry
├── run_id.cpp                # new_run_id / resolve_run_id
├── structure_registry.cpp    # Tree traversal and path computation
└── gpu/
    ├── shared_prng.cpp       # Embeds shared_prng.wgsl at build time (#embed)
    ├── shared_prng.wgsl      # PCG-1D32 stateless hash PRNG
    ├── smoke_main.cpp        # wgpu bootstrap smoke test (executable)
    ├── min_kernel_main.cpp   # End-to-end compute-kernel verification (executable)
    ├── spike_main.cpp        # Framework spike measurements (executable)
    └── backends/
        ├── null/backend_null.cpp          # Host-only, dependency-free gpu_device
        └── webgpu/
            ├── backend_webgpu.cpp         # The single TU that sees <webgpu/webgpu.h>
            ├── gpu_context.{h,cpp}        # wgpu lifecycle: instance → adapter → device → queue
            ├── wgpu_util.h                # Backend-private inline helpers
            └── wgsl_module.{h,cpp}        # WGSL source → WGPUShaderModule
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

### Random seeds (`seed.h`)

Every RNG in COAST takes a 32-bit seed: `mt19937_simd`, `mt11213b_simd`,
`ngm2::stream_seed`'s result, and the `u32` seeds and keys of the WGSL shaders
(WGSL has no 64-bit integer). A seed is therefore a `uint32` end to end, and an
entity's `effective_seed()` — reported in the run manifest's `node_seeds` and in
the checkpoint's configuration record — is exactly the value its RNG was seeded
with.

`seed_t` and `seed_offset_t` are distinct strong types (`enum class … :
std::uint32_t`). Nothing converts to them implicitly, so the compiler finds every
place a seed is produced or consumed, and a seed and an offset cannot be swapped.
The raw value is `std::to_underlying(s)`, taken where the seed enters an RNG or a
hash.

- **Composition.** `seed_t + seed_offset_t → seed_t` and `seed_offset_t +
  seed_offset_t → seed_offset_t`, modulo 2³² (unsigned, so well-defined). The
  effective seed of a node is config seed + `--seed-offset` + the node's
  `random_seed_offset`. Distinct (seed, offset) pairs can land on the same seed;
  that is accepted, since the composition is deterministic.
- **Input.** A config seed or offset, and `--seed-offset`, accept an integer in
  [0, 2³² − 1], the value the RNG is seeded with. A negative number, a
  non-integer (`1.0`, `"3"`) or an integer above 2³² − 1 is refused at load. (Until
  #199 a negative value was taken as its two's complement; that made the stored
  seed differ from the written one.) The nlohmann hooks (`from_json`/`to_json`)
  live in `seed.h` itself, so they are visible wherever `seed_t` is; they cannot
  see their key, so `json_loader` first checks every seed key of the config with
  `seed_bits_of()` and names a bad value by its JSON path, e.g.
  `/datasets/mnist/default/random_seed: a random seed must be …`.

## Key Macros

| Macro | Purpose |
|-------|---------|
| `ELEMENT_TYPE(cls, name)` | Generates `type_id()` override |
| `ELEMENT_CHILDREN_1(member, Type)` | Generates hierarchy for one child vector |
| `ELEMENT_CHILDREN_2(m1, T1, m2, T2)` | Generates hierarchy for two child vectors |
| `ELEMENT_PROBES(...)` | Generates `probe_points()` returning a static list |
| `PROBE(name, value)` | Auto-registers probe name, calls `entity_element::probe()` |
