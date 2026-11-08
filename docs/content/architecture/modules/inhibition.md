+++
title = "inhibition"
weight = 35
+++

# inhibition — Swappable Group Inhibition

The `inhibition` module provides a first-class, **swappable inhibition abstraction**
owned by neuron groups. It replaces the former ad-hoc `ngm2::local_inhibition()`
free function — which was shared awkwardly between the models (HD-NGM3 reached
into `ngm2::local_inhibition`) and hardcoded as a GPU kernel — with a polymorphic
interface selected by name from an independently-keyed config block. The module is
model-agnostic: HD-NGM2, HD-NGM2.5 and HD-NGM3 (CPU and GPU) all use it.

The default kind is `max_softmax`. The former default, the entropy-modulated soft
winner-take-all `soft_wta` (a port of `local_inhibition`), was removed together with
the normalized Shannon entropy it depended on; a config that still names
`"soft_wta"` is rejected as an unknown kind.

## CPU Interface

`inhibitor_t` derives `core::entity_element` so an inhibitor can be a probe-able
child of its group (uniform with how `neuron_t` / `dendrite_t` are entity_elements
for per-component probing).

```cpp
class inhibitor_t : public core::entity_element {
    // the resolved params it runs with (a seed drawn from time is the drawn seed)
    const inhibition_params_t& params() const;
    // raw and out may alias (ngm2 inhibits in place); when distinct the
    // implementation copies raw -> out first. Precondition: equal sizes == neuron count.
    virtual void inhibit(std::span<const float> raw, std::span<float> out) = 0;
    virtual void save_state(logging::state_writer&) const {}   // no-op default
    virtual void load_state(logging::state_reader&) {}         // (stateless kinds)
};
```

- **`random_index_t`** — **CPU-only** lateral inhibition via random recurrent
  connections (no modelled interneurons). Each group owns a random `N×N` permanence
  matrix `M` (entries `U[0,1)`, zero diagonal), generated once on first `inhibit()`
  from `ri_seed` (`N` taken from the neuron-span size). `ri_seed` composes with the
  group's offsets like every seed — config value + `--seed-offset` + the node's
  `random_seed_offset`, modulo 2³² — so a seed sweep varies `M` too; the factory
  (`make_inhibitor`, given the group's offset) does the addition, and a seed drawn
  with `ri_seed_from_time` replaces the composed one. The group reports the seed as
  `inhibition_seed()`, recorded in the run manifest's `node_seeds`. The `threshold_linear`
  variant derives a fixed weight matrix `T` from `M` — entries `< ri_reach` become
  `ri_strength`, the rest `0`, diagonal forced `0` — then each step computes
  `inh = T·raw` and `out = clamp(raw − inh, 0, 1)`. Two knobs: the *breadth*
  (`ri_reach`) and *strength* (`ri_strength`) of inhibition. **Stateful**:
  `save_state` persists `M` (plus the resolved seed and hyperparameters) so a restore
  is bit-identical on the same machine without depending on RNG reproducibility; a
  corrupt/absent matrix on load falls back to lazy regeneration. On load the seed
  comes back with `M` (random state), the hyperparameters stay as configured. The `ri_variant`
  enum leaves room for further variants.
- **`max_softmax_t`** — the default kind: softmax over the group output, rescaled so
  the leader keeps its pre-inhibition activity. The softmax normaliser cancels in
  that rescaling, leaving a closed form in the gap to the leader:

  ```text
  out[i] = max * softmax(beta*v)[i] / max_j softmax(beta*v)[j]
         = max * exp(-beta * (max - v[i]))
  ```

  `beta` (the `strength` field) is in units of 1/activity. The leader is never
  weakened and nothing exceeds it, so a weak group response stays at or below its
  own (weak) leader rather than being blown up to 1 as a plain softmax would. It is
  not damping-only: a neuron is raised when `exp(-beta*gap) > v[i]/max` (small
  `beta`, or a weak leader with small gaps), and `beta = 0` sets every neuron to the
  max.
  Stateless (`max_softmax_apply()` holds the math).

Each group owns one `std::unique_ptr<inhibitor_t>` (per-group state) and calls
`inhibitor_->inhibit(...)` at the single existing inhibition call site. The group's
checkpoint configuration record (`inhibition.*`) is written from `params()`, except a
seed drawn with `ri_seed_from_time` (32 bits, like every seed): that is random state, not a setting, so it is
saved with `M` and left out of the record, and resuming such a run under an unchanged
configuration reports nothing. State is
persisted string-keyed under a scoped `begin_group("inhibition")`, exactly as the
group scopes its RNG state.

## GPU Interface — Kernel Injection

`gpu_inhibitor_t` is a kernel **emitter**, not an `entity_element`. The host (a GPU
neuron group) describes the buffers it owns; `emit()` registers the emitter's own
self-contained WGSL kernel + uniform on the device and appends one compute op into
the host's `step_plan`:

```cpp
class gpu_inhibitor_t {
    virtual void emit(coast::gpu::gpu_device& dev,
                      const gpu_host_handles& h,   // act_in, out_page, neuron_cnt, dispatch
                      coast::gpu::step_plan&  sp) = 0;
};
```

The injected WGSL carries its own uniform layout, so it never couples to the host
entity's `Params` struct. `gpu_max_softmax_t` ports `max_softmax_apply` (one max reduction,
then `max * exp(-beta * gap)`, pass-through unless the max is a normal float);
`unit.inhibition.max_softmax_gpu` checks CPU/GPU parity. This "kernel injection" is the GPU swappability mechanism.

Not every CPU kind has a GPU counterpart: `make_gpu_inhibitor` rejects
`random_index` with a clear CPU-only error rather than silently falling back, so a
GPU config referencing it fails fast.

## Configuration

Inhibition parameters live in a separate name-keyed dictionary so adding kinds does
not bloat the group config. The config is **typed** (mirroring `dendrite_params_t`):
a `type` discriminator plus a superset of per-kind fields, (de)serialized via the
same NLOHMANN macros and editable in the config editor with no bespoke code.

```jsonc
"ngm2": {
  "inhibitions": {
    "strong": { "type": "max_softmax", "strength": 20.0 },
    "ridx":   { "type": "random_index", "ri_variant": "threshold_linear",
                "ri_reach": 0.1, "ri_strength": 0.5, "ri_seed": 0,
                "ri_seed_from_time": false }
  },
  "neuron_groups": {
    "L4-mnist": { "inhibition_cfg": "strong", ... }
  }
}
```

An **empty** `inhibition_cfg` synthesizes a `max_softmax` with beta = the group's
`local_inhibition_strength` (default 10; the repo configs set 10). A non-empty name
that is not a key in `inhibitions` is a construction error on both CPU and GPU
(`make_inhibitor` / `make_gpu_inhibitor` throw), so a typo fails fast instead of
silently running the default.

## Files

| File | Responsibility |
|------|----------------|
| `inhibition/inhibition.h` / `.cpp` | `inhibitor_t` interface + `make_inhibitor` factory |
| `inhibition/inhibition_cfg.h` | typed `inhibition_params_t` + `inhibition_type_t` enum |
| `inhibition/random_index.h` / `.cpp` | CPU-only `random_index_t` (random recurrent inhibition; stateful matrix) |
| `inhibition/max_softmax.h` / `.cpp` | `max_softmax_t` (max-rescaled softmax) |
| `inhibition/gpu_inhibition.h` | `gpu_inhibitor_t` emitter interface + `gpu_host_handles` |
| `inhibition/gpu_max_softmax.h` / `.cpp` | GPU `max_softmax` (injected WGSL kernel) |
| `json_cfg/inhibition_json.h` | JSON adaptors (enum + params), included by both model json headers |

## Adding a Kind

The factory is a central switch (matching the `dendrite_params_t` precedent rather
than a self-registering registry). Adding a kind touches: a value in
`inhibition_type_t`, a case in `make_inhibitor` (and `make_gpu_inhibitor` for GPU),
a JSON enum mapping, and the new subclass file(s).
