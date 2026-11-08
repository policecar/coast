+++
title = "hd_ngm2"
weight = 30
+++

# hd_ngm2 — Neural Model

The HD-NGM2 (Hierarchical Dendritic - Neuron Group Model 2) is the core neural model. It implements biologically-inspired neurons with permanence-based synapses, a binary tree of dendritic segments, and competitive learning with stochastic winner selection.

## Directory Layout

```
include/hd_ngm2/
├── hd_ngm2_neuron_group.h    # Top-level entity: group of neurons
├── hd_ngm2_neuron.h          # Single neuron with dendrites
├── hd_ngm2_dendrite.h        # Dendrite with synapse tree
├── hd_ngm2_cfg.h             # Configuration structs
└── hd_ngm2_tools.h           # Utility functions (sigmoid, Hoyer, stats, noise gate)

src/hd_ngm2/
├── hd_ngm2_neuron_group.cpp  # process() implementation
├── hd_ngm2_neuron.cpp        # get_response() and adapt()
├── hd_ngm2_dendrite.cpp      # Match computation, learning, branching
└── (hd_ngm2_tools is header-only)
```

## Hierarchy

```
neuron_group_t (io_entity)
  └── neuron_t [0..N-1]  (entity_element)
        └── dendrite_t [0..M-1]  (entity_element, leaf)
              └── synapses_t  (SOA: permanence[], mismatch[], segment_idx[], ...)
                    └── Binary tree of segments [1..max_segment_idx]
```

## neuron_group_t

Implements `io_entity`. The `process()` method orchestrates the full simulation step:

1. **Compute responses**: Each neuron computes activity from its dendrites (parallel)
2. **Local inhibition**: a swappable `inhibition::inhibitor_t` (default `max_softmax`) suppresses weaker activations — see [inhibition]({{% relref "/architecture/modules/inhibition" %}})
3. **Stochastic winner selection**: Any neuron within top ~20% of max can win, each with equal probability: a second draw picks uniformly among them, not by index (SPEC.md M.4.4)
4. **Winner adaptation**: Primary learning with sigmoid-filtered weight
5. **Common adaptation**: Secondary learning for all neurons (weak, global drift)

`input_channels` must name exactly the channels the group is fed, each once (a group fed by a single channel may leave it empty); their order is the layout of `population/in`.

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `local_inhibition_strength` | 10.0 | Beta of the default max_softmax inhibition (no `inhibition_cfg`) |
| `stochastic_win_thres` | 0.8 | Winner selection range (fraction of max) |
| `common_learning_rate` | 0.0001 | Secondary adaptation rate |
| `weight_filter` | (0.5, 0.33) | Sigmoid shape for adaptation weight |
| `frozen` | false | Freeze learning — skip winner selection, adaptation, branching, and max_activity updates (see [Frozen Inference Mode](#frozen-inference-mode)) |
| (learning windows) | tracked | Learning windows that follow the activity they gate (see [Activity Window Tracking](#activity-window-tracking)) |

## neuron_t

Each neuron has apical (top-down modulatory) and proximal (main) dendrites.

**Activity computation**: `activity = clamp(apical * proximal + noise, 0, 1)` — the apical dendrite acts as a multiplicative gate on the proximal response.

**Learning window**: Adaptation is strongest at medium activity. Very low or very high activity neurons receive little learning, preventing over-specialization. `adapt(weight, stats, window)` gates with the window the group passes in ([activity window tracking](#activity-window-tracking)).

### Probes

- `"activity"` — final neuron activity
- `"dendrite_count"` — number of dendrites
- `"representation_count"` — total leaf segments across dendrites

## dendrite_t

The most algorithmically complex component. Each dendrite maintains:

### Synapse Data (SOA Layout)

| Array | Purpose |
|-------|---------|
| `permanence` | Connection strength [0,1] — thresholded to connected/disconnected |
| `mismatch` | IIR-filtered tracking of input/permanence disagreement |
| `adapt_history` | Accumulated learning effort (sum of theta) |
| `segment_idx` | Which binary tree segment this synapse belongs to |
| `input_inc` | 0 or 1 — allows multiple synapses per input dimension |

The synapses cover the dendrite's input channels one after another, in the order its `input_channels` lists them, so that list decides which input each permanence meets. The order also affects the response, because a segment's sum is clamped at 0 after each penalised synapse. `input_channels` must name exactly the channels mapped to the dendrite, each once, and at least one, or `init_io` throws; a dendrite with a single channel may leave it empty. Two channel mappings onto the same `to_channel` are refused too, since the second would replace the first. From format version 4 a checkpoint records each dendrite's channel order and refuses a load that declares another ([channel record]({{% relref "/architecture/modules/logging" %}})); the name it records for each channel is the group-level channel name it resolves to through the group's and the neuron's own `channel_mappings`, not this dendrite's own `input_channels` key, since a mapping hop can rename the channel again on the way here without that key changing. Checkpoints older than format version 3 were written while ngm2 took its channel order from the wiring, so ngm2 refuses them if any dendrite has more than one channel.

### Binary Segment Tree

Standard heap-style indexing: root=1, left child=2i, right child=2i+1. For `max_branch_level=3`, there are 15 segments (8 leaves at indices 8-15).

`max_branch_level` must be in [0, 4], the same upper bound as ngm3 and the ngm25 GPU path; a larger value is refused at construction. The response, the adaptation and both representation-count probes walk the whole segment array on every step, so the per-step cost grows as 2^(L+1) per dendrite whatever the tree actually holds (measured with 16 neurons of 2 dendrites on a 64-value input and unbranched trees: 81 µs per step at depth 3, 1.9 ms at depth 15, the old `u16` limit). Loops over segment ids count in `std::size_t` and representation counts summed over dendrites are `std::size_t`: depth 15 used to be accepted, and a `u16` loop counter wrapped there and hung `process()` (#237).

### Match Computation (`get_response()`)

1. Linear sweep: accumulate input × permanence into segment activities
2. Stochastic penalty: connected synapses with weak input reduce activity
3. Root-to-leaf propagation: parent activities cascade to children
4. Noise-gate attenuation: noisy inputs get suppressed responses (see *Noise gate* below)
5. Returns maximum leaf segment activity

### Noise gate

Both `get_response()` and `adapt_synapses()` attenuate input that looks like noise, i.e. flat and dim:

```text
attenuation = 1 - sigmoid((noise - start) / width, shape) * (1 - sigmoid(avg))
```

`noise` is 1 − Hoyer sparsity (`input_stats_t::noise_hoyer`: 0 for one active entry, 1 for a flat
input; 0 for an empty, all-zero or single-entry input, which carry no shape evidence — at 0 the gate
passes a channel at ≥ 0.99 whatever its brightness, since the brightness term can only open a
noisy-looking input, never close a sparse one). The windows are compile-time constants in `hd_ngm2_tools.h` (`response_noise_window`,
`learning_noise_window`, applied by `noise_attenuation()`), shared by ngm2, ngm2.5 and ngm3; the
ngm3 GPU kernels receive them as generated WGSL constants:

| Gate | start | width | shape | Fit target for signal / "neither" / noise |
|------|-------|-------|-------|--------------------------------------------|
| response (`get_response()`) | 0.125 | 0.84 | {0.25, 0.5} | ≥ 0.9 / 0.35–0.65 / ≤ 0.1 |
| learning (`adapt_synapses()`) | −0.25 | 1.405 | {0.5, 0.5} | ≥ 0.9 / 0.15–0.35 / ≤ 0.1 |

The gate is graded: few strong winners (signal) pass, many moderate winners ("neither") are
partly damped, flat and weak input (noise) is suppressed. The windows were fitted on simulated
neuron-group outputs of 50 to 400 entries (`explore/nse_alt/hoyer_fit2.py`) against the targets
in the table. They are targets, not guarantees: on the fit data the medians meet them except
learning on signal at 200–400 entries (0.86–0.88) and on "neither" at 50 (0.40), and 44–77 % of
the signal samples fall inside their band. 1 − Hoyer replaced the normalized Shannon entropy
because, at a fixed fraction of active entries, it moves little with the input size between 50
and 400 entries, where the NSE gate's window (start 0.8, width 0.2) shifted by more than its
width. Inputs of only a few entries read sparser: a 30 % active input of 8 entries passes the
response gate at about 0.9, so the windows do not carry over to very small groups.

First-layer dataset inputs were not part of the fit. A noisy mnist digit (0–5 % noise, as in
`configs/mnist.json`) reads 1 − Hoyer ≈ 0.43, which the gates place at response ≈ 0.74 and
learning ≈ 0.56 — in the "neither" range. A retina OFF channel on mnist reads ≈ 0.79 and learns
at ≈ 0.03.

`get_response()` takes the minimum of `noise` and `avg` over the dendrite's input channels that
carry shape evidence (`noise_gate_min_t`: a channel with a zero or non-finite sum, or of a single
entry, is left out, else its noise 0 would open the gate for a noisy sibling; with none left the
gate is open);
`adapt_synapses()` judges each input channel separately, so one channel can be learned while
another is treated as noise.


### Learning (`adapt_synapses()`)

- **Hebbian permanence update**: high input → permanence drifts to 1; low input → decays to 0
- **Winner gets primary learning rate**, others get secondary (much weaker)
- **Mismatch tracking**: IIR filter detects synapses that inconsistently match input

### Branching (`adapt_branches()`)

When enough synapses are "ambiguous" (high mismatch + sufficient learning history):
1. Each ambiguous synapse is **cloned** into two child segments
2. Permanences are wiggled (±0.1 random)
3. History is reset, allowing the two copies to specialize

This enables a single dendrite to learn multiple representations organized in a tree structure — common features at the root, discriminative features at the leaves.

### Probes

- `"segment_activity"` — maximum leaf segment activity
- `"synapse_count"` — total synapses (grows with branching)
- `"representation_count"` — distinct occupied leaf segments

## Utility Functions (hd_ngm2_tools.h)

| Function | Purpose |
|----------|---------|
| `hoyer_noise(vec)` | 1 − Hoyer sparsity in [0,1]: 0 = one-hot, 1 = flat; needs only Σv and Σv² |
| `compute_input_stats(span)` | Reduce a buffer to `input_stats_t` {sum, avg, min_val, max_val, noise_hoyer} |
| `sigmoid(x, shape)` | Custom sigmoid with steepness and transition point; `inv_step = 1/(1 - steepness^0.1)`, so a configured steepness must lie in [0, 1) (checked by the `sigmoid_shape_t` JSON adapter) |
| `softmax(vec, beta)` | Standard softmax with temperature |
| `normalize(vec)` | Min-max normalization to [0,1] |

### Input Statistics (Refactor A)

`input_stats_t` and `compute_input_stats()` were moved into `hd_ngm2_tools.h` during the S0 refactor of the *ngm3-gpu-port* cycle, as the **shared** definition used by both ngm2 and [ngm3]({{% relref "/architecture/modules/hd_ngm3" %}}):

```cpp
struct input_stats_t { float sum, avg, min_val, max_val, noise_hoyer; };
input_stats_t compute_input_stats(std::span<const float> buf);   // max_val = true data max (no 1.0 floor)
```

Previously the per-buffer statistics lived *inside* `io_buffer` and were recomputed on every swap. They now belong to the **consumer**: at the top of `neuron_group_t::process()` the group computes one `input_stats_t` per input buffer (an `input_stats_map_t` keyed by output id) and passes it **by const reference** down into `neuron_t::get_response()` / `adapt()` and the dendrites — a single computation per buffer per step, no fan-out recomputation. This removed the `sim_core → hd_ngm2` dependency that `io_buffer`'s stats helper had created (see [IO Buffer]({{% relref "/key-abstractions/io-buffer" %}})).

## Frozen Inference Mode

In a frozen run (the config's top-level `frozen`, SPEC.md M.7.6; there is no per-group key) every neuron group switches to pure inference: `process()` returns after inhibition, skipping winner selection, adaptation, branching, and `max_activity` updates. The `max_activity` update is part of the group's learning phase (`neuron_t::update_max_activity()`, called after that return), so it is skipped with the rest.

**Zero group output.** A step whose post-inhibition output is all 0 (possible with `random_index` inhibition) is handled the same way, decided per step: the output is written, and nothing in the learning state changes — no winner, adaptation, branch counter, branching or `max_activity` update (SPEC.md M.4.15). No neuron produced output to attribute learning to.

**Why zero learning rates are not enough.** Three mechanisms mutate persisted state even at lr=0: the IIR mismatch-smoothing EMA, the decaying running-activity max (`*= 0.99999; max(...)`), and the accumulated-theta branch gate (pre-existing theta plus still-rising mismatch fires splits). Measured on a 40 k-step MNIST run + 10 k-step lr=0 window: branching fired (64 rewrites), mismatch drifted on every synapse (max |Δ| 0.90), and `max_activity` moved on every neuron. `frozen=true` gates all three.

**On `load_state` the configuration wins** (SPEC.md M.7.8), frozen or not. A checkpoint carries the learned state, the RNG streams and a configuration record (`config` subgroup, see [logging]({{% relref "/architecture/modules/logging" %}})), never the settings themselves: learning rates, thresholds, weight filter and inhibition strength keep the values the group was built with. `neuron_group_t::current_config()` lists them — the group's own, then once per neuron cfg (`neuron[<cfg>].…`) and dendrite cfg (`dendrite[<cfg>].…`) it uses, read from the first instance of that cfg — and a load reports each difference from the record in one warning. The neuron and dendrite setters (`set_branch_interval`, `set_primary_learning_rate`, …) change one instance alone, so every later instance also records the settings in which it differs from its cfg's first, under its index: `neuron[<cfg>]#<n>.…`, `dendrite[<cfg>]#<n>:<d>.…` (the checkpoint's `neurons/<n>/dendrites/<d>`). Nothing differs in a group as configured, so its record has no such entries. `frozen` isolates the model at step time, not at load time.

A checkpoint of another shape (`neuron_count`, a neuron's `dendrite_count`, or — from format version 5 on — a dendrite's type `apical` or tree depth `max_branch_level`) or with a missing or mis-sized piece is refused with an error naming the neuron and dendrite (see [logging]({{% relref "/architecture/modules/logging" %}})). The group, its neurons and its dendrites pass a `who` string down `load_state` for that. Each dendrite's own shape check reads and validates all five synapse arrays — their lengths agree, the input walk (`input_inc`, which advances once per input, at that input's last synapse) covers exactly the dendrite's input width, and every segment id exists in its tree — before assigning any of them (`dendrite_t::check_shape`); a checkpoint from another input width or a deeper tree is refused, e.g. `… dendrite 0: checkpoint does not fit this dendrite — input width 16 (checkpoint) vs 12 (configured)`.

That same function is what a group-level, read-only pre-pass runs over every neuron and dendrite before `load_state` assigns anything at all — not only once its own loop, neuron by neuron, reaches the offending one. Without the pre-pass, a tree-depth or input-width mismatch on, say, neuron 5's dendrite 2 was found only there, after the group's rng and neurons 0 through 4 had already loaded; a checkpoint refused that way left the group partially resumed, not unchanged (SPEC.md M.7.18). The type and the depth have to be recorded: equal counts and widths say nothing about the apical/proximal split, and a shallower tree's segment ids all fit a deeper one. A checkpoint older than format version 5 is checked for depth against its configuration record (`dendrite[<cfg>].max_branch_level`, by the cfg names the current configuration gives its dendrites) and refused on a difference; its types are recorded nowhere, so it loads unchecked for them, with one notice per checkpoint. The pre-pass does not cover a checkpoint missing a whole neuron or dendrite subgroup — that is still found, and reported, only once the main load loop reaches it, so a refusal for a *missing* piece (as opposed to one of the wrong shape) can still leave the neurons before it loaded.

**Seed provenance** (SPEC.md M.2.4). A load restores the group's RNG streams, so it also restores `effective_seed()` — the seed the run manifest reports and the next checkpoint records — from the checkpoint's `effective_seed` field (format version 5 on), else from its configuration record's entry; one with neither keeps the configured seed. The stream itself is untouched: a group resumed under another `--seed-offset` continues exactly as one resumed under the checkpoint's own. The per-cfg seeds (`neuron[<cfg>].random_seed`, `dendrite[<cfg>].rnd_seed`) are not restored: they stay the configuration's, so a re-saved record names the configured ones even though the restored per-neuron and per-dendrite streams descend from the checkpoint's.

## Activity Window Tracking

The learning window (SPEC.md M.4.2, M.4.2a) is not configured: each group keeps two windows of its own and passes them to `adapt()`, one for the step's winner and one for every other neuron. Each window's edges (its two sigmoid transition points) are running quantiles of that population's raw activity — the combined activity before inhibition, the value the window gates. Once per learning step, after adaptation, every edge `q` moves by

```
q = clamp(q + rate * (tau - share of the population's activities below q), 0, 1)
```

so it settles where a share `tau` of the activities lies below it (a Robbins–Monro quantile estimate). The winner's edges see one activity per step, the other edges all non-winners' activities at once; a step without a winner moves only the other edges. A step adapts with the edges the previous learning step left; frozen and zero-output steps move nothing.

ngm2, ngm25 and ngm3 (CPU and GPU) share one implementation, `include/ngm/window_tracking.h` (namespace `ngm::window_tracking`): the `constexpr` constants (with `static_assert`s on their ranges), the edge state, the update, and checkpoint save/load. All arithmetic is f32 so the GPU runs the same rule. The GPU groups prepend `window_tracking_wgsl()` — the constants and the edge step, generated from the header — to their WGSL; `winner_main` advances a 4-float device buffer of edges and hands the current ones to the adapt kernels in `adapt_scalars[4..7]`, so tracking adds no dispatch. `integration.impl_equivalence{,_gpu}` compares the four edges of all five implementations after every step, alongside the synapse state.

| Constant | Value | Meaning |
|----------|-------|---------|
| `rate` | 0.001 | Edge step size per learning step |
| `quantile_lower`, `quantile_upper` | 0.19, 0.94 | `tau` of the lower and upper edge |
| `steep_lower`, `steep_upper` | 0.57, 0.78 | Fixed steepness of the lower and upper edge |
| `initial_lower`, `initial_upper` | 0.33, 0.66 | Starting edges for both windows |

There is no learning-window setting: the former `activity_learning_window` (neuron cfg in ngm2, group cfg in ngm25/ngm3) is removed. The four edges are learning state, saved under `activity_window_tracking/` in every implementation's checkpoint. `window_tracking::read()` checks them before a load assigns anything: from checkpoint format 2 (`logging::k_format_version`) on, a missing group, a missing field or an edge outside [0, 1] is a corrupt checkpoint; a format-1 checkpoint predates tracking and starts the trackers over at the initial edges. The GPU groups throw if the edge readback for a checkpoint fails. The winner's edges follow the drawn winner even when it does not adapt (ngm3 skips neurons with degenerate bank maxima). Every group (ngm2, ngm25, ngm3; CPU and GPU) probes the edges after each learning step as `"window winner lower"`, `"window winner upper"`, `"window other lower"` and `"window other upper"` (the GPU groups read them back only while a probe is active); ngm2's `neuron_group_t::get_tracked_windows()` also returns both windows.

A quantile window is relative by construction: it lets roughly `upper − lower` of its population learn fully, whatever the absolute activity level.

## Configuration Structs

All in `hd_ngm2_cfg.h`. Organized hierarchically:

- **`dendrite_params_t`** — type (apical/proximal), declared input channels (their order is the input layout), permanence threshold, max branch level, learning rates, mismatch parameters
- **`neuron_params_t`** — dendrite sets, channel mappings, branch interval, noise range
- **`neuron_group_params_t`** — declared input channels (their order is the layout of `population/in`), neuron sets, channel mappings, inhibition, learning rates; `frozen`, set from the run's by `cfg_store::group_or_default`
- **`cfg_store`** — named maps of the above, with `"default"` entries pre-populated
