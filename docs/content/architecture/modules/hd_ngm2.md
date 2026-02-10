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
└── hd_ngm2_tools.h           # Utility functions (sigmoid, inhibition, NSE)

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
2. **Local inhibition**: Soft winner-take-all suppresses weaker activations
3. **Stochastic winner selection**: Any neuron within top ~20% of max can win
4. **Winner adaptation**: Primary learning with sigmoid-filtered weight
5. **Common adaptation**: Secondary learning for all neurons (weak, global drift)

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `local_inhibition_strength` | 5.0 | Power law exponent for suppression |
| `stochastic_win_thres` | 0.8 | Winner selection range (fraction of max) |
| `common_learning_rate` | 0.0001 | Secondary adaptation rate |
| `weight_filter` | (0.5, 0.33) | Sigmoid shape for adaptation weight |

## neuron_t

Each neuron has apical (top-down modulatory) and proximal (main) dendrites.

**Activity computation**: `activity = clamp(apical * proximal + noise, 0, 1)` — the apical dendrite acts as a multiplicative gate on the proximal response.

**Learning window**: Adaptation is strongest at medium activity. Very low or very high activity neurons receive little learning, preventing over-specialization.

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

### Binary Segment Tree

Standard heap-style indexing: root=1, left child=2i, right child=2i+1. For `max_branch_level=3`, there are 15 segments (8 leaves at indices 8-15).

### Match Computation (`get_response()`)

1. Linear sweep: accumulate input × permanence into segment activities
2. Stochastic penalty: connected synapses with weak input reduce activity
3. Root-to-leaf propagation: parent activities cascade to children
4. NSE-based attenuation: noisy inputs get suppressed responses
5. Returns maximum leaf segment activity

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
| `local_inhibition(vec, strength)` | Soft WTA: `val *= (val/max)^(1+(strength-1)*nse_factor)` |
| `normalized_shannon_entropy(vec)` | NSE in [0,1]: 0=single spike, 1=uniform/noise |
| `sigmoid(x, shape)` | Custom sigmoid with steepness and transition point |
| `softmax(vec, beta)` | Standard softmax with temperature |
| `normalize(vec)` | Min-max normalization to [0,1] |

## Configuration Structs

All in `hd_ngm2_cfg.h`. Organized hierarchically:

- **`dendrite_params_t`** — type (apical/proximal), permanence threshold, max branch level, learning rates, mismatch parameters
- **`neuron_params_t`** — dendrite sets, channel mappings, branch interval, learning window
- **`neuron_group_params_t`** — neuron sets, channel mappings, inhibition, learning rates
- **`cfg_store`** — named maps of the above, with `"default"` entries pre-populated
