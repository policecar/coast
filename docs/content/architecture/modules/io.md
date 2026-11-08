+++
title = "io"
weight = 40
+++

# io — Dataset Loaders

The `io` module implements dataset entities — source entities with no inputs that present training data to the simulation one sample at a time. Entities that transform a dataset's output before the neuron groups see it, such as the [retina]({{% relref "/architecture/modules/retina" %}}) node, live in their own modules.

## Directory Layout

```
include/io/
├── sequence_params.h          # Shared base: dataset::sequence_params_t (change_interval, noise_range, random_seed, shuffle, index_manifest; plus the run binding)
├── dataset_run.h              # dataset::run_binding_t (identity record); save_run_binding / restores_sequence (SPEC S.5.14)
├── sample_sequencer.h         # Sample ordering: shuffle, index manifests, epoch tracking; build_sequencer/restore_sequenced_state helpers
├── idx_io_base.h              # Shared base class for IDX-format loaders (mnist, emnist, kmnist, color_mnist)
├── cifar_io_base.h            # Shared base class for CIFAR binary-batch loaders (cifar10, cifar100)
├── dataset_cfg.h              # Common dataset type enum
├── mnist_io.h                 # idx_gray_io<Tag>: MNIST, EMNIST, KMNIST (28x28x1); mnist_io/emnist_io/kmnist_io aliases
├── mnist_cfg.h / emnist_cfg.h / kmnist_cfg.h  # their configs
├── cifar10_cfg.h / cifar10_io.h  # CIFAR-10 (32x32x3 RGB)
├── cifar100_cfg.h / cifar100_io.h  # CIFAR-100
├── dsprites_cfg.h / dsprites_io.h  # dSprites (64x64x1, HDF5)
├── chairs3d_cfg.h / chairs3d_io.h  # 3D Chairs (no sequencer — excluded from shared base)
├── color_mnist_cfg.h / color_mnist_io.h  # Colorized MNIST
└── (each _cfg.h has a matching _param_t struct)

src/io/
├── sample_sequencer.cpp       # sample_sequencer, load_index_manifest, ensure_manifest_readable
├── mnist_db.cpp               # IDX file format reader (unsigned-byte data only); rejects truncated headers, data that does not match the header and empty datasets
├── idx_io_base.cpp            # Shared implementation for IDX-format loaders
├── cifar_io_base.cpp          # Shared implementation for CIFAR loaders
├── mnist_io.cpp               # idx_gray_io process, status_str; explicit instantiations for MNIST/EMNIST/KMNIST
├── cifar10_io.cpp             # CIFAR-10 ctor: lists its batch files (1 label byte)
├── cifar100_io.cpp            # CIFAR-100 ctor: lists its file (2 label bytes)
├── dsprites_io.cpp            # HDF5-based loader
├── ... (one .cpp per dataset)
```

## Common Pattern

All dataset entities follow the same pattern. They are **source entities** with no inputs:

### Constructor
1. Load dataset from disk into memory (normalized to `[0,1]` floats)
2. Initialize the sample sequencer via `dataset::build_sequencer(params, image_count)` — resolves `change_interval`, `shuffle`, the configured seed, and any `index_manifest`
3. Initialize the input-noise RNG and distribution from `random_seed` and `noise_range`

### process()
1. Get write buffer via `output_mem()`
2. Copy current sample into buffer
3. Add noise: `clamp(val + noise, 0, 1)` per element
4. Advance the sequencer; wrap at epoch end and reshuffle the next epoch when enabled

### Configuration Parameters

Each `*_param_t` struct contains:

| Parameter | Purpose |
|-----------|---------|
| Data path(s) | Where to find the dataset files |
| `change_interval` | Steps between sample changes (0 = hold) |
| `noise_range` | Uniform noise bounds (min, max) |
| `random_seed` | RNG seed for reproducibility |
| `shuffle` | Present a fresh per-epoch sample permutation without replacement |
| `index_manifest` | Optional path to an index-manifest file restricting presentation to a sample subset |

The sequencing parameters live once in `dataset::sequence_params_t` (`sequence_params.h`); every family's `*_param_t` inherits from it. Dataset-specific parameters include `use_test_set` (CIFAR), `use_coarse_labels` (CIFAR-100), and `image_width`/`image_height` (Chairs3D — no sequencer, so no sequencing fields).

### Index Manifests

An **index manifest** is a plain-text file of sample indices into the loaded dataset (whitespace/comma-separated, `#` comments allowed). When `index_manifest` is set, the sequencer presents exactly those samples — in manifest order when `shuffle` is false, reshuffled per epoch *within the manifest* when true. This lets an external harness declare an auditable subset (e.g. a validation set) against the canonical dataset files instead of shipping re-sliced binaries. coast attaches no policy to the subset (no split roles, no eval-config validation — that belongs to the harness).

Support lives in `sample_sequencer.{h,cpp}`:

- `load_index_manifest(path, source_count)` parses the file and rejects empty manifests, duplicate indices, and out-of-range values.
- `ensure_manifest_readable(path)` is a fail-fast open check that two-phase loaders (CIFAR, dSprites) run *before* their expensive bulk reads; the manifest itself is parsed once the real `image_count` is known.
- `build_sequencer(params, image_count)` / `restore_sequenced_state(params, reader, seq, rng)` are the two shared wiring helpers every loader funnels through (via the format bases, or directly for `dsprites_io` / `mnist_io_gpu`).

When a manifest is active, the sequencer's `image_count` becomes the manifest size and `refresh_cur_idx` maps the logical position through the manifest to a source index; shuffle and epoch tracking compose unchanged. Restoring a checkpoint saved under a *different* manifest **throws** rather than resuming at an unrelated sample — either a legacy cursor (no `seq_pos`) whose source index is absent from the manifest, or a modern cursor whose restored source index doesn't match the one saved. The diagnostic says so: a checkpoint continues its sequence only on the manifest it was saved with.

`configs/mnist_ngm3_frozen_eval.json` (with `configs/mnist_ngm3_frozen_eval.manifest.txt`) is a worked example wiring `index_manifest`, `shuffle=false` and the top-level `frozen` into a forward-only pass over a fixed subset — the shape an external harness loads a trained checkpoint into for evaluation.

### Shared Format Bases

Two base classes factor out the repeated state and method bodies:

**`dataset::idx_io_base`** (`include/io/idx_io_base.h`, `src/io/idx_io_base.cpp`) — shared base for the IDX-format loaders: `dataset::idx_gray_io<Tag>` (instantiated as `mnist::mnist_io`, `emnist::emnist_io`, `kmnist::kmnist_io`) and `color_mnist_io`. Holds `output_buf`, `id`, `seq_`, `rgen`, `dis` (protected); inherits `sim::io_entity` and `mnist::mnist_db`. Implements `set_output_buffer`, `get_outp_id`, `get_outp_size` (returns `get_image_size()`), and `save_state`/`load_state` (seq + rng + identity record, restored only as `dataset::restores_sequence` decides). `idx_gray_io<Tag>` implements `process()` and `status_str()` once; the tag supplies the config type, the `ELEMENT_TYPE` display name and the status label, so each instantiation stays a distinct C++ type with its own `element_type`. `color_mnist_io` implements its own `process()`, rejects image files that are not 28×28 at construction, and overrides `get_outp_size()` (RGB output) and `save_state`/`load_state` (to also persist `current_color`).

**`dataset::cifar_io_base`** (`include/io/cifar_io_base.h`, `src/io/cifar_io_base.cpp`) — shared base for `cifar10_io` and `cifar100_io`. Holds `images`, `image_count`, `output_buf`, `id`, `seq_`, `rgen`, `dis` (protected); defines the 32×32×3 image geometry constants. Implements `process()`, `status_str()`, the dataset accessors, `set_output_buffer`, `get_outp_id`, `get_outp_size`, `save_state`, `load_state`, `load_batch_file()` (one batch file → planar-to-interleaved floats, label bytes skipped; throws on a short file or trailing data) and `rebuild_sequencer()` (called after file loading once `image_count` is known). A subclass is only a constructor that sizes the buffers, lists its batch files with its label byte count, and calls `rebuild_sequencer()`.

### Common API

All datasets provide (informally, not via virtual base; consumed by `gui_core/dataset_access.cpp` through one `resolve_as<T>` template):
- `get_image_info()` → (width, height, channels)
- `sample_count()`
- `get_sample(idx)` → `span<const float>`

### State Persistence

`save_state()` / `load_state()` persist the sequencer position (`cur_epoch`, `cur_idx`, `ci_cnt`, `seq_pos`) plus `seq_perm` and `seq_rng` in shuffle mode, so the dataset resumes from the same point in the same realised epoch order. Whether a load restores at all is the run's business, not the dataset's (`include/core/run_id.h`, `include/io/dataset_run.h`, SPEC S.5.14). A config names its run with the top-level `run_id`; `"auto"` (the default) means a fresh id — coastd fixes it once when it writes a job's run config, so every relaunch of the job is the same run, and coast draws one per process otherwise (`--resume`, and the GUI's resume from a snapshot, of an `"auto"` config adopts the checkpoint's: `sim_instance::prepare_resume`). Every checkpoint records its run's id (`metadata/run_id`), and a dataset restores its sequencer and input-noise RNG exactly when that equals the loading run's; the load says which it did. So a frozen trial from a trained state (another run) starts fresh, a coastd resume continues, and training on another subset from a trained checkpoint starts the subset's sequence at its beginning. To continue a checkpoint's sequence in a new run, name its `run_id` in the config. Each dataset also records its resolved parameters (paths, manifest, sequencing; not the seed, which is random state) in a `config` record; a restoring load compares them and warns, naming both, on a difference. The rule covers chairs3d and the GPU mnist loader as well. `dataset::run_binding_t` is not a configured setting: `sim::register_entity_handlers` fills it in at construction. A saved `seq_perm` of the loaded dataset's length must be a permutation of `[0, N)` — every entry in range, none twice — or `load_state` throws (a corrupt checkpoint; `save_state` only ever writes the realised permutation). One of another length is a cross-dataset or pre-shuffle checkpoint and is regenerated from the restored `seq_rng`, as before. The sequencer reads and checks everything before it assigns any member, so a refused load leaves it unchanged; color_mnist likewise refuses a `current_color` outside `[0, 2]` before restoring anything.

## Dataset Specifics

| Dataset | Image Size | Format | Base class | Notes |
|---------|-----------|--------|------------|-------|
| MNIST | 28x28x1 | IDX binary | `idx_io_base` | |
| EMNIST | 28x28x1 | IDX binary | `idx_io_base` | Same format as MNIST |
| KMNIST | 28x28x1 | IDX binary | `idx_io_base` | Same format as MNIST |
| Color MNIST | 28x28x3 | IDX binary | `idx_io_base` | Overrides get_outp_size + save/load for color state |
| CIFAR-10 | 32x32x3 | Binary batches | `cifar_io_base` | Planar→interleaved RGB |
| CIFAR-100 | 32x32x3 | Binary | `cifar_io_base` | Optional coarse/fine labels |
| dSprites | 64x64x1 | HDF5 | — | Binary images (0/1 → 0.0/1.0); standalone |
| Chairs3D | configurable | Images | — | Loaded via stb_image; no sample_sequencer; standalone |

## `mnist_io_gpu` — the GPU Sibling

`mnist_io_gpu` (compiled only under `COAST_WITH_GPU`) is a flat sibling of `mnist_io`: it derives from both `sim::gpu_io_entity` and `mnist_db` (reusing the IDX reader), and serves the same MNIST stream from the GPU. It is the simplest end-to-end GPU entity and a good worked example of the [gpu_io_entity]({{% relref "/key-abstractions/io-entity" %}}) contract.

- **Construction** — the entire normalized dataset (~188 MB f32 for full MNIST) is uploaded to a `dataset_gpu` `WGPUBuffer` once, via a single `wgpuQueueWriteBuffer`. Keeping the whole dataset resident (rather than streaming one image per step) keeps the entity wholly on the GPU, honouring the entity-device rule.
- **`process()`** (host bookkeeping) — advances the shared sample sequencer exactly as the CPU version does, and writes a small params uniform (image index + a PRNG key for the per-pixel noise).
- **`record()`** appends one compute kernel: read image by index + add stateless PCG noise + clamp into the output buffer. The entity emits no probes.

Because the dataset is GPU-resident and the kernel writes the output buffer directly, a downstream GPU neuron group reads it with **no copy** (`io_buffer_gpu`); a CPU consumer triggers a `gpu_to_cpu` readback at swap.
