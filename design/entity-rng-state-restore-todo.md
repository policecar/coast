# TODO: Incomplete entity RNG state serialization (save/load reproducibility)

Status: fixed — 98122325 ("state_io: persist fast_rng_t PRNG state across save/load", 2026-06-21)
Found: 2026-06-12, while implementing the io_buffer read-page restore fix
(commit `c7f95f0`, see `design/quiesced-state-access.md` §5.3).

## Problem

State save/load (`--save-state` / `--load-state` / `--resume`, and the runtime
GUI load) is **not bit-reproducible for CPU entities that carry a stateful RNG
advanced during `process()`**. After a load, those RNGs restart from their seed
instead of resuming where they were, so the first resumed step — and every step
after — draws different random numbers than an uninterrupted run. The divergence
is silent: outputs stay plausible (e.g. noise within `noise_range`), so it is
easy to miss.

This is **not** a defect in the simulation system or the state-IO framework.
That machinery is complete and correct:

- `state_logger` iterates entities and calls `entity.save_state(writer)` /
  `entity.load_state(reader)`; the abstract `state_writer`/`state_reader`
  interface can carry any typed field.
- The io_buffer half (read-page restore) is fixed in `c7f95f0`.

The gap lives entirely **inside specific entities' own `save_state`/`load_state`
bodies**: they serialize their "data" state but omit their RNG state, which is
mutable simulation state that determines the trajectory.

### Verified instances

- `mnist_io` (`src/io/mnist_io.cpp`): persists `cur_epoch` / `cur_idx`, **not**
  the noise RNG `fast_rng_t rgen` (and its `dis`). Noise diverges by up to
  `noise_range` after load.
- CPU `ngm3::neuron_group_t` (`src/hd_ngm3/hd_ngm3_neuron_group.cpp`): persists
  permanences / segment_ids / adapt_history / mismatch, **not** `rgen` or the
  per-neuron `neuron_rgens`. `process()` uses them every step
  (`win_dis(rgen)` stochastic winner, `low_dis(nrng)` per-synapse,
  `noise_dis(nrng)`).

### Likely additional instances (NOT yet audited — audit is part of the work)

- Other dataset loaders that share `mnist_io`'s noise-RNG pattern
  (cifar10/cifar100, dsprites, emnist, kmnist, chairs3d, color_mnist).
- CPU `ngm2` neuron group / neurons / dendrites, if they hold stateful RNGs
  advanced in `process()`.

### Exempt (no work needed)

- All **GPU** entity variants (`mnist_input_gpu`, `ngm3_neuron_group_gpu`, …).
  They use a counter-based PRNG keyed on the step count
  (`core/gpu/shared_prng`), which is stateless given the step — and the step
  count *is* restored. This is why the GPU step-once reproducibility test passes
  while a CPU one would not (and why the CPU test was scoped down to a
  deterministic read-page assertion in `c7f95f0`).

### Why it matters

- Breaks exact `--resume` / save-load reproducibility, a stated goal of the
  `fix/state-io-sim-pause` work.
- Falsifies the assumption in `design/quiesced-state-access.md` §5.3 that
  "entity state is restored correctly" — true for GPU entities, false for CPU
  stateful-RNG entities. **§5.3 should be corrected when this is addressed.**

## Solution outline

The fix is self-contained (no framework change) but spans three layers, because
the RNG type does not currently expose its state.

### 1. Give the RNG a serializable-state capability

`fast_rng_t` is `rng_t<mt19937_simd>` (`include/core/rng/fast_rng.h`). Today it
exposes only `operator()` and `seed()` — no state get/set; the state is private.

- Add state accessors to `mt19937_simd` (`include/core/rng/mt19937_simd.h`):
  expose the generator state `mt_[624]` and the consumption index `mti_`, plus a
  way to reinject them.
- Add matching pass-through accessors to `rng_t`
  (`include/core/rng/rng_wrapper.h`).

**Wrinkle (must handle for bit-exactness):** `mt19937_simd` is SIMD-batched — it
precomputes a *batch* of tempered outputs into `tempered_` and serves them via
`mti_`. So the classic "624 state words + index" is **not** sufficient to
restore mid-batch exactly. Options:
  (a) also serialize the not-yet-consumed entries of `tempered_` (plus `mti_`),
      or
  (b) restore `mt_` + a normalized index and recompute the tempered batch
      deterministically (verify it reproduces the same stream),
  (c) snapshot at batch boundaries only (simplest, but constrains when a save may
      occur — likely unacceptable).
Pick after a small experiment that confirms which representation reproduces the
stream bit-for-bit (write a focused test: draw K values, snapshot mid-batch,
restore into a fresh generator, draw K more, assert identical).

### 2. Wire each affected entity's save_state/load_state

For every entity holding a stateful RNG, persist/restore its RNG state via the
new accessors — as ordinary `state_writer`/`state_reader` fields (e.g. a
`uint32` vector for the state words + a scalar index, under a named subgroup).
Keep the existing data fields unchanged; just add the RNG fields.

- `mnist_io`: `rgen`.
- CPU `ngm3::neuron_group_t`: `rgen` **and** every entry of `neuron_rgens`
  (order matters — they are seeded in neuron order at construction).
- Any others surfaced by the audit.

Backward compatibility: on load, treat a missing RNG field as "not present"
(older saves) and fall back to the current behavior (reseed) rather than
failing the load — i.e. read with a sentinel/`has_field` guard.

### 3. Audit the full entity set

Grep for stateful RNG members advanced in `process()` across `src/io/` and the
CPU neural-model entities; confirm each is either serialized or genuinely
stateless. Document the exempt GPU variants explicitly.

## Verification

- **RNG round-trip unit test** (layer 1): the focused stream-reproduction test
  above, covering a mid-batch snapshot.
- **CPU step-once reproducibility** (the test deferred in `c7f95f0`): save at
  step N, load into a fresh env, step once, assert the post-step io_buffer state
  matches an uninterrupted run's step N+1 — now expected to pass on a CPU config.
  (Mirror `tests/integration_ngm3/state_resume_repro_test.cpp`, which currently
  runs GPU-only for exactly this reason.)
- **B-tier / A-tier**: unchanged contracts; A-tier goldens unaffected (no
  compute change to an uninterrupted run).

## Pointers

- Read-page fix this follows: commit `c7f95f0`;
  `design/quiesced-state-access.md` §5.3.
- Framework: `src/logging/state_logger.cpp`, `include/logging/state_io.h`.
- RNG: `include/core/rng/{fast_rng.h,rng_wrapper.h,mt19937_simd.h}`.
- Existing tests to mirror: `tests/integration_ngm3/state_resume_repro_test.cpp`
  (GPU step-once), `state_restore_readpage_test.cpp` (CPU read-page).
