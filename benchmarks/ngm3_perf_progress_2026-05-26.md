# ngm3 Performance Progress — 2026-05-26

This document captures the state of the ngm3 parallelization work done on
2026-05-26. It is intended as a self-contained handoff: a fresh session
(human or AI) should be able to read this and pick up the GPU port without
re-deriving the analysis below.

## TL;DR

- ngm3 was rewritten from a flat-loop scalar implementation into a TBB-parallel
  per-neuron kernel with per-neuron RNG, work-stealing scheduling, and a fused
  response+combine loop. Four commits on branch `feature/ngm3`:
  `ed7a2ac` → `5eb7907` → `44eecd3` → `f2c1965`.
- **At HEAD `f2c1965` on the matched config (50 × 5 × 8 reps × 784 inputs),
  ngm3 runs at ~1,163 SPS** on Strix Halo. ngm2 mature (1M-step warmup, same
  hardware, `configs/mnist.json`) runs at ~5,195 SPS. **Per synapse-cell touched,
  ngm3 is at parity with ngm2** (ngm3 is actually ~2.3 % faster per synapse).
  The 4.5× per-step gap is entirely explained by ngm3's 4.58× more synapse
  cells (1,568,000 fixed vs 342,223 measured for mature ngm2).
- A misconfigured `build/` tree (empty `CMAKE_BUILD_TYPE`, effectively `-O0`)
  was the root cause of multiple confusing earlier results. After fixing,
  ngm2 went from 470 → 5,195 SPS, ngm3 from 150 → 1,163 SPS. **Always verify
  the build type before measuring.**

## Critical: build configuration

Before running ANY measurement, verify:

```bash
grep '^CMAKE_BUILD_TYPE' build/CMakeCache.txt
# Must print: CMAKE_BUILD_TYPE:STRING=Release

grep 'hd_ngm3_neuron_group' build/compile_commands.json | head -1 \
  | grep -oE '\-O[0-9s]+|\-DNDEBUG|\-march=[^ ]+'
# Must include: -O3, -DNDEBUG, -march=native
```

If the cache shows an empty `CMAKE_BUILD_TYPE`, the default compile is
effectively `-O0` (no optimization, no NDEBUG). Performance numbers from
such a build are 7–13× too slow and **not comparable to anything**. When
in doubt, reconfigure explicitly:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Optimization tests done on `-O0` builds are not just smaller-magnitude —
the *direction* of the effect can be wrong. Two optimizations (#2 modulo
decomposition, #3 response+combine fusion) read as "no effect" under `-O0`
but turned out to be small but real wins under `-O3`.

## What ngm3 looks like now (HEAD `f2c1965`)

`compute_step` runs the following per step:

1. **Serial:** `compute_input_stats` for each bank (apical + proximal) — small
   reductions over `input_buf`. Computes `sum`, `avg`, `max_val`, normalized
   Shannon entropy.
2. **Serial:** hoist bank-wide constants from the stats — `attenuation`,
   `inv_sum`, `low_dis_max` for each bank. These become read-only uniforms
   for the parallel loop below.
3. **`tbb::parallel_for` (auto_partitioner) over neurons — FUSED response + combine:**
   For each neuron `n`, in one thread:
   - Apical response (if `apical_cnt > 0`): for each of its `apical_cnt × rep_cnt`
     cells, `transform_reduce` over the `input_size` permanences with the
     input buffer, gated by `permanence_threshold` and a dropout-like
     random roll (`low_dis(neuron_rgens[n])`). Writes `rep_act` cell.
   - Proximal response: same, for `proximal_cnt × rep_cnt` cells.
   - Combine: `reduce_neuron_max` over the just-written `rep_act` slices to
     get `neuron_apical_max[n]` and `neuron_proximal_max[n]`; multiply +
     `noise_dis(nrng)` to get `neuron_act[n]`; decaying running max into
     `neuron_mx_act[n]`.
4. **Serial:** copy `neuron_act` → `group_outp`; `ngm2::local_inhibition`
   modifies `group_outp` in place.
5. **Serial:** find `max_element`, `std::reduce` to get sum, pick a winner
   index (first `n` with `group_outp[n] ≥ win_act`).
6. **`tbb::parallel_for` (auto_partitioner) over neurons — ADAPT:**
   For each neuron `n` (with `isnormal` predicate skipping degenerate
   activity), call `adapt_bank_neuron` for apical and proximal banks.
   `adapt_bank_neuron` pre-computes per-segment weights (tree-folded), then
   sweeps `(r, i)` over the dendrite block — modulo-free indexing, hoisted
   `cur_inp`. Updates `permanences`, `adapt_history`, `mismatch`.
7. **Serial (every `branch_interval` steps):** `branch_bank` for each bank.
   Variable control flow, scatter writes. Stays single-threaded because it
   fires ~1/5000 of the time.

Two `parallel_for` barriers per step; ~6 serial sections of varying size
between them.

### Data layout (flat SoA per bank)

For each bank (apical, proximal) — sizes are 0 for apical when
`apical_cnt == 0`:

- `permanences`: `[n × d × rep × input_size]` float
- `segment_ids`: same shape, uint8
- `adapt_history`: same shape, float
- `mismatch`: same shape, float
- `rep_act`: `[n × d × rep]` float

Plus per-neuron scratch:

- `neuron_apical_max`, `neuron_proximal_max`, `neuron_act`, `neuron_mx_act`,
  `group_outp` — each `[n]` float

Plus per-neuron RNG state:

- `neuron_rgens`: `[n]` `fast_rng_t` (currently `mt19937_simd`), seeded
  deterministically from the master `rgen` at construction. Used by
  `compute_bank_response`'s `low_dis` and `compute_step`'s `noise_dis`.

The master `rgen` is still used for `init_storage`, `win_dis`, and
`branch_bank` — all single-threaded contexts.

## Numbers (Release / -O3 / -march=native)

### Matched-config head-to-head

| | ngm2 mature (loaded from 1M warmup) | ngm3 TBB+auto+fused (HEAD f2c1965) |
|---|---|---|
| Config | `configs/mnist.json` | `configs/ngm3_mnist_matched.json` |
| Neurons × dendrites × depth | 50 × 5 × max_branch_level=3 | 50 × 5 × branch_depth=3 (→ 8 reps fixed) |
| Effective synapses | 342,223 (measured from `.h5`) | 1,568,000 (fixed) |
| Threads used | ~30 of 32 | ~23 of 32 |
| **SPS (window @ step 1010k / 3-run mean)** | **5,195** | **1,163** |
| µs / step | 192 | 860 |
| **Per-synapse cost** | 0.561 µs / synapse | 0.548 µs / synapse |
| **Per-step ratio** | 1× | 4.47× slower |
| **Per-synapse ratio** | 1× | **~0.98× (parity, ngm3 marginally faster)** |

### Optimization journey (all numbers Release / -O3 unless noted)

| Step | Change | Mean SPS | Δ | Status |
|---|---|---|---|---|
| (start) | single-thread, no TBB (estimated from `13.55 / -O0 build × 7.6` build-fix speedup) | ~100? | — | replaced by TBB |
| `ed7a2ac` | TBB `parallel_for` over neurons + per-neuron RNG, `static_partitioner` | ~1,058 (std 36) | +TBB speedup | committed |
| `5eb7907` | `static_partitioner` → `auto_partitioner` (work-stealing) | 1,138 (std 2.4) | +7.5 % | committed |
| `44eecd3` | `syn % v.input_size` → nested `(r, i)` loops | 1,154 (std 2.0) | +1.25 % paired | committed |
| `f2c1965` | Fuse apical response + proximal response + combine into one `parallel_for` | 1,163 (std 6.3) | +1.2–1.6 % paired | committed |

`auto_partitioner` is OPPOSITE the static-partitioner-wins guidance for
ngm2 in the memory notes. The reason: ngm2's per-neuron work is small
enough that cache locality dominates load balance; ngm3's per-neuron work
is much larger (8× rep duplication × 5 dendrites × 784 synapses) so the
50-neuron / 32-thread imbalance (some threads get 2 neurons, others 1)
dominates and work-stealing is the right call.

### Things tried that were correctly rejected

- **Both #2 (modulo) and #3 (fusion) initially appeared as "no effect / negative"
  under `-O0`** (+0.9 % and −0.4 %). They were reverted, then re-tested under
  `-O3` after the build was fixed. Under `-O3` both showed small, statistically
  clean positive effects (+1.25 %, +1.2-1.6 %) and were committed. Lesson:
  optimization measurements are only valid on the build configuration you'd
  actually ship.
- **Response → adapt fusion across the global step boundary**: blocked by the
  winner-selection reduction that needs all neurons' `group_outp` before adapt
  can run. Not implementable without changing the algorithm.
- **Decomposed-by-i-outer / r-inner loop order**: would let `cur_inp` be
  loop-invariant, but breaks the stride-1 access pattern on permanences. Not
  tried (the cache cost dominates the modulo cost in the opposite direction).

## On the user's 80–90 % per-thread utilization observation

After all optimizations, htop still shows per-thread utilization at ~90 %
during ngm3 runs (ngm2 hits 100 %). After `auto_partitioner` and the
response+combine fusion, this remaining 10 % is **not** from within-`parallel_for`
load imbalance (auto_partitioner steals work) and **not** from inter-`parallel_for`
barriers (the fusion test of cutting one out had ~no SPS effect on `-O0`
and a +1.3 % effect on `-O3`).

The remaining sources, ranked by guess:

- The adapt `parallel_for`'s tail of idle workers waiting for the slowest
  neuron's adapt to finish (variable due to `isnormal` skips).
- Serial sections between the two `parallel_for` calls: `compute_input_stats`,
  `local_inhibition`, winner-select reduce, `gather` input copying, output_mem
  copy. Each individually small (≤ few %).
- `branch_bank` firing every 5000 steps; over the 5000-step measurement runs
  this fires exactly once and is fully serial.
- TBB overhead at the work-stealing decision points.

This was NOT investigated further — a proper `perf record` with a Release
build (we did one with the `-O0` build, which was useless because of all
the un-inlined std symbols) is the obvious next diagnostic. But the
remaining headroom is small enough that other questions (GPU port, larger
configs, accuracy/learning quality) likely have higher leverage.

## Reproducibility recipe

```bash
# 1. Verify (or set) build configuration
grep '^CMAKE_BUILD_TYPE' build/CMakeCache.txt   # must say Release
# if not:
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. ngm2 warmup (~3 min) — produces the mature checkpoint
./build/coast -n 1000000 --save-state /tmp/ngm2_warm_1M.h5 configs/mnist.json

# 3. ngm2 mature measurement (~5 sec)
./build/coast --load-state /tmp/ngm2_warm_1M.h5 -n 20000 configs/mnist.json
# Take the SPS window value at step 1010000+ — that's the mature steady-state rate

# 4. ngm3 measurement (3-run mean, ~15 sec total)
for i in 1 2 3; do
  START=$(date +%s%N)
  ./build/coast -n 5000 configs/ngm3_mnist_matched.json > /tmp/ngm3_$i.log 2>&1
  END=$(date +%s%N)
  awk "BEGIN{w=($END-$START)/1e9; printf \"run %d: sps=%.2f\n\", $i, 5000/w}"
done

# 5. Effective ngm2 synapse count from the saved state
h5ls -r /tmp/ngm2_warm_1M.h5 | grep 'permanence Dataset' \
  | awk '{print $NF}' | tr -d '{}' \
  | awk 'BEGIN{n=0;s=0} {s+=$1;n++} END{printf "dendrites=%d total=%d mean=%.1f\n", n, s, s/n}'
```

`configs/ngm3_mnist_matched.json` matches `configs/mnist.json` shape:
50 neurons, 5 proximal dendrites, branch_depth=3, no apical, MNIST input,
seed 0.

## GPU port — recommended starting structure

The fusion commit `f2c1965` brought `compute_step` close to the shape a
hand-written GPU kernel would naturally take. The natural mapping:

1. **CPU (or a tiny GPU prelude kernel)**: `compute_input_stats` for each
   bank — small reductions over `input_buf` (~3 KB). Bandwidth-trivial.
   Bank-wide constants (`attenuation`, `inv_sum`, `low_dis_max`) become
   GPU uniform/constant-buffer values.

2. **GPU kernel `response_combine`** — one workgroup per neuron, threads
   cooperating over `(d × r × syn)`:
   - Threads collaboratively load `input_buf` into LDS (~3 KB, hot for
     entire kernel).
   - Each thread computes one (or a tile of) `(d × r)` cell's
     `transform_reduce` against permanences and input. Permanences are
     stride-1 in DRAM; coalesced loads.
   - Output `rep_act[n × d × r]` cells written by the thread that
     computed them.
   - Workgroup barrier; then a final per-neuron reduce over `d × r`
     cells to get `neuron_apical_max[n]`, `neuron_proximal_max[n]`. This
     reduce is small (40 floats) and stays in LDS.
   - Thread 0 of workgroup computes `neuron_act`, decaying `neuron_mx_act`,
     writes them to DRAM.
   - **RNG**: replace `mt19937_simd` with Philox-4x32 keyed on
     `(step, n, d, r, syn_idx)`. Stateless, deterministic, no per-thread
     state to manage.

3. **CPU**: copy `neuron_act` → `group_outp`, run `local_inhibition` (50
   floats), pick winner index. Single-threaded, ~µs.

4. **GPU kernel `adapt`** — one workgroup per neuron, threads cooperating
   over the `(r, i)` sweep:
   - Workgroup-shared LDS for `segment_weights[32]`, `segment_activity[32]`,
     and `input_buf` (3 KB). Thread 0 (or a small subset) does the tree-fold
     into LDS; barrier.
   - Each thread iterates over its assigned `(r, i)` strip, computing
     `theta`, updating `permanence`, `adapt_history`, conditionally
     `mismatch`. All writes go to its assigned slice, no atomics needed.

5. **CPU**: `branch_bank` every 5000 steps. Stays on CPU until/unless
   it shows up in profiles.

### What to watch out for in the port

- **Register pressure** in the fused `response_combine` kernel — it now
  does both banks plus the combine. Set `launch_bounds`/`maxnreg` and tune.
- **`isnormal` predicate** in the adapt loop creates thread divergence
  within a workgroup. Probably tolerable on RDNA 3.5 but worth measuring.
- **The winner-selection step is currently "first neuron above threshold
  by index"** — the user has confirmed this can be relaxed to any winner
  (e.g. argmax of `group_outp`) without changing the algorithm's intent.
  argmax is easier to parallelize.
- **Goldens**: ngm3 has no A-tier goldens yet (only a smoke test). The
  RNG swap to Philox WILL change bit-exact output but is OK because of
  this; no infrastructure to break.

### Expected speedup (honest, not aspirational)

The earlier "8–26× from GPU" estimate in the memory notes was against an
unoptimized CPU baseline. Against the current optimized CPU (1,163 SPS),
the realistic estimate on Strix Halo's unified LPDDR5x is **2–4× wall-time
speedup**, possibly less. The work is bandwidth-bound; GPU and CPU compete
for the same 116 GB/s DRAM on this machine. On a bigger machine with
dedicated HBM/GDDR (where the GPU has its own ~1+ TB/s) the upside is much
larger and the port is more compelling.

The port is worth doing if:
- We expect to scale ngm3 to neuron counts where CPU per-neuron parallelism
  plateaus (probably >> 50 neurons per group);
- We expect to run on machines with dedicated GPU memory;
- We want the option of latency hiding (GPUs hide the RNG + branch costs
  better than OOO CPUs do — the memory notes flag ngm2's "52.9 cycles per
  synapse vs 8 ideal" as latency-stall-bound).

The port is probably NOT worth doing if Strix Halo is the only target
and the current 1,163 SPS at this scale is sufficient.

## Open questions / not done

- The CPU `perf record` profile that would tell us where the remaining
  ~10 % per-thread idle goes — only done once on the broken `-O0` build.
  Re-do under `-O3` if optimizing further is desired.
- The `branch_bank` cost is unmeasured at this scale. Single-step
  measurements would tell us if it's worth GPU-porting later.
- Larger ngm3 configurations (more neurons, larger input, depth=4) — no
  scaling study done.
- ngm3 learning quality / accuracy comparisons with ngm2 — out of scope
  for this perf work, but a hard requirement before deciding ngm3 is
  the production model.

## Memory references (in `~/.claude/projects/.../memory/`)

These were updated during this session and are the authoritative records:

- `MEMORY.md` — index, mentions the stale-baseline-resolved finding
- `ngm3_vs_ngm2_2026-05-26.md` — measurement methodology and conclusions
- `ngm3_gpu_assessment.md` — original structural analysis
- `feedback_verify_release_build.md` — the bitter lesson on build types
- `feedback_job_launching.md` — duplicate-job-launch lesson from this session
