# ngm3 improvement plan — reclaiming the 4.6× cell-inflation cost

Status: proposed (2026-07-09, from ngm2/ngm3 review on branch `research/ngm-par`)
Audience: implementing agent. Read this whole document before writing code.

## 1. Context and measured baseline

ngm3 replaced ngm2's grown object-graph synapse storage with a dense flat SoA
`[neuron_cnt × dendrite_cnt × rep_cnt × input_size]` per bank, where
`rep_cnt = 2^dendrite_branch_depth` is pre-allocated at init
(`src/hd_ngm3/hd_ngm3_neuron_group.cpp`, `init_storage`). This removed the
properties that blocked SIMD/GPU (data-dependent iterator advance, ragged
growth), at the cost of paying worst-case tree width from step 0.

Measured head-to-head (`benchmarks/ngm3_perf_progress_2026-05-26.md`,
Release/-O3, Strix Halo, matched 50 neurons × 5 dendrites × 784 inputs,
depth 3 → 8 reps):

| | ngm2 mature (1M warmup) | ngm3 |
|---|---|---|
| SPS | 5,195 | 1,163 |
| synapse cells | 342,223 (grown, measured from .h5) | 1,568,000 (fixed) |
| µs per synapse cell | 0.561 | 0.548 |

**Key fact: per cell touched the two models are at parity.** The 4.47× per-step
gap is entirely the 4.58× cell inflation. The cells are padding: duplicated
copies of rows that have not (yet) branched apart. Therefore the gap is
attackable by (A) not touching padding cells and (B) shrinking bytes per cell —
without giving up the dense layout that makes ngm3 SIMD/GPU-friendly.

Configs/scripts of record: `configs/ngm3_mnist_matched.json`,
`configs/mnist.json`, reproduction recipe in
`benchmarks/ngm3_perf_progress_2026-05-26.md` (bottom).

## 2. Gating measurements — do these FIRST, they size everything

We measure, we don't guess. None of the workstreams below should be
implemented before its gating measurement exists as a committed script +
result markdown in `benchmarks/` (project convention: cite script + commit
hash).

**M1 — confirm ngm3 is memory-bound on a Release build.**
The IPC-0.985 / ~85%-DRAM-stall profile in `benchmarks/perf_baseline.md` is an
*ngm2* measurement; ngm3 has never been profiled under -O3 (the only
`perf record` was on a broken -O0 build). Run
`perf stat` (cycles, instructions, LLC-loads/misses, stalled-cycles) on
`./build/coast -n 5000 configs/ngm3_mnist_matched.json`.
Expected: IPC ≪ 1, high LLC miss rate. If ngm3 turns out compute-bound
instead, Workstream B's payoff collapses and A's shrinks — stop and reassess.
Cost: ~30 min. **Verify `grep CMAKE_BUILD_TYPE build/CMakeCache.txt` says
Release before trusting any number.**

**M2 — realized-structure distribution (sizes Workstream A; fork criterion).**
From the existing ngm2 1M-step warmup checkpoint (or regenerate:
`./build/coast -n 1000000 --save-state /tmp/ngm2_warm_1M.h5 configs/mnist.json`,
~3 min), extract per dendrite:
- per-input leaf counts: reconstruct each synapse's input via a prefix sum
  over the saved `input_inc` array (the delta encoding), then collect each
  input's set of `segment_idx` values = the leaves of its personal split
  tree; and
- the **union-tree leaf count** per dendrite: each input's leaf set induces a
  partition of the `rep_cnt` slots into aligned intervals (§3.1 explains the
  alignment); the union tree is the **common refinement** of all inputs'
  partitions, and its block count is the number of rows a live-row scheme
  must keep distinct. Note this can be far larger than any per-input leaf
  count when splits are scattered across *different* inputs — that scatter is
  precisely the failure mode M2 exists to detect.
Report:
1. the retained-traffic ratio
   `sum(union_leaves × input_size) / (dendrites × rep_cnt × input_size)` —
   the fraction of today's ngm3 traffic that Workstream A keeps paying.
   Known bounds: ≥ 342,223/1,568,000 ≈ 0.218 (per-input sum; also ngm2.5's
   ratio by construction), ≤ 1.0. The gap between the union ratio and 0.218
   is what A1 leaves on the table versus ngm2.5 — the fork number.
2. the same ratio with `union_leaves` rounded to `2^⌈log2⌉` — prices the
   GPU-simpler uniform `live_depth` bound variant.
3. the trajectory: run the extraction at ~100k / 500k / 1M checkpoints (one
   warmup with `--checkpoint` yields all three). 1M is the live-row
   worst case; wall time integrates the average over training.
4. saturation: count of dendrites with any input at the depth cap (was the
   mature structure cap-limited or converged?).
Decision rule: union ratio ≲ 0.35 → A1 captures most of the win at a fraction
of ngm2.5's cost → improve ngm3. Union ratio ≈ 1.0 → A1 is not worth
building; only ngm2.5 reaches realized-structure traffic. In between: price
the residual factor against a new module's cost.
Caveat (validate-mechanisms rule): this is a data-dependent property measured
on one dataset. mnist's branching statistics are not cifar's; if the fork
decision is close, rerun the extraction on a cifar/dsprites checkpoint — the
script is dataset-agnostic once written.

**M3 — update-magnitude histograms (sizes Workstream B and A2).**
Instrument a temporary build to histogram, over ~50k mnist steps, per adapt
pass: `theta`, `|Δpermanence|`, `|Δmismatch|`, and the fraction of cells whose
`Δpermanence` is absorbed (no-op) at fp32. Motivation: for non-winner neurons
`theta ≈ segment_weight ~ synapse_weight · secondary_lr`, where
`synapse_weight` already contains `common_learning_rate` (1e-4) — the product
is plausibly ~1e-8, i.e. below even the fp32 quantum near perm≈1.0. If a large
fraction of adapt work is *already* a numerical no-op, A2 (θ-gating) is nearly
free semantically, and it tells us what precision B actually has to preserve.

## 3. Workstream A — reduce actual work

### 3.1 A1: per-dendrite live-row iteration

**Insight (verify before relying on it):** rows of one dendrite that have never
been split apart are **bit-identical**, and stay bit-identical:

- `init_storage` mirrors row 0 across all `rep_cnt` rows.
- `branch_bank`'s split writes the *same* `new_perm_a` (resp. `_b`) to every
  row of the a-half (b-half) and resets adapt_history/mismatch identically
  (`hd_ngm3_neuron_group.cpp`, pass 4). Splits always cover aligned
  power-of-two rep intervals (`split_size = 1 << (depth - bit_width(sid))`,
  starting from seg 1 covering the whole axis), so the per-input row
  partitions are aligned binary partitions and so is their union.
- `adapt_bank_neuron` updates are deterministic given (seg_id, input, weights)
  — identical rows receive identical updates. The RNG draws in the *response*
  pass differ per row but the response writes only `rep_act`, never bank
  state.

Hence each dendrite's rows fall into equivalence classes (aligned intervals);
one representative per class carries all information. The scheme:

- Keep allocation and checkpoint layout unchanged (arrays stay full-size; the
  cost we are attacking is per-step bandwidth, not RAM).
- Add per-dendrite metadata: an ordered list of class-start rows (≤ `rep_cnt`
  entries of u8; ~one cache line per dendrite). Maintained only by
  `branch_bank` (splits refine classes; nothing ever merges).
- `response_one_bank`: iterate representative rows only; write the computed
  activity to `rep_act` for the representative. The per-neuron max then
  reduces over representatives (weight the max? No — max over distinct values
  only; duplicates cannot exceed their representative except via RNG, see
  semantic note below).
- `adapt_bank_neuron`: iterate representative rows only. Duplicate rows go
  stale in memory; that is fine because the *only* consumer that ever
  re-materializes them is `branch_bank`'s split, which now copies from the
  representative's current values instead of the stale row. Introspection
  accessors (`get_*_representation`) must also redirect through the class map.
- `branch_bank`: the three stats passes already skip duplicate cells via
  seg_id comparison; with the class list they simply iterate representatives
  (cheaper and simpler than the current skip logic).

**Semantic delta — state it in the commit and test it (see §5):** today the
per-neuron activity is the max over `rep_cnt` *independent stochastic
evaluations*, including bit-identical duplicate rows whose only difference is
the RNG draw in the low-input penalty. Iterating representatives means one
draw per class, not per row — an extreme-value shift *downward*, i.e. **toward
ngm2**, which draws once per physical synapse. This is a fidelity improvement
by the "ngm2 is the reference" standard, but it changes ngm3's output
distribution, most strongly early in training when every dendrite is one
class. All existing ngm3 goldens and the CPU↔GPU parity expectations move.

**Expected gain:** step cost ∝ retained-traffic ratio from M2. If union leaves
track the per-input average (~1.75 of 8 rows), that is ~3.5–4.5× on the
matched config and ~rep_cnt× at step 0. Do not promise numbers before M2.

**RNG determinism:** draw counts become structure-dependent (fewer classes →
fewer draws). Per-neuron streams stay deterministic for a fixed run, but
run-to-run bit-comparison against pre-change builds is meaningless. Same
caveat as every prior semantic change; the noise-floor methodology in
`benchmarks/ngm3_aggregate_noise_floor.md` applies.

### 3.2 A2: gate negligible-θ adapt work (contingent on M3)

If M3 confirms that non-winner θ is ~1e-8 (numerically absorbed anyway):

- In `adapt_bank_neuron`, hoist a segment-level check: if
  `segment_weights[sid] · attenuation < θ_min` for all sids of a dendrite,
  skip the permanence/adapt_history part of the sweep for that dendrite.
- **Constraint 1:** `adapt_history += θ` must stay bounded-correct. Choose
  θ_min so that the accumulated skipped mass over one `branch_interval`
  (default 5000 steps) is ≪ `accumulated_theta_thres` (2.0). E.g.
  θ_min = 1e-7 → worst-case error 5e-4 per interval. Document the bound in
  the code.
- **Constraint 2:** the mismatch IIR update is *independent of θ* (it is gated
  by `act_ratio ≥ mismatch_act_thres`, and it reads `dendrite_perm`). Skipping
  a cell entirely would silently change branching statistics. Either keep the
  mismatch pass unconditionally (saving only perm-write + adapt_history
  traffic), or demonstrate via the §5 comparison that the combined skip is
  invisible in learning quality. Start with the conservative variant.

A2 composes with A1 (it gates work *within* the representative rows, mostly on
non-winner neurons — which is `neuron_cnt − 1` of every step).

## 4. Workstream B — shrink bytes per cell

Current cell: `permanence` f32 + `adapt_history` f32 + `mismatch` f32 +
`segment_ids` u8 = 13 B. The response pass reads only `permanence` (4 B/cell);
adapt touches everything.

Recommendation: **u16 fixed-point, not fp16**, for all three float arrays.
Rationale: all three quantities live in [0, ~4]; u16 gives a *uniform* quantum
(1.5e-5 on [0,1]) whereas fp16 has eps ≈ 9.8e-4 near 1.0 — worse than typical
primary-path updates, so fp16 would systematically absorb learning near
saturated permanences. Per array:

- `permanence` [0,1] → u16, scale 1/65535. Compute in f32 (load → convert →
  update → round-to-nearest → store). M3 tells us how much update mass falls
  below the 1.5e-5 quantum; if material, use stochastic rounding (cheap — an
  RNG draw is already in scope in response; adapt would need one, or reuse a
  hash of (cell, step)).
- `adapt_history` [0, ~4] (threshold 2.0, reset on split) → u16, scale 2^-14
  (resolution 6.1e-5, max 4.0). Increments below resolution interact with A2's
  θ_min — pick θ_min ≥ the quantum so gating and quantization tell one story.
- `mismatch` [0,1], IIR with smoothing 0.001 → u16, scale 1/65535. The IIR
  step is `mm · 0.999 + target · 0.001`; near equilibrium the per-step delta
  is ≤ 1e-3 · |target − mm|, which quantizes to zero once within ~1.5e-2 of
  the target. That plateau shifts the branch threshold statistics
  (`mm_thres = avg + var·k + 1/N`); this is the riskiest of the three — gate
  it on the §5 learning-quality comparison, and be prepared to keep mismatch
  at f32 (cell drops to 9 B instead of 7 B; still −31%).

Net: 13 B → 7 B per cell (−46%), response-pass reads 4 B → 2 B (−50%).
Combined with A1 the effects multiply.

**Checkpoint compatibility:** widen to f32 in `save_state`/`load_state` so the
HDF5 schema and `coast_compare` tolerances are untouched, or version the
schema deliberately. Do not let the storage type leak into the state files by
accident.

**GPU note:** WGSL has no u16 storage type; pack two cells per u32 (the GPU
already widens seg_ids u8→u32 for alignment, so precedent exists), or use
`shader-f16` where available. This is a real cost of B on the GPU path —
scope it before porting.

## 5. Validation protocol (non-negotiable)

1. **Perf:** interleaved A/B, never sequential blocks (a sequential-block
   comparison in this repo once produced a false −1.78%). Pair runs by round,
   3+ rounds, report paired deltas. Release build verified via CMakeCache
   before every campaign. Baselines: the §1 table's configs.
2. **Fidelity:** aggregate parity is a weak test — the documented seed-noise
   floor on the mean aggregate is ~49%
   (`benchmarks/ngm3_aggregate_noise_floor.md`). Use the learning-quality
   pipeline instead: `scripts/ngm2_ngm3_compare.sh` +
   `scripts/analyze_ngm3_comparison.py` (representation-growth timelines,
   permanence histograms, specialization heatmaps), before/after each
   workstream. For A1 specifically, add the targeted step-0 test: matched
   initial permanences, combine noise `[0,0]`, compare the neuron-activity
   distribution old-ngm3 vs new-ngm3 vs ngm2 — the extreme-value shift (§3.1)
   should be visible and should move toward ngm2.
3. **Tests:** ngm3 integration tests are `mnist_ngm3_cpu_to_gpu`,
   `mnist_ngm3_gpu_to_cpu`, `mnist_ngm3_heterogeneous`, `mnist_ngm3_philox`,
   `ngm3_noise_floor` (`tests/integration/`). The CPU↔GPU transfer tests
   assume CPU and GPU share semantics — a CPU-only change breaks them.
   Therefore: either land each workstream CPU+GPU together, or gate the new
   behavior behind a group config flag (e.g. `"live_rows": false` default)
   so existing configs/goldens stay green until the GPU port lands, then flip
   the default and re-baseline goldens **consciously** (commit the new
   goldens, `coast_update_goldens`). GPU `_gpu` tests must run outside the
   sandbox (no GPU access inside).
4. **Docs:** update `docs/content/architecture/modules/hd_ngm3.md` (layout and
   per-step algorithm sections) and record the semantic deltas of §3.1/§3.2
   as explicit decisions — this plan exists partly because the ngm2→ngm3
   deltas were never written down.

## 6. Ordering and expected end state

1. M1 + M2 + M3 (independent; parallelizable; ~a day total including scripts).
2. **Prerequisite cleanup:** delete or reunify the dead
   `compute_bank_response` (`hd_ngm3_neuron_group.cpp:281`,
   declared `hd_ngm3_neuron_group.h:155`) — it duplicates the fused
   `response_one_bank` in `compute_step` and any A1 change would otherwise
   have to be made twice. It has no callers (verified 2026-07-09). Confirm
   deletion with the user first (project rule: no silent removal of unused
   capability).
3. A1 (biggest lever, if M2 supports it) → measure → learning-quality gate.
4. B on `permanence` + `adapt_history` (straightforward) → measure; `mismatch`
   quantization only if the quality gate stays green.
5. A2 (only if M3 shows the θ mass is negligible; conservative variant first).

If M2's retained-traffic ratio is ~0.25 and B lands at 7 B/cell, the compound
effect on the matched config is roughly (0.25 traffic) × (0.54 bytes) ≈ 7×
less adapt-side traffic and ~8× less response-side — comfortably past the
4.47× gap to ngm2, i.e. ngm3 would beat ngm2 per step *while keeping the
dense, GPU-portable layout*. Treat that arithmetic as a hypothesis to be
destroyed by measurement, not a promise.

## 7. Relationship to the flat-sparse alternative

`design/ngm2_5_flat_sparse.md` describes a competing path: instead of
reclaiming padding inside ngm3's dense layout, re-lay ngm2 as a flat-sparse
CSR arena (explicit input indices, stateless hash RNG, re-pack at branch
events) so traffic is proportional to realized structure by construction.

**Fork resolved by user decision, 2026-07-09: ngm2.5 will be built regardless
of M2.** Consequences for this document: Workstream A (live rows, A1/A2) is
**on hold** — revisit only if ngm3's dense path is needed again (e.g. as the
GPU-simpler variant) after ngm2.5 results are in. Workstream B (u16
quantization) remains live: it applies to ngm2.5 unchanged and is still gated
by M3. The M1/M2 specs above are retained as reference, not as prerequisites
for ngm2.5.

## 8. Open questions for the user

- Is bit-exact continuation of existing ngm3 checkpoints across these changes
  a requirement, or is "load-compatible + statistically equivalent" enough?
  (A1 changes draw counts; exact stream continuation across the change is
  impossible.)
- Priority of CPU ngm3 vs GPU ngm3: if the GPU path is the strategic target,
  A1's per-dendrite indirection should be designed kernel-first (per-dendrite
  row lists are awkward in WGSL; a per-dendrite `live_depth` uniform bound —
  iterate `2^live_depth` rows — is GPU-simpler and recovers most of the win
  when trees are balanced; M2 can report both variants' retained-traffic
  ratios).
