# ngm2.5 — flat-sparse neuron group (design for a future agent)

Status: implemented in `src/hd_ngm25/` (from 65cf3942, 2026-07-09); proposed 2026-07-09 as a follow-up to the ngm2/ngm3 review
Companion: `design/ngm3_improvement.md` (the dense-path alternative; shares gating
measurements M1–M3 — read that document first, especially §2).
Companion: `design/ngm2_5_transcription_map.md` — region-by-region map of ngm2
code to ngm2.5 kernels with permitted/forbidden deltas. **Implementation must
follow the map**, not a fresh reading of the ngm2 source.
Companion: `design/ngm2_5_implementation_plan.md` — the ordered increments
(I1–I10) with per-increment gates; start there when implementing.
Companion: `design/ngm2_5_gpu.md` — the GPU design (device layout,
growth-via-plan-rebuild, CPU/GPU RNG parity, the GI-track). Read it for
anything GPU; §8/P10 below are only the seed of that document.
Scaffold: `include/hd_ngm25/*.h` — contract-bearing headers (commit 0952c8d).
Audience: implementing agent. This is a design-space document, not a spec —
every quantitative claim marked *hypothesis* must be measured before it is
relied on.

## 1. Why this exists

ngm3 solved ngm2's parallelization blockers by densifying the dendrite tree to
worst-case width (`rep_cnt = 2^depth` full rows), which costs a measured 4.58×
cell inflation → 4.47× step time on the matched mnist config
(`benchmarks/ngm3_perf_progress_2026-05-26.md`). But ngm2's *object graph* and
its *sparsity* are independent properties, and only the object graph blocks
SIMD/GPU. ngm2.5 is the un-explored middle point: **flat, group-owned,
GPU-uploadable arrays that stay proportional to realized structure.** Each of
ngm2's three serial blockers falls to a targeted fix:

| ngm2 blocker | root cause | ngm2.5 mechanism |
|---|---|---|
| serial synapse sweep | input index stored as delta (`input_inc`) → prefix-sum coupling | explicit `input_idx` per synapse (u16) → random access |
| serial RNG | sequential mt19937 stream, consumption data-dependent | stateless counter-based hash keyed on structural identity (§5) — already validated on GPU (ADR-6, PCG-1D32) |
| ragged growth | per-dendrite `vector` resize in `adapt_branches` | group-owned CSR arena, re-packed only at branch events (~1/5000 steps, §7) |

The semantics target is **ngm2** (it is the reference model), not ngm3.

## 2. Decision status and remaining Demonstrate step

**The fork is resolved: user decision 2026-07-09 — ngm2.5 will be built
regardless of M2's outcome.** M2 is no longer a prerequisite for this
document (its spec stays in `ngm3_improvement.md` §2 for the day the dense
path is revisited; its per-input leaf distribution retains minor value for
arena capacity planning and load-balance expectations, but nothing here gates
on it). M3 still gates the later u16-quantization workstream
(`ngm3_improvement.md` §4), which applies to ngm2.5 unchanged.

The hash-RNG micro-benchmark (the surviving Demonstrate step) was **run on
2026-07-09** — `benchmarks/rng_hash_bench.cpp`, results in
`benchmarks/perf_hash_rng.md`. Outcome, in an ngm2-response-shaped
cache-resident loop, single-threaded, interleaved rounds:

- **Latency-chain hypothesis REFUTED.** The RNG accounts for only ~10% of the
  loop's cost (a no-RNG floor variant is 1.10× the mt19937 baseline);
  `mt19937_simd` amortizes generation into SIMD-filled buffers, so there is
  no per-draw serial chain to relieve. The scalar hash is ~5.5% *slower*
  than the buffered stream (hash_pcg 0.946×, hash_sm32 0.945×, tight across
  rounds).
- **Hash selected: `pcg_hash`** (production-identical with the GPU path,
  `src/core/gpu/shared_prng.wgsl:12`; quality χ²[df=255] = 244.9, mean/var
  nominal). splitmix32 offered no advantage.
- The hash RNG's justification is therefore **purely structural**: order-free
  draws make any parallel granularity safe, keys survive re-packs, and
  checkpoint stream-position state disappears. Budget its measured price
  (~5% on the response sweep) rather than expecting a win. If it ever
  matters, draws are independent → a batched SIMD hash path can recover it;
  build only if profiling of the real implementation says so.

## 3. Data layout

One arena per group (or per dendrite-type bank, see pitfall P8). All arrays
index by global synapse slot; dendrites are contiguous ranges.

```
// synapse arena (SoA), total_syn entries, capacity-doubled on growth
f32  syn_perm        [total_syn]
u16  syn_input_idx   [total_syn]   // explicit input position — THE key change
u8   syn_seg         [total_syn]   // segment id, heap-indexed (root=1); u8 caps depth ≤ 6
f32  syn_adapt_hist  [total_syn]
f32  syn_mismatch    [total_syn]

// dendrite directory
u32  dend_syn_offset [dend_cnt+1]  // CSR offsets; dendrite d owns [off[d], off[d+1])
u8   dend_type       [dend_cnt]    // apical | proximal
u16  dend_neuron     [dend_cnt]    // owning neuron

// per-dendrite scratch, seg_cnt = 2^(depth+1) entries each (≤128 for depth 6)
f32  seg_activity    [dend_cnt × seg_cnt]
f32  seg_weights     [dend_cnt × seg_cnt]

// per-neuron scratch: type-max activities, combined act, decaying mx_act — as ngm2/ngm3
```

Within a dendrite, synapses stay **input-major** (ascending `input_idx`,
clones of the same input adjacent) — matches ngm2's order, gives monotonic
(prefetch-friendly) input access, and makes re-packing a stable partition.

Sizing on the matched config: 342k synapses × 15 B ≈ 5.1 MB resident;
response-pass reads (perm + input_idx + seg) ≈ 7 B/syn ≈ 2.4 MB/step vs
ngm3's 6.3 MB (perm alone over 1.57M cells). The u16-quantization workstream
from `ngm3_improvement.md` §4 applies here unchanged if more is needed.

## 4. Per-step algorithm

Parallel granularity: **dendrite** (250 on the matched config — ample for CPU;
see §8 for GPU). Response and adapt inner loops stay serial per dendrite in
v1; that is what preserves ngm2's within-segment accumulation order (P1).

**Response (per dendrite):**
```
clear seg_activity[d]
for s in [off[d], off[d+1]):
    p = syn_perm[s];        if (p <= perm_thres) continue
    i = syn_input_idx[s];   inp = input[i]
    seg_activity[seg[s]] += inp
    if (penalty_policy::draw(key(d, seg[s], i), step) * half_max[i] > inp) {
        seg_activity[seg[s]] -= strength(p) * (1 - inp * inv_sum[i])
        if (seg_activity[seg[s]] < 0) seg_activity[seg[s]] = 0     // ngm2 floor
    }
push tree to leaves; leaf max; attenuation; → dendrite activity
```
Then neuron combine (type-max, apical×proximal + noise), inhibition, winner
selection, adapt sweep (same structure, `segment_weights` precomputed per
dendrite exactly as `hd_ngm2_dendrite.cpp::adapt_synapses`), branch check —
all as ngm2. The winner logic and inhibition stay serial; they are true data
dependencies and are small.

`half_max[i]`, `inv_sum[i]`: see §6.

## 5. RNG design — the most important correctness detail

Use a stateless hash (PCG-1D32 per ADR-6) in hot loops; keep engine-based RNGs
only for cold paths (init, winner draw, branch wiggle can also be hashed).

**Key by structural identity, never by arena index.** A synapse's key must be
`(group_seed, dendrite_id, seg_id, input_idx, step)` — properties that survive
re-packing. If you key by arena slot, every branch-event re-pack silently
reshuffles all subsequent draws and run-to-run reproducibility dies in a way
no test will attribute correctly. Structural keying gives a property neither
ngm2 nor ngm3 has: results independent of memory layout.

Consequences to accept and write down:
- Bit-parity with ngm2 is **unattainable and must not be chased** — the draw
  sequence differs by construction. Acceptance is learning-quality
  equivalence (§9), same as the ngm3 CPU/GPU precedent.
- After a split, children get fresh keys (new seg_id) → fresh streams; this
  mirrors ngm2's clones diverging, just deterministically.

## 6. Input statistics — replicate ngm2, not ngm3

ngm2 computes stats **per input channel** and min-combines nse/avg across
channels; ngm3 pools the concatenated bank and silently changed the semantics
for multi-input configs. ngm2.5 must keep ngm2's behavior, and can do it
branch-free: once per step, expand the per-channel constants into
per-position arrays over the concatenated input —
`inv_sum[input_size]`, `half_max[input_size]`, `high_thres`-family for adapt —
built from each channel's stats (a few hundred floats; negligible). Hot loops
then index by `input_idx` with zero channel logic. Attenuation uses the
min-combined nse/avg exactly as `hd_ngm2_dendrite.cpp:264-267`.

## 7. Growth: branch events and re-packing

Branching keeps ngm2's per-dendrite decision logic (mm_avg / variance /
qualify predicate — note `mm_std` is actually a variance in ngm2; replicate
first, question later). Execution changes:

1. Per dendrite, count clones (parallelizable, read-only).
2. Prefix-sum new per-dendrite sizes → new `dend_syn_offset`.
3. If total exceeds capacity, grow with capacity doubling (amortized O(1);
   this is why growth stopped being a hot-path concern).
4. Write the re-packed arena: per dendrite (parallelizable), copy synapses in
   input-major order, materializing each qualifying synapse as two children
   with new seg ids and hash-derived permanence wiggle. The delicate
   back-to-front in-place trick from `hd_ngm2_dendrite.cpp::adapt_branches`
   is unnecessary — with explicit `input_idx` there is no `input_inc`
   encoding to maintain (and the "pack_size in high bits" comment at
   `hd_ngm2_dendrite.cpp:464` describes an encoding that never existed —
   don't import it).

Depth cap: u8 seg ids allow depth ≤ 6 (seg_cnt 128); scratch sizing follows.
ngm2's practical configs use depth 3. Decide the cap consciously and validate
it at construction (as ngm3 does).

## 8. Stochasticity as a compile-time policy (the research seam)

Prior experiments showed removing stochasticity outright hurt learning — so
the penalty draw is presumed load-bearing until shown otherwise, and ngm2.5
should make the question cheap to research rather than hard-code an answer.

Mechanism: template the response/adapt kernels on a policy type —

```cpp
struct stochastic_penalty {           // ngm2 semantics, hash RNG
    static float draw(uint64_t key, uint64_t step);            // U[0,1) hash
};
struct expected_penalty {             // deterministic expectation
    // draw ~ U(0, max/2)  =>  P(draw > inp) = max(0, 1 - 2·inp/max)
    // apply penalty scaled by that probability instead of sampling
};
struct no_penalty {};
```

Instantiate the kernel for each policy; dispatch **once per group per step**
(function pointer or `switch` at the top). Hot loops contain zero runtime
branches on the mode; the JSON config still selects the policy at runtime
because all instantiations exist in the binary. This is the standard
policy-template pattern — cheap, and exactly what the user's compile-time
instinct points at. (A pure compile-time-only switch via `#ifdef` would force
rebuilds per experiment and prevent A/B in one process — don't.)

WGSL has no templates: the backend assembles WGSL by **plain string
concatenation** (verified — no preprocessor; `shared_prng_wgsl()` is prepended,
then the kernel body). Realize the policy variants by concatenating a **policy
prelude string** ahead of the shared body, one `register_kernel`/pipeline per
policy. See `design/ngm2_5_gpu.md` §3/§6 for the full GPU treatment.

Research matrix this enables (learning-quality pipeline, §9): stochastic vs
expected-value vs none, and noise *placement* variants (per-synapse vs
per-dendrite vs relying on the existing per-neuron combine noise). Note the
early negative result was for *removal*; the expected-value and coarser-
placement variants are untested middle ground. If `expected_penalty` holds up,
the response kernel becomes a deterministic segmented sum — trivially SIMD on
CPU and the GPU port simplifies further.

## 9. Validation and acceptance

- **Reference:** ngm2, via the learning-quality pipeline
  (`scripts/ngm2_ngm3_compare.sh`, `scripts/analyze_ngm3_comparison.py`):
  representation-growth timelines, permanence histograms, specialization
  heatmaps. Aggregate-output parity alone is too weak (documented seed-noise
  floor ~49% on the mean, `benchmarks/ngm3_aggregate_noise_floor.md`).
- **Perf protocol:** Release build verified in CMakeCache; interleaved A/B
  paired by round (sequential blocks once produced a false −1.78% in this
  repo); 3+ rounds. Baselines: ngm2 5,195 SPS / ngm3 1,163 SPS on the §1
  matched config.
- **Structural invariants:** write a consistency checker early (offsets
  monotone and bounded; input-major order per dendrite; seg ids form a valid
  tree cut per input, i.e. for each (dendrite, input) the set of seg ids is
  an antichain covering the root; clone adjacency). Run it after every branch
  event in debug runs — re-packing bugs corrupt silently and surface as
  "learning got worse", the most expensive failure mode to localize.
- **Step-0 distribution test:** matched Poisson init, combine noise `[0,0]`,
  compare neuron-activity distributions ngm2 vs ngm2.5 (they should differ
  only through the RNG scheme; a shift beyond that indicates a kernel bug).

## 10. Pitfalls checklist (each one is real; read before coding)

- **P1 — clamp order-dependence.** ngm2's per-synapse floor
  (`seg_activity < 0 → 0`) makes segment accumulation order-sensitive.
  Per-dendrite serial inner loops preserve ngm2's order; any finer-grained
  parallelism or SIMD reordering of the accumulate changes semantics. If SIMD
  of the accumulate is attempted later, it is a *semantic change* requiring
  the §9 gate — same class of decision as ngm3's per-rep floor (issue #40).
- **P2 — RNG keying** (§5). Structural identity, never arena index.
- **P3 — winner double-adapt.** In ngm2 the winner receives a primary
  `adapt()` *and* participates in the secondary all-neurons pass
  (`hd_ngm2_neuron_group.cpp:217-235`), and per-neuron `input_count` advances
  per adapt call (winner's branch cadence drifts ahead). ngm3 silently
  changed both. ngm2.5 must replicate ngm2 exactly in v1; deviate only as a
  documented decision afterward.
- **P4 — per-channel stats** (§6). Do not pool; expand per-position.
- **P5 — introspection/probes.** The `entity_element` hierarchy
  (neuron→dendrite children, PROBE sites, `get_representation`/leaf-mask
  logic in `hd_ngm2_dendrite.cpp:578-643`) disappears with the objects, as it
  did for ngm3. Visualizers and probe paths need a CSR-backed
  reimplementation; budget this — it is boring and larger than it looks. Do
  not drop capabilities silently (project rule); list any deferred ones.
- **P6 — persistence.** `save_state`/`load_state` must persist the arena +
  offsets + per-neuron carried state (`input_count`, `max_activity`) + RNG
  keys/epoch. Ragged is fine in HDF5 (1-D datasets + offsets dataset).
  Checkpoint-resume must be exact: same arena, same step → same hash draws —
  structural keying makes this easy, but `step` must be part of the persisted
  state used in keys.
- **P7 — capacity growth invalidates spans/pointers.** Anything caching
  `std::span` over the arena across a branch event dangles. Hand out offsets,
  not spans, across step boundaries.
- **P8 — apical/proximal.** ngm2 dendrite types differ in params and combine
  via type-max. Either one arena with `dend_type` (simpler) or two banks like
  ngm3 (cleaner stats separation given §6's per-position expansion covers
  channels already). One arena is recommended; banks were an ngm3 artifact of
  pooled stats.
- **P9 — sample batching is off the table.** The simulation's linked,
  double-buffered io_buffers constrain each entity to one sample in flight;
  do not architect for batched inference (confirmed by the model's author,
  2026-07-09).
- **P10 — GPU v1 scope.** One workgroup per dendrite; shared-memory
  `seg_activity` bins; the P1 clamp-order decision (final-only clamp vs
  serial-per-segment) is made and documented in `design/ngm2_5_gpu.md` §6.
  Re-pack at branch events on the **host**; because the dendrite set is fixed,
  the dispatch grid never changes and branch events are *data* updates (no plan
  rebuild except on capacity doubling) — the full growth design is
  `design/ngm2_5_gpu.md` §5. Device-side scan/compaction is a later
  optimization, not v1. **The full GPU design lives in `design/ngm2_5_gpu.md`;
  this pitfall is only a pointer.**
- **P11 — the -O0 trap.** Every perf claim: verify
  `grep CMAKE_BUILD_TYPE build*/CMakeCache.txt` first. This repo has been
  burned twice.

## 11. What NOT to attempt

- Bit-parity with ngm2 (impossible past the RNG change; chasing it wastes
  weeks — the acceptance standard is §9).
- SIMD inside the segment accumulation in v1 (P1).
- Device-side re-packing in v1 (P10).
- "Improving" the model while porting (variance-vs-std, double-adapt,
  attenuation shapes): replicate first, change as separately measured
  decisions after the port is validated. One variable at a time.

## 12. Expected outcome (hypotheses, to be destroyed by measurement)

- Traffic ∝ realized structure → per-step cost tracks ngm2 (~5,200 SPS on the
  matched config) instead of ngm3's 1,163, while keeping flat GPU-portable
  arrays. *Hypothesis; the gather-and-hash overhead per synapse is unmeasured.*
- ~~The hash RNG may relieve ngm2's 52.9 cyc/synapse latency chain~~ —
  **measured 2026-07-09 and refuted** (`benchmarks/perf_hash_rng.md`): the
  RNG is ~10% of the response-loop cost and the hash costs ~5% vs the
  buffered mt19937 stream. Expect ngm2-parity minus ~5% on the response
  sweep, not a win.
- With `expected_penalty` validated, response becomes SIMD-friendly and the
  remaining gap to hardware limits (see `perf_bottleneck_facts`) reopens.

## 13. Decisions (cleared with the user, 2026-07-09)

1. **One arena** (confirmed by the user 2026-07-09, after two questions were
   resolved). First: can one arena serve *different* input sets for
   apical vs proximal dendrites? Yes — this is exactly what the explicit
   per-synapse `input_idx` buys. All routed channels are concatenated once
   into a single group input buffer with per-channel offsets; a dendrite's
   synapses simply hold indices into the positions of the channels routed to
   its type. The response/adapt kernels are type-agnostic; only *init*
   (which positions get synapses) and the *per-type stat scalars* know about
   routing. Two consequences to implement:
   - the min-combined `nse`/`avg` and `last_max_inp` scalars are **per routed
     channel set** (per dendrite type when types differ), not global — R1 of
     the transcription map computes one scalar set per type;
   - `input_size` in the branch formulas (`1/input_size`,
     `min_mismatch_percentage · input_size`) is the **dendrite's own routed
     width**, stored per dendrite type.
   To be explicit: the one-arena choice unifies *synapse storage only*. The
   apical/proximal separation is preserved at the reduction stage — the
   neuron combine reduces per-dendrite activities into two per-neuron
   accumulators selected by `dend_type[d]` (type-max, apical −1.0 sentinel,
   `act = clamp(apical_max · proximal_max + noise)`), and the adapt pass
   hands each dendrite the type-max of its own type. See transcription map
   R3/R6.
2. **Depth cap: u8 seg ids, depth ≤ 6** (128-entry scratch, L1-resident).
3. **Config schema: new flat schema** (user decision, taken knowing the
   trade-off). Consequence: ngm2-vs-ngm2.5 comparisons need matched config
   pairs. Mitigation, part of v1 tooling: a generator script that translates
   an ngm2 group config into the equivalent ngm2.5 flat config, so pairs are
   derived, never hand-maintained — the matched-config benchmarks have
   already been an error source once.
4. **Positioning: coexist with ngm2** as a third model; ngm2 remains the
   untouched, runnable semantic reference throughout validation. Replacement
   is a separate future decision.

Remaining open: priority order vs `ngm3_improvement.md` after M1–M3 land
(the M2 fork criterion).
