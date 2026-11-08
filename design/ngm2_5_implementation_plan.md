# ngm2.5 implementation plan — for the implementing agent

Status: authored 2026-07-09 on `research/ngm-par` (scaffold commit 0952c8d).
Read order before writing any code:
1. `design/ngm2_5_flat_sparse.md` — the design; decisions §13 are final.
2. `design/ngm2_5_transcription_map.md` — **binding**: regions R1–R9,
   permitted/forbidden deltas, RNG inventory D1–D5, constant sheet.
3. `include/hd_ngm25/*.h` — the scaffold; contracts in the headers are the
   spec for each signature. Extend it; don't redesign it silently.
4. `benchmarks/perf_hash_rng.md` — why the hash RNG is a structural choice
   with a ~5% price, not a performance feature. Don't re-litigate.
5. `design/ngm2_5_gpu.md` — the GPU design (device layout, growth-via-rebuild,
   CPU/GPU RNG parity, GI-track). A **separate track** that starts after the
   CPU model is validated (I9), except its premise-check (GI0) which runs early.

## Ground rules

- **One increment at a time, each ending green.** Every increment below has a
  gate; do not start increment N+1 with N's gate red. Commit per increment.
- **The map is law.** When implementation reality conflicts with the map,
  stop and surface the conflict — do not improvise a resolution. "Replicate
  first, improve later" applies to everything, including things that look
  wrong (variance-as-std, winner double-adapt).
- **No bit-parity chasing against ngm2 — at all** (user decision 2026-07-09;
  reinforces design §5/§9). It is impossible in stochastic paths past the RNG
  change, and it is *not pursued* in the deterministic paths either. The
  deterministic fixture below is a **debugging aid** (catch gross transcription
  errors by close numerical comparison + the consistency checker + the step-0
  distribution test), **not a bit-gate**. Do not widen it into a
  tolerance-chase, and do not make bit-equality-with-ngm2 a passing condition
  anywhere.
  - Bit-exactness IS still a gate in two places, but only as *self-consistency*
    of ngm2.5 against **itself** — categorically different from parity with
    ngm2: (1) run-to-run / thread-count determinism (I3/I4), (2)
    checkpoint→resume exactness (I7). These are reproducibility properties of a
    single deterministic model; keep them.
- **v1 parallel granularity is fixed**: dendrite-serial inner loops (P1),
  TBB `parallel_for` across dendrites (response) / neurons (secondary adapt).
  Start with `static_partitioner` (per-dendrite work is ngm2-sized); the
  static-vs-auto question is measured in I9, not guessed.
- **Perf is measured once, in I9.** No optimization work in I1–I8 beyond not
  writing obviously wasteful code. Every perf claim: Release verified in
  CMakeCache, interleaved A/B, paired rounds.
- Unit tests live in `tests/unit/hd_ngm25/` (pattern:
  `tests/unit/hd_ngm3/`), built with `-DBUILD_TESTING=ON` in a separate
  build dir. New sources join `coast_lib`'s list in `CMakeLists.txt`
  (ngm3 precedent at :477).
- **Testing private kernels:** implement kernel bodies as file-local free
  functions (anonymous or `ngm25::detail` namespace) taking arena/ctx views;
  the class methods are thin callers. Unit tests hit the detail functions
  directly — no friend hacks, and it matches the "kernel-shaped helper"
  precedent that made the ngm3 GPU port mechanical.

## The deterministic fixture (debugging aid for I3–I5 — NOT a bit-gate)

A fixture where ngm2 and ngm2.5 are *both deterministic*, useful for localizing
gross transcription errors by **close numerical comparison** — explicitly not a
bit-equality gate (per the ground rule above).

Construction: `neuron_noise_range = [0,0]`; input vector where every position
is either `0.0` or one global value `M`; permanences arranged so every
above-threshold synapse sits on an `M`-valued position. Then the penalty
never fires in either model (draw `U(0, M/2)` can never exceed `M`), ngm2's
draws are consumed but inert, and both response and adapt become deterministic
float pipelines. Because the map pins operation *order* (R2 accumulation order,
R7 statement order) and R1 reuses `ngm2::compute_input_stats` verbatim, the two
should agree closely.

Both models link into the same test binary (`coast_lib` contains both).
Build the ngm2 side from real `ngm2::dendrite_t`/`neuron_t` objects, copy
their post-init permanences into an arena (input-major order matches 1:1 for
unbranched dendrites), run both, compare **distributions and per-field
magnitudes** — a large divergence points at a reordered/omitted operation;
investigate it. Do **not** treat a low-order-bit difference as a failure, and
do **not** tighten toward bit-equality. The authoritative correctness gate is
the step-0 distribution test + the consistency checker + the learning-quality
pipeline (design §9). A manually-constructed *branched* fixture (known clone
layout on both sides) extends the aid once I5 exists.

## Increments

### I1 — arena, Poisson init, consistency checker (~400 LOC + tests)
Files: `src/hd_ngm25/hd_ngm25_arena.cpp`, tests.
- `check_arena_consistency` first (all five invariants from the header
  contract — the antichain-cut check is the subtle one).
- Arena construction from `group_shape_t`: one synapse per (dendrite,
  routed-input position), input-major, seg=1, Poisson(100·thres)/100
  permanences from a per-group cold engine (D5) with documented order.
Gate: unit tests — a freshly built arena passes the checker; each invariant
has a test that corrupts it and sees the checker fail with a useful message.

### I2 — repack_with_splits (~350 LOC + tests)
- Count → prefix-sum offsets → capacity-doubled rebuild per the header
  contract; children keyed D4 (dend, NEW seg, input_idx, branch_epoch).
Gate: hand-built fixtures verify structure and values; a property test runs
randomized split sequences (hundreds of rounds) asserting the checker stays
green and input-major/adjacency invariants hold; same inputs → bit-identical
output (D4 determinism).

### I3 — step context + response/combine (~600 LOC + tests)
Map R1, R2, R3. Policy-templated (`stochastic_penalty` first; the other two
instantiations compile but need no dedicated tests yet).
- R1: per-channel `ngm2::compute_input_stats` verbatim → per-position
  expansions + per-TYPE min-combined scalars (§13.1).
- R2: the sweep with per-synapse floor, tree push, `{0.25,0.5}` attenuation
  post-sweep on leaves. R3: type-max with the −1.0 apical sentinel mechanism,
  combine, frozen-gated running max.
Gate: (a) step-0 activity-distribution agreement vs ngm2 on the deterministic
fixture (close, not bit-exact — §ground-rule), single dendrite and
multi-dendrite/multi-neuron, plus the consistency checker green; (b) a two-type
fixture (apical + proximal on different channels) verifying per-type stats
routing; (c) run-to-run **bitwise** determinism across 1/4/32 TBB threads (this
one IS exact — ngm2.5 vs itself).

### I4 — winner + adapt path (~500 LOC + tests)
Map R5 (loops 3 and 4, in that order, loop 3 serial-before), R6, R7.
- The adapt sweep contains no draws — the deterministic fixture is at its most
  informative here (any divergence is a pure transcription error).
Gate: (a) adapt distribution/magnitude agreement vs ngm2 (permanence,
adapt_hist, mismatch arrays) after 1 and after 100 steps on the fixture — close,
not bit-exact — including the winner's double-adapt and per-neuron `input_count`
cadence; (b) the "no-primary-leaf dendrite" case has an explicit test; (c)
thread-count determinism (ngm2.5 vs itself, exact) as in I3.

### I5 — branch integration (~250 LOC + tests)
Map R8: per-adapt-call gate, decide predicate (variance-as-std verbatim),
`repack_with_splits` wiring, `branch_epoch_` increment.
Gate: synthetic high-mismatch drive forces branching; checker green after
every event; representation growth on the fixture tracks ngm2's schedule on
the same drive (closely, on the deterministic fixture — not asserted bit-exact);
extend the deterministic fixture across a branch event using the
manually-constructed branched pair.

### I6 — io_entity + config + generator script (~500 LOC + script)
- `process()` assembled per map §0; policy dispatch once per step.
- JSON: `include/json_cfg/ngm25_json.h` (pattern: `ngm3_json.h`), loader
  wiring in `src/json_cfg/json_loader.cpp`.
- Builder: `io_entity_type_t::ngm25_neuron_group` in
  `include/tools/sim_builder_cfg.h` (:17 enum + the switch sites),
  `sim_builder_interface.h`, `src/tools/sim_builder.cpp`.
- **The config generator** (decision §13.3 obligation):
  `scripts/ngm2_to_ngm25_config.py` — translates an ngm2 group config into
  the flat schema; used to derive every comparison config from now on.
Gate: `./build/coast -n 1000 <generated matched config>` runs headless,
output buffer sane (non-NaN, in [0,1], not all-zero after warmup);
`status_str` reports counts; generator round-trips `configs/mnist.json`.

### I7 — persistence + exact resume (~300 LOC + tests)
Map R9. Arena + directory + per-neuron carried state + `step_` +
`branch_epoch_` + D3/D5 engine states; structural validation on load.
Gate: the hash-RNG payoff test — save at step k, load, run n more steps,
compare against an uninterrupted k+n run **bit-exactly** (h5 buffer dump or
in-process fingerprint). This must hold with the stochastic policy on real
mnist input; if it doesn't, a key ingredient (usually `step_`) isn't
persisted.

### I8 — observe / probes / introspection (~400 LOC; scope gate with user)
Map R9 / pitfall P5. population in/out observe (ngm2 channel names + tiers),
group-level PROBEs, representation accessors over CSR (built on checker
invariants).
**Ask the user before starting**: which GUI/vis surface v1 needs (full
ngm2-style dendrite visualizers vs introspection-only). Do not guess; do not
silently drop capabilities (project rule).
Gate: probes visible in the probe browser; observe output readable by the h5
tooling; representation accessor cross-checked against the checker on a
branched arena.

### I9 — validation campaign (scripts + `benchmarks/perf_ngm25.md`)
Design doc §9 in full:
1. Step-0 activity-distribution test (matched init, noise [0,0]) — ngm2 vs
   ngm2.5 should differ only through RNG scheme.
2. Learning-quality pipeline vs ngm2 on mnist (adapt
   `scripts/analyze_ngm3_comparison.py`; configs from the I6 generator):
   representation-growth timelines, permanence histograms, specialization.
3. Perf: matched config, interleaved A/B vs ngm2 AND ngm3, static vs auto
   partitioner, 3+ paired rounds. Expectation to test: ngm2-parity minus ~5%
   (`perf_hash_rng.md`); ngm3 comparison quantifies the realized-structure
   advantage.
Gate: results markdown committed citing script + commit hash; learning
quality judged equivalent by the user (this call is not the agent's to make
alone); perf table complete. Deviations from expectation get analyzed, not
narrated away.

### I10 — docs, goldens, hygiene
- Hugo module page `docs/content/architecture/modules/hd_ngm25.md` (+ file
  reference, decisions entry); `cd docs && hugo` zero warnings.
- A-tier integration test + goldens for an ngm25 config (pattern:
  `tests/integration/`; commit goldens consciously).
- clang-tidy target clean vs baseline; ASan/UBSan pass on the unit +
  integration subset (`-E 'integration\.'` rule for FP-drift tests).

## GPU track (I1–I10 are the CPU track; the GPU track is `design/ngm2_5_gpu.md`)

The GPU implementation is a **separate increment track (GI0–GI6)** specified in
full in `design/ngm2_5_gpu.md` §8. It is not folded into I1–I10 because it has a
different reference (the *validated CPU ngm2.5 model*, not ngm2) and a different
dependency: **GI1+ start after CPU I9** so there is a correct model to compare
against. The one exception is **GI0, the premise-check micro-benchmark**, which
runs early — it measures whether the sparse arena on device actually beats the
measured ngm3-GPU compute floor (~1,774 SPS; still 3× slower than ngm2-CPU),
and its go/no-go is a **standing user-decision gate**. Key GPU-specific
properties (all detailed in the GPU doc): growth rides the dormant
`mark_dirty`/`build_plan` rebuild seam (dispatch grid is per-dendrite and
fixed, so only data changes at branch events); CPU and GPU draw **identical**
`pcg_hash` values when the WGSL `syn_key` transcribes `hd_ngm25_prng.h`
line-for-line (eliminating ngm3's 16–25% CPU/GPU bias by construction); no
bit-parity with ngm2 on GPU either — acceptance is CPU↔GPU learning-quality
equivalence, expected *tighter* than ngm3's.

## Definition of done

All ten gates green; the three §8 penalty policies compile and dispatch
(stochastic validated end-to-end; expected_value/none reserved for the
research matrix — NOT part of this effort's acceptance); memory + design docs
updated with outcomes, including honest recording of any deviation from the
map that was approved along the way.

## Standing user-decision gates (do not resolve alone)

1. I8 vis scope.
2. I9 learning-quality equivalence judgment.
3. Any map conflict discovered at any point.
4. Whether ngm2.5 replaces ngm2 — explicitly out of scope here (§13.4).
5. GI0 GPU premise-check go/no-go (`design/ngm2_5_gpu.md` §2) — a negative
   result stops the GPU port and is a valid outcome.
6. GI6 CPU↔GPU learning-quality equivalence judgment.

### Resolutions

- **Gates 2 (I9) + 6 (GI6) — learning-quality campaigns: consciously DEFERRED.**
  Decided by the user 2026-07-10 (drive the implementation to a complete,
  correct, checkpointable, GUI-visualizable model without gating on the
  learning-quality campaign) and re-confirmed 2026-07-19 for the PR #119 merge.
  The model ships with structural/self-consistency correctness fully gated
  (unit + observe + A-tier golden regression, checker-in-step-path, checkpoint
  round-trips) but WITHOUT the vs-ngm2 / CPU↔GPU learning-quality comparison.
  This is a deliberate, approved deviation, not an oversight: the campaign
  remains open follow-up work. Anyone treating ngm2.5 as learning-validated is
  reading more into the merge than was gated.
