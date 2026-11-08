# ngm2.5 transcription map — ngm2 semantics → ngm2.5 kernels

Status: authored 2026-07-09 against `research/ngm-par` HEAD (b917f32), with both
implementations read in full. Line numbers refer to that state; if the ngm2
files change, re-verify before trusting them.
Companion to `design/ngm2_5_flat_sparse.md` (layout, RNG, pitfalls P1–P11).

Purpose: ngm2 is the semantic reference for ngm2.5, but "replicate ngm2" is
spread over ~1000 lines with order-sensitive details. This map lists, per code
region: the target kernel, what the semantics *are*, which deviations are
**permitted** (they follow from the ngm2.5 design) and which are **forbidden**
(they would silently change the model). When in doubt: forbidden.

Verification granularity: after implementing each region, the §9 step-0
distribution test plus the consistency checker should pass before moving on.
Do not implement all regions and debug at the end.

---

## 0. Step-order skeleton (per `process()` call)

ngm2 order (`src/hd_ngm2/hd_ngm2_neuron_group.cpp:140-246`), which ngm2.5 must
reproduce exactly:

1. per-channel input stats (R1) →
2. response, parallel over neurons/dendrites (R2, R3, R4) →
3. inhibition, in place, serial →
4. [frozen: return here] →
5. group max + winner threshold draw, serial (R5) →
6. winner primary adapt, serial (R5 → R6, R7) →
7. secondary adapt of ALL neurons incl. winner, parallel (R5 → R6, R7) →
8. per-neuron branch gate → branch decide + re-pack (R8).

## 1. Carried per-step state (crosses region boundaries)

| ngm2 state | set at | read at | ngm2.5 form |
|---|---|---|---|
| `last_max_inp` (per dendrite) | response `:195-199` | adapt `:419` (`inv_last_max_inp`) | it equals max over `stats.max_val` of the channels **routed to that dendrite's type** — one constant per channel set (per type), computed in R1. *Permitted simplification (value-identical).* |
| `dendrite_type_activity[2]` (per neuron) | response `hd_ngm2_neuron.cpp:98-101` | adapt `:171-173` (type-max passed per dendrite) | scratch `f32[neuron_cnt × 2]` |
| `neuron_activity`, `max_activity` (per neuron) | response `:116-133` | adapt `:163-166` | scratch + **persisted** (`max_activity` is checkpointed, `hd_ngm2_neuron.cpp:223`) |
| `input_count` (per neuron) | `adapt()` `:177` | branch gate `:177` | **persisted**; increments once per `adapt()` CALL, not per step — winner increments twice (see R5) |
| `observed_segment_activity_`, `observed_attenuation_` (per dendrite) | response `:324-325` | `perform_observations` | per-dendrite scratch; NaN on invalid step (`:209-214`) |

## 2. RNG draw-site inventory

Every stochastic site in ngm2, with its ngm2.5 replacement. Keys per
`ngm2_5_flat_sparse.md` §5 (structural identity, never arena index).

| # | site | ngm2 stream | consumption | ngm2.5 replacement |
|---|---|---|---|---|
| D1 | low-input penalty, response `hd_ngm2_dendrite.cpp:275` | dendrite rgen | one draw per **above-threshold** synapse, in sweep order | `hash(seed, dend, seg, input_idx, step)` → U[0,1) × `half_max[i]` |
| D2 | combine noise, `hd_ngm2_neuron.cpp:113-117` | neuron rgen | 1 per neuron per step (also in frozen mode!) | `hash(seed, neuron, step)` scaled to noise range |
| D3 | winner threshold, `hd_ngm2_neuron_group.cpp:208-209` | group rgen | 1 per step | serial section — keep a group engine or hash(seed, step); either fine |
| D4 | permanence wiggle on clone, `hd_ngm2_dendrite.cpp:526,558-564` | dendrite rgen | 2 per cloned synapse | `hash(seed, dend, new_seg, input_idx, branch_epoch)` per child |
| D5 | Poisson init, `hd_ngm2_dendrite.cpp:123-130` | dendrite rgen | 1 per synapse at init | cold path — a seeded engine with defined order is fine; document the order |

Note the frozen-mode simplification: ngm2's D2 keeps advancing engine state
even when frozen (matters for checkpoint stream continuity). With hash draws,
"stream position" ceases to exist — one whole class of resume bugs disappears.

---

## R1 — per-channel input statistics

**Source:** `ngm2::compute_input_stats` (shared tools; ngm3 delegates to it) +
group `process()` `:153-156`.
**Target:** reuse `ngm2::compute_input_stats` **verbatim**, once per channel
per step; then expand to per-position arrays (`inv_sum[i]`, `half_max[i]`,
adapt-side `high_thres[i]`-family) and the min-combined `nse`/`avg` scalars.
The min-combined scalars and `last_max_inp` are **per routed channel set**
(one set per dendrite type when apical and proximal read different channels)
— design doc §13.1.
**Permitted:** the per-position expansion (value-identical lookup).
**Forbidden:** pooling stats over the concatenated input (that is ngm3's
undocumented delta — see design doc §6/P4).

## R2 — dendrite response sweep

**Source:** `dendrite_t::get_response`, `hd_ngm2_dendrite.cpp:149-328`.
**Target:** `response_dendrite(d)` kernel.
Semantics, in order:
1. clear `seg_activity[d]` (`:188`);
2. `inp_sum` = Σ channel sums; if `!isnormal(inp_sum)` → activity 0, observed
   snapshots = NaN (`:194-214`);
3. sweep synapses in arena order (= input-major):
   skip if `perm <= perm_thres` (`:273` — note: strict `>` to include, i.e.
   `perm > thres`); accumulate `seg_activity[seg] += inp`; draw D1; if
   `draw > inp` subtract `perm_strength·(1 − inp·inv_sum[i])` where
   `perm_strength = (perm − thres)·inv_perm_range`, then **clamp the running
   segment value to ≥ 0 immediately** (`:275-281`);
4. push root→leaf: `child += parent`, ascending `si` from 1 (`:289-293`);
5. attenuation `= 1 − sigmoid((nse−0.8)/0.2, {0.25,0.5}) ·
   (1 − sigmoid(avg, {0.5,0.5}))` using the **min-combined** nse/avg (`:304`);
6. leaves only: `clamp(act · attenuation · inv_inp_sum, 0, 1)`, track max
   (`:306-311`). Inner segment entries stay raw — nothing reads them later.

**Permitted:** explicit `input_idx` lookup; D1 hash draw; per-position stat
arrays; computing the sweep per dendrite in parallel with other dendrites.
**Forbidden / traps:**
- The step-3 clamp is order-dependent within a segment → serial inner loop in
  v1 (P1).
- ngm2 draws D1 for *every* above-threshold synapse (draw happens before the
  comparison). With hash draws consumption doesn't matter, but if you ever
  count draws for validation, that's the count.
- Attenuation shape here is `{0.25, 0.5}` — **different from the adapt-side
  attenuation** (R6). Copy-pasting one into the other is the single easiest
  transcription error in this codebase; ngm3 preserved the asymmetry
  (`hd_ngm3_neuron_group.cpp:294 vs :628`).
- Attenuation is applied **post-sweep, leaves only**, from min-combined stats;
  adapt-side attenuation is per-channel *inside* the sweep. Not
  interchangeable.

## R3 — neuron combine

**Source:** `neuron_t::get_response`, `hd_ngm2_neuron.cpp:82-140`.
**Target:** `combine_neuron(n)` (fusable with R2's per-dendrite loop as ngm3
did, provided the dendrite results for neuron n are complete).
Semantics: apical type-activity starts at **−1.0 as an absent-sentinel**,
proximal at 0 (`:93-95`); type-max over dendrites (`:98-101`); sentinel → 1.0
if no apical dendrite responded/exists (`:104-105`); clamp both to [0,1];
`act = clamp(apical·proximal + D2_noise, 0, 1)`; frozen-gated running max
`mx = max(mx·0.99999, act)` (`:130-133`).
**Forbidden:** replacing the sentinel with a "has apical dendrites" flag —
the sentinel semantics differ subtly if apical dendrites exist but all return
negative/zero; replicate the sentinel mechanism itself.

## R4 — inhibition

**Source:** group `process()` `:179`, delegating to the swappable
`inhibition::inhibitor_t`.
**Target:** reuse the `inhibition/` module unchanged (ngm3 already does).
No transcription needed.

## R5 — winner selection and the two adapt passes

**Source:** group `process()` `:199-236`.
**Target:** serial section + `adapt` dispatch.
Semantics, exactly:
1. frozen → return before any of this (`:199-201`);
2. `mx_act` = max over post-inhibition out; `win_act` = D3 draw from
   `U(mx_act·stochastic_win_thres, mx_act)` (`:204-209`);
3. **loop 3** (`:216-222`): scan ascending index, first neuron with
   `out[idx] + FLT_EPSILON >= win_act` gets `adapt(sigmoid(1−out[idx], wf))`,
   then `break`. Absolute epsilon, ascending order — both load-bearing.
4. **loop 4** (`:226-236`): `act_sum` = Σ out; **every** neuron, winner
   included, gets `adapt(sigmoid(1 − out[i]/act_sum, wf) · common_lr)`.

**Forbidden / traps:**
- **The winner adapts twice per step** (primary in loop 3, secondary in
  loop 4). ngm3 silently collapsed this to if/else — do not import that.
  Consequence: winner's dendrites run R6+R7 twice, and its `input_count`
  advances twice (see R8 cadence).
- `wf_params` is precomputed once per step (`:213`) — keep the hoist (it
  removed a `pow` per call).
- Loop 4 is parallel over neurons in ngm2; the primary adapt in loop 3 runs
  serially *before* it. Preserve that ordering: loop 4 reads permanences that
  loop 3 already updated for the winner. Running them concurrently is a race
  ngm2 does not have.

## R6 — per-neuron adapt envelope

**Source:** `neuron_t::adapt`, `hd_ngm2_neuron.cpp:156-184`.
**Target:** `adapt_neuron(n, weight)` prologue.
Semantics: `synapse_weight = weight · min( max(1 − sigmoid(mx_act),
sigmoid(act, window.first)), 1 − sigmoid(act, window.second) )` (`:163-166`);
then per dendrite: `adapt_dendrite(d, type_max_activity, synapse_weight)`
where `type_max_activity` is R3's per-type max (`:171-173`).
**Trap:** `sigmoid(mx_act)` uses the *default* sigmoid shape (no params) —
three different sigmoid parameterizations appear within five lines here.

## R7 — dendrite adapt sweep

**Source:** `dendrite_t::adapt_synapses`, `hd_ngm2_dendrite.cpp:342-468`.
**Target:** `precompute_segment_weights(d)` + `adapt_dendrite(d)`.
Semantics, in order:
1. `!isnormal(type_max_activity)` → whole dendrite skips adapt (`:345-347`);
2. leaf weights (`:365-377`): ascending leaf index, the **first** leaf with
   `act + FLT_EPSILON >= max_activity` gets `weight · primary_lr` and sets a
   flag so no second leaf qualifies; all others get
   `act · weight · secondary_lr / max_activity`. A dendrite may contain **no**
   primary leaf (the type-max may live in a sibling dendrite) — the flag can
   stay false; do not "fix" this.
3. fold to root (`:381-385`): `parent = max(children)` for both weights and
   activities, level by level from the leaves. After this, inner
   `segment_activity` entries mean "max leaf activity in my subtree".
4. sweep all synapses (`:422-467`), per synapse:
   - `theta = clamp(seg_weights[seg] · ramp(inp) · attenuation_ch, 0, 1)`
     where `ramp` is the two-sided distance from `high_thres[ch] =
     avg_ch/2 + FLT_EPSILON` with the two reciprocal slopes (`:440-443`);
   - permanence IIR: `perm = clamp(perm·(1−theta) + (inp > high_thres ?
     theta : 0), 0, 1)` (`:445`);
   - `adapt_history += theta` (`:449`);
   - mismatch, only if `act_ratio = seg_activity[seg]/max_activity >=
     mismatch_act_thres` (`:455-461`): sign chosen by **the just-updated
     permanence** vs threshold, magnitude `inp/last_max_inp`-based, scaled by
     `act_ratio`, folded into the IIR with `mismatch_smoothing`.

**Permitted:** explicit `input_idx`; per-position `high_thres`/attenuation
arrays (per-channel values expanded, R1); per-dendrite 128-entry scratch for
the weight/activity folds; parallelism across dendrites/neurons.
**Forbidden / traps:**
- Adapt-side attenuation uses the **default-shape** nse sigmoid
  (`:410` — no `{0.25,0.5}`), per channel, inside the sweep. See R2 trap.
- The mismatch sign reads the **post-update** permanence. Reordering the
  permanence update after the mismatch computation flips signs near the
  threshold. Keep the statement order.
- `theta` is computed from the *pre-fold-consistent* `seg_weights` of step 2/3
  — i.e. one precompute per (dendrite, adapt-call), and the winner's second
  adapt call recomputes it with the new weight. Do not cache across the two
  calls.

## R8 — branch gate, decide, and re-pack

**Source:** gate `hd_ngm2_neuron.cpp:177-183`; logic
`dendrite_t::adapt_branches`, `hd_ngm2_dendrite.cpp:476-573`.
**Target:** `branch_decide(d)` + `repack_arena()` (design doc §7).
Semantics to keep:
- **Gate cadence:** `(++input_count % branch_interval)` checked **per adapt
  call, per neuron** — winners check twice per step and their cadence drifts
  ahead of non-winners. This is per-neuron state, not a group counter (ngm3
  changed this; do not import).
- Decide (`:481-506`): `mm_avg`; `mm_std` is **variance** (mean of squared
  deviations, no sqrt — replicate the misnomer, question it later);
  `mm_thres = mm_avg + variance·min_mismatch_deviation + 1/input_size`;
  qualify = `adapt_history >= accumulated_theta_thres && mismatch >= mm_thres
  && seg·2+1 <= max_seg`; proceed only if
  `count >= input_size · min_mismatch_percentage`.
- Split effect per qualifying synapse: two children at `seg·2` and `seg·2+1`,
  `adapt_history = mismatch = 0`, permanences = parent ± D4 wiggle
  (independent draws per child), same `input_idx`.
**Permitted (by design):** the entire *execution* changes — ngm2's
back-to-front in-place clone (`:522-572`) is replaced by the count →
prefix-sum → re-pack scheme; the `input_inc = 0` bookkeeping (`:567`)
disappears with explicit `input_idx`. Only the *decide predicate and split
effect* transcribe.
**Trap:** ignore the comment at `hd_ngm2_dendrite.cpp:464-465` ("pack_size in
high bits of input_inc") — it describes an encoding that does not exist in
the code (verified 2026-07-09); `input_inc` only ever holds 0/1.

## R9 — persistence, observe, introspection

**Source:** `save_state`/`load_state` (`hd_ngm2_dendrite.cpp:670-736`,
`hd_ngm2_neuron.cpp:215-256`, group `:396-465`); `perform_observations`
(three levels); introspection accessors incl. leaf-mask logic
(`hd_ngm2_dendrite.cpp:578-643`).
**Target:** arena-shaped equivalents. Must round-trip: arena arrays + offsets,
per-neuron `input_count`/`max_activity`, tunables, inhibition group, and
whatever the hash-RNG scheme needs (seed, branch_epoch) — engines only where
engines remain (D3/D5). Observe channels should keep ngm2's names/granularity
tiers (population/neuron/dendrite) so existing analysis tooling works.
**Trap:** the leaf-mask / `get_representation` logic assumes per-input
synapse chains in input-major order — it transcribes naturally to CSR ranges,
but write the consistency checker (design doc §9) *first* and build these
accessors on top of it, not in parallel with it.

---

## Formula constant sheet (quick-reference, all load-bearing)

| quantity | formula | source |
|---|---|---|
| response attenuation | `1 − sig((nse−0.8)/0.2, {0.25,0.5})·(1 − sig(avg,{0.5,0.5}))`, min-combined stats, post-sweep, leaves only | dendrite `:304` |
| adapt attenuation | `1 − sig((nse−0.8)/0.2)·(1 − sig(avg,{0.5,0.5}))`, **default shape**, per channel, in-sweep | dendrite `:410,432` |
| perm strength | `(perm − thres) / (1 − thres)` (precomputed reciprocal) | dendrite `:83,277` |
| high threshold | `avg_ch/2 + FLT_EPSILON`, per channel | dendrite `:414` |
| running-max decay | `mx ·= 0.99999` per step, frozen-gated | neuron `:130-133` |
| winner primary weight | `sig(1 − out[idx], weight_filter)` | group `:219` |
| secondary weight | `sig(1 − out[i]/act_sum, weight_filter) · common_learning_rate` | group `:231-232` |
| adapt envelope | `min(max(1 − sig(mx), sig(act, win.first)), 1 − sig(act, win.second))` | neuron `:163-166` |
| branch threshold | `mm_avg + variance·k + 1/input_size` | dendrite `:488` |
| epsilon comparisons | absolute `FLT_EPSILON` in winner scan and primary-leaf scan | group `:218`, dendrite `:371` |
