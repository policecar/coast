# ngm2.5 — GPU design (companion to the flat-sparse design)

Status: implemented in `src/hd_ngm25/` (from 0bc98b9f, 2026-07-10); proposed 2026-07-09. Extends
`design/ngm2_5_flat_sparse.md` (§8 policy seam, P10 GPU-v1 scope) into a full
device design. Grounded in the *measured* ngm3-GPU reference, not in
speculation — every performance claim cites a benchmark or is marked
*hypothesis, to be measured*.

Read first:
- `design/ngm2_5_flat_sparse.md` — the model, the arena layout (§3), the
  per-step algorithm (§4), the RNG (§5), the pitfalls (P1–P11).
- `design/ngm2_5_transcription_map.md` — R1–R9 semantics; the GPU kernels
  transcribe the *same* regions, so the map is binding here too.
- The reference implementation this mirrors: `src/hd_ngm3/hd_ngm3_gpu.wgsl`,
  `src/hd_ngm3/hd_ngm3_gpu_neuron_group.cpp`, and the abstraction seam
  `include/core/gpu/gpu_device.h`.

The bit-parity stance is unchanged and reinforced by user decision
(2026-07-09): **bit-identity with ngm2 is not a goal and must not be chased.**
Acceptance is distribution / learning-quality equivalence (§7), the same
standard as the CPU path and the ngm3 CPU/GPU precedent.

## 1. What is known, and why this is worth doing

Measured facts (do not re-derive; cite):

- ngm3-GPU on RADV runs **~1,774 SPS** on the matched mnist config, and is
  **GPU-compute/occupancy-bound, not sync-bound**
  (`benchmarks/.../gpu_dispatch_bottleneck_2026-06-13`, WAIT_IDLE≈508µs of
  564µs/step). Dispatch/submit/WebGPU overhead was ruled out as dominant.
- ngm3-GPU (1,774 SPS) is **still ~3× slower than ngm2-CPU (5,195 SPS)**. The
  GPU port did not beat the sparse CPU model; it beat *dense* ngm3-CPU
  (1,163 SPS). This is the number ngm2.5-GPU must improve on, and the reason
  the premise-check (§2) exists.
- The occupancy lever is real: the single-workgroup passes (stats, winner,
  inhibition) serialize the whole GPU; a prior adapt-occupancy fix bought
  ~7.6×. Kernel occupancy, not sync removal, is the RADV lever.
- Both ngm3-GPU response/adapt kernels are structurally **bandwidth-bound**
  (~1.5 and ~0.7 FLOP/byte vs the Radeon 8060S ~128 FLOP/byte balance);
  the GPU's win over CPU is **latency hiding**, not raw FLOPs
  (`ngm3_gpu_assessment`, `perf_bottleneck_facts`).

The ngm2.5 thesis on GPU: the dense ngm3 kernel touches `rep_cnt = 2^depth`
inflated arrays (4.58× the realized cell count). A sparse arena that stays
proportional to realized structure moves **~4.6× less data per step** through
the bandwidth-bound kernels. If the ngm3-GPU cost is bandwidth-dominated, that
should translate toward ngm2-CPU parity or better on the same adapter — *while
keeping the flat, GPU-uploadable arrays*. **This is the entire value case, and
it is unmeasured. §2 measures it before the full port is committed.**

## 2. GI0 — the premise-check (measure before you port)

The whole GPU effort rests on "sparse arena on device beats the dense
ngm3-GPU compute floor." Validate that on a throwaway harness **before**
writing the production kernels — a few hundred lines that either green-lights
the port or kills it cheaply. This is the honest ordering (measure the thing
the plan rests on, first); it is GI0 in §8.

Harness: a synthetic realized-sparse arena (matched-config sizes: ~342k
synapses across 256 dendrites, input-major), the response sweep only
(R2 accumulate + penalty + tree-push + attenuate), one workgroup per dendrite,
`pcg_uniform` draws. Register buffers and a one-pass `step_plan` through the
real `gpu_device` façade; drive `begin/enqueue/submit/rendezvous` in a tight
loop with `COAST_GPU_TIMING`-style WAIT_IDLE instrumentation.

Gate (RADV, outside the sandbox — the sandbox has no GPU):
- measured GPU exec time per sweep vs the ngm3-GPU response-pass share;
- **go criterion:** the sparse sweep's per-step device time is below the
  ngm3-GPU response share by a margin consistent with ~4.6× less traffic
  (allow for fixed launch overhead). If it is not — if occupancy or launch
  overhead dominates at this granularity — record the number and **stop**;
  the port does not pay, and that is a valid outcome.

## 3. Mapping onto the abstraction seam

Use the existing `coast::gpu::gpu_device` façade and `step_plan` POD
(`include/core/gpu/gpu_device.h`) exactly as ngm3-GPU does. Do **not** touch
`<webgpu>` — the type is `ngm25::neuron_group_gpu_t : sim::gpu_io_entity`,
compiled under `COAST_WITH_GPU`, mirroring `ngm3::neuron_group_gpu_t`.

Fixed patterns inherited from the seam (verified against the backend):

- **WGSL assembly is string concatenation**, not a template engine. Prepend
  `coast::gpu::shared_prng_wgsl()` (gives `pcg_hash`/`pcg_uniform`), then the
  kernel source. The penalty-policy variants (§6) are therefore realized by
  concatenating a **policy prelude string** ahead of the shared body — one
  `register_kernel(source, entry)` per policy → one pipeline per policy. All
  three compile; the config selects which `kernel_id` the plan references.
- **Cross-device data movement rides in the plan**, because the façade's
  `enqueue_transfers()`/`complete_transfers()` are still stubbed-to-throw in
  the webgpu backend (node F2). Follow ngm3: per-channel **gather copies**
  (`plan_op{kind=copy}`, `src_side=read` for the one-step-latency upstream
  page) into a resident `input_concat` buffer at the top of the plan, and a
  **publish copy** of the post-inhibition page into a stable `group_outp_`
  buffer mid-plan so the winner/adapt passes read a non-ping-pong source.
- **Readback is on-demand and gated** (`gpu_device::readback`, staging-buffer
  copy + map + pump), never per-step in steady state. Buffers that are read
  back need `CopySrc` (the seam sets it for all non-uniform buffers).
- **Faults** surface as `core::step_error{device_fault}` via the backend
  firewall (`check()`); WGSL compile errors throw at `register_kernel` time
  with the naga diagnostic. No change needed.
- **The null backend** (`COAST_GPU_BACKEND=null`) drops compute passes and
  keeps copies — use it to test the plumbing (registration, plan build,
  swap, generation flip, checkpoint round-trip) with zero GPU.

## 4. Device data layout

One caveat dominates the layout: **WGSL storage is 32-bit-granular**. ngm2.5's
CPU arena packs `input_idx` as u16 and `seg` as u8; on device both **widen to
u32** (ngm3 already widens its u8 `segment_ids` to u32 for the same reason).
The bandwidth math must use the device widths.

Resident buffers (registered `topology::gpu_resident`, capacity-sized — see §5):

```
f32  syn_perm        [cap]        // permanence
u32  syn_input_idx   [cap]        // widened from u16
u32  syn_seg         [cap]        // widened from u8
f32  syn_adapt_hist  [cap]
f32  syn_mismatch    [cap]
u32  dend_offset     [dend_cnt+1] // CSR; changes at branch events (data, not plan)
                                  // dend_type / dend_neuron: fold into a small
                                  // per-dendrite uniform buffer (fixed size)
f32  seg_scratch     [dend_cnt * seg_cnt]   // workgroup-shared in-kernel; a
f32  seg_weights     [dend_cnt * seg_cnt]   // resident copy only if adapt needs it
f32  input_concat    [input_size]           // gather target (§3)
f32  group_outp      [neuron_cnt]           // stable publish target
// per-neuron scratch: type-max×2, neuron_act, neuron_mx_act, winner scalars — as ngm3
```

Response-pass read set per synapse on device = `perm`(4) + `input_idx`(4) +
`seg`(4) = **12 B/syn**; matched config ≈ 342k × 12 ≈ **4.1 MB/step** vs
ngm3-GPU's dense `perm`-alone ≈ 6.3 MB over 1.57M cells, over ~4.6× fewer
elements. The u16-repack of `input_idx` (two positions per u32, manual
bit-unpack in WGSL) from `ngm3_improvement.md` §4 is a **deferred** bandwidth
optimization — gate it on M3 quantization results, not v1.

Per-step host→device traffic: the per-channel input stats (§6/R1) and the
policy/step uniforms only — a few hundred floats, mirroring ngm3's two
uniform uploads.

## 5. Growth: the one real structural difference from ngm3-GPU

ngm3-GPU is fixed-topology (dense), so its branch pass rewrites seg-ids in
place and the graph never rebuilds. ngm2.5 **grows the arena** at branch
events. The design turns on one structural fact:

**The dispatch grid is one workgroup per dendrite, and the dendrite set is
fixed at construction** (`neuron_cnt × (apical+proximal)`; branching splits
*synapses within* a dendrite, never adds dendrites). Therefore:

- **The `step_plan` dispatch dimensions never change at a branch event.** Only
  `dend_offset[]` contents, the arena contents, and the active synapse count
  change — all of which are *data* (uploads), not plan structure.
- **Register the arena buffers at capacity, not at current size.** A
  `syn_count` uniform bounds every per-dendrite grid-stride sweep. A branch
  event that stays within capacity is a pure data update: upload the repacked
  arena + new `dend_offset` + new `syn_count`; **no plan rebuild.**
- **Plan rebuild happens only when capacity is exceeded** (capacity-doubling,
  ~log₂ times over a whole run, not per branch event). That is exactly the
  dormant `mark_dirty(plan_id)` → `build_plan()` seam ngm3 never exercised;
  `build_plan` is re-callable and drains before releasing old resources.
  Re-register the larger buffers, rebuild the plan, re-upload. Rare and
  explicit (a synchronizing barrier, as the abstraction design notes).

**Where the repack runs (v1): on the host** (P10). On a branch step the host
already tracks the per-neuron gate cadence (`input_count_`, R8 — winners drift
ahead, replicate exactly). When a neuron gates:
1. readback that neuron's dendrites' `mismatch` + `adapt_hist` (small);
2. run the CPU `branch_decide` + `repack_with_splits` (§7 of the flat-sparse
   doc) — **the same host code the CPU model uses**, so split semantics have a
   single source of truth;
3. upload the changed arena ranges + updated `dend_offset` + `syn_count`.

Honest caveat: per-neuron gates are **staggered**, not synchronized to one
global 1/5000 step, so repack fires on more than 1/5000 of steps (bounded by
~neuron_cnt/branch_interval fraction, and only when synapses actually qualify).
A full-arena readback+repack+reupload per event is a few MB; it amortizes, but
if I9 perf shows repack cost is material, the optimization path is (a)
per-dendrite partial repack with per-dendrite capacity slack (breaks the single
contiguous arena — measure the trade), or (b) device-side scan/compaction —
**both deferred past v1** (P10). Device-side branch *decide* (producing
`clone_flags` on device, like ngm3's `branch_main`) is a cheap independent
optimization that can land before device-side *repack*; note it, don't build
it in v1.

## 6. Per-step dispatch plan

Ordering follows the map §0 skeleton and the ngm3-GPU plan, adapted to the
sparse arena. Passes marked **[host]** run CPU-side; **[dev]** are GPU compute
passes in the `step_plan`.

1. **[host] per-channel input stats (R1).** ngm2.5 keeps ngm2's per-channel
   min-combined semantics (not ngm3's pooled bank stats, P4). These are a few
   hundred floats; compute them on the host and upload as a small buffer +
   the per-position expansions. **This deletes ngm3-GPU's single-workgroup
   `stats_main` pass** — one fewer serializing pass, and it restores ngm2
   semantics for free. (ngm3 computed stats on device *because* it had already
   pooled them; ngm2.5 has no reason to.)
2. **[dev] gather copies** → `input_concat` (§3).
3. **[dev] response + combine (R2 + R3),** one compute pass, **one workgroup
   per dendrite** (finer than ngm3's per-neuron — more workgroups, better
   occupancy on the bandwidth-bound sweep). Within a dendrite workgroup:
   grid-stride the synapses into workgroup-shared `seg_activity` bins
   (≤128 f32, L1/LDS-resident), tree-push root→leaf, attenuate leaves
   (min-combined stats, `{0.25,0.5}` shape — **not** the adapt shape, R2
   trap), write dendrite activity. Then a per-neuron reduction pass (or thread-0
   in a per-neuron workgroup, as ngm3 does) applies the apical −1.0 sentinel,
   `clamp(apical·proximal + D2 noise)`, and the frozen-gated running max (R3).
4. **[dev] inhibition** — reuse `inhibition::gpu_inhibitor_t::emit(...)`
   unchanged (R4; ngm3 already does). Single workgroup (occupancy cost; shared
   with ngm3, out of scope to fix here).
5. **[dev] publish copy** → `group_outp_`.
6. **[frozen] stop here** — the plan omits the learning passes when
   `params.frozen` (as ngm3 does); with hash draws there is no stream to keep
   advancing.
7. **[dev] winner (R5).** Single-workgroup argmax + D3 threshold draw →
   `{winner_idx, act_sum, win_act, mx_act}`. Small; keep on device (a host
   round-trip would add a sync point).
8. **[dev] adapt (R5 loops 3+4 → R6 → R7).** As shipped this is two distinct
   dispatches, honoring the winner's double-adapt (P3) and the
   serial-loop-3-before-parallel-loop-4 ordering (R5 trap; loop-4 must read the
   perms loop-3 wrote): a single-workgroup **primary** pass (the winner only),
   then an all-neuron **secondary** pass dispatched **one workgroup per neuron**
   (each looping its own dendrites), with a storage barrier between them. Note the
   secondary granularity is per-neuron, not per-dendrite (unlike the response
   pass at step 3): the sweep updating perm (IIR) / adapt_hist / mismatch and the
   `seg_weights` fold run inside each neuron's workgroup.
9. **[dev, uniform_gated] branch decide** *(optional v1)* or **[host] branch**
   per §5. If the decide runs on device, mark the pass `uniform_gated` and
   early-out on the per-neuron cadence, emit `clone_flags`; the host reads them
   back only when a gate fires. v1 may do the whole gate+decide on host from a
   small readback — simpler, and the cadence bookkeeping already lives there.

### The P1 clamp-order decision on GPU (must be made and written down)

ngm2's per-synapse penalty subtract + immediate `≥0` clamp makes the segment
accumulation order-sensitive (P1). The accumulate *without* the clamp is
order-free (addition); **only the intermediate clamp is order-sensitive.**
v1 GPU choice, consistent with "no bit-parity":

- **Recommended:** parallel grid-stride accumulate into the shared seg bins,
  applying the penalty per synapse but clamping the bin to ≥0 **once, after
  the sweep** (not per-synapse). This is a documented semantic delta from
  CPU-serial ngm2.5, of the **same class** as ngm3's per-rep floor decision
  (issue #40), and it is validated by the §7 distribution / learning-quality
  gate — *not* by bit-comparison. State it in the kernel header.
- **Fidelity fallback** (only if learning quality regresses): serialize the
  accumulate per segment (one thread per segment, or an ordered subgroup
  reduction), recovering ngm2's per-synapse clamp order at an occupancy cost.

**Superseded 2026-10-19 (SPEC.md M.0.6, M.7.31).** The
implementations must compute the same model, so the final-only clamp is gone. The
per-synapse clamp is recovered *without* the occupancy cost: each synapse acts on
its segment sum as x ↦ max(x + a, b), and such maps compose associatively —
(a₁,b₁) then (a₂,b₂) = (a₁+a₂, max(b₁+a₂, b₂)). Each thread composes the maps of a
contiguous synapse run, and an order-preserving tree combines adjacent runs, so
the result equals the serial CPU sum up to floating-point summation order. Checked
step by step against the CPU implementations by `integration.impl_equivalence_gpu`
(max output difference ≤ 5e-7); throughput unchanged (interleaved A/B, +0.7 %,
within noise). See the kernel header in `src/hd_ngm25/hd_ngm25_gpu.wgsl`.

Under the `expected_penalty` policy (§6 of the flat-sparse doc) the penalty is
deterministic and this question softens considerably — another reason that
policy is attractive for the GPU path.

## 7. RNG: identical draws on CPU and GPU (a real advantage, use it)

ngm3's CPU used mt19937 and its GPU used `pcg_hash`; the two draw *different*
numbers, and the result is a documented **systematic 16–25% CPU/GPU mean
bias** (`ngm3_gpu_parity_2026-06-05`, the open fidelity thread). ngm2.5 does
not inherit this, **by construction**: ngm2.5-CPU already draws with
`pcg_hash(syn_key(seed, step, dend, seg, input_idx))` (`hd_ngm25_prng.h`).

**Hard requirement:** the WGSL `syn_key` / D2 / D3 / D4 key composition must be
a **line-for-line transcription of `include/hd_ngm25/hd_ngm25_prng.h`** — the
same multipliers (`3266489917`, `2246822519`, `668265263`, `374761393`), the
same XOR structure, the same `pcg_hash`, the same `to_unit` 2^-32 scaling. Do
**not** copy ngm3's WGSL key form (`(base+i)*2654435761 ^ …`) — it is a
different composition. When this holds, CPU and GPU draw **bit-identical random
values** for the same synapse at the same step, and CPU↔GPU divergence reduces
to floating-point **reduction-order** differences only (the P1 clamp choice,
parallel reductions) — no RNG-distribution gap. Expect **tighter** CPU/GPU
agreement than ngm3 achieved.

Guard it cheaply: a unit test that evaluates `syn_key` over a grid of
`(seed, step, dend, seg, input_idx)` on the host and inside a tiny WGSL kernel
and asserts equality. This is a GPU gate (§8, GI2) — it is *self-consistency*
(ngm2.5-CPU vs ngm2.5-GPU key math), which is legitimate and cheap; it is not
bit-parity with ngm2.

## 8. GPU increments (GI-track)

The GPU track depends on a validated CPU reference: **GI1+ start after CPU I9**
(the CPU learning-quality gate) so there is a correct model to compare against
— **except GI0, which runs early** to de-risk the premise. Each increment ends
green; commit per increment; the map is law here too.

- **GI0 — premise-check micro-benchmark** (§2). Throwaway. Gate: measured
  device sweep time vs the ngm3-GPU response share; go/no-go recorded in
  `benchmarks/perf_ngm25_gpu_premise.md` (script + commit hash). *This gate is
  the user's call, not the agent's.*
- **GI1 — device entity + static plan plumbing.** `neuron_group_gpu_t`,
  capacity buffer registration, arena upload, gather/publish copies, the
  response `step_plan`. Gate: builds under `COAST_WITH_GPU`; the **null
  backend** runs the full register→build→step→swap→checkpoint loop; buffer
  sizes and offsets match the CPU arena for the same config.
- **GI2 — response + combine WGSL (R2/R3) + RNG parity guard.** Policy-prelude
  concatenation (stochastic first). Gate: (a) the §7 host/WGSL `syn_key`
  equality test; (b) CPU↔GPU response **distribution** agreement on the §9
  deterministic fixture (close, not bit-exact — no widening-to-bit-parity);
  (c) run-to-run determinism (fixed-order reductions, no atomics) as ngm3-GPU
  achieves.
- **GI3 — winner + adapt WGSL (R5/R6/R7).** Winner double-adapt and loop
  ordering honored. Gate: CPU↔GPU adapt distribution agreement over 1 and 100
  steps on the fixture; the P1 clamp-order decision recorded in the kernel
  header.
- **GI4 — growth on host + capacity-doubling rebuild (§5).** `syn_count`
  uniform bounds sweeps; branch gate cadence on host; `repack_with_splits`
  reused; `mark_dirty`/`build_plan` only on capacity crossing. Gate:
  consistency checker green after every event (on readback); a forced
  capacity-doubling exercises the rebuild path (pattern: ngm3's G7
  forced-rebuild test); representation growth tracks the CPU model on a matched
  drive.
- **GI5 — observe / probes / checkpoint on GPU.** `read_back_sync` for
  GPU-only `group_outp`; OBSERVE population channels (ngm2 names) + group
  PROBEs (discovered-by-execution, gated readback); `save_state` reads the
  arena back to host and writes the CPU schema; `load_state` refuses on
  structural mismatch and re-uploads. Gate: checkpoint→resume **bit-exact
  against an uninterrupted run** — this is ngm2.5-vs-itself reproducibility
  (the hash-RNG payoff), legitimate and cheap; observe output readable by the
  h5 tooling.
- **GI6 — GPU validation campaign** (`benchmarks/perf_ngm25_gpu.md`).
  1. CPU↔GPU learning-quality parity on mnist (expect tighter than ngm3 per
     §7); 2. occupancy/perf: device time breakdown (WAIT_IDLE instrumentation),
     vs ngm3-GPU and vs ngm2-CPU, on RADV **outside the sandbox**;
     3. the single-workgroup passes (winner, inhibition) quantified as the
     residual occupancy cost. Gate: results markdown committed (script +
     commit hash); learning-quality equivalence is the **user's** judgment;
     perf table complete; deviations analyzed, not narrated away.

Deferred past v1 (record, don't build): device-side repack/compaction;
device-side branch decide; u16 bit-packing of `input_idx`; bf16 storage of the
large arrays (`ngm3_gpu_assessment` — attacks the bandwidth bound, but only
after fp32 is stable); keep-warm / native Metal/Vulkan backends (gated on the
PR #57 W1/M2 measurements, `gpu_abstraction_layer_design_2026-06-14`).

## 9. Validation and acceptance (no bit-parity)

- **Reference:** the validated ngm2.5-CPU model (not ngm2, not bit-exact).
  Same learning-quality pipeline as the CPU path
  (`scripts/analyze_ngm3_comparison.py`): representation-growth timelines,
  permanence histograms, specialization heatmaps. Aggregate-output parity
  alone is too weak (seed-noise floor ~49% on the mean).
- **Deterministic fixture** (reused from the CPU plan, §9 of the impl plan) is
  a *debugging aid* for CPU↔GPU comparison, **not a bit-gate**: compare
  distributions and spot gross transcription errors; do not widen to a
  tolerance-chase and do not require bit-equality.
- **Self-consistency gates that ARE exact** (legitimate, cheap, kept): the §7
  `syn_key` host/WGSL equality; run-to-run determinism of the GPU path;
  checkpoint→resume bit-exactness of the GPU path against itself. These are
  ngm2.5-vs-itself reproducibility, categorically different from parity with
  ngm2.
- **Perf protocol:** RADV, outside the sandbox; Release verified in
  CMakeCache (P11); `COAST_GPU_TIMING`-style WAIT_IDLE breakdown; interleaved
  A/B where CPU comparison applies; **exclusive GPU** (a background load
  produced a 2.5× outlier once). Baselines to beat/contextualize: ngm3-GPU
  1,774 SPS, ngm2-CPU 5,195 SPS.

## 10. GPU pitfalls (additional to P1–P11)

- **G1 — single-workgroup serialization.** stats (deleted here, §6),
  winner, inhibition serialize the GPU. Keep them small; do not add more.
  This is the RADV occupancy lever — the reason per-dendrite (not per-neuron)
  granularity is chosen for the big sweeps.
- **G2 — 32-bit storage widening.** `input_idx`/`seg` are u32 on device
  (§4). The bandwidth budget and any bit-packing plan must use device widths.
- **G3 — key composition must match `prng.h` exactly** (§7), or the CPU/GPU
  fidelity advantage evaporates and ngm3's bias returns. Guard with the
  equality test.
- **G4 — buffers can't resize in place.** Capacity-doubling with a `syn_count`
  uniform (§5); plan rebuild only on capacity crossing, and it is a
  synchronizing barrier — explicit, never silent.
- **G5 — the transfer methods are stubbed.** `enqueue_transfers`/
  `complete_transfers` throw in the webgpu backend; move data via in-plan
  copies + on-demand `readback` (§3), as ngm3 does. Do not call them.
- **G6 — readback needs `CopySrc` + sim-thread + device-idle.** Any buffer you
  observe/checkpoint must be `CopySrc`; all readback runs on the device-owning
  sim thread at a step boundary (`ngm3_gpu_observability`), never the GUI
  thread.
- **G7 — UMA is not zero-copy.** WebGPU forbids STORAGE+MAP_READ, so readback
  always stages+copies even on Strix Halo's shared memory
  (`gpu_dispatch_bottleneck_2026-06-13`). Budget the copy; don't assume UMA
  elides it.
- **G8 — wake-from-idle tax on some backends.** The synchronous
  submit-then-block pattern costs ~1.46ms/step on a fanless M3 (RADV ~22µs).
  The fix (keep-warm + double-buffer) is deferred and backend-private; do not
  design around it in v1, but know GPU SPS is adapter-sensitive.

## 11. Expected outcome (hypotheses, to be destroyed by measurement)

- Sparse device traffic (~4.6× less than dense ngm3-GPU) on bandwidth-bound
  kernels → per-step device time well below ngm3-GPU's ~508µs, plausibly into
  ngm2-CPU-parity territory on RADV. *Hypothesis; GI0 measures the response
  sweep, GI6 the whole step.*
- CPU↔GPU learning-quality agreement **tighter** than ngm3's (§7 shared-hash
  argument) — ideally within the seed-noise floor with no directional bias.
  *Hypothesis; GI6 measures it.*
- The premise may fail: if occupancy or launch overhead dominates at
  per-dendrite granularity for realized-sparse sizes, GI0 says stop. Recording
  a negative here is a win, not a failure.
```
