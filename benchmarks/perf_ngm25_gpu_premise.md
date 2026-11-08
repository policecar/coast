# ngm2.5 GPU premise-check (GI0)

**Verdict: GO (default decision — the go/no-go is the user's call, design
`ngm2_5_gpu.md` §2/§8). The premise holds; recommend proceeding to GI1.**

Date: 2026-07-10. Branch `research/ngm-25-opus`.
Harness: `src/hd_ngm25/hd_ngm25_gpu_premise_main.cpp` (target
`coast_ngm25_gpu_premise`, webgpu backend). Throwaway; inlines its WGSL.
Adapter: **AMD Radeon 8060S Graphics (RADV STRIX_HALO) [Vulkan]**, Release,
`COAST_GPU_FAST_WAIT` on. Run outside the sandbox.
Invocation: `./build/coast_ngm25_gpu_premise 4000` (4000 iters, 300 warmup,
256 dendrites).

## What was measured

The whole GPU port rests on one unmeasured claim (design §2): *a realized-sparse
arena on device beats the dense ngm3-GPU compute floor because it moves ~4.6×
less data through a bandwidth-bound response sweep.* GI0 measures the **response
sweep only** (R2 accumulate + per-synapse penalty draw + workgroup reduce), one
workgroup per dendrite, through the real `gpu_device` façade
(`register_buffer`/`register_kernel`/`register_plan`/`build_plan`, then
`begin`/`enqueue`/`submit`/`rendezvous` in a tight loop; device-wait = the
`rendezvous_step` = `wait_idle` portion).

The kernel uses ngm2.5's structural-identity `syn_key` (constants
`3266489917/2246822519/668265263/374761393`) and shared_prng's `pcg_uniform`, so
the per-synapse draw cost is representative. It reads `perm` first and skips
`input_idx`/`seg` for sub-threshold synapses, modeling the 12 B/syn device read
set faithfully. It omits the per-seg binning + tree-push (O(128)/dendrite,
negligible vs the O(syn/dendrite) sweep) — a perf proxy, not a semantic kernel.

Rather than compare against a separately-measured ngm3 number with a different
method, the harness runs a **size sweep** on the same adapter with the same
kernel, so a fixed+linear fit separates per-dispatch overhead from per-synapse
(bandwidth) cost.

## Result

256 dendrites throughout; device-wait µs/step:

| synapses  | /dendrite | read set | device-wait µs |
|-----------|-----------|----------|----------------|
| 200,704   | 784       | 2.4 MB   | 34.2           |
| 342,000   | 1336      | 4.1 MB   | 40.9 ← realized-sparse anchor |
| 684,000   | 2672      | 8.2 MB   | 53.5           |
| 1,605,632 | 6272      | 19.3 MB  | 94.6 ← dense ngm3 (rep=8) anchor |

**Fit: device-wait ≈ 25.5 µs fixed + 0.043 ns/synapse.** The linear term is
constant across an 8× size range → the traffic component is bandwidth-linear as
the thesis predicts. The 25.5 µs fixed term is the per-dispatch floor (kernel
launch + 256-workgroup dispatch + barrier reductions).

Reference floors (design §1): ngm3-GPU **~508 µs/step** device-wait (WHOLE step),
ngm2-CPU **~193 µs/step** (5195 SPS).

## Interpretation (honest)

1. **Absolute headroom is large.** The realized-sparse response sweep is 40.9 µs
   — ~12× below ngm3-GPU's 508 µs full step and ~5× below ngm2-CPU's 193 µs full
   step. Even the dense arena's response sweep is only 94.6 µs.

2. **The sparse win on the sweep is 2.31×, not the 4.69× traffic ratio** — and
   the fit says exactly why: the *traffic* cost IS ~4.5× smaller for sparse
   (≈15 µs vs ≈68 µs of linear term), but the shared ~25.5 µs fixed floor dilutes
   the ratio. This is precisely the "occupancy/launch overhead at this
   granularity" the design flagged as a possible stop condition (§2). Here it is
   a bounded ~25 µs floor, not a wall.

3. **The decisive cost is NOT the response sweep.** Since even the *dense*
   response sweep is only ~95 µs, ngm3-GPU's 508 µs full-step floor must be
   dominated by the **adapt pass + the single-workgroup stats/winner/inhibition
   passes**, not response. GI0 does not measure those (GI6 does). The sparse
   arena cuts the traffic-bound portion of the adapt pass by the same ~4.5×,
   which is where the real win should land — but that is a **hypothesis GI0 does
   not close.**

## Why GO (and the caveat carried forward)

The premise — sparse-on-device response beats the dense compute floor — holds:
2.31× faster on the sweep, traffic thesis confirmed on the linear term, and
~12× absolute headroom under ngm3-GPU. The one real risk (a ~25.5 µs fixed floor
per compute pass, recurring across ~4 passes → maybe 50–100 µs of pure dispatch
overhead in the full step) is comfortably inside the 508 µs ngm3 budget, and the
adapt pass — ngm3's actual bottleneck — should get the same ~4.5× traffic cut.

Even a pessimistic full-step estimate (4× the response sweep for adapt + the
serializing passes ≈ 160–200 µs) lands at ngm2-CPU parity and ~2.5–3× faster
than ngm3-GPU. The uncertainty is *how big* the win is (ngm2-CPU parity vs.
merely beating ngm3-GPU), not *whether* there is one.

**Carried to GI6:** measure the full step — especially the adapt pass and the
single-workgroup winner/inhibition floors — to confirm the whole-step win and
size it. That is where a negative could still appear; the response sweep does
not.
