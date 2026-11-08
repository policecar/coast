# ngm2.5 GPU validation campaign (GI6)

Date: 2026-07-10. Branch `research/ngm-25-opus` (GI0-GI5 committed:
c174e32/f2f4104/6f69cb7/e8b174c/8e2510a/d0e87b2).
Adapter: **AMD Radeon 8060S Graphics (RADV STRIX_HALO) [Vulkan]**, Release
(verified `CMAKE_BUILD_TYPE=Release` in CMakeCache, P11), `COAST_GPU_FAST_WAIT` on.
Run **outside the sandbox** (no GPU in sandbox). Exclusive machine.
Method: end-to-end wall-clock over N steps of `./build/coast -n N <config>`
(SPS = N / wall). This is the honest throughput number — it includes the host
gather, R1 stats, per-step uploads, and the GI4 per-step winner readback, not just
device compute. Two runs each; warmup discarded.

## 1. Correctness / reproducibility (validated in GI2-GI5, cited here)

- **RNG parity (§7):** WGSL `syn_key` == host `ngm25::syn_key` over 16384 keys, 0
  mismatches. CPU and GPU draw bit-identical values -> no ngm3-style 16-25% CPU/GPU
  bias by construction.
- **CPU<->GPU forward (GI2):** on the uniform §9 fixture (penalty inert) outputs
  agree to reduction-order FP (max|d|=7.4e-8).
- **CPU<->GPU adapt (GI3):** perm BIT-IDENTICAL at 1 step; at 100 steps perm
  |mean d|=1.8e-6, adapt_hist identical.
- **Growth (GI4):** GPU 200704->201354 syn over 7 branch events, consistency
  checker green after every event, growth tracks CPU (ratio 1.01).
- **Checkpoint->resume (GI5):** BIT-EXACT vs an uninterrupted run (max|d|=0) — the
  hash-RNG payoff.

Learning-quality equivalence (representation-growth timelines, specialization
heatmaps) is the **user's judgment** — deferred exactly as CPU I9 was. The near-
bit-exact CPU/GPU agreement above + the bit-identical RNG make a directional
learning-quality gap very unlikely, but the campaign is the user's to run/steer.

## 2. Throughput (SPS)

### Matched ngm2.5-CPU vs ngm2.5-GPU
`mnist_ngm25.json` and `mnist_ngm25_gpu.json` are the SAME config, differing only
in the entity type — a clean matched comparison. The 400-neuron variants scale
neuron_cnt 50->400 (~8x the dendrites/synapses).

| config           | neurons x prox | ~synapses | ngm2.5-CPU SPS | ngm2.5-GPU SPS | GPU/CPU |
|------------------|----------------|-----------|----------------|----------------|---------|
| mnist_ngm25(_gpu)| 50 x 5         | ~196k     | 3701           | 2731           | 0.74x   |
| big (n=400)      | 400 x 5        | ~1.57M    | 860            | 1258           | **1.46x** |

Scaling from 50->400 neurons (8x the work):
- **CPU: 3701 -> 860 SPS = 4.3x slower.**
- **GPU: 2731 -> 1258 SPS = 2.17x slower** — the GPU amortizes its fixed
  per-dispatch cost, so it scales ~2x better than the CPU.

### ngm2.5-GPU vs ngm3-GPU (context; different configs)
| model      | config   | SPS  |
|------------|----------|------|
| ngm3-GPU   | 16 x 4   | 1007 |
| ngm2.5-GPU | 50 x 5   | 2731 |

ngm2.5-GPU is **2.7x faster than ngm3-GPU** despite a *larger* config (50x5 vs
16x4) — the sparse arena decisively beats the dense ngm3-GPU floor. This is the
GI0 premise, now confirmed on the full step, not just the response sweep.

## 3. Verdict (honest)

- **Premise CONFIRMED.** The sparse-arena device model beats dense ngm3-GPU (2.7x)
  and, at scale, beats the matched CPU model (1.46x at 400 neurons). GI0's
  hypothesis holds end-to-end.
- **Crossover ~100-150 neurons.** Below it the GPU's fixed per-dispatch overhead
  (~25 us/dispatch x ~6 passes, GI0 fit) + the GI4 per-step winner readback
  dominate, so the GPU LOSES to the CPU on small configs (50n: 0.74x). Above it the
  GPU wins and the margin grows with scale. State this plainly: **ngm2.5-GPU is a
  scale play** — it is not a win for small groups on this UMA adapter.
- **This matches the design's honest framing** (§1, §11): the value is at scale /
  on dedicated-GPU machines; on Strix Halo's shared LPDDR5x both compete for the
  same bandwidth, and fixed launch/sync overhead sets a small-config floor.

## 4. Optimization levers (measured-motivated, not yet done)

1. **Remove the GI4 per-step winner readback** (a synchronous device->host round
   trip every step). Move `input_count` on-device (winner/adapt increment a device
   buffer; read back only when a branch gate fires). This is the single clearest
   host-sync cost and would lower the crossover + lift GPU SPS at every size.
2. **Single-workgroup passes** (winner, inhibition) serialize the GPU (G1). Winner
   is a small serial scan; inhibition is shared with ngm3. Quantify and, if
   material, widen.
3. **Device-time WAIT_IDLE breakdown** (COAST_GPU_TIMING) to attribute the ~795
   us/step (400n) across gather-copy / response / combine / winner / inhibition /
   adapt / readback — not done here (wall-clock SPS was the gating number).
4. **GPU->GPU dataset gather** (design §6.2) to remove the host gather + upload
   (currently host-visible-input only, GI1 v1 limitation).
5. bf16 storage of the large arrays (attacks the bandwidth bound) — after fp32 is
   the accepted baseline.
