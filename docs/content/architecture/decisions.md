+++
title = "Decisions (ADR)"
weight = 20
+++

# Architecture Decisions — Heterogeneous Execution

These Architecture Decision Records capture the load-bearing choices made during the *ngm3-gpu-port* cycle (the work that gave COAST CPU/GPU heterogeneous execution). Each records the forces at play, the decision, its consequences, and the alternatives that were weighed and rejected. They are the "why" behind the [Heterogeneous Execution]({{% relref "/key-abstractions/heterogeneous-execution" %}}) mechanism.

The overriding design constraint across all of them: **minimal disruption to the CPU-only path.** Where a choice existed between modifying existing CPU code and adding a sibling that leaves the existing path untouched, the sibling won by default.

{{% callout info %}}
This page covers the *ngm3-gpu-port* cycle only. The [coastd]({{% relref "/coastd" %}}) daemon has its own, separately-arbitrated decision record — including four choices that **supersede** its design document — at [coastd → Decisions]({{% relref "/coastd/decisions" %}}).
{{% /callout %}}

---

## ADR-1 — GPU framework: wgpu-native

**Status:** Accepted

**Context.** COAST must build and run on both AMD/Linux (Vulkan) and Apple Silicon/macOS (Metal). A single source tree should cover both without dual implementations.

**Decision.** Use **wgpu-native v29.0.0.0** (the prebuilt Rust WebGPU implementation, via its C header), pulled by CMake `FetchContent` with per-platform SHA256-pinned binaries.

**Consequences.** One portable WGSL/compute API spans Vulkan and Metal; ~10 MB shared library added as a runtime dependency. Backend is chosen at compile time (`#ifdef __APPLE__`). We do not control the wgpu release cadence (version is pinned).

**Alternatives rejected.** A single-source C++ GPU API (e.g. a Vulkan/Metal abstraction we maintain) — more control but far more code and a portability burden the cycle explicitly did not want.

---

## ADR-2 — `io_buffer` as the sole device seam, with four variants

**Status:** Accepted

**Context.** Data must cross between CPU and GPU somewhere. COAST already routes *all* inter-entity data through `io_buffer`, under three structural invariants (single-writer, buffer-only communication, one-step latency) — verified against the actual code during Demonstrate.

**Decision.** Make `io_buffer` an abstract base with **four concrete variants** (`io_buffer_cpu`, `io_buffer_cpu_to_gpu`, `io_buffer_gpu_to_cpu`, `io_buffer_gpu`), selected per buffer from `(writer.device, reader devices)`. The seam is exactly the existing buffer boundary; cross-device movement happens at swap time.

**Consequences.** Each variant carries only the fields it needs (clear responsibility, easy isolated testing). Heterogeneous placement requires no new synchronization. `io_buffer` ownership moved to `unique_ptr` (needed for the abstract base).

**Alternatives rejected.** A single fat `io_buffer` holding optional CPU+GPU storage with mode flags (γ-1) — simpler map but every buffer pays for fields it does not use, and the branching obscures responsibility.

---

## ADR-3 — Device carried by type, not a runtime tag

**Status:** Accepted

**Context.** The dispatcher must know which entities run on the GPU.

**Decision.** Introduce an intermediate base `gpu_io_entity : public io_entity` carrying the GPU-work hook (originally a per-step `record(encoder)`). Identify GPU work with `dynamic_cast<gpu_io_entity*>`. Establish the **entity-device rule**: every entity runs entirely on one device; all communication crosses an `io_buffer`.

> **Evolved (gpu-abstraction-layer).** The per-step `record(encoder)` hook was replaced by `build_step_plan(gpu_device&)`, which builds a **static** backend-agnostic `step_plan` **once** at construction; the dispatcher replays it each step and the entity only refreshes uniforms in `process()`. The entity-device rule and the `dynamic_cast`-by-type dispatch are unchanged. The hook is now agnostic — it crosses the seam through `gpu_device`, not a `WGPUCommandEncoder`.

**Consequences.** The type system enforces that a GPU entity implements the GPU-work hook; no device state can drift out of sync. The `dynamic_cast` runs at entity-loop scale (tens per step), dwarfed by kernel work.

**Alternatives rejected.** A `device_kind` enum/tag on `io_entity` (DM2) — duplicates information the class hierarchy already encodes and risks the tag disagreeing with the actual type.

---

## ADR-4 — Two encoder phases, two submits per step

**Status:** Superseded by the gpu-abstraction-layer refactor (see note below) — originally Accepted

**Context.** GPU launch overhead must be bounded — O(1) submissions per step regardless of GPU entity count (goal MH7).

**Decision (original mechanism).** Record all phase-1 kernels (entity compute) into one encoder and submit once in `process()`; record all phase-2 cross-device movement into a second encoder and submit once in `swap_io()`. The `process()` submit does **not** block: GPU recording stays serial (one shared encoder), but after the submit the CPU entities run via TBB `parallel_for` (with a persistent `affinity_partitioner`) concurrently with the in-flight GPU batch, rendezvousing at `wait_idle()` only before probe readback. `swap_io()` still blocks on `wait_idle()` after its submit.

**Consequences.** In the original mechanism there were two submits per step, however many GPU entities/buffers existed. `process()` overlapped the GPU batch with CPU-entity compute (CPU∥GPU); `swap_io()` remained a blocking rendezvous. The overlap is latent on entity-sparse graphs (one dominant entity) and is realized as the graph gains substantial concurrent CPU work — see [Simulation Loop]({{% relref "/data-flow/simulation-loop" %}}).

**Alternatives rejected.** Per-entity submission — simple but O(entities) host cost, failing MH7.

**Superseded (gpu-abstraction-layer).** The two-submit encoder mechanism was replaced by the backend-agnostic `gpu_device` facade. `process()` now calls `begin_step` → per-entity `enqueue(plan)` → `submit_step` (no wait) → dispatches CPU entities concurrently → `rendezvous_step` (the **one** mandatory per-step coherence wait, BARRIER 1) → `read_probes` — preserving the CPU∥GPU overlap but with a single coherence wait rather than a recorded-encoder submit. `swap_io()` no longer records and submits a second encoder; it does a direct `gpu_device::upload` (cpu→gpu), a ping-pong `advance_generation()` flip, and an `io_buffer::device_to_cpu_readback` (gpu→cpu) before the buffer swap. The MH7 O(1)-host-cost goal still holds; the encoder/submit machinery is now confined to the WebGPU backend behind the facade.

---

## ADR-5 — Parity is statistical-aggregate, not per-step numerical (MH3)

**Status:** Accepted (reframe of the original goal)

**Context.** Correctness of the GPU ngm3 must be verifiable against the CPU ngm3. But the GPU kernels use a *stateless* PCG hash PRNG, while the CPU uses the `mt19937_simd` stream — so the two diverge by **design**, not by floating-point error.

**Decision.** Define parity as **statistical-aggregate**: compare per-field `(mean, max, std)` over an N-step run within tolerances, plus sanity floors (determinism, no NaN/Inf, bounded values). A per-step `rel=1e-4` numerical match is structurally infeasible and was abandoned.

**Consequences.** `coast_compare` gained an `aggregate` mode and `tolerances.json` an `aggregate` block (see [tools]({{% relref "/architecture/modules/tools" %}}) and [Golden Protocol]({{% relref "/testing/golden-protocol" %}})). The noise floor was pinned with a CPU-vs-CPU-different-seed reference run. A known consequence: the GPU mean runs ~16–25 % below the CPU reference on Strix Halo (documented, accepted).

**Re-measured (2026-10, macOS/Metal).** The gap does not come from the GPU model. `ngm_parity_probe` (tests/integration; then `ngm3_parity_probe`) ran 8 seed offsets × 20 000 steps per condition and compared them as two samples (Welch and exact Mann–Whitney per metric). With the dataset and data order held fixed and only the group's device changed (`mnist_ngm3_cpu_to_gpu.json` vs the same config with a CPU group), no output or learned-state metric differed significantly (all p > 0.1; output mean over the last 200 steps within 0.7 %). The fully GPU pipeline (`mnist_ngm3_gpu_full.json`) likewise matched once its dataset was pinned to `shuffle: false`. A 16–25 %-style gap reappears only when the two runs present *different images* — `mnist_ngm3_gpu_full.json` leaves `shuffle` at its default (on) while the CPU fixtures pin it off — and a single final step is compared. The original Strix Halo measurement should be re-checked with matched data order before the gap is treated as a model property.

**Alternatives rejected.** Forcing bit-equivalence by porting `mt19937` to WGSL — defeats the stateless-hash simplicity and buys nothing for a memory-bound POC. An earlier idea of a runtime RNG selector on the CPU ngm3 (D5) was **retracted**: with statistical parity, the CPU ngm3 stays literally unchanged.

---

## ADR-6 — Stateless WGSL hash PRNG (PCG-1D32)

**Status:** Accepted

**Context.** GPU kernels need per-synapse pseudo-randomness without carrying ~25 KB of mt19937 state per neuron across the device boundary.

**Decision.** Use a stateless PCG-1D32 hash (`shared_prng.wgsl`), keyed per call by entity coordinates and step — e.g. `pcg_uniform((neuron ^ dendrite ^ rep ^ step ^ slot) …)`.

**Consequences.** No RNG state on the GPU; deterministic per `(seed, step)`; ~6 ALU ops; portable across Vulkan/Metal. Not bit-equivalent to the CPU (this is the root cause behind ADR-5).

---

## ADR-7 — GPU observability: kernel always writes, readback only when watched

**Status:** Accepted

**Context.** Probes, OBSERVE, and visualization need GPU-resident values on the CPU, but the GPU→CPU readback is not free.

**Decision.** Kernels write probe outputs **unconditionally** every step (cheap on-GPU). The CPU-side readback is **gated**: `read_probes()` copies back only when `any_probe_active()`; the OBSERVE population readback (now in a `perform_observations` override) only when `acc.group_active("population")`; the visualization mirror is **consumer-pulled** via `refresh_vis_mirror()` (the visualizer asks, on its own cadence; the entity never auto-refreshes).

**Consequences.** Zero readback cost when nothing is observing; pay-as-you-look otherwise. Probes remain discoverable because the `PROBE()` statements still execute every step.

---

## ADR-8 — Step-0 CPU refactor (Refactors A and B)

**Status:** Accepted

**Context.** Two pre-existing structures in the CPU path obstructed a clean GPU port: per-buffer statistics living *inside* `io_buffer`, and entities receiving buffers as `std::function` closures.

**Decision.** Before any GPU work, **Refactor A** moved statistics out of `io_buffer` — the consuming neuron group computes `input_stats_t` once per step via `ngm2::compute_input_stats()` and passes it down by reference. **Refactor B** dropped the closure factories — entities now hold `io_buffer&` directly, with the virtual base providing the loose coupling.

**Consequences.** `io_buffer::swap_buffer()` became a pure index flip; the seam abstraction is clean; the `sim_core → hd_ngm2` dependency inversion (buffer needing stats helpers) was removed. The single deliberate deviation from "don't touch the CPU path" — justified visibly, and guarded by re-baselining the A-tier goldens (ngm2 numerics held; ngm3 shifted only by the documented true-max change).

**Alternatives rejected.** Leaving stats/closures in place and bolting GPU support beside them — would have propagated the contract violation into every new variant.
