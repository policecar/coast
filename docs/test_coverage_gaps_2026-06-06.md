# Test-coverage gap analysis — `feature/ngm3` @ `ceea36f`

Date: 2026-06-06
Scope: this branch is a major change — it adds the **hd_ngm3** model, a **GPU execution layer**
(core/gpu, wgpu/WGSL), **heterogeneous CPU/GPU** execution, an **observe** instrumentation
system, an **hdf5 executor**, **run manifests**, and CLI/app restructuring. This report maps
what new/changed functionality is *not* covered by tests, ranked by risk.

Method: read-only analysis of `tests/**` and the new `include/`+`src/` modules. Headline
claims below were spot-verified (see "Evidence verified").

## Current test inventory (16 registered ctest)

- **Unit (7):** `smoke`, `unit.blob_writer`, `unit.compare`, `unit.cli.parse`,
  `unit.probe.spsc_queue`, `unit.probe.probe_instance`, `unit.json_cfg.loader`.
- **A-tier integration (8, golden fingerprint):** `mnist_seed{0,1,2}_baseline` (ngm2 CPU),
  `mnist_ngm3_philox` (ngm3 CPU), `mnist_ngm3_cpu_to_gpu`, `mnist_ngm3_gpu_to_cpu`,
  `mnist_ngm3_heterogeneous` (GPU-marked), `ngm3_synthetic_smoke` (structural asserts, no golden).
- **B-tier (1):** `integration_b` — CLI surface, exit codes, save/load/checkpoint/log-buffers,
  `--emit-run-manifest`.

## Coverage by module

| Module | Tested? | Via | Gap | Note |
|---|---|---|---|---|
| cli/parse | direct | unit.cli.parse | LOW | well-covered; `--emit-run-manifest` only tested end-to-end |
| app/build_env | incidental | all A-tier | LOW | thin wrapper, exercised by construction |
| app/run_headless | incidental | integration_b | MED | error paths (build throws) not explicit |
| app/run_gui | none | — | n/a | not headless-testable |
| **hd_ngm3 CPU** (`hd_ngm3_neuron_group`) | incidental | mnist_ngm3_philox golden | **HIGH** | only "output is finite & matches golden"; no unit test of learning / branching / permanence bounds |
| **hd_ngm3 GPU** (`hd_ngm3_gpu_neuron_group`) | incidental | 3 GPU A-tier (aggregate tol) | **HIGH** | logic unit-untested; relies on unregistered `coast_ngm3_gpu_probe` |
| **core/gpu/gpu_context** | incidental | GPU A-tier (implicit) | **HIGH** | fault callback / device-lost / limit validation untested; `coast_gpu_smoke` not registered |
| core/gpu/wgsl_module | incidental | GPU A-tier | MED | shader-compile error path untested |
| core/gpu/shared_prng | incidental | GPU A-tier | MED | no CPU↔GPU Philox parity test |
| sim_core/io_buffer_gpu | incidental | GPU A-tier | MED-HIGH | swap/readback/staging mechanics not isolated; `coast_io_buffer_gpu_probe` unregistered |
| sim_core/gpu_io_entity | incidental | GPU A-tier | MED | record/read_probes error paths untested |
| **observe/** (1541 LOC) | **none (functional)** | — | **CRITICAL** | registry/macro/site/trigger/writer/dtype-lock all unexercised; see below |
| hdf5/executor | incidental | B-tier save/load | MED | queue saturation, async error propagation, worker lifecycle untested |
| sim_core/run_manifest | happy-path | integration_b | MED | write-failure / unwritable-path / iso8601 edges untested |
| io/mnist_io_gpu | incidental | GPU A-tier | MED | dataset upload, cursor wrap, noise PRNG not isolated; `coast_mnist_io_gpu_probe` unregistered |
| json_cfg/ngm3_json | incidental | A-tier load configs | MED | no error-path test (typo'd field → silent default) |
| logging/state_io | happy-path | B-tier save/load | MED | corruption / version-mismatch / HDF5 error untested |
| sim_core/sim_instance, sim_thread_controller | incidental | all tests | LOW | lifecycle exercised; atomic ordering unit-untested (also a TSan follow-up) |

## Top gaps, ranked

### CRITICAL
1. **observe system has no functional test (1541 LOC).** The only test mentions are a *comment*
   in `unit.json_cfg.loader` and a manifest-mirroring check in `integration_b` (asserts the
   run-manifest JSON contains the resolved `observe` config block). Nothing exercises actual
   capture: registry/site registration, the `OBSERVE` dispatch, dtype/shape locking, trigger
   windows/decimation, or the async HDF5 writer + its output layout. A bug in site resolution
   or trigger logic silently drops data or crashes the writer thread with no test to catch it.
   - Missing: `unit/observe/*` (registry lifecycle, dtype-lock-on-second-call, trigger modes,
     config error cases) **and** an A-tier `integration.mnist_observe_test` that runs
     `configs/mnist_observe_test.json` headless, then reads back the `.observe.h5` and checks
     dataset count / shape / row count vs decimation. (That config exists but is **not**
     registered as a test.)

### HIGH
2. **hd_ngm3 model logic untested (CPU + GPU).** A-tier goldens catch gross regression but not
   correctness of learning/branching or invariants (permanence ∈ [0,1], segment-id bounds,
   branch-interval counter). Add `unit/ngm3/*` exercising one deterministic step against hand
   computed references + invariant asserts. Applies to both `hd_ngm3_neuron_group` and the GPU
   variant.
3. **No CPU↔GPU numerical parity test.** The 3 GPU A-tier tests use *aggregate* tolerance to
   absorb reduction-order differences — a systematic GPU bias (e.g. truncated intermediate)
   would still pass. Add a focused test running identical seed/input on CPU and GPU ngm3 for N
   steps and comparing element-wise within an FP32 ULP band.
4. **GPU fault handling untested.** `gpu_context`'s device-lost / uncaptured-error callback and
   `check_or_throw` propagation have no test. A broken fault path means GPU failures surface at
   an unpredictable later point. (Hard to inject without a mock device — at minimum test the
   happy-path init + limit-validation that `coast_gpu_smoke` does today unregistered.)

### MEDIUM
5. **io_buffer_gpu swap/readback mechanics** — unit-test index-swap invariants and readback
   completion for the cpu_to_gpu / gpu_to_cpu / gpu variants.
6. **Register the 5 standalone dev harnesses as smoke ctests** (guarded by GPU availability):
   `coast_dispatcher_probe`, `coast_gpu_smoke`, `coast_io_buffer_gpu_probe`,
   `coast_mnist_io_gpu_probe`, `coast_ngm3_gpu_probe`. They already encode meaningful
   assertions but run only by hand — they're not in CI.
7. **ngm3_json error paths** — a test that a typo'd/missing field is rejected (or at least
   asserted), not silently defaulted.
8. **run_manifest / state_io / hdf5_executor error paths** — write-permission failure, disk
   full, corrupted/version-mismatched state, executor queue saturation.

## Evidence verified
- observe LOC: `cat src/observe/*.cpp include/observe/*.h | wc -l` → **1541**.
- observe in tests: only `tests/unit/json_cfg/test_json_loader.cpp:20` (comment) and
  `tests/integration_b/test_b_tier.cpp:368-372` (manifest contains `observe` block;
  `observe_outputs` intentionally absent). **No capture/writer test.**
- `configs/mnist_observe_test.json` exists but is referenced by **no** `tests/**/CMakeLists.txt`.
- Unregistered harness executables (built, not ctest): `coast_dispatcher_probe`,
  `coast_gpu_smoke`, `coast_io_buffer_gpu_probe`, `coast_mnist_io_gpu_probe`,
  `coast_ngm3_gpu_probe`.

## Caveat on scope
This is a static read of test↔code mapping. "Incidental" means a module is executed by an
integration test but has no assertion targeting its specific logic/invariants — gross
regressions are caught, unit-level correctness bugs are not. I did not measure line/branch
coverage (no gcov/llvm-cov run); doing so would quantify the "incidental" rows precisely.
