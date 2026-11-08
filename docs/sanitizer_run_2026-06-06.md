# Sanitizer run — `feature/ngm3` @ `ceea36f`

Date: 2026-06-06
Builds: `cmake -B build-{asan,ubsan,tsan} -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER={asan,ubsan,tsan}` (all exit 0).
Per `SANITIZERS.md`, A-tier integration tests are excluded under sanitizers (FP drift from instrumentation trips the 1e-6 golden tolerance — not a real bug).

## Summary

| Sanitizer | Scope run | Result |
|---|---|---|
| **ASan** | unit (7) + B-tier (`integration_b`), `detect_leaks=1` | **CLEAN** — 8/8 passed, no memory errors, no leaks (11.3 s) |
| **UBSan** | unit (7) + B-tier (`integration_b`) | **CLEAN** — 8/8 passed, zero UB diagnostics (9.4 s) |
| **TSan** | bounded 300-step headless run on `configs/mnist_ngm3_philox.json` (CPU ngm3) | **576 warnings, dominated by known-noise classes** (see below) — consistent with the documented ngm2 triage, now reproduced for the new ngm3 code |

ASan + UBSan are a clean gate. TSan remains a manual-inspection tool, not a gate (matches `SANITIZERS.md`).

## TSan detail (300-step `mnist_ngm3_philox`)

576 distinct `WARNING: ThreadSanitizer` reports (1136 data-race + 16 vptr-race access frames). Categorized by source area (frame frequency):

| Class | Evidence | Status (per SANITIZERS.md) |
|---|---|---|
| TBB internals (`oneapi/tbb/*`, `libtbb.so.12`) | ~8000 frames across partitioner/parallel_for/_utils | **False positive.** TBB uses its own task-system synchronization TSan can't model. Suppression-file candidate. |
| AVX-512 SIMD intrinsics (`avx512fintrin.h`) | ~800 frames | **False positive.** Read-only SIMD loads instrumented as 8-byte reads. |
| First-party inside `tbb::parallel_for` lambdas — **NEW: now `hd_ngm3_neuron_group.cpp`** | top sites `:411 :420 :442 :337 :70` (the ngm3 `process()` path) | **Likely false positive** *iff* the TBB partition gives each worker a disjoint neuron/dendrite range — same assumption as ngm2, **unverified for ngm3**. |
| Lock-free atomic plumbing in sim threading | `sim_thread_controller.cpp:112` (`atomic<unsigned long>::load`), `:117` (`atomic<sim_mode>::load`), `sim_env.cpp:258`, `sim_env.h:140`, `sim_instance.cpp:28/69` | **Needs review.** Reported as the ctor's member writes racing the worker thread's atomic loads — classic TSan happens-before gap at thread launch; probably benign but worth confirming the thread-start synchronizes-with the atomic init. |

### Honest caveats
- These are **frame-frequency counts**, not distinct-bug counts — one warning contributes many frames. The 576 figure is the warning count; the distinct *first-party* sites are a handful (the ngm3 `process()` body + ~6 sim-threading atomic sites), exactly the two classes `SANITIZERS.md` lists as open follow-ups.
- I did **not** prove any of these are real races, nor that they are all false positives. The honest status is "matches the known-noise profile; the ngm3 `parallel_for` disjointness and the sim_thread_controller atomic-init ordering are the two things that would need a closer read to fully close."
- TSan full-ctest was intentionally **not** run: ~40 min wall, and B-tier subprocess tests fail on TSan's exit code 66 regardless of correctness (documented).

### Follow-ups (carry-over from SANITIZERS.md, now also applicable to ngm3)
1. Verify TBB partition disjointness in `hd_ngm3_neuron_group.cpp::process()` (lines ~337–442). If disjoint, the ngm3 `parallel_for` races are noise → write TBB suppressions.
2. Audit `sim_thread_controller.cpp:112/117` — confirm the worker-thread atomic loads have a happens-before edge from the controller ctor's atomic initialization (they almost certainly do via thread creation, but TSan flags it).
3. Add `tests/sanitizer_suppressions/tsan.supp` for the TBB + AVX-512 classes once confirmed; keep first-party sites visible.
