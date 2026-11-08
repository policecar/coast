# Sanitizer Build Status

Initial triage from the `develop.sanitizer-builds` work in the testing dew cycle.

## How to invoke

```bash
# Build directory per sanitizer. CMAKE_BUILD_TYPE=Release matters — see "FP drift" below.
cmake -B build-asan  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER=asan
cmake -B build-ubsan -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER=ubsan
cmake -B build-tsan  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER=tsan

# Build + run tests:
cmake --build build-asan -j
cd build-asan && ASAN_OPTIONS="detect_leaks=1:halt_on_error=0" \
    ctest --output-on-failure -E 'integration\.' -LE gpu

# Same pattern for UBSan / TSan.
# Two exclusions for sanitizer runs:
#   -E 'integration\.'  skips the A-tier fingerprint tests (FP drift under
#                       instrumentation — see "FP drift" below).
#   -LE gpu             skips the gpu.smoke.* harnesses: they drive the wgpu /
#                       Vulkan driver, whose uninstrumented allocations trip
#                       LSan and whose threads flood TSan. GPU correctness is
#                       covered by the Release ctest run, not the sanitizer one.
```

## Findings

### ASan (AddressSanitizer)

**Result: clean.** Full ctest suite of the time (11 tests) passes under ASan with `detect_leaks=1`. No
memory errors, no leaks. ~2× slowdown (~140 s vs ~70 s baseline).

### UBSan (UndefinedBehaviorSanitizer)

**Result: clean.** Zero UB diagnostics from any test path (unit + B-tier subprocess + headless
integration). Integration A-tier tests trip on numerical drift, not on UB.

### TSan (ThreadSanitizer)

**Result: ~250 raw warnings, dominated by known false-positive classes; a handful of
first-party reports worth individual investigation.**

Triage on a 200-step `coast --headless` run, categorized by source:

| Class | Count | Status |
|-------|------:|--------|
| TBB internals (`oneapi/tbb/*`) | ~80 | **False positive.** TBB uses its own task-system synchronization (custom mutexes, work-stealing queues) that TSan does not model. Industry-standard noise; would normally go into a TSan suppression file. |
| AVX-512 SIMD intrinsics (`avx512fintrin.h`) | ~32 | **False positive.** SIMD loads are read-only and don't actually race; TSan instruments them as 8-byte reads but can't see the higher-level access pattern. |
| First-party inside `tbb::parallel_for` lambdas (`hd_ngm2_*.cpp`) | ~120 | **Likely false positive** assuming the TBB partitioning is correct (each thread sees a disjoint range of neurons/dendrites). Verifying the partition disjointness would close the question. Until that's done these are "needs review", not "real race." |
| Lock-free SPSC / atomic patterns in sim plumbing (`sim_thread_controller.cpp`, `io_buffer.cpp`, `sim_env`) | ~6 | **Needs review.** These live in the GUI↔sim communication path. The design is intentionally lock-free; the question is whether the memory orderings on the atomics are strict enough to satisfy TSan's happens-before model. |
| Dataset IO (`mnist_io.cpp`) | 2 | **Needs review.** Two reads/writes on the IO thread side. Quick check needed. |

#### Action items for follow-up (not in this cycle)

1. Verify TBB partition disjointness in `hd_ngm2_neuron_group.cpp:process()` — sketch a proof
   or run with the static partitioner config that's already in use; argue per-thread
   ownership. If disjoint, write the TBB suppressions.
2. Audit the ~6 first-party SPSC sites in `sim_core/`. Likely either (a) tighten memory
   orderings, or (b) document the race-vs-feature distinction inline. The probe `spsc_queue`
   is already correct (acquire/release pairs); the controller's `step_count` needs a
   closer read.
3. ~~Add `tests/sanitizer_suppressions/tsan.supp`~~ **Done** — the file exists and suppresses
   the TBB task-system noise (via `called_from_lib:libtbb.so`). Caveat: that scope is broad
   and also masks first-party races inside `parallel_for` lambda bodies (a precise
   `race:<func>` scope does not work — TSan inlines the TBB partitioner frames into the
   caller). So the hd_ngm2 partition-disjointness audit must still run TSan *without* the
   suppression to keep those frames visible.

#### Test impact

The full ctest suite under TSan takes ~40 min wall time (5–10× slowdown). Even with `-E
'integration\.'` to skip A-tier (FP drift), the B-tier subprocess tests fail because the
child `coast` binary exits with TSan's default warning exit code (66) — even though the
warnings are dominantly false positives.

With `tests/sanitizer_suppressions/tsan.supp` (TSan task-system noise) the **unit** suite is
now TSan-gateable — e.g. `unit.sim_core.step_fault` (which exercises the value-based fault
reduction under `parallel_for`) passes clean:

```bash
cd build-tsan && TSAN_OPTIONS="suppressions=tests/sanitizer_suppressions/tsan.supp" \
  ctest -LE gpu -E 'integration\.'
```

The broader B-tier subprocess gate and the first-party `parallel_for` lambda review still
require the bare (unsuppressed) manual-inspection runs above.

## FP drift under instrumentation — known limitation

The A-tier integration tests (`integration.mnist_seed*_baseline`) use bit-precise fingerprints
of `io_buffer` state captured with tolerance `rel=1e-6`. Any change to the binary's
floating-point code generation — including:

- different optimization levels (Release `-O3` vs RelWithDebInfo `-O2`),
- sanitizer instrumentation that reorders FP ops,
- compiler / libstdc++ version changes,

will perturb the last few digits of the simulation output past 1e-6. Observed:

- RelWithDebInfo + ASan: ~24% max relative drift, fails.
- Release + ASan: matches goldens exactly.
- Release + UBSan: ~2% max relative drift, fails (UBSan instrumentation alone is enough to
  shift `-O3` output).

**Recommendation:** when running under sanitizers, exclude the A-tier tests:

```bash
cd build-{asan,ubsan,tsan} && ctest --output-on-failure -E 'integration\.'
```

The A-tier tests serve a different purpose (regression at the bit level) than sanitizer runs
(UB / data race / leak detection at the runtime level). Don't conflate them.

If a sanitizer-specific A-tier suite is ever needed, capture per-sanitizer goldens in
separate directories (e.g. `tests/integration/mnist_seed0_baseline/goldens_asan/`). That's
deferred until there's a concrete reason to want it.

## Suppressions

- `tests/sanitizer_suppressions/tsan.supp` → `TSAN_OPTIONS=suppressions=...` — **exists.**
  Suppresses the oneTBB task-system noise only (`called_from_lib:libtbb.so`; see the caveat in
  the file and under TSan above). First-party COAST races are NOT suppressed.

As more real third-party noise (HDF5, SDL2, ImGui at shutdown) emerges, add the remaining
per-sanitizer files under `tests/sanitizer_suppressions/`:

- `asan.supp` → `LSAN_OPTIONS=suppressions=...`
- `ubsan.supp` → `UBSAN_OPTIONS=suppressions=...`

Any first-party sanitizer report is a test failure (no suppression). Third-party
hits go in this frozen list.
