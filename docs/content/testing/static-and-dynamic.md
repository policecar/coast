+++
title = "Static & Dynamic Analysis"
weight = 30
+++

# Static and Dynamic Analysis

The four test tiers ([Tier model]({{% relref "tier-model" %}})) cover regression in observable behavior. Two further CMake-only quality gates cover the dimensions tests cannot reach: static code drift (`clang-tidy`) and runtime undefined behavior (the sanitizer trio).

Neither gate is automatic. Both are on-demand runs, invoked by the developer when the work warrants. Wiring them into the default build was rejected — `clang-tidy` over the whole codebase is slow, and sanitizer builds change the binary in ways that perturb A-tier fingerprints (see below).

## clang-tidy

**Target**: `coast_tidy`. Reports only — never gates the build.

```bash
cmake --build build --target coast_tidy
```

### Rule subset

`.clang-tidy` at the repository root explicitly opts into a conservative subset:

```yaml
Checks: >
  -*,
  bugprone-*,
  performance-*,
  readability-redundant-*,
  modernize-use-nullptr,
  -bugprone-easily-swappable-parameters,
  -bugprone-narrowing-conversions
```

The four enabled categories are the ones that produce real-bug or real-perf signal with low noise on numerical C++23 code. Two `bugprone` rules are disabled inline:

- `bugprone-easily-swappable-parameters` — false-positives pervasively in HD-NGM2 (functions naturally take "neuron_idx, dendrite_idx, segment_idx" triples; these are not swappable in practice but trip the lint).
- `bugprone-narrowing-conversions` — overlaps with compiler warnings and adds noise without finding bugs.

Explicitly *not* enabled:

| Category | Why disabled |
|----------|--------------|
| `cppcoreguidelines-*` | Aggressive; many false positives on tight numerical loops |
| `hicpp-*` | Duplicates other categories with stricter framing |
| `google-*` | Google-internal style |
| `cert-*` | Security-focused; mostly aliases |
| `fuchsia-*`, `llvm-*` | Project-specific conventions |
| `readability-magic-numbers` | Pervasively false-positive in scientific code |

The intent is **expand the rule set as warnings drop to zero in the current set**, not "fix everything tidy emits at once."

### Baseline

The current baseline (captured in `.clang-tidy-baseline.txt` at repo root): **66 warnings**. This is the "frozen for now" reference. A future `coast_tidy` run that produces more than 66 warnings on first-party code is the actionable signal. Driving the baseline to zero is a follow-up, not a precondition for shipping changes.

### What is excluded

Third-party code under `third_party/`, `tests/third_party/`, and the vendored `doctest.h` is excluded — these are trusted black boxes for the purposes of this project's static analysis.

## Sanitizers — ASan, UBSan, TSan

```bash
cmake -B build-asan  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER=asan
cmake -B build-ubsan -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER=ubsan
cmake -B build-tsan  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER=tsan
```

Each sanitizer lives in its own out-of-tree build dir. The `COAST_SANITIZER` option is single-valued because the three sanitizers are not all mutually compatible (ASan and TSan in particular cannot share a binary).

| Sanitizer | What it catches |
|-----------|-----------------|
| **ASan** | Use-after-free, heap-buffer-overflow, stack-buffer-overflow, double-free, memory leaks (with `ASAN_OPTIONS="detect_leaks=1"`) |
| **UBSan** | Signed integer overflow, null deref, misaligned access, invalid enum, float-to-int overflow, etc. |
| **TSan** | Data races on shared state across the sim and main threads |

`SANITIZERS.md` at repo root holds the operational details and the current known-clean/known-noisy state of each run.

### Two gotchas worth pinning in your mental model

#### 1. Release, not RelWithDebInfo

Sanitizer builds *must* use `CMAKE_BUILD_TYPE=Release` (i.e. `-O3`), not `RelWithDebInfo` (`-O2`). The codegen difference between `-O2` and `-O3` perturbs floating-point output past the A-tier `rel=1e-6` tolerance. Mixing optimization levels between the goldens build and the sanitizer build produces fake regressions that have nothing to do with the sanitizer's findings.

This is a project-specific constraint that does not generalise — most codebases do not have a fingerprint-comparison test suite that is sensitive to `-O2`-vs-`-O3` differences. COAST does, and the result is that all release-shaped builds (including sanitizers) pin to `-O3`.

#### 2. Skip A-tier tests under sanitizers

A-tier integration tests compare against retained goldens with `rel=1e-6` tolerance. Sanitizer instrumentation (ASan shadow memory, UBSan integer-overflow guards) perturbs floating-point evaluation order enough to trip that tolerance — *without there being a real bug*. The recommended invocation skips them:

```bash
cd build-asan && ASAN_OPTIONS="detect_leaks=1" ctest --output-on-failure -E 'integration\.'
```

What you lose: confirmation that the A-tier sim runs cleanly under instrumentation. What you keep: the unit tier, the B-tier subprocess tests, and the GUI smoke link-test — all of which exercise enough of the codebase to surface the kinds of bugs the sanitizers find.

A future refinement (not in this iteration) would be to add **structural-only** A-tier checks for sanitizer runs: run the full sim under ASan, but assert only "no crash, no leak, no race", not "fingerprint matches golden". The same N-step run still exercises the code; only the comparison is relaxed.

## Workflow combination

A realistic developer cadence (none of this is automated locally; CI runs the suite on every push, see [Decisions]({{% relref "decisions" %}}), D10):

| Trigger | Run |
|---------|-----|
| Before a small commit | `ctest -R 'unit\.'` (sub-second sanity) |
| Before a non-trivial commit | full `ctest --output-on-failure` |
| After touching threading or shared state | `cd build-tsan && ctest -E 'integration\.'` |
| After touching memory ownership or pointer math | `cd build-asan && ctest -E 'integration\.'` |
| Periodically, or after touching first-party headers | `cmake --build build --target coast_tidy`, eyeball the delta from 66 |

Locally there is no enforcement layer behind this. The discipline is "the developer who skips it gets to debug the regression in the next PR."
