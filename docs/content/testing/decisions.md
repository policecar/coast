+++
title = "Decisions"
weight = 40
+++

# Decision Record

The load-bearing choices made during the testing-infrastructure cycle. Each entry names the decision, the alternative that was considered, and the reason one was chosen over the other.

---

## D1 — `libcoast` is a static library, `coast` is a thin executable

**Decision**: Move all current `src/` code (except `main.cpp` and the new `cli`/`app` modules) into a static library `libcoast`. The `coast` executable becomes a ~50-line entry point that parses argv and dispatches.

**Alternative**: Keep `main.cpp` as the sole compilation unit, write tests as separate `.cpp` files that include the project's headers.

**Why this**: There is no other way to give unit tests in-process access to internals without duplicating sources or shelling out for every test. Forcing the `libcoast` extraction was also a needed cleanup independent of testing — `main.cpp` had grown a `build_env` and runner logic that was never going to be tested in its original form. The cycle made the refactor unavoidable, which is a feature.

**Consequence**: One additional CMake target. The whole-archive linker flag (`-Wl,--whole-archive`) is needed for the `coast` executable so static-initializer registrars in `gui_vis/` are not dropped; tests do not need this flag because they reference the symbols they use directly.

---

## D2 — doctest, vendored as a single header

**Decision**: doctest 2.4.11, vendored at `tests/third_party/doctest/doctest.h`.

**Alternative**: GoogleTest, Catch2, build-system-managed dependency.

**Why this**: doctest's single-header drop-in has the smallest dependency footprint of any C++ test framework. GoogleTest's headline features (GMock, death tests) target use cases this project does not have. Catch2 v3 requires a real build dependency and is slower to compile. Vendoring rather than fetching at configure time keeps the build offline-safe and avoids version drift between contributor environments.

**Consequence**: a single ~12 kLOC header gets included from every test TU. Doctest's "tests next to production code" feature is unused but harmless — production code is free of `TEST_CASE` macros.

---

## D3 — JSON blobs with binary sidecars, not HDF5

**Decision**: A-tier integration tests write JSON files with optional binary `.bin` sidecars for arrays too large to inline.

**Alternative considered**: HDF5, which COAST already uses for its production logger and state snapshots.

**Why this**: HDF5's C API is not thread-safe; the simulation already routes all HDF5 access through a serializing `hdf5::executor`. Reusing that path from tests would couple test infrastructure to executor lifecycle (start, drain, stop around every capture). JSON has none of that coupling, `git diff` shows real numeric changes on a golden update, and the readability constraint of "JSON has to fit a PR review" pushes test authors toward summary statistics rather than full-state dumps — a discipline the test infrastructure benefits from.

**Consequence**: large arrays need a sidecar; the JSON-plus-sidecar format is custom (not a standard schema). `coast_compare` knows how to dereference the binary refs.

---

## D4 — In-process A-tier, subprocess B-tier

**Decision**: Two integration tiers. A-tier tests build `sim::env` directly inside the test process and run the simulation; B-tier tests fork the built `coast` binary and assert on its CLI surface.

**Alternative considered**: One unified integration tier — either everything in-process (and skip CLI testing) or everything subprocess (and lose direct env access).

**Why this**: A-tier's value is direct probe access and fast iteration; B-tier's value is exercising the user-facing `./coast` binary including argv parsing, exit codes, and file side-effects. These are different surfaces with different failure modes, and the design's framing — "the CLI has real combinatorial interactions; a single full-app test is testing-cosplay" — argued explicitly against collapsing them.

**Consequence**: Two harnesses to maintain. The B-tier harness is small (one shared executable hosting many `TEST_CASE`s, since subprocess fork dominates per-case cost).

---

## D5 — Strict array comparison

**Decision**: Any single element of an array exceeding tolerance fails the whole field.

**Alternative considered**: "X% of elements within tolerance", "mean delta below threshold", or other averaging modes.

**Why this**: Looser modes turn the tripwire into a noise filter. A real bug that shifts every element by exactly the wrong amount averages to zero. The point of A-tier is to catch things, not to smooth over them. The cost is paid through per-field tolerance overrides when a specific field is genuinely noisier than the default, not through globally weakening comparison.

**Consequence**: per-field tolerance overrides become the only knob; `tests/integration/<test>/tolerances.json` is the place to use it.

---

## D6 — N=5 retained goldens, β-naming

**Decision**: Each A-tier test directory keeps up to 5 historical golden blobs, named `YYYYMMDD_<commit-short-sha>.json`. A test passes if its new blob matches any retained golden within tolerance.

**Alternative considered**: A single golden per test, replaced on every intentional update. Or a much larger retention window.

**Why this**: One-golden-per-test loses the "this PR reverts a previous fix" signal — the moment a fix lands, the previous behavior is unrecoverable. Five gives enough history to detect revert patterns without ballooning the repo. The chronological filename sort gives ordering for free with no rename churn on rotation; the script that rotates the oldest out is a few lines.

**Consequence**: silent passes when new behavior happens to match a retained-but-older golden. See [Golden Protocol]({{% relref "golden-protocol#multi-version-match-semantics" %}}) — that's by design, and the diagnostic is the "closest golden" report.

---

## D7 — Default tolerance `rel=1e-6, abs_floor=1e-9`

**Decision**: All A-tier comparisons use that default, with per-field overrides in `tolerances.json` added only after a real rerun trips the default.

**Alternative considered**: Pre-populating per-field tolerance overrides during golden capture.

**Why this**: Pre-emptive overrides defeat the tripwire — a field whose tolerance was widened in advance no longer tells you anything when it would have caught a regression. The discipline is "the developer who hits the trip sees the diagnostic, sees the magnitude, and decides whether to widen the tolerance or to fix the bug." Forcing that decision to be a deliberate, post-failure act is the whole point.

**Consequence**: First runs on a new platform may produce tolerance trips that are pure FP drift, not regressions. The mitigation is the multi-version-match logic (D6): a freshly-captured golden from the new platform joins the retained set, and the test stabilises.

---

## D8 — `BUILD_TESTING` defaults to OFF

**Decision**: Adding tests is opt-in via `cmake -B build-tests -DBUILD_TESTING=ON`. The conventional `build/` directory remains identical to the pre-test-infrastructure state.

**Alternative considered**: On by default once `BUILD_TESTING` is wired up.

**Why this**: An external developer cloning the repo and running the canonical `cmake -B build && cmake --build build` should still get a working `./coast` without surprises. Tests are a quality-of-life feature for people changing the code, not a precondition for using it. The cost of one extra build directory is trivial; the cost of every clone-and-build invoking the test compilation is real.

**Consequence**: Documentation has to lead with the `-DBUILD_TESTING=ON` flag; if it's missed, ctest fails with "no tests found".

---

## D9 — Demonstrate stage skipped

**Decision**: No standalone Demonstrate stage. Every Demonstrate-flagged item from Discover either collapsed during Design or moved into Develop.

**What this looked like in practice**:

| Discover Demonstrate item | Disposition |
|---------------------------|-------------|
| Byte-equality of reruns | Collapsed: tolerance-based comparison is the design; the "byte-equal?" question is no longer load-bearing |
| Fingerprint regression sensitivity (synthetic-bug injection) | Reframed: fingerprints are a localizer, not a coverage proof; no value in pre-validating |
| Headless GL for GUI smoke | Dropped: GUI smoke runs in a real SDL/GL window on the developer workstation |
| ASan/UBSan runtime cost | Moved to informational; sanitizers are on-demand, not always-on |
| clang-tidy baseline triage | Folded into `develop.coast-tidy` work item |

**Why this**: the Demonstrate stage's job is to verify undocumented library or hardware behavior that the design depends on. The design depends on neither. Running a Demonstrate stage anyway would have produced PASS-stamped documents that confirmed nothing not already confirmed by the implementation itself.

---

## D10 — No CI, no enforcement layer

**Decision**: Manual invocation. Run `ctest` before commits with larger edits. No GitHub Actions, no pre-commit hook, no required-status-check.

**Alternative considered**: Wiring up GitHub Actions or equivalent.

**Why this**: CI was explicitly out of scope per Discover. The two-developer workflow with manual discipline is what the team will actually use; the test suite's design optimises for that workflow (fast, intelligible, doesn't require infrastructure to run).

**Consequence**: A developer who skips `ctest` and pushes a regression breaks subsequent runs for everyone else. The mitigation is the small team size and the cheap re-run cost of catching it on the next contributor's machine. If team size grows or the suite slows down, this decision will need to be revisited.

**Superseded**: CI now exists. `.github/workflows/linux-check.yml` runs on every push on a self-hosted Linux runner: a gcc build running the full ctest suite and the B-tier binary, a clang build doing the same minus the A-tier golden tests (there are no clang goldens), a headless gcc build (`COAST_WITH_GUI=OFF`) running `unit.cli.*` and the B-tier binary, and a `python-eval` job that lints, type-checks and tests `eval/`.

---

## Items not yet decided (or deliberately deferred)

- **Per-component GUI smoke** (construct each gui_* component, tick one frame). Deferred during Develop; revisit on next GUI refactor or after a crash regression.
- **Image-diff regression tests** for renderers. Phase-2 — cost of headless GL + golden staleness + cross-driver variance vs. value delivered argued against phase-1.
- **`imgui_test_engine`-driven interaction tests**. Phase-2.
- **100-run statistical baseline** for outlier detection on fingerprints. Phase-2 — calibration work (statistic choice, threshold, false-positive rate) is non-trivial and the fixed-seed scheme delivers most of the value sooner.
- **Wider A-tier matrix** (every `configs/*.json` × seed-offsets 0/1/2). Scoped down during Develop to the three mnist seeds for runtime budget, since widened by one seed each of `mnist_ngm3_*` and `mnist_ngm25`; one-line CMake addition per pair when needed.
