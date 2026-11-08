# Plan — closing the test-coverage gap on `feature/ngm3`

Date: 2026-06-06
Companion to `docs/test_coverage_gaps_2026-06-06.md`. Ordered by ROI (impact ÷ cost), not by
module.

> **Status (branch `feature/ngm3-tests`):** Phase 1 **complete** — 1.1, 1.2, 1.5 (commit
> `c7311f5`); 1.3, 1.4, 1.7 (commit `b47537a`). Item **1.6 deliberately skipped**: the existing
> `integration.ngm3_synthetic_smoke` already asserts the permanence/seg-id/output invariants, so
> a separate invariant test would be cargo-cult. **Phase 2 complete** — 6 GPU validator harnesses
> registered as `gpu.smoke.*` (commit `4284874`); the two `*_spike` perf harnesses excluded (not
> pass/fail). **Phase 3 complete** — e2e observe capture, ngm3 config error paths, hdf5 executor +
> run_manifest error paths, and a corrupt-state B-tier case (commit `e497a91`). **All three phases
> done.** Suite: 31/31 green (incl. 6 GPU on AMD Radeon 890M). Each item states the **tier and why that tier**, the **concrete assertion** (so it
catches a real bug, not a tautology), and an **effort** estimate. A closing section lists what
we deliberately do **not** test, with reasons — the anti-cargo-cult discipline.

Grounding facts that shaped the tiering (verified against the code):
- The `observe` *logic* (config parse, dtype/shape lock, trigger evaluation, writer→HDF5) is
  reachable without a running sim → mostly **unit**-testable. Only the full `OBSERVE`-from-sim
  capture path needs a subprocess. (`observe_registry.{h,cpp}`, `observe_writer.cpp`,
  `observe_cfg.h`.)
- ngm3 single steps are **not hand-checkable** — each neuron carries its own `fast_rng` whose
  draws interleave with the permanence/threshold math (`hd_ngm3_neuron_group.cpp:298,420,453`).
  So ngm3 targets are **invariants/properties**, not exact values.
- `observe_registry` is a process **singleton** (`instance()`). Tests sharing a process must use
  **distinct group ids per case** (or we add a test-only `reset()`), else state leaks between
  cases. This is a real hazard — encoded below.

---

## Phase 1 — high impact, low cost, CPU-only (runs in CI everywhere)

Closes the CRITICAL observe gap and the HIGH ngm3-invariant gap. No GPU, no MNIST, fast.

### 1.1 `unit.observe.config_parse` — observe config deserialization *(unit; ~1.5h)*
- **Why unit:** `observe_cfg_t` ↔ JSON is context-free nlohmann (`observe_cfg.h:41-70`).
- **Assert:** valid JSON → expected groups/ranges/combine; `combine:"and"|"or"` parsed
  case-insensitively; **malformed throws** — a 1-element step range and a bad combine string
  each `CHECK_THROWS_AS(..., std::runtime_error)` (`observe_cfg.h:48,67`). This is the gap that
  would otherwise let a typo'd capture window silently misconfigure.
- Mirror the existing `tests/unit/json_cfg/` fixture-file pattern.

### 1.2 `unit.observe.dtype_lock` — site dtype/shape locking *(unit; ~2h)*
- **Why unit:** `resolve_site()` runs without a sim; needs only a registry + group.
- **Assert (happy path):** first `resolve_site(float, shape=[], 4B)` creates the site; read back
  `site->h5_dtype/shape/value_bytes` and verify they are locked to the first call
  (`observe_registry.cpp:131-150`).
- **The mismatch path `abort()`s** (`:167-169`), not throws — so it cannot be a doctest
  `CHECK_THROWS`. Cover it as a tiny **B-tier death test** (1.3) instead. Do **not** fake it
  with a signal handler — brittle.
- **Singleton hygiene:** unique group id per case.

### 1.3 `b_tier.observe_dtype_mismatch_aborts` — guard actually fires *(B-tier subprocess; ~1h)*
- **Why B-tier:** the only honest way to assert an `abort()` is a child process with a non-zero
  exit. The existing B-tier harness already does subprocess + exit-code checks.
- **Assert:** a minimal driver that resolves the same site twice with conflicting dtype exits
  abnormally (signal/abort), and a matching dtype does not. Proves the invariant guard is live,
  which a unit test structurally cannot.

### 1.4 `unit.observe.trigger_eval` — capture-window/trigger logic *(unit; ~2h)*
- **Why unit:** `evaluate_triggers()` is a pure function of (step, ranges, combine, prog-flags)
  (`observe_registry.cpp:311-329`). Gate arming uses `probe::set_current_step()` (testable).
- **Assert:** range-hit in/out of `[start,stop]`; `OR` vs `AND` combine; programmatic
  `on_trigger(START/STOP/SINGLE_CAPTURE)` flips `gate_this_step` as specified; sticky-trigger
  stays armed (`:365-373`). These are the off-by-one-prone decisions that drop or duplicate
  captured steps.
- Needs friend access or driving via the public surface (`on_trigger` + group-field readback) —
  prefer a `friend` test declaration over widening the API.

### 1.5 `integration.observe_writer_roundtrip` — writer → HDF5 layout *(in-process integration; ~3h)*
- **Why integration, not unit:** exercises the real `hdf5::executor` worker thread + a temp
  `.h5` file. No sim, but not "pure" — so it lives beside the other in-process integration
  tests, not in `unit/`.
- **Assert:** construct `observe_writer_t` on `/tmp/…XXXX.h5`; `start()` →
  `enqueue(site, step)` a handful of rows → `stop_and_join()` (synchronous drain,
  `observe_writer.cpp:243-270`); reopen with the HDF5 C API and verify: value dataset at
  `/{flat_id}`, parallel `/{flat_id}__step_index`, row count == enqueues, dtype preserved
  (`:99-100`). This is the test that proves captured data actually lands on disk correctly —
  the single highest-value observe test.

### 1.6 `integration_ngm3.invariants` — ngm3 CPU property assertions *(in-process integration; ~2h)*
- **Why integration, not unit:** reuse the existing synthetic-group harness
  (`tests/integration_ngm3/synthetic_smoke_test.cpp` builds an 8-neuron group via `build_env` +
  `test_ngm3_synthetic.json` — no MNIST). Hand-wiring io_buffers in a pure unit test is fiddlier
  and more brittle than reusing this path.
- **Assert after K steps** (read via `get_proximal_representation` / `get_*_segment_ids` /
  `get_outp`, `hd_ngm3_neuron_group.h:109-120`):
  - permanences finite and ∈ [0,1] (clamp at `:639` — catches a broken clamp the golden's
    aggregate tolerance could mask),
  - segment_ids ∈ [1, `max_seg_id` = 2^(depth+1)−1] (`:55,93-95`),
  - output size == neuron_cnt, all finite.
- These are real invariants whose violation corrupts downstream sigmoid/tree-fold math, yet are
  invisible to a finite-tolerance output fingerprint.

### 1.7 `integration_ngm3.branch_timing` — dendrite branching fires on schedule *(in-process integration; ~1.5h)*
- **Why integration:** branching only triggers after `branch_interval` inputs
  (`:532-533`); needs a multi-step run. Use a **small `branch_interval`** synthetic config to
  keep it fast.
- **Assert:** before the interval, all segment_ids == 1; after crossing it, child ids (2S, 2S+1)
  appear and stay ≤ max_seg_id (`:751-771`). Catches off-by-one / interval-counter bugs the
  smoke test only grazes.

**Phase 1 total ≈ 13h.** Deliverables: `tests/unit/observe/` (3 cases), one B-tier case, one
writer round-trip, two ngm3 property tests. After this the two worst gaps (observe = 0 functional
tests; ngm3 = no invariant tests) are closed with CPU-only tests that run on every machine.

---

## Phase 2 — register existing GPU harnesses as CI smoke tests *(low cost, GPU-gated; ~2h total)*

Five harnesses already encode real assertions but run only by hand. Wire them into ctest behind
the **existing** GPU gate (`coast_add_integration_test(... GPU)` skips when `COAST_WITH_WGPU=OFF`
and labels `gpu` so sanitizer/CPU runs exclude with `-LE gpu`). Near-zero authoring cost, high
ROI: it puts the entire GPU path under CI wherever a device exists.

- `gpu.smoke.context` ← `coast_gpu_smoke` (device init + limits)
- `gpu.smoke.mnist_io` ← `coast_mnist_io_gpu_probe` (dataset upload + NSE/MAX/AVG probe)
- `gpu.smoke.ngm3` ← `coast_ngm3_gpu_probe` (mnist→ngm3 chain, finite/non-degenerate output)
- `gpu.smoke.io_buffer` ← `coast_io_buffer_gpu_probe` (swap/readback)
- `gpu.smoke.dispatcher` ← `coast_dispatcher_probe` (heterogeneous dispatch ordering)

Mechanical change: each `*_main.cpp` returns non-zero on failure already; register via
`add_test` with a `gpu` label and a `WILL_FAIL`-free pass/fail on exit code. No new assertions to
write.

---

## Phase 3 — full-path & error-path coverage *(medium cost; ~8h, do after 1–2)*

Lower marginal value than Phase 1 but closes the robustness gaps.

### 3.1 `b_tier.observe_capture_e2e` — config → capture → file *(B-tier subprocess; ~3h)*
- **Why B-tier:** the full `OBSERVE`-from-sim path needs a real run; B-tier already asserts HDF5
  side-effects of `--save-state`/`--log-buffers`, so this fits. Prefer a **small synthetic
  observe config** over `mnist_observe_test.json` (MNIST = slow); the latter can be a second,
  `gpu`/slow-labeled case.
- **Assert:** run `coast -n N <observe_config>` headless; the emitted `.observe.h5` exists and
  contains the configured group's datasets with row counts matching the trigger window. This is
  the end-to-end proof that complements the isolated writer test (1.5).

### 3.2 `unit.json_cfg.ngm3` — ngm3 config error paths *(unit; ~1.5h)*
- **Assert:** a valid ngm3 config deserializes to the expected `hd_ngm3_cfg` fields; an
  out-of-range/typo'd value is rejected or defaulted **as documented**. Note: today unknown
  fields silently default — if that's undesired, the test pins the decision and the fix
  (reject-unknown) becomes a tracked change rather than a silent surprise.

### 3.3 error-path tests for `run_manifest`, `state_io`, `hdf5/executor` *(unit/B-tier; ~3.5h)*
- run_manifest: write to an unwritable path → returns failure, no crash (unit).
- state_io: load a truncated/corrupt `.h5` → clean error, not UB (B-tier subprocess).
- hdf5 executor: submit_sync of a throwing job → exception propagates to caller; queue drains in
  order (unit, real executor + temp file).

---

## Deliberately NOT testing (and why)

Honest scoping — these would be cargo-cult or poor ROI:

- **CPU↔GPU per-synapse numerical parity.** The two models use *different* RNG (CPU mt19937 per
  neuron vs GPU stateless PCG hash) and different FP reduction order by design
  (`hd_ngm3_gpu_probe_main.cpp:14-17` states S6 is sanity-only, parity is aggregate-only). A
  tight-tolerance parity test would assert something the design guarantees to be false. The
  existing GPU A-tier tests already use *aggregate* tolerance — that's the correct bar; no new
  parity test.
- **GPU device-lost / fault injection.** Exercising `gpu_context`'s fault callback needs a mock
  device we don't have; a real device won't lose itself on command. Not worth a brittle harness
  now — track as a known gap. The happy-path init/limits are covered by Phase 2's `gpu.smoke.context`.
- **`run_gui` / ImGui event loop.** Not headless-testable; out of scope.
- **`build_env` / `sim_instance` happy-path wrappers.** Already exercised by every integration
  test; a dedicated test would be a tautology.
- **Exact ngm3 step values.** Per-neuron RNG interleaving makes hand-derived expected values
  impossible; we assert invariants/properties instead (1.6/1.7), which is the testable truth.

---

## Suggested execution order
1. Phase 1 (1.1→1.7) — the gap that matters, CPU-only, ~13h.
2. Phase 2 — cheap GPU CI coverage, ~2h.
3. Phase 3 — robustness/error paths, ~8h.

Each item is independently landable. Recommend starting with **1.1 + 1.2 + 1.5** (the observe
config/lock/writer trio) — together they take the observe subsystem from zero functional
coverage to "the capture pipeline is proven end of disk," which is the single biggest risk
reduction on this branch.
