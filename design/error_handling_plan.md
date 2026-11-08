# Error-handling redesign: `std::expected` on the hot path

Status: **merged** (started 2026-06-27; landed on main via #83, 2525dfaa, 2026-06-28)

## Motivation

The project builds `-O3` with `NDEBUG`, so every plain `assert()` is a no-op in the
binaries that actually run. The hot per-step path (`env::process()` → each entity's
`process()`) therefore has **no means to report an error at all**: a violated
invariant is either silently ignored (assert compiled out) or, for the few real
runtime checks, propagated by an exception that unwinds out of the TBB worker /
sim thread. We want a *visible, value-based* error channel on the hot path.

Decision (see conversation 2026-06-27):
- **Cold paths** (constructors, setup, dataset loaders) keep exceptions — they are
  cold, single-threaded, and unwinding is fine there.
- **Hot path** (`process()` and everything reachable per step) gets a value-based
  channel: `std::expected<void, core::step_error>`, reduced across the TBB fan-out,
  surfaced to the controller as a clean halt.
- In-place contract checks **stay in place** (they document the assumption at the
  site that depends on it, and survive a property later becoming per-step dynamic).
  They convert `assert(...)` → `COAST_REQUIRE(...)`, which is always compiled in.

## Why this is cheap (the granularity argument)

`env::process()` makes **K virtual `process()` calls per step, K = entity count
(single digits), not synapse count**. So the `expected` returns, the `if(!r)`
checks, and the combinable reduction all happen at K-granularity — negligible
against the million-scale synapse loop *inside* `process()`. The `expected`
machinery never enters the inner loop. Per-element checks (e.g. a future finiteness
guard) use a branchless monoid (`ok &= isfinite(x)`) tested once after the loop —
never a per-synapse `expected`. Phase 1 validates this empirically.

## Vocabulary

- `core::error_kind` — `{ contract_violation, non_finite, device_fault, … }`.
- `core::step_error` — `{ error_kind kind; std::size_t entity_id; const char* site; }`.
  Lean by design: the **step index is stamped by the controller** at the halt
  boundary (it owns the counter); `entity_id` is stamped by `env::process()` during
  reduction; `device_fault` uses `entity_id == no_entity`.
- `COAST_REQUIRE(cond, kind)` (in `include/core/require.h`) — always compiled in;
  on violation returns `std::unexpected` from the enclosing function. Only valid in
  a function returning `std::expected<…, core::step_error>`. Success path = one
  `[[unlikely]]` branch; payload built only on the cold branch.

## Reduction under TBB (allocation-free)

`dispatch_cpu_entities` reduces per-entity `step_result`s with an **allocation-free
sink**: a single `std::atomic<bool> cpu_fault_` (reset once + read once per step,
relaxed) plus a mutex-guarded `std::optional<step_error>` written **only on the cold
`!res` branch**. The happy path does no per-step container construction, no
per-thread slots, and no shared writes beyond the two relaxed atomic ops. The
smallest `entity_id` wins (scheduling-independent → deterministic, matters for the
bit-determinism / golden contract).

> Design note: the first implementation constructed a `tbb::combinable` per step.
> A non-interleaved A/B looked like a significant −1.8% regression; an *interleaved*
> A/B showed it was NOT significant (t=−1.67) — the first result was confounded by
> CPU thermal/boost drift between the two measurement windows. The point estimate
> stayed consistently ~−1.5% though, and per-step `combinable` construction was the
> obvious culprit, so the sink was rewritten allocation-free (above) and re-measured.

## Controller halt

`do_step()` consumes `env::process()`'s (and, after Phase 2, `swap_io()`'s)
`expected`. On `unexpected`: record `{step_error, current_step}` under a fault
mutex (sticky — first fault wins), set a `halted_` flag, set `mode_ = paused`.
`thread_main` re-forces paused while `halted_`. New getters: `is_halted()`,
`get_fault()`.
- **Headless** (`run_headless`): after the poll loop, if `get_fault()`, report to
  **stderr** (so it can be logged) and return non-zero; manifest `exit_reason`
  stays `fatal_error`.
- **GUI**: already reflects paused; a fault banner reading `get_fault()` is a
  follow-up (the key win — no more sim-thread `std::terminate` — lands in Phase 1).

## The all-or-nothing conversion set (verified)

`io_entity::process()` is pure-virtual; its signature change forces every override
in one commit.

Production overrides (coast_lib): `mnist_io`, `color_mnist_io`, `cifar10_io`,
`cifar100_io`, `dsprites_io`, `chairs3d_io`, `emnist_io`, `kmnist_io`,
`mnist_io_gpu`, `hd_ngm2_neuron_group`, `hd_ngm3_neuron_group`,
`hd_ngm3_gpu_neuron_group`.

Standalone harness subclasses (separate exes — break the build if skipped):
`dispatcher_probe_main` (×3), `mnist_io_gpu_spike_main`, `mnist_io_gpu_probe_main`,
`hd_ngm3_gpu_probe_main` → mechanical `return {};`.

Call sites consuming the new return: `sim_thread_controller.cpp:118` (driver),
`sim_env.cpp:298` (parallel_for → combinable), `sim_env.cpp:338` (GPU serial loop),
3× `env.process()` in probe-mains.

## Phases

### Phase 0 — Vocabulary (no behavior change) ✅ DONE
- [x] `include/core/step_error.h`, `include/core/require.h`, `COAST_SITE`.
- [x] doctest unit test (`tests/unit/core/test_step_error.cpp`, `unit.core.step_error`) — passes;
      its TU includes both doctest `REQUIRE` and `COAST_REQUIRE`, proving the prefix avoids collision.
- Gate: builds; zero production use. ✅

### Phase 1 — Interface conversion + plumbing + ngm3 first consumer ⭐ GO/NO-GO
- [x] `io_entity::process()` → `sim::step_result` (alias); all 12 production overrides +
      1 integration-test override + 6 harness stubs converted; **ngm3 gets real
      `COAST_REQUIRE`** at all 5 in-place sites (`process()`, `gather` lambda, `compute_step`).
- [x] `env::process()` → `step_result`; `tbb::combinable` reduction (deterministic,
      zero happy-path shared writes); GPU serial loop checks results.
- [x] `do_step()` halt + sticky `halted_`; `get_fault()`/`is_halted()`; `thread_main`
      re-pause; `run_headless` stderr + exit 1.
- [x] (CPU config; GPU `check_or_throw` + `swap_io` untouched this phase — Phase 2.)
- [x] Build green (build-tests, Release, 100%, 0 errors — all-or-nothing set complete).
- [x] **Behavior-neutral**: 22/22 non-GPU unit tests pass (incl. `unit.core.step_error`);
      CPU ngm3 goldens bit-identical (`mnist_ngm3_philox`, `ngm3_synthetic_smoke`,
      `ngm3_branch_timing`).
- [x] **SPS within noise** — interleaved A/B (alternating order, paired by round),
      ngm3 CPU `mnist_ngm3.json`, N=40k. Allocation-free reduction: −1.29%, paired
      t=−1.79 (df=9) → NOT significant (p≈0.10), below the ±3% noise floor. The
      combinable→atomic rewrite did not move the point estimate (−1.48%→−1.29%), so
      the residual is irreducible `expected`-return plumbing or binary-layout noise,
      not per-step work. (Non-interleaved first pass showed −1.78%/t=−2.99 but was
      confounded by CPU thermal drift between measurement windows — interleaving fixed it.)
- Gate: **GO** — within noise. Caveat: consistently slight-negative point estimate
      (~−1.3%); granularity argument predicts it shrinks on denser configs. Optional
      follow-up: confirm on a synapse-dense config + more rounds if certainty wanted.

### Phase 2 — GPU harmonization (throw-free `env`) ✅ functional gate passed
- [x] `gpu_device::check_or_throw()` → `check() → std::expected<void, core::step_error>`
      (core-layer type, no sim_core dep); webgpu override is the single try/catch
      firewall (logs driver detail to stderr, returns `device_fault`); null → `{}`.
- [x] Both `env` consumers converted (`process()` check + `swap_io()` check);
      `env::swap_io()` → `step_result`; `do_step()` checks swap_io too. `env::process()`
      and `swap_io()` are now genuinely throw-free.
- [x] Build green; CPU goldens still bit-identical; 22/22 unit.
- [x] **GPU gate (outside sandbox, real Radeon 8060S): 14/14 GPU-labeled tests pass** —
      incl. `mnist_ngm3_heterogeneous` (CPU+GPU), GPU ngm3 state roundtrip/resume parity
      (bit-deterministic), `ngm3_gpu_forced_rebuild` (converted test entity), `gpu.smoke.*`.
- [x] GPU SPS within noise — interleaved A/B (pre-Phase1 vs Phase2), `mnist_ngm3_gpu.json`:
      −1.15%, within GPU noise (pre std 2.9% > delta). Host-side change on a compute-bound
      path → structurally negligible, as predicted.
- Gate: **GO**. Cumulative pre-Phase1→Phase2: CPU ~−1.3%, GPU ~−1.15%, both within noise.

### Phase 3 — Real `COAST_REQUIRE` into remaining hot-path entities ✅ DONE
- [x] ngm2 `:136,141` (the last genuinely *dead* hot-path asserts — no-ops in Release) → `COAST_REQUIRE`.
- [x] io loaders (mnist/emnist/kmnist/color_mnist/cifar10/cifar100/dsprites/chairs3d):
      in-`process()` setup-guard throws (`output_buf`, size mismatch) AND the inline
      `out_of_range` index-bounds throws (cifar/dsprites/chairs3d) → `COAST_REQUIRE`.
      User decision: plain `COAST_REQUIRE` (terse), accepting loss of the "expected X got Y"
      message detail. This also fixes a *pre-existing* latent bug: a throw inside `process()`
      re-thrown by TBB on the sim thread would `std::terminate` (do_step/thread_main have no
      catch); the value channel halts cleanly instead.
- [x] GPU entities (`mnist_io_gpu`, `hd_ngm3_gpu`): `output_buf` guard throw → `COAST_REQUIRE`.
- [x] inhibition throws are all in `make_inhibitor`/`make_gpu_inhibitor` factories (cold
      setup) — correctly LEFT as throws.
- [x] Gate: build green; 6/6 CPU goldens bit-identical (ngm2 + ngm3); 22/22 unit; dsprites
      config smoke-runs clean (exercises the converted cifar-family negations).
- Residual (flag for Phase 5): `mnist_db::get_norm_image()` still throws `std::out_of_range`
  (a shared accessor with a `.at()`-style throwing contract, not a `process()` guard) — and
  stdlib throws (bad_alloc) remain. `process()` is throw-free w.r.t. its own explicit
  defensive guards, not 100% (converting the accessor is a separate signature-ripple call).

### Phase 4 — Cold-path `assert → throw` + fix broken asserts ✅ DONE
- [x] ngm3 ctor (`proximal_cnt`, `dendrite_branch_depth` range) + `init_storage`
      (`proximal_in_size`) asserts → throws, consolidated with the existing
      `branch_interval` throw. Config contracts now fire in Release.
- [x] ngm2 `get_avg_acc_theta` — the `isnormal(x)+epsilon` asserts that can never fire
      → **fixed** to `assert(std::isfinite(x))` (the evident intent: allow 0, reject NaN/inf),
      not deleted, preserving the author's numerical-health intent.
- [x] `random_index` `ri_fatal`: `std::abort()` → `throw std::runtime_error`. A bad
      `ri_reach`/`ri_strength` config is constructed on the cold build path, so it's now
      caught by `build_env` and reported gracefully like every other config error, instead
      of hard-aborting the process.
- [x] **Left as-is (deliberate, per-site)**: GUI-vis asserts (`gl_texture`, `vec_vis`,
      `vec_group_vis`, `vec_ring_buffer`) — debug-only render invariants, converting to throw
      risks crashing the GUI on a transient; `sim_thread_controller:116` self-deadlock check —
      a dev-usage assert; `observe.h`/`observe_registry` — already `abort()` in Release.
- [x] Gate: build green; CPU goldens bit-identical; 22/22 unit (incl. inhibition/random_index).

### Phase 5 — Dedicated re-analysis (capstone) ✅ DONE
- [x] **Census diff** (src+include): plain `assert()` 24 → 14 (12 real, rest comments);
      `COAST_REQUIRE` 0 → 31; `throw` 148 → 132; `std::abort` 4 → 2. **Plain asserts on the
      per-step hot path = 0** — the remaining real asserts are all GUI-vis render invariants,
      the ngm2 `get_avg_acc_theta` stats query (now `isfinite`), `observe` (already `abort`),
      and the `run_quiesced` dev-check.
- [x] **Throw-freedom**: `env::process()`/`swap_io()`/`do_step()` are throw-free; the only
      `throw` reachable from a valid `process()` is the shared `mnist_db::get_norm_image()`
      `out_of_range` accessor (documented residual) + stdlib (`bad_alloc`).
- [x] **Fault-injection test** (`unit.sim_core.step_fault`): a minimal faulting entity proves
      entity → `env::process()` unexpected (entity_id stamped) → deterministic min-id reduction
      (both orders) → controller sticky-halt (kind/entity_id/step correct, step frozen, paused).
- [x] **Perf** → `benchmarks/perf_error_handling.md` (cites `scripts/bench_error_handling_sps.sh`):
      CPU −1.29% (t=−1.79, not sig), GPU −1.15% (within noise). Within the ±3% floor.
- [x] **Sanitizers**: ASan (no leaks) + UBSan (no UB) + TSan all clean on the fault test.
      TSan needed the newly added `tests/sanitizer_suppressions/tsan.supp` (broad
      `called_from_lib:libtbb.so` — a precise `race:` scope doesn't work, TSan inlines the
      TBB partitioner frames into the caller; caveat documented). My `cpu_fault_` reduction
      shows NO race by provenance (all race frames were TBB-internal). `SANITIZERS.md` updated.
- [x] **Docs**: `docs/content/threading/_index.md` — fixed the stale `check_or_throw` line and
      added §6 "Hot-Path Fault Channel" + `halted_` atomic. `hugo` clean (only pre-existing
      theme-deprecation WARNs).

#### Residuals (deliberately out of scope; candidate follow-ups)
- `observe` dtype/shape-lock `std::abort()` ×2 (`observe_registry.cpp:168`, `observe.h:75`) —
  reachable from the hot path via `OBSERVE()`, but converting routes `expected` through the
  OBSERVE/PROBE macros + observe machinery (its own phase). Already fires in Release.
- `mnist_db::get_norm_image()` `out_of_range` — a `.at()`-style throwing accessor; converting
  is a signature ripple to all callers.
- GUI-vis / `run_quiesced` asserts kept as debug-only by design.
