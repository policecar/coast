# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COAST (Coherent Active State Dynamics in Self-Organizing Systems) is a C++23 neural simulation framework with an interactive SDL2/OpenGL3 GUI for visualizing the HD-NGM (Hierarchical Dendritic - Neuron Group Model) neural models: `hd_ngm2`, `hd_ngm25` and `hd_ngm3`.

## Build Commands

**IMPORTANT:** Always use `-DCMAKE_BUILD_TYPE=Release` to enable `-O3` optimization. Debug builds are ~10x slower.

```bash
# Linux
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# macOS — default: Homebrew LLVM Clang (libc++). Builds the full project incl.
# the embedded clang-repl console. Confirm `Console: clang-repl via LLVM …` at
# configure time. The console is the hard reason for this toolchain: Homebrew's
# libLLVM is libc++-built, so the interpreter can only be linked by a matching
# clang/libc++ build. (An older note here claimed Apple's system Clang lacks
# C++23 features COAST needs. The one known instance was std::generator, removed
# in favour of the hand-rolled entity_range — see include/sim_core/sim_env.h:55.
# A console-less AppleClang build works — verified 2026-07-23, see the headless
# section below.)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
  -DCMAKE_PREFIX_PATH="$(brew --prefix llvm)"
cmake --build build -j

# macOS — lighter alternative: GCC 15 (no console). libLLVM is libc++-built, so
# a GCC/libstdc++ build can't link the console (skipped with a harmless warning).
# Pin the versioned gcc@15 keg: brew's `gcc` formula rolls forward (16+), so
# $(brew --prefix gcc)/bin/g++-15 no longer resolves.
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$(brew --prefix gcc@15)/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=$(brew --prefix gcc@15)/bin/g++-15
cmake --build build -j

# Run (use System > Open Configuration menu to load a config)
./build/coast

# Headless benchmark (500k steps)
./build/coast -n 500000 configs/mnist.json
```

**Headless-only build.** `-DCOAST_WITH_GUI=OFF` (default ON) drops the whole GUI
layer: `coast_gui` is never created, SDL2 and OpenGL are never even searched for,
ImGui/ImPlot/ImNodeFlow are not compiled, and the embedded clang-repl console is
forced off (it *is* an ImGui panel). A headless build therefore configures on a
machine with no graphics stack installed at all — needs only TBB and HDF5.

```bash
cmake --preset headless          # → build-headless/  (COAST_WITH_GUI=OFF)
cmake --build build-headless -j

# Leanest possible build: no GUI and no GPU backend
cmake -B build-min -DCMAKE_BUILD_TYPE=Release \
  -DCOAST_WITH_GUI=OFF -DCOAST_GPU_BACKEND=none
```

Measured on one Linux box (Release, gcc): full build 114 s / 15.5 MB / 30 shared
libs; headless 21 s / 4.1 MB / 13 libs; headless + no GPU 20 s / 3.8 MB / 10 libs.
A headless binary produces **byte-identical** simulation state to the full build.

The `headless` preset deliberately pins **no** compiler — a headless build needs
no Homebrew LLVM, since the console (the reason the `macos` presets pin one) is
off anyway. On macOS it therefore selects AppleClang, and that path **works**:
verified 2026-07-23 on macOS 26.5.2 / Apple Silicon with AppleClang 21.0.0 —
configures and builds clean in 25 s, no SDL2/OpenGL in `otool -L`, and adding
`-DCMAKE_DISABLE_FIND_PACKAGE_SDL2=ON -DCMAKE_DISABLE_FIND_PACKAGE_OpenGL=ON`
reports both as *unused*, confirming they are genuinely never searched for.

But do **not** re-baseline goldens from `headless-tests` on macOS: goldens
generated under a different compiler are not comparable with the `macos/` ones
(A-tier fingerprints diverge ~99% across toolchains). That divergence is what the
same run measured — 84/92, the 8 failures all A-tier golden mismatches
(`mnist_seed{0,1,2}_baseline`, `mnist_ngm3_philox`, `mnist_ngm3_heterogeneous`,
`mnist_ngm25`, `mnist_ngm3_{cpu_to_gpu,gpu_to_cpu}`) against `macos/` goldens
built by a different toolchain. They trip the `coast_compare`-nonzero branch at
`tests/integration/integration_helpers.cpp:195-196`, i.e. floating-point drift, not
crashes or logic errors. Everything else, GPU included (state roundtrip/resume,
`gpu.smoke.*`, checkpoint — real Metal via webgpu), passed.

A headless binary runs headless implicitly and prints a one-line notice saying so
— unless `-H/--headless` was given, which suppresses it. `--headless` is still
accepted everywhere; it just has nothing left to switch off.

**Run length is explicit — there is no sentinel.** Every headless run must say
how long it lasts, in either build:

```bash
coast -n 1000 config.json     # finite: 1000 steps (count must be >= 1)
coast --unlimited config.json # unbounded: runs until interrupted
coast --headless config.json  # ERROR: needs -n or --unlimited
coast -n 0 config.json        # ERROR: 0 no longer means "forever"
coast -n abc config.json      # ERROR: exits 1 (used to abort the process)

# --unlimited refuses outputs it cannot deliver: an unlimited run ends only by
# being killed, so --save-state and --emit-run-manifest would never be written
# and --log-buffers would be left unclosed (an unreadable HDF5 file).
# --save-on-term lifts the refusal: SIGTERM then ends the run through the normal
# exit path, writing them (coastd runs unbounded jobs this way). SIGINT (Ctrl-C)
# is still not caught, so an interactively killed run loses them regardless.
coast --unlimited --save-state s.h5 config.json                 # ERROR
coast --unlimited --save-on-term --save-state s.h5 config.json  # OK — written on SIGTERM
coast --unlimited --checkpoint 5000 config.json                 # OK — snapshots complete
                                                                # during the run
```

`-n 0` used to mean "run forever", which made a mistyped or computed-to-zero
count an unbounded compute job. `cli::options::steps_mode` (`unset`/`finite`/
`unlimited`) now carries the mode and `max_steps` is read only through it. This
matters most in a headless build, where a bare `coast config.json` — the command
that opens a window in a GUI build — would otherwise have started that unbounded
run silently.

**macOS shortcut — CMake presets** (presets need CMake ≥ 3.21, but the project
floor is **3.24** — see `cmake_minimum_required`). The clang recipe above
is also a preset, so you don't pass the compiler flags by hand:
`cmake --preset macos` (→ `build/`) and `cmake --preset macos-tests` (→
`build-tests/`, `BUILD_TESTING=ON`) pin the Homebrew LLVM toolchain (via
`$HOMEBREW_PREFIX`, gated to Darwin — on Linux use the plain `cmake -B build`
flow). Prefer these so the macOS build stays on one compiler — the A-tier goldens
diverge ~99% between clang and GCC codegen, so a consistent toolchain keeps the
locally-generated `macos/` goldens reproducible. The manual gcc recipe still
works.

## Code Navigation

For C/C++ code navigation (where a symbol is used, call hierarchies, resolved types), load and use the LSP tool (findReferences, incomingCalls, hover) 
and use rg only to find symbol positions or as a cross-check. If results look incomplete, check whether compile_commands.json exists.

### Setting up the LSP server (clangd)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
ln -sf build/compile_commands.json compile_commands.json
```

- Use the widest configuration (`COAST_WITH_GUI=ON`, `COAST_GPU_BACKEND=webgpu`, `COAST_WITH_CONSOLE=ON`, `BUILD_TESTING=ON`).
- Reconfigure after adding TUs.
- After changing the database, restart clangd (`pkill clangd`).
- References through a namespace-scope using-declaration (e.g. unqualified `seed_t`, `fast_rng_t`) are not found by `findReferences`; use rg for those.

## Testing

**A test's outcome must never depend on the load of the machine it runs on.**
A test either probes its outcome deterministically and independently of wall-clock
time, or it does not exist. This rules out, without exception:

- asserting a measured duration against a bound (`CHECK(dt < 2s)`, `CHECK(latency < 5000.0)`),
- `sleep_for(X)` followed by a check that something happened by then (or did *not* happen yet),
- fixed observation windows ("watch the queue for 3 s, expect exactly one event"),
- relying on a scripted sleep to hold a state open long enough for a fault to land in it,
- `RUN_SERIAL` (or any scheduling property) used to keep such a test from flapping —
  that hides the defect, it does not fix it.

Instead: make the system under test *observable* (an event, a state transition, a
handshake, a sentinel file) and wait for that, and make timeouts and clocks
*injectable* so the test drives them. Performance/latency *measurements* belong in
benchmarks that report numbers, not in pass/fail tests.

**Hang detection is by progress, not by timeout.** A process we control (coastd, a
fake child, the sim thread) must expose a *heartbeat* — a monotonically increasing
progress counter (reconcile/poll passes, steps, `@coast/progress` lines). Tests
express expectations in that logical time ("after the child is reaped, the job has
left `running` within k passes"), which is load-independent: under load the passes
come later, the outcome is the same, and a wrong outcome fails deterministically.
A process is *hanging* only when its heartbeat stops advancing — that is the one
place a wall-clock threshold remains, and it never decides whether correct
behaviour passes. Timeouts are sensible only where we do not control the other
side (e.g. a network resource that may be down).

The helpers for this are `tests/support/await_progress.h` (`await_progress`,
`await_condition`, stall limit 60 s) and `tests/support/sim_sync.h`
(`sim_iterations`); the heartbeats and seams are listed in
`docs/content/testing/_index.md`. Tests that could not be made load-independent
were removed (2026-11): the discovery rate-cap and idle-stop cases, proc_util's
"output before the deadline" case, and the gate's kill + auto-resume case.

**Nothing goes to GitHub untested: every code change is followed by ALL the tests before it is pushed** — the full suite with gcc (GUI and headless builds) and with clang, all passing. This holds for every push, not only the first one of a PR: a follow-up fix, however small, gets the full run again, never just the tests it touched.

**Both compilers (on Linux).** CI builds both, and the compilers disagree on real code (PR #240: clang parses `T(std::numeric_limits<float>::quiet_NaN())` inside a doctest macro as a declaration; gcc accepts it). Clang has no A-tier goldens, so exclude them as CI does (`ctest_filter` in `.github/workflows/linux-check.yml`), and run the B-tier binary from the repo root:

```bash
cmake -S . -B build-clang -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_TESTING=ON
cmake --build build-clang -j
ctest --test-dir build-clang --output-on-failure -E '^integration\.(mnist_seed[012]_baseline|mnist_ngm3_(philox|cpu_to_gpu|gpu_to_cpu|heterogeneous)|mnist_ngm25)$|^integration_b$'
./build-clang/tests/integration_b/test_b_tier
```

**No test may depend on the NAS's timing.** The NAS is shared and in use: its latency varies by a factor of two and more, and it can be unreachable (on 2026-11-02 the runner lost it for hours). Availability is deliberately *not* hidden: CI reads the datasets from the NAS (`data -> /mnt/NAS/datasets`), so a NAS outage fails the dataset tests there instead of passing against a stale copy. CI also sets `COAST_NO_NAS=1`, under which a test that reads the NAS directly must skip (as `campaign.real` does). Never assert on a time the NAS determines: report it with `MESSAGE` and check the property behind it (e.g. "no file opened") offline. ctest `TIMEOUT`s are hang guards and keep a wide margin over the time measured **under heavy load** (2-4x nproc busy loops at the same nice), not over an idle run; one that a heavier load could reach is raised to ctest's default 1500 s.

The test suite is opt-in via `-DBUILD_TESTING=ON` (default off). Use a separate build dir so the default `build/` stays clean.

```bash
# Configure + build with tests   (macOS shortcut: cmake --preset macos-tests)
cmake -B build-tests -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-tests -j

# Run the whole suite (the integration.mnist* tests dominate the runtime)
cd build-tests && ctest --output-on-failure

# Run just the unit tests
cd build-tests && ctest --output-on-failure -E 'integration'

# Run a specific test
cd build-tests && ctest -R 'unit.cli.parse' --output-on-failure

# Headless test build (no GUI): cmake --preset headless-tests → build-headless-tests/
cmake --preset headless-tests
cmake --build build-headless-tests -j
cd build-headless-tests && ctest --output-on-failure
```

The headless suite is the full suite minus the GUI-layer tests that cannot exist
without `coast_gui`: the `unit.gui_plot.*` and `unit.gui_replay.*` cases and (when built) `unit.console.*`,
plus the 10 `b_tier.simc_*` cases (compiled out of `test_b_tier.cpp` via
`COAST_WITH_GUI`, since `coast::simc` lives in the GUI layer). Everything else —
all A-tier, all `_gpu`, the rest of B-tier — runs in both configurations. The 24
`unit.coastd.*` tests run in both; `coastd` does not depend on `coast_gui`.

**Totals depend on the build options, so quote them with the options.** Tests
registered (`ctest -N`), Linux, 2026-11-08:

| Build | Options | Tests |
|---|---|---|
| `build-tests` | `WITH_GUI=ON`, `GPU_BACKEND=none`, `WITH_CONSOLE=OFF` | **149** |
| `build-headless-tests` (preset) | `WITH_GUI=OFF`, `GPU_BACKEND=webgpu` | **165** |
| `build-min`-style + `BUILD_TESTING=ON` | `WITH_GUI=OFF`, `GPU_BACKEND=none` | **141** |
| widest (defaults + `BUILD_TESTING=ON`) | `WITH_GUI=ON`, `GPU_BACKEND=webgpu`, `WITH_CONSOLE=ON` | **176** |

The first two differ by more than the GUI layer: headless *adds* 24 GPU tests
(`_gpu`, `gpu.smoke.*`, `ngm25_gpu_*`, the CPU↔GPU checkpoint and rebuild cases)
the first build never registers, and drops 8 (7 `unit.gui_plot.*`, 1
`unit.gui_replay.*`); the console adds 3 (`unit.console.*`). So a `GPU_BACKEND=none`
build **is not the widest suite** — with `WITH_CONSOLE=OFF` as well, it reports a
green 149/149 while never compiling
the GPU or console tests at all. Check `CMakeCache.txt` before reading a pass count
as coverage. The default backend is `webgpu`, so a plain configure does compile the
GPU tests; run them outside the Claude Code sandbox (no GPU in there — see below).

On macOS/AppleClang the headless suite scored **84/92** — the 8 A-tier golden
mismatches are cross-toolchain FP drift against the `macos/` goldens, not failures;
see the headless build section above. That run predates the `coastd` suite, so its
denominator is the older 92.

**Test tiers**:
- **Unit** (`tests/unit/`): pure modules (cli::parse, spsc_queue, probe_instance decimation, json_loader, blob_writer, coast_compare). Sub-second.
- **A-tier integration** (`tests/integration/`): in-process tests that build a real sim via `coast::app::build_env`, run for N steps, fingerprint `io_buffer` state, compare against retained goldens.
- **B-tier integration** (`tests/integration_b/`): subprocess tests against the built `coast` binary — CLI surface, exit codes, file side-effects of `--save-state`/`--load-state`/`--checkpoint`/`--log-buffers`.

**Golden management**. A-tier tests fingerprint the simulation's `io_buffer` state and compare against `tests/integration/<test>/goldens/<YYYYMMDD>_<sha>.json`. To re-baseline (e.g. after an intentional change):

```bash
cmake --build build-tests --target coast_update_goldens_mnist_seed0_baseline
# or all at once:
cmake --build build-tests --target coast_update_goldens
git add tests/integration/<test>/goldens/   # commit the new baseline
```

The script keeps the newest 5 goldens per test (β-naming sorts chronologically). A test passes if it matches *any* of the retained goldens within tolerance.

**Reading a `coast_compare` failure**. The compare report is severity-sorted:
- `STRUCT  <path>  <reason>` — structural error: missing field, dtype mismatch, NaN, missing sidecar, etc. Almost always means the test contract changed; re-baseline.
- `FAIL    <path>  golden=X  new=Y  Δ=%  tol=…` — numerical drift past the per-field tolerance. Highest-Δ items first.

Tolerance is `rel=1e-6 abs_floor=1e-9` by default. Per-field overrides go in `tests/integration/<test>/tolerances.json`:

```json
{"default": {"rel": 1e-6, "abs_floor": 1e-9},
 "overrides": {"buffer_2/spike_counts": {"rel": 1e-4}}}
```

Add overrides **only after a real rerun trips the default**, not pre-emptively.

## Static analysis & sanitizers

**clang-tidy** (on-demand, reports only):

```bash
cmake --build build --target coast_tidy
```

Rule subset is conservative (bugprone-*, performance-*, readability-redundant-*, modernize-use-nullptr). Baseline at the time of writing: 66 warnings, captured in `.clang-tidy-baseline.txt`. Configure rules in `.clang-tidy` at repo root.

**Sanitizers** (out-of-tree builds, one per mode). See `SANITIZERS.md` for full details.

```bash
cmake -B build-asan  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER=asan
cmake -B build-ubsan -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER=ubsan
cmake -B build-tsan  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER=tsan
```

**IMPORTANT**: sanitizer builds need `CMAKE_BUILD_TYPE=Release`, not `RelWithDebInfo` — the -O2 vs -O3 codegen difference perturbs floating-point output past A-tier's 1e-6 tolerance.

Under sanitizers, skip A-tier integration tests (they trip on FP drift from instrumentation, not real bugs):

```bash
cd build-asan && ASAN_OPTIONS="detect_leaks=1" ctest --output-on-failure -E 'integration\.'
```

## The Claude Code sandbox (GPU + network)

Two classes of command must run **outside** the sandbox (`dangerouslyDisableSandbox`):

**GPU.** GPU tests (anything `_gpu`, e.g. the `integration.ngm3_*_gpu` suite) need
a real graphics adapter. **The sandbox has no GPU access**, so they fail with
`gpu_context: adapter request failed ... no suitable graphics adapter found`.
This is a sandbox artefact, **not** a code or environment failure — the dev
machines have working GPUs (Linux: Radeon 8060S via RADV/Vulkan; macOS: Apple
Silicon via Metal). Run `_gpu` ctests, headless GPU configs
(e.g. `configs/mnist_ngm3_gpu_full.json`), and checkpoint/resume parity on GPU
outside the sandbox. The GPU ngm3 path is bit-deterministic run-to-run, so exact
`h5diff` comparison is valid there.

**Network.** The sandbox blocks egress to most hosts, so anything that talks to
GitHub fails inside it: `git push`/`git fetch`/`git pull` (ssh to github.com),
`gh` (PRs, issues, API), and `cmake -B …` reconfigures that hit github.com via
FetchContent. Run these outside the sandbox. The only such download, the
wgpu-native zip, is kept in `COAST_DEPS_CACHE` (default `~/.cache/coast/deps`,
`cmake/coast_dep_cache.cmake`); once a verified copy is there, a new build tree's
configure needs no network.

**Gotcha (Linux):** `$TMPDIR` is unset outside the sandbox there — use an explicit
`mktemp -d` for scratch files (a bare `$TMPDIR/foo` collapses to `/foo` and fails).
macOS sets `$TMPDIR` per user, so it is defined outside the sandbox too.

## Architecture

The codebase builds as a thin `coast` executable that parses CLI args and links two static libraries: `coast_lib` (the headless core) and, in a GUI build, `coast_gui` (the SDL2/OpenGL/ImGui layer).

```
┌──────────────────────────┐
│  src/main.cpp            │  ~160 lines: parse → dispatch
└────────────┬─────────────┘
             │ links
             ▼
┌─────────────────────────────────┐
│   coast_gui (static archive)    │  ONLY when -DCOAST_WITH_GUI=ON (default)
├─────────────────────────────────┤
│   app/run_gui                   │  SDL2 + ImGui event loop
│   gui_core/                     │  dialogs, browsers, config editor
│   gui_vis/                      │  OpenGL renderers, vis_registry factory
│   gui_plot/ gui_layout/ …       │  ImPlot bridge, force layout, replay, tools
│   probe/observers/  sim_console/│  ImGui-rendering observers, simc dashboard
└────────────┬────────────────────┘
             │ links (one-directional — coast_lib never links coast_gui)
             ▼
┌─────────────────────────────────┐
│   coast_lib (static archive)    │  TBB + HDF5 only; no SDL2, no OpenGL, no ImGui
├─────────────────────────────────┤
│   cli/  app/                    │  CLI parsing + build_env + run_headless
├─────────────────────────────────┤
│   Simulation Core (sim_core)    │  env container, sim_thread_controller, sim_instance
├─────────────────────────────────┤
│   Neural Models (hd_ngm2/25/3)  │  neuron_group, dendrite components
├─────────────────────────────────┤
│   io/ observe/ logging/ h5browse│  datasets, observe sites, HDF5 logging
└─────────────────────────────────┘
```

The split is **structural, not conventional**: the dependency only ever points
coast_gui → coast_lib. Precisely what that catches:

- A core TU including an **ImGui-bearing** header fails to **compile** in every
  configuration — the imgui/implot/ImNodeFlow/SDL2 roots are on coast_gui's
  include path only. This is the common case.
- A core TU including an **ImGui-free** coast_gui header (`gui_plot/figure.h`,
  `console_iface.h`) still compiles, because `include/` is one project root for
  both archives. It fails to **link** only where coast_gui is absent — a headless
  build, or a unit test linking bare `coast_lib`. A default GUI build would not
  notice it.

**So the headless build is what enforces the boundary** — run `cmake --preset
headless` (and `headless-tests`) before trusting the split; a GUI-only build
cannot prove the core is GUI-free. If a core module seems to need a GUI header,
the header is either misfiled (as `vis_hints.h` and `demand_published.h` once
were — both GUI-free, now in `core/`) or the code belongs in coast_gui.

**Note on whole-archive linking**: several TUs (vis_registrars in `gui_vis/`) register themselves with `vis_registry` via static-initializer side effects only. A static library's default linker behavior would drop these object files because nothing references their symbols. Force-linking the whole archive into the `coast` exe makes the registrars fire. Both archives get the treatment, via `$<LINK_LIBRARY:WHOLE_ARCHIVE,…>` (CMake ≥ 3.24), which expands to `-Wl,--whole-archive` for GNU ld and `-Wl,-force_load` for Apple's ld64. Tests do NOT use whole-archive — they stay slim and pull only what they reference.

**Key modules** (GUI-layer modules — `gui_*`, `probe/observers/`, `sim_console/` — build only with `COAST_WITH_GUI=ON`):
- `cli/` - CLI options + argv parser (`coast::cli::parse`)
- `app/` - Application entry points: `build_env`, `run_headless` (core); `run_gui` (coast_gui)
- `gui_core/` - SDL2 app framework, GUI components (entity/buffer/dataset browsers), config editor
- `gui_vis/` - Visualization system with factory pattern (vis_registry), colormaps, GL textures
- `sim_core/` - Simulation environment (env), IO buffers, threading controller, `sim_instance` (RAII container)
- `hd_ngm2/` - Neural model implementation (neurons, dendrites, neuron groups)
- `hd_ngm25/` - HD-NGM2.5: ngm2's semantics transcribed into a new arena-based implementation (CPU + GPU)
- `hd_ngm3/` - HD-NGM3: ngm2 re-implemented on a flat, group-owned Structure-of-Arrays layout (CPU + GPU)
- `io/` - Dataset loaders (MNIST, CIFAR, dSprites, etc.) implementing io_entity interface
- `json_cfg/` - JSON configuration loading for reproducible experiments
- `tools/` - `sim_builder` for constructing simulations from config; `coast_compare` golden-blob comparator

## Threading Model

- **Main thread**: GUI event loop (ImGui, SDL input)
- **Simulation thread**: Dedicated thread for `process()` and `swap_io()` calls
- **Synchronization**: Lock-free atomics for mode control; hook callbacks for data updates

## Key Design Patterns

- **RAII**: Resources managed via `unique_ptr`; GPU resources need explicit `free_resources()` before context destruction
- **Factory pattern**: `vis_registry` for creating visualization instances
- **Hook system**: Deferred hook registration in `sim::env` for thread-safe GUI-sim communication
- **Type erasure**: `entity_vec` base for heterogeneous `io_entity` storage

## Live Probe System

The probe system enables real-time observation of simulation values with minimal overhead.

**Core components** (`include/core/`, `include/probe/`):
- `entity_element`: Base class providing hierarchical structure (neurons, dendrites)
- `element_type`: Type-safe identifier using `typeid`
- `structure_registry`: O(1) lookup from element pointer to structural path
- `entity_probe_set`: Thread-safe probe registration (TBB concurrent_hash_map)
- `probe_instance`: Per-probe state with SPSC queue and decimation

**Usage in simulation code:**
```cpp
// In entity_element subclass (neuron, dendrite, etc.)
probe("activity", activity_value);  // Records value if probe is active
```

**GUI** (`include/gui_probe/`):
- Probe Browser: Discover and activate probes via Introspection menu
- Observer Panel: View active probe visualizations (scalar, time series, histogram)

**Threading**: Each probe instance has a single producer (one sim thread) and single consumer (GUI thread). Thread-safe registration handled by TBB concurrent_hash_map.

## Style Conventions

- All naming uses `snake_case`: classes, methods, members
- Type aliases use `_t` suffix (e.g., `neuron_group_t`)
- Header guards: `COAST_FILENAME_H`
- Headers in `include/<module>/`, implementation in `src/<module>/`
- Use forward declarations in headers to reduce compile times
- Mark intentionally unused members/params with `[[maybe_unused]]`

## Documentation

The `docs/` directory contains a Hugo static site (Relearn theme) that documents the project's architecture, data flow, threading model, key abstractions, and a full file reference. Its purpose is to support the formation of a detailed mental model of the project and its code structure in the reader's mind.

**When making changes to the codebase, keep the documentation in sync.** If you add, rename, or remove modules, classes, data flows, or threading patterns, update the corresponding pages under `docs/content/`. Build with `cd docs && hugo` and verify zero warnings.

## Pull requests

**Do not edit a PR's description after it is opened.** Report follow-up work
(review fixes, CI findings, new commits) by **replying to the reviewer's
comments** (inline replies in the review thread) or by **adding a new PR
comment**. The description stays what reviewers read first; changes to it are
easy to miss, while a comment notifies them and keeps the history.

**Submit a review as a review, with a verdict.** Post findings on a PR as a
GitHub review (`gh pr review <n> --approve` or `--request-changes`, with the
body and inline comments in the same review), not as a plain PR comment, and
choose one of the two: approve, or request changes. GitHub refuses both verdicts
on a PR opened by the account that reviews it; say so and post the review with
`--comment` only in that case. A question for clarification is not a review:
ask it as a plain PR comment (`gh pr comment <n>`), and give the verdict once it
is answered.

## Datasets

Download datasets using `scripts/download_datasets.sh` or manually into `data/`.
Configuration files in `configs/` reference these datasets.

**In a git worktree, link `data/` from the main checkout.** `data/` is
gitignored, so a new worktree has none, and the configs and A-tier tests load it
by the relative path `./data/...`. Without it the dataset-backed integration
tests fail within ~0.01 s each, which looks like a broken build rather than
missing data. From the worktree root:

```bash
ln -s "$(git worktree list --porcelain | sed -n '1s/^worktree //p')/data" data
```

(The first entry of `git worktree list` is always the main checkout.)
