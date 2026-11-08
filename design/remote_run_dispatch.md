# Remote Run Dispatch — Design

**Status:** proposal / not yet implemented
**Scope:** let `coast::simc` (the console sim-control façade) execute headless
runs on a remote machine over SSH, instead of only spawning a local child.
**Author's context:** written against `sim_console_api.{h,cpp}` as of commit
`79fecb7` (worktree `cimc-coast-consim`).

---

## 1. Problem

The console can already drive headless simulations: `run()` stages a run
directory (config + input state), and `step()` / `step_async()` spawn a child
`coast` process against it (`posix_spawnp`, stdout/stderr → per-step log, state
chained across calls via `--load-state`/`--save-state`). See
`design/sim_console_api.md`.

We want the *same* workflow to target a **different machine over SSH** — e.g.
kick off long/heavy runs on a dedicated Linux box (more cores, a real GPU) from
a console running on a laptop — without changing the console-facing API shape.

The open sub-question that motivated this doc: *how does the remote machine get a
`coast` binary that matches "the current version" the user is working on?*

---

## 2. Reframing — this is not a cross-compilation problem

The instinct was "cross-compile a Linux binary on the dev machine and ship it."
That is the wrong tool here. The task decomposes into two independent
sub-problems, and cross-compilation is a poor fit for the one it touches:

1. **Transport of a run** (per dispatch). Nearly free: the console↔headless
   boundary is *already* a file-in / process / file-out contract. Remoting it is
   an `rsync` + `ssh` bracket around the existing spawn.

2. **Deployment of "the current version"** (per version, not per run). This is
   where cross-compilation was proposed. Building **on the remote** dominates it:

   - **`-march=native`** (`CMakeLists.txt:159,167`, applied globally, incl. the
     AVX-512 PRNG path) is correct for free when the compile happens on the
     target CPU. Cross-compiling forces a pinned target arch → either underused
     SIMD or `SIGILL` on a feature mismatch. Real perf/correctness lever, not a
     nicety.
   - **Deps + GPU drivers already live on the remote** (SDL2 / TBB / HDF5 / LLVM
     / Vulkan + the matching `wgpu-native` `.so`). Cross-compiling means
     replicating all of that as a sysroot on the dev machine for *zero* runtime
     benefit — the headless run needs the remote's real driver stack regardless.
   - **Provenance stays honest.** `coast_build_info.cmake` already bakes git SHA
     + dirty + build-type into the run manifest. A remote that builds from our
     source self-certifies what it ran. A cross-shipped artifact needs a
     *separate* ledger mapping binary → (commit, sysroot, flags) — re-inventing
     provenance we already have.
   - **Latency is a one-time cost.** A persistent remote build tree makes only
     the first deploy a full build; every "current version" after that
     recompiles just the changed TUs — the same incremental loop as local dev.
     In this feature the *config* churns far more than the *binary*.

**Decision:** deploy by syncing source + building on the remote; dispatch by
bracketing the existing spawn seam with `rsync`/`ssh`. Cross-compilation is
explicitly rejected for the controlled-remote case (see §7 for when it is not).

---

## 3. Assumptions (make them explicit, test before relying)

| # | Assumption | Why it matters | How to test |
|---|-----------|----------------|-------------|
| A1 | The remote is a machine the user **controls**: SSH key access, a persistent home dir, a full toolchain + deps installable/installed. | Enables build-on-remote. If false, §7 applies. | `ssh remote 'cmake --version && which g++'` once, at setup. |
| A2 | The remote is the same OS **family** as the source (Linux ↔ Linux, or the CMake handles it). | `cmake --build` on the synced source must succeed there. | First deploy is the test. |
| A3 | Dev machine and remote share **no filesystem**. | The run dir's ABSOLUTE paths (`sim_console_api.cpp:332`, `run()` doc in the header) are host-local — they will not resolve remotely as-is. This is the one real code change on the dispatch path. | Inspect a staged `config.json`: its `snapshot_dir`/`observe.output_dir` are dev-absolute. |
| A4 | "Current version" often means **uncommitted** changes. | Rules out git-fetch-a-ref as the sync mechanism; argues for `rsync` of the working tree. | Trivially true during active dev. |
| A5 | Runs are **coarse-grained** (seconds–hours), so per-dispatch `rsync` latency (tens of ms to seconds) is negligible relative to the run. | Justifies the file-shipping model over a persistent remote daemon. | Measure a representative run's wall time vs. rsync of its run dir. |

If A1 or A2 is false, the whole approach flips — see §7.

---

## 4. Design — deploy and dispatch, decoupled

Two operations, deliberately separate. The user (or a helper) runs **deploy**
only when the binary/source changed; **dispatch** happens per run.

### 4.1 Deploy (per version)

`rsync` the working tree (not a git ref — carries dirty state, A4) to a remote
source dir, then incremental-build there:

```
rsync -a --delete \
      --exclude build/ --exclude build-*/ --exclude data/ --exclude .git/ \
      ./  remote:~/coast-src/
ssh remote 'cmake -B ~/coast-src/build -DCMAKE_BUILD_TYPE=Release \
              && cmake --build ~/coast-src/build -j'
```

- `--exclude build/`: build trees are host-specific; the remote keeps its own,
  incremental across deploys (only changed TUs recompile).
- `--exclude data/`: datasets are large and gitignored; sync once out-of-band or
  symlink the remote's own copy (mirrors the local `data/` symlink convention).
- The remote build's run manifest will carry the remote's own SHA/dirty — which
  is exactly the provenance we want (§2).

This step is a shell helper, not console API — it is version-granularity, not
run-granularity. A thin `scripts/deploy_remote.sh remote:~/coast-src` wrapper is
enough. (Whether to also expose it as a `simc::deploy(host)` convenience is an
open question, §8.)

### 4.2 Dispatch (per run) — extend the existing seam

The spawn seam is already isolated behind `COAST_SIMC_BINARY` +
`spawn_child()` (`sim_console_api.cpp:49-118`). Generalize "spawn a local child"
to "run a child, possibly over SSH":

- **New env / config knob** `COAST_SIMC_REMOTE=user@host:/abs/path/to/coast`.
  Absent → current local behavior, unchanged (default path = the running exe).
- When set, `step()` / `step_async()` become:

  ```
  1. rsync  <local run_dir>/  →  remote:<remote_run_dir>/     # config + input state
  2. ssh    user@host  <remote coast>  -n N  --load-state …  --save-state …  config.json
  3. rsync  remote:<remote_run_dir>/  →  <local run_dir>/     # output state + logs
  ```

  Step 2 replaces the `posix_spawnp` call; steps 1 and 3 are new brackets. The
  child's stdout/stderr still land in the per-step log — now via `ssh`'s stdio,
  redirected locally exactly as today, so `last_log()` / `tail -f` keep working.

- **State chaining** is preserved: the `--load-state`/`--save-state` filenames are
  already relative-to-run-dir; they just need to be run-dir-relative *on the
  remote*. This forces the fix flagged in A3 (§4.3).

### 4.3 The one real code change: path portability (A3)

Today `run()` writes `snapshot_dir` and `observe.output_dir` as **dev-absolute**
paths (`sim_console_api.cpp:332-340`; rationale in the `run()` header doc: coast
resolves relative `snapshot_dir` against the config's parent but `output_dir`
against CWD, so absolute was the only locally-unambiguous choice).

For remote dispatch those absolutes are meaningless on the remote. Two options:

- **(preferred) run-dir-relative + fixed remote CWD.** Write the config with paths
  relative to the run dir, and have the SSH invocation `cd <remote_run_dir>`
  before exec'ing `coast`. Then the same run dir is valid on either machine —
  and, as a bonus, makes local run dirs relocatable too.
- **(fallback) remote-absolute rewrite.** Keep absolutes, but when
  `COAST_SIMC_REMOTE` is set, compute them against the *remote* run-dir root at
  stage time. More special-casing; keeps the two modes divergent.

Preferred option is the cleaner invariant. It is a change to how `run()` stages
config paths, gated so local-only behavior is bit-identical unless the relative
form is proven equivalent first (test: local run with relative paths must match
the golden of the same run with absolute paths).

---

## 5. What stays unchanged (the value of the current seam)

- Public API (`run_config`, `run`, `sim_run::step`/`step_async`, the async
  registry `poll_runs`/`wait_all_runs`/`render_runs_window`) — **no signature
  changes.** Remote vs local is a spawn-strategy detail behind `step()`.
- The async monitoring model: an SSH child is still "a child process with a log
  we poll"; `poll_runs()` reaps it and chains state exactly as for a local child.
- Containment: a native fault still kills only the (now remote) child, never the
  console/GUI. SSH adds a second failure axis (see §6).

---

## 6. Failure modes (what happens when it fails)

| Failure | Local today | Remote — new handling |
|--------|-------------|----------------------|
| Nonzero child exit | `step()` throws `std::runtime_error` | SSH propagates the child's exit code; same throw. Must distinguish *SSH* failure (255) from a *coast* nonzero exit. |
| Binary missing | `posix_spawnp` → ENOENT throw | `ssh … coast` → 127; map to a clear "remote binary not found — deploy first" error. |
| Transport failure (host down, key rejected, disk full) | n/a | New: `rsync`/`ssh` nonzero. Surface as a distinct error kind, not a coast failure. |
| Version skew (remote binary ≠ synced source) | n/a | The run manifest records the remote SHA/dirty; a mismatch vs. the dev tree is detectable post-hoc. **Open:** whether to assert match pre-run (§8). |
| Partial result on interrupted run | local files present | Step-3 rsync may be partial; treat the run dir as authoritative only after a clean step-3. |

The interesting new invariant: an SSH exit of 255 means *the transport* failed,
not the simulation — these must not be conflated, or a network blip reads as a
sim crash.

---

## 7. When cross-compilation / a shipped artifact *would* win

If A1/A2 are false — the remote is a **locked-down** box (shared HPC login node,
thin cloud worker, no toolchain, no permission to install deps) — build-on-remote
is off the table and a self-contained artifact is required. Even then, prefer a
**container image** over raw cross-compilation: the thing that bites is
glibc/driver-version skew between the dev sysroot and the remote userland, and a
container pins the whole userland. Raw cross-compilation is the last resort, only
for a CI-style pipeline that must emit a Linux binary from a non-Linux host with
no Linux stage available. This design targets the controlled-remote case; the
locked-down case is out of scope here and would be a separate doc.

---

## 8. Open questions

1. **Deploy ergonomics.** Shell script only, or also a `simc::deploy(host)`
   console function? Console-native deploy is convenient but drags SSH orchestration
   into the parse-boundary header.
2. **Version-skew assertion.** Should dispatch refuse to run if the remote
   binary's SHA ≠ the last-deployed source SHA? Safer, but couples dispatch to a
   deploy-tracking side-channel. Leaning: *warn, don't block* (mirrors the
   manifest's `dirty` flag philosophy — record truth, don't gate).
3. **Remote run-dir lifecycle.** Where do remote run dirs live, and who cleans
   them? Proposal: a configurable remote root (`~/coast-runs/<run-stem>_NNN`),
   same suffix-claiming logic as local `run()`.
4. **Datasets on the remote.** Sync once, symlink, or assume pre-present? Config
   references them by path; the remote's paths must resolve. Likely: assume a
   remote `data/` mirror, documented as a setup precondition.
5. **Multiplexing.** Reuse one SSH `ControlMaster` connection across steps to
   avoid per-step handshake cost? Matters only if A5 is violated (many short
   steps).
6. **GPU runs.** The remote's `wgpu-native` `.so` + Vulkan driver must match the
   remote build (they will, if built there). Confirms build-on-remote for the GPU
   path specifically; no extra work beyond §4.1.

---

## 9. Summary

- **Reject cross-compilation** for the controlled-remote case; it loses on
  `-march=native` correctness, dep/driver replication cost, and provenance.
- **Deploy** = `rsync` working tree + incremental build on the remote (carries
  dirty state; native-correct; honest provenance).
- **Dispatch** = bracket the existing spawn seam with `rsync`/`ssh`, behind a new
  `COAST_SIMC_REMOTE` knob; **no public API change**.
- **One real code change:** make run-dir paths run-dir-relative so a run dir is
  valid on either machine (A3 / §4.3).
- **New failure axis:** transport vs. simulation failure must stay distinct.
