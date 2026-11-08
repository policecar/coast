# Installing `coastd` on a server

This document walks through building and installing `coastd`, the resident
per-machine COAST headless-job daemon, on a Linux server. Every step states
*what* it does and *why*, so you can adapt it rather than follow it blindly.

Linux/systemd is the **only** supported *deployment* for the daemon. On macOS the
daemon runs when started by hand, but not as a service — see
[Appendix A](#appendix-a--macos). On any other platform it refuses to start. The
*clients* (GUI panel, console façade, `coastd_client`) are portable and can drive a
remote daemon from any platform.

If you have done this before and want only the sequence of steps, use
[`QUICK_INSTALL.md`](QUICK_INSTALL.md) — the same install as a checklist, with
each step linking back to the section here that explains it.

This document is the **operator's manual**: how to get a daemon running and keep
it running. For *why* it is built the way it is — the pure state machine, the
recovery model, the persistence layout, the wire protocol and the known
boundaries — see the **coastd** section of the documentation site
(`docs/content/coastd/`, or `cd docs && hugo server`).

---

## 0. What you are actually installing

Before touching a shell, get the model right — most installation mistakes are
mistakes about the layout, not about commands.

`coastd` is a long-lived process that accepts *job submissions* over a TCP
socket and, for each job:

1. **Resolves** the requested git ref (branch/tag/sha) to a commit SHA against a
   git repository you point it at (`--repo`), fetching first so the ref is
   current. The SHA is frozen at submit time, so a job is pinned to exactly the
   code it was submitted for, even if the branch moves later.
2. **Builds** that SHA natively: `git worktree add` into a per-SHA directory,
   a shallow `git submodule update` of every submodule except the docs theme,
   `cmake` configure, then
   `cmake --build --target coast`. Worktrees are shared — ten jobs on the same
   SHA build once.
3. **Schedules** the job against a fixed number of `slots` (concurrent running
   children), highest `priority` first, FIFO within a priority.
4. **Supervises** the resulting one-shot `coast` child process, and
5. **Auto-resumes** it from its newest checkpoint after a crash, a reboot, or a
   daemon restart.

So `coastd` is a *build-and-run orchestrator*. It never embeds the simulator; it
always runs `coast` as a subprocess. This has three consequences that shape the
whole install:

- **The server needs a full COAST build toolchain at runtime**, not just at
  install time — `coastd` invokes `cmake` and a compiler for every new SHA.
- **The server needs the git repository locally**, and the daemon user needs
  whatever credentials reach its remote — `coastd` fetches that repo before
  resolving a ref (§5.2).
- **The server needs the datasets** any submitted config references.
- **The server does *not* need the config files.** A client sends the config's
  *content* over the wire, never a path — so you never stage configs on the
  server, and submitting from a laptop works with the file that is open in front
  of you. See [§5.4](#54-configs-travel-as-content-not-as-paths).

### The three directory roots

`coastd` keeps three deliberately independent roots, all derived from a single
`--data-root` unless you override them:

| Root | Default | Contents | Size | Backed up? |
|---|---|---|---|---|
| `state_root` | `<data-root>/state` | one JSON file per job, git-versioned | tiny | **yes** — this is the queue |
| `runs_root` | `<data-root>/runs` | one directory per job: config, logs, checkpoints, results | **large** | your call |
| `worktrees_root` | `<data-root>/worktrees` | one checked-out + built source tree per SHA | large, regenerable | no |

They are separate because they have completely different size, durability, and
backup characteristics. If you put `runs_root` on a big scratch volume and
`state_root` on your reliable one, that is the intended usage.

Inside `state_root`, jobs live in *bucket directories* (`active/`, `done/`,
`failed/`, `cancelled/`, `quarantine/`). **The filesystem is the truth**: a
lifecycle transition is an atomic `rename` between bucket directories. A git
repository is initialised in `state_root` as an *audit trail* only — it may lag
behind the filesystem after a crash and is never on the correctness path.

Inside `runs_root`, each job gets `<runs_root>/<job-id>/` containing:

```
config.json          the sim config the job runs (snapshot_dir rewritten by coastd)
config.submitted.json  what the client sent, byte-for-byte (provenance only)
snapshots/           checkpoint .h5 files
logs/run_*.log       one log file per launch attempt
final_state.h5       final saved state
coast_runtime.json   run manifest: build SHA, seeds, step counts, exit reason
```

---

## 1. Prerequisites

### 1.1 Build and runtime dependencies

**A server install needs no graphics stack at all.** `coast` builds in two
configurations, and `-DCOAST_WITH_GUI=OFF` (§3.1) drops the entire GUI layer:
SDL2 and OpenGL are then never even *searched for*, so they need not be
installed. `coastd` itself is GUI-free by construction — it links only the
headless `coast_lib`, never `coast_gui`. Verified on one Linux box: a `coastd`
built with the §3.1 flags links exactly 8 shared libraries — `libtbb`, `libhdf5`
(plus its `libz`/`libsz`), `libstdc++`, `libm`, `libgcc_s`, `libc` — and nothing
resembling SDL or GL. Stronger still, adding
`-DCMAKE_DISABLE_FIND_PACKAGE_SDL2=ON -DCMAKE_DISABLE_FIND_PACKAGE_OpenGL=ON`
makes CMake report both as *unused variables*: the lookups genuinely do not
happen, rather than happening and being harmless.

This matters twice over: once for the host build below, and again for every
**per-SHA build the daemon runs at runtime** — which needs the same treatment
via `build_cmake_args` (§6.2), or each job's build will pull in a graphics stack
the server has no use for.

Required:

| Dependency | Why |
|---|---|
| C++23 compiler: GCC 15+, or Clang with libstdc++ 15+ (see the README's Dependencies) | `CMAKE_CXX_STANDARD 23`; configure checks the library features it needs |
| CMake ≥ 3.24 | build system (`cmake_minimum_required`); also invoked by `coastd` at runtime for each SHA |
| `git` | ref resolution, worktrees, submodules, and the state audit trail |
| TBB (dev) | simulation parallelism |
| HDF5 C library (dev), 1.12+ | checkpoints and observe outputs |
| `make` or `ninja` | whatever CMake generates for the per-SHA builds |

Debian/Ubuntu:

```sh
sudo apt install build-essential cmake git ninja-build libtbb-dev libhdf5-dev
```

The distribution's packages must meet the minimums above: Ubuntu 24.04's do not
(GCC 13/14, HDF5 1.10), and Debian 13 ships GCC 14.

Arch:

```sh
sudo pacman -S base-devel cmake git ninja onetbb hdf5
```

You do *not* need an X server, a display, a GPU, or SDL2/OpenGL for a CPU-only
deployment. If you *do* build with the GUI on for some reason (you should not on
a server), add `libsdl2-dev libgl1-mesa-dev` — and note that CMake ≥ 3.24 is a
hard floor in either case.

Optional: `python3` — used only by the opt-in real-coast gate test in §4.3.

### 1.2 Network access

The per-SHA build clones submodules and runs a CMake configure that may hit the
network via `FetchContent`. Turning the GPU backend off (§3.1) removes the one
FetchContent download that is otherwise unavoidable.

Each build worktree clones its own submodules from the URLs in `.gitmodules`,
independently of this clone's. It initialises every submodule except the docs
theme (`docs/themes/hugo-theme-relearn`, 1.1 GiB of history, never compiled), at
depth 1.

To clone them from local mirrors (an air-gapped or slow link), set
`submodule.<path>.url` in `<repo_root>/.git/config`, which every build worktree
shares, to a `file://` URL. Git ignores `--depth` for a bare-path URL and clones
the full history. Git refuses the `file` transport for submodule clones unless
`protocol.file.allow=always` is set (its CVE-2022-39253 mitigation), and the
daemon's git reads that key only from the environment: it ignores `~/.gitconfig`
(§9.1), and repo-local config does not reach a submodule clone. Set it in the
daemon's environment, e.g. in the systemd unit:

```ini
Environment=GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always
```

`coastd` does not set this itself, because that would switch the mitigation off in
every deployment, including those without local mirrors.

### 1.3 A dedicated user account

Run `coastd` as an ordinary, unprivileged user — **not** root. It builds and
executes arbitrary git refs, so it should own nothing it does not need. The
recommended deployment is a *systemd user service* under a dedicated account
(§7), which inherits that user's git credentials and environment.

```sh
sudo useradd --create-home --shell /bin/bash coast
sudo -iu coast          # everything below runs as this user
```

---

## 2. Get the source

```sh
git clone --recurse-submodules <your-coast-remote> ~/src/coast
cd ~/src/coast
```

`--recurse-submodules` matters: COAST vendors nlohmann/json as a submodule and
the build of `coastd` fails without it. (imgui, implot and ImNodeFlow are
vendored too but are not compiled in a `COAST_WITH_GUI=OFF` build. The per-SHA
builds clone their own submodules (§1.2), so this clone's submodules are needed
only to build `coastd` itself.) If you already cloned without it:

```sh
git submodule update --init --recursive
```

This clone serves **two distinct purposes**, and it helps to keep them apart in
your head:

- It is the source you build the `coastd` binary *itself* from (once, now).
- It is the `--repo` that `coastd` resolves refs against and creates worktrees
  from (continuously, at runtime).

They can be the same directory — that is the simple, recommended setup — but
`coastd` never builds *in* it. It only reads refs from it and adds worktrees
elsewhere, so your working copy is never disturbed.

### 2.1 Datasets

Configs reference datasets by path. `coastd` runs each child with its working
directory set to the job's run directory, so **a config using relative dataset
paths like `./data/mnist/...` will not resolve on its own.** You have three
options, in order of preference:

- **Set `dataset_base` in `coastd.json`** (recommended) to the machine's shared
  dataset directory. `coastd` then drops a `data` symlink to it into every job's
  run dir, so the conventional relative `data/<set>/...` path every stock config
  uses just works — submitters need not touch their configs at all. See §6.2.
- put absolute dataset paths in the configs you submit, or
- keep the relative form with no `dataset_base` and accept that jobs will fail to
  find their data.

Fetch the datasets once:

```sh
./scripts/download_datasets.sh
```

They land in `data/` and are gitignored, so worktrees created by `coastd` will
**not** contain them — which is exactly why the `data` symlink (or an absolute
path) is needed: the datasets live outside the per-SHA worktree the child runs.

---

## 3. Build `coastd`

### 3.1 Configure

```sh
cd ~/src/coast
cmake -B build-coastd \
      -DCMAKE_BUILD_TYPE=Release \
      -DCOAST_WITH_GUI=OFF \
      -DCOAST_GPU_BACKEND=none \
      -DCOAST_WITH_CONSOLE=OFF
```

Each flag, and why:

- **`-DCMAKE_BUILD_TYPE=Release`** — enables `-O3`. A Debug build of COAST is
  roughly 10× slower. This is never optional for anything that runs simulations.
- **`-DCOAST_WITH_GUI=OFF`** — drops the whole GUI layer (`coast_gui`). SDL2 and
  OpenGL are never searched for, ImGui/ImPlot/ImNodeFlow are not compiled, and
  the build succeeds on a machine with no graphics stack installed. `coastd`
  never touched the GUI, but with this ON the *simulator* half of the build
  still would. There is an equivalent `--preset headless` for interactive use;
  the explicit flag is used here so it composes with the two below.
- **`-DCOAST_GPU_BACKEND=none`** — compiles out the GPU path entirely. The
  default is `webgpu`, which downloads a prebuilt `wgpu-native` asset at
  configure time. On a CPU-only server that download is pure cost and a network
  dependency you do not need. Use `webgpu` **only** if the server has a real
  graphics adapter and you intend to run GPU model variants.
- **`-DCOAST_WITH_CONSOLE=OFF`** — skips the embedded `clang-repl` C++ console,
  which needs LLVM/Clang development libraries. The daemon does not use it. (The
  console's `find_package(LLVM CONFIG)` is non-required, so leaving it ON only
  wastes configure time if LLVM is absent — but being explicit documents intent.
  `COAST_WITH_GUI=OFF` already forces the console off, since it *is* an ImGui
  panel; the flag is kept for clarity and for a GUI-on build.)

A separate build directory (`build-coastd`) keeps this server build from
colliding with any `build/` you use interactively.

There is one more flag you do **not** want here but should know about:

- **`-DCOAST_WITH_COASTD_DAEMON=OFF`** — builds everything except the `coastd`
  executable itself. Useful on a machine where the daemon cannot run, or where
  you do not want it: it refuses to start wherever `start_token_supported()` is
  false (anything but Linux and macOS), because without a recycle-safe process
  start token its duplicate-run guard is inoperative. A developer working only on
  the *client* or *console* side can set this and stop paying build time for the
  binary. The client and console libraries are still built, so the GUI panel and
  the REPL façade are unaffected; the tests that launch a real daemon are dropped
  from the suite (they key off `if(TARGET coastd)`). Default is ON everywhere —
  compiling it on an unsupported platform still catches portability breakage, and
  only *running* is refused.

### 3.2 Build

```sh
cmake --build build-coastd -j --target coastd
```

Targeting `coastd` specifically avoids building the host's `coast` executable,
which you do not need on the server — `coastd` builds its own per-SHA `coast`
binaries at runtime. Measured on one Linux box with the §3.1 flags: 34 s from a
cold build directory.

The binary lands in a **subdirectory** of the build tree, `build-coastd/coastd/coastd`
— `coastd` is its own CMake subproject (`add_subdirectory(coastd)`) and the project
sets no `RUNTIME_OUTPUT_DIRECTORY`, so its output stays beside its own CMake files.
Only the top-level `coast` executable sits at the build root. Verify:

```sh
./build-coastd/coastd/coastd --version     # -> coastd 0.1.0-dev
./build-coastd/coastd/coastd --help
```

### 3.3 Install the binary somewhere stable

The systemd unit will point at an absolute path, and that path must not move
under it. Copy the binary out of the build tree:

```sh
mkdir -p ~/.local/bin
install -m 755 build-coastd/coastd/coastd ~/.local/bin/coastd
```

Use `install`, not `cp`, here and on every upgrade (§10.2): `cp` writes into
the existing file and fails with `Text file busy` while the daemon is running
it.

Do this deliberately rather than pointing the service at `build-coastd/coastd/coastd`:
a later rebuild would then swap the binary underneath a running daemon.

---

## 4. Test the build before installing it

Do not install an unvalidated binary as a service. The test suite is opt-in.

### 4.1 Build the tests

```sh
cmake -B build-tests \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DCOAST_WITH_GUI=OFF \
      -DCOAST_GPU_BACKEND=none \
      -DCOAST_WITH_CONSOLE=OFF
cmake --build build-tests -j
```

Same flags as §3.1 plus `-DBUILD_TESTING=ON`, deliberately: you want to test the
configuration you will install, not a different one.

### 4.2 Run the coastd suite

```sh
cd build-tests
ctest --output-on-failure -R '^unit\.coastd\.'
```

**They need not run serially.** Not for port reasons: the cases that
launch a real `coastd` (`comms`, `console`, the `fault` matrix below, and
`unit.console.repl`) pass `--tcp-port 0 --udp-port 0` and read the ports the
kernel assigned from `<data-root>/coastd.ports` (`tests/unit/coastd/test_daemon.h`),
so they collide neither with each other, nor with a real daemon on the host, nor
with a second `ctest` run, nor with the host's own outbound connections. Nor for
timing: no coastd test depends on it — states are held by construction and waits
follow the daemon's heartbeat (`tests/unit/coastd/CMakeLists.txt`, "Serialization").

24 tests (`ctest -N`, 2026-11-08). They are the same 24 in
every build configuration — `coastd` does not depend on the GUI layer, so nothing
here is compiled out by `COAST_WITH_GUI=OFF`.

What this actually covers, roughly in increasing order of realism:

- `reconcile`, `job_json`, `job_store`, `config_load`, `atomic_write`,
  `protocol_guard` — the pure scheduling/recovery state machine, the durable
  store and its crash-safe write primitive, config layering, and the wire
  contract. No subprocesses, no sockets. These are the primary reliability lever:
  the whole policy is a pure function, so it is testable without a daemon.
- `child_cmd`, `line_tailer`, `proc_util`, `child_proc`, `supervisor` — command
  derivation (including the resume step budget), subprocess spawning, wall-clock
  deadlines, and the SIGTERM-grace-then-SIGKILL escalation.
- `builder` — worktree creation against this very repository.
- `reality_probe`, `log_sweep`, `machine_info`, `concurrency` — the uncached
  world-rescan, age-based log housekeeping (§10.4), the discovery/`machine_info`
  projection, and concurrent access to the shared structures.
- `core_lifecycle`, `child_exit`, `recovery` — full submit → resolve → build →
  spawn → exit → terminal flows against a fake `coast`, plus boot-time orphan
  detection and the crash-loop cap.
- `comms`, `console` — launch the **real** `coastd` binary and drive the wire
  protocol through the thin client.

Two further `coastd` ctest entries are deliberately **outside** the
`^unit\.coastd\.` filter because they are slow and environment-dependent:
`gate.coastd.real_coast` (§4.3) and `fault.coastd.matrix` — the {kill client,
kill daemon} × {building, queued, running} fault matrix plus the latency gates,
which waits out real builds, real runs and a real SIGTERM grace. Run the matrix
once on a new server if you want the recovery guarantees exercised end to end:

```sh
ctest --output-on-failure -R '^fault\.coastd\.'
```

If you want the whole COAST suite, drop the `-R` filter: **141 tests** in
exactly this configuration (`ctest -N`, 2026-11-08); the runtime is dominated by the
fault matrix and the A-tier mnist integration tests. That number is lower than the
176 a full desktop build reports, and the
difference is entirely expected and was checked by diffing the two `ctest -N`
listings: `COAST_GPU_BACKEND=none` removes 24 GPU tests (5 `gpu.smoke.*`, 14
GPU `integration.*` cases, 2 `ngm25_gpu_*`, 3 GPU unit tests) and `COAST_WITH_GUI=OFF`
removes 11 GUI-layer ones (7 `unit.gui_plot.*`, 1 `unit.gui_replay.*`, 3
`unit.console.*`), plus the `b_tier.simc_*`
cases compiled out inside the B-tier binary. Nothing is skipped at runtime;
those tests are not built at all, which is why you will *not* see the
environmental `_gpu` failures a GPU-less server would otherwise produce.

### 4.3 Optional: the real end-to-end gate

This runs a genuine mnist job *through* `coastd` and asserts the final sim state
is bit-identical to a direct `coast` run. It needs a built `coast` binary, the
datasets, `h5diff`, and `python3` — and it is a **registered ctest**, so its
outcome is visible rather than a silent green tick:

```sh
cmake --build build-tests -j --target coast
cd build-tests && ctest --output-on-failure -R '^gate\.coastd\.real_coast$'
```

It reports one of three distinguishable outcomes: `Passed`, `Failed`, or
`Skipped` (exit 77 — no dataset, no `h5diff`, or no `coast` binary). A *Skipped*
result means the gate never ran; do not read it as a pass. The driver script can
also be run by hand, which is what the ctest entry does for you:

```sh
tests/unit/coastd/run_real_gate.sh \
    "$PWD" \
    "$PWD/build-tests/coast" \
    "$PWD/build-tests/tests/unit/coastd/test_coastd_core_lifecycle"
```

It is worth running once on a new server: it validates the entire chain —
config preparation, child launch, checkpointing, and output placement — on the
actual hardware.

---

## 5. Choose the layout and prepare the roots

### 5.1 Data root

```sh
mkdir -p ~/.local/share/coastd
```

`--data-root` defaults to `$XDG_DATA_HOME/coastd`, else `~/.local/share/coastd`,
else `./.coastd`. Relying on the default is fine, but the service unit records
an explicit absolute path anyway, so pick it consciously.

The three roots default to `<data-root>/state`, `<data-root>/runs` and
`<data-root>/worktrees`, but each can be **moved independently** with a config-file
key — no symlink needed. To put the large `runs_root` on a scratch volume while the
small, backed-up `state_root` stays on the reliable one:

```json
{ "runs_root": "/mnt/scratch/coastd-runs" }
```

The literal token `<data-root>` is expanded in these values, so you can anchor a
moved root without repeating the prefix — e.g. `"worktrees_root": "<data-root>/wt"`.
Give absolute paths (or use the token); a relative value resolves against the
daemon's working directory, which under systemd is not where you expect. `data_root`
itself has no config key — the config file is discovered at `<data-root>/coastd.json`,
so it must be known before the file is read. See §6.2 for the full key list.

`coastd` creates everything it needs on first boot, including `git init` in
`state_root`; `runs_root` and `worktrees_root` are created lazily, the first time a
job runs or a SHA builds. You do not need to pre-create any subdirectory.

### 5.2 Repo root and ref freshness

Whatever you pass to `--repo` must be a git clone of the project. `--repo`
defaults to the daemon's working directory; you can instead set `"repo_root"` in
`coastd.json` (the `<data-root>` token expands there too), but `--repo` on the
command line always overrides that key. `--install-service` bakes the *effective*
repo — CLI or config — into the unit as an absolute `--repo`, so a service started
with no working directory still finds it. By default `coastd` keeps the clone
current itself: before resolving a submitted ref it runs
`git fetch --all --prune --tags` in that repo, then resolves.

Two properties of that fetch are worth understanding, because they are what make
a submitted ref name mean what you expect:

**It prefers the remote's branch over a same-named local one.** `git fetch`
updates *remote-tracking* refs (`origin/main`); it does **not** move your local
`main`. Since nothing ever checks anything out in `repo_root`, that local branch
is frozen at whatever the clone was created with — possibly months old. So for a
plain branch name, `coastd` resolves `<remote>/<ref>` first and only falls back
to the literal ref. Submitting `main` therefore runs the remote's current main,
not a stale local copy.

The preference applies **only** to plain branch-ish names. These always resolve
verbatim, because prefixing them would change their meaning:

| Ref you submit | Resolves as |
|---|---|
| `main`, `feature/x` | `origin/main`, `origin/feature/x` — falls back to the literal name if the remote has no such branch |
| `HEAD` | this repo's HEAD (never `origin/HEAD`, which is the remote's *default branch*) |
| `refs/heads/main` | the local branch, explicitly pinned |
| `v1.2`, any tag | the tag |
| a raw SHA | itself |
| `main~3`, `x^{}`, `@{u}` | verbatim revision syntax |

**A fetch failure is never fatal.** No network, no remote, or a transient
`cannot lock ref` race against a concurrent `git worktree add` all produce a
warning in the daemon log and resolution proceeds against the local clone. The
ref usually already exists there; turning a network blip into a terminally-failed
job would trade a freshness feature for a reliability regression. If the ref
genuinely is not present, you get the ordinary
`could not resolve ref '<name>'` error.

Fetches are throttled: at most one per `fetch_min_interval` (default 30 s) across
all concurrent resolutions, so a burst of fifty submissions produces one fetch
rather than fifty. A resolution that arrives while that fetch runs waits for it,
so in a burst for a commit the clone has not seen yet no job fails for having
looked before the fetch finished.

To turn all of this off and go back to purely local, local-first resolution —
for an air-gapped server, or one where you want to control exactly what code is
runnable by curating the clone by hand:

```sh
coastd --no-fetch ...        # or "fetch_enabled": false in coastd.json
```

With fetching disabled you are back to needing your own `git fetch` discipline:

```sh
# crontab -e, as the coast user
*/10 * * * * git -C $HOME/src/coast fetch --all --prune --quiet
```

Related knobs: `--git-remote <name>` (default: auto — `origin` if present, else
the sole remote; with several remotes and no `origin`, remote-first is skipped
entirely and resolution stays local), `--fetch-timeout <s>` (default 300 — the
fetch is network-bound, so it gets its own generous budget rather than sharing
the 60 s `git_op_timeout` used for quick local calls), and
`--fetch-min-interval <s>` (default 30; `0` = fetch on every resolve).

### 5.3 A note on disk

`coastd` does **not** monitor free disk space. Checkpoints and per-SHA worktrees
both grow without a hard bound. Idle worktrees are garbage-collected after
`worktree_cooloff` (default 6 h), but run directories are kept forever. Monitor
the volume holding `runs_root` externally, and prune old run directories on a
policy you decide.

### 5.4 Configs travel as content, not as paths

There is nothing to set up here — this section exists because the behaviour is
easy to assume wrongly, and assuming wrongly is expensive.

When a client submits a job it sends the **config file's bytes**, not its
location. The daemon never opens a client-named path. Consequences:

- **You do not stage configs on the server.** Submitting from a laptop against a
  remote daemon works with the file you have locally. No `scp` step, no shared
  mount, no convention about where configs live on the server.
- **The GUI can submit at all.** Its config editor holds an edited buffer that
  may not correspond to any saved file; there is no path it could have sent.
- **A same-named file on the server can never shadow yours.** This is the real
  reason for the design. If the client sent `~/work/configs/mnist.json` and the
  server happened to have a *different* file at that path — two developer boxes,
  near-certain — the job would have run the server's version, successfully, with
  no signal to anybody that the wrong config had been used. A wrong answer that
  looks right is worse than a failure, so the path form is not offered at all.

Two files land in the run directory as a result:

| File | What it is |
|---|---|
| `config.json` | what `coast` actually runs — your config with `snapshot_dir` rewritten to the job's absolute snapshot directory, re-serialized (so key order and indentation are the JSON library's, not yours) |
| `config.submitted.json` | exactly what you sent, byte for byte — the provenance record. Nothing reads it; it is there so that months later you can answer "what did I actually submit?" |

Size bounds: a config is capped at `comms_max_config_bytes` (4 MiB) and must fit
in one request line, `comms_max_request_bytes` (10 MiB, kept above the config cap
automatically — the daemon raises it and warns if you configure them the other
way round). Both limits produce a clear error reply, never a dropped connection.

An ad-hoc `nc` caller must inline the config as a JSON string, for example with
`jq`:

```sh
jq -cn --arg ref main --rawfile cfg ./configs/mnist.json \
   '{cmd:"submit", ref:$ref, config_text:$cfg, config_name:"mnist.json", steps_target:500000}' \
  | nc -q1 <server> 51900
```

`--rawfile` is what keeps the file's formatting intact; JSON string escaping is
lossless, so the bytes that arrive are the bytes you sent.

---

## 6. Configure the daemon

### 6.1 Precedence

Configuration is layered: **compiled-in defaults < config file < CLI flags.**
The config file is `--config <file>`, or, if that is not given,
`<data-root>/coastd.json` when it exists. An explicitly named file that fails to
load is fatal; an auto-discovered one that fails is a warning and boot continues.

Unknown keys are ignored, and a wrong-typed or out-of-range value is skipped with
a warning rather than being fatal — so a slightly stale config file still boots.
The ranges mirror the CLI's bounds, which exist to stop values that would wedge
the daemon (e.g. `scan_period_ms <= 0` would busy-spin the core loop at 100 % CPU).

Prefer the config file over flags for anything you might tune later: changing it
does not require rewriting the service unit.

### 6.2 A starting config

`~/.local/share/coastd/coastd.json`:

```json
{
  "slots": 2,
  "checkpoint_interval": 500000,
  "build_jobs": 8,
  "build_cmake_args": [
    "-DCOAST_WITH_GUI=OFF",
    "-DCOAST_GPU_BACKEND=none",
    "-DCOAST_WITH_CONSOLE=OFF"
  ],
  "fetch_enabled": true,
  "fetch_min_interval_seconds": 30,
  "fetch_timeout_seconds": 300,
  "worktree_cooloff_seconds": 21600,
  "term_grace_seconds": 30,
  "build_timeout_seconds": 3600,
  "max_launch_failures": 5,
  "max_build_failures": 3
}
```

Key by key:

- **`slots`** (default 1) — how many jobs run concurrently. This is the single
  most important knob. COAST simulations are CPU-bound and TBB-parallel
  internally, so more slots is *not* automatically more throughput; oversubscribing
  cores makes every job slower. Start at 1–2 and measure before raising it.
- **`checkpoint_interval`** (default 500000) — passed to the child as
  `coast --checkpoint N`. This sets how much work a crash can lose *and* how
  much a resume must redo. It also interacts with a guard: a job that is
  repeatedly interrupted after running real steps but before ever reaching a
  checkpoint makes zero durable progress and would resume from the same base
  forever. After `max_launch_failures` such launches it is failed with a distinct
  reason naming this interval. Set it well below your typical `steps_target`.
- **`build_jobs`** (default 0 = half the detected cores) — `cmake --build -j N`.
  The default deliberately leaves half the machine to running simulations. Raise
  it only if builds are your bottleneck and no jobs are running.
- **`build_cmake_args`** — extra `-D` flags for each per-SHA configure. `coastd`
  always passes `-DCMAKE_BUILD_TYPE=Release`; everything else is up to you.
  **Match these to §3.1.** The host build's flags do not propagate — each SHA is
  configured from scratch in its own worktree, with only what is listed here. Two
  consequences if you leave it empty or partial: a missing
  `-DCOAST_GPU_BACKEND=none` makes every job's build download and link a GPU
  backend the server cannot use, and a missing `-DCOAST_WITH_GUI=OFF` makes it
  search for SDL2 and OpenGL — which on a server that (correctly) has neither
  installed is not a slower build but a **failed** one. It dies at the *configure*
  step (`find_package` for module SDL2 called with REQUIRED), so the SHA never
  builds and every job on it is failed after `max_build_failures`. The error is
  the first thing in that SHA's `coastd-build.log`. The equivalent CLI
  flag is `--cmake-arg` (repeatable), but prefer the config file — you will
  revisit this setting.
- **`fetch_enabled`** (default `true`) — fetch before resolving a ref, and prefer
  the remote's branch over a same-named local one. See §5.2 for the full
  semantics; `git_remote`, `fetch_timeout_seconds` and
  `fetch_min_interval_seconds` tune it.
- **`worktree_cooloff_seconds`** (default 21600 = 6 h) — how long an idle
  worktree survives before GC. Lower it if disk is tight; raise it if you
  frequently re-submit the same SHA and want to skip rebuilds.
- **`term_grace_seconds`** (default 10) — how long a SIGTERM'd child gets to
  write its final `--save-on-term` checkpoint before SIGKILL. Applies to both job
  cancellation and daemon shutdown. **Raise this for large models** whose final
  checkpoint write takes longer than 10 s; otherwise a shutdown loses that work.
  Keep systemd's `TimeoutStopSec` above it (the installer handles this — §7.1).
- **`build_timeout_seconds`** (default 3600) — a *bound*, not a hang detector.
  A slow build (cold submodule clone on a poor link) is externally
  indistinguishable from a hung one, so `coastd` caps total duration and treats
  the overrun as a build failure rather than letting one build park a worker
  thread forever. With `build_concurrency` at 1, an unbounded build would block
  *all* future builds for the daemon's lifetime — this is the guard against that.
- **`max_launch_failures`** (default 5) — consecutive relaunches with no forward
  progress before the job is moved to `failed/`. Relaunches back off
  exponentially: `relaunch_backoff_base_seconds * 2^(failures-1)`, capped at
  `relaunch_backoff_max_seconds` (defaults 2 s → 60 s), so a transient fault
  (fork `EAGAIN` under memory pressure, a briefly-unavailable dataset mount) does
  not burn all five strikes in a few seconds. Widen the base/max on a box prone to
  such transients; the daemon warns if you set the base above the max.
- **`max_build_failures`** (default 3) — consecutive build failures on a SHA
  before jobs on it are failed.
- **`min_free_disk_bytes`** (default 2 GiB) — the disk-pressure floor. Each scan
  measures free space on `runs_root`, `worktrees_root` and `state_root` (usually
  one filesystem) and takes the minimum. Below the floor the daemon **stops
  starting new work**: no builds, no job launches. Running children keep going and
  worktree GC keeps running — GC is the only thing that gives space back, and
  killing a child would discard compute without freeing anything its last
  checkpoint had not already written. Queued jobs simply wait, with no strike and
  no failure; they start as soon as space returns. The onset and the recovery are
  each logged once (not once per second). Sizing: a built worktree is ~238 MB, so
  2 GiB leaves room for a build plus checkpoints and time to notice. Set `0` to
  disable the check entirely — appropriate when the roots live on storage
  `statvfs` reports uselessly (a quota-managed share, some network mounts).

- **`log_compress_after_seconds`** (default 604800 = 7 d), **`log_delete_after_seconds`**
  (default 7776000 = 90 d), **`log_sweep_period_seconds`** (default 21600 = 6 h) —
  age-based log housekeeping, the only thing `coastd` deletes on its own (§10.4).
  Every `log_sweep_period`, run logs and per-SHA build logs idle longer than
  `log_compress_after` are gzipped, and `.gz` files older than `log_delete_after`
  are removed. Age is judged by **mtime**, which is what keeps the sweep off a
  live file: a log being written has a fresh mtime, so a threshold of days can
  never select the file the supervisor is tailing. Set the delete threshold long
  if a run's logs are part of your provenance record.
- **`build_log_max_bytes`** (default 64 MiB) — the per-SHA `coastd-build.log` is
  appended across every build attempt on that SHA; past this size it is restarted
  rather than growing without bound.

**Directory roots** (`state_root`, `runs_root`, `worktrees_root`, `repo_root`)
are also config keys — see §5.1 for moving them onto separate volumes and the
`<data-root>` token. `repo_root` is additionally settable with `--repo`, which
wins over the key.

- **`dataset_base`** (default *unset*) — the machine's shared dataset directory.
  When set, `coastd` creates a `data` symlink to it inside every job's run dir, so
  a submitted config's conventional relative `data/<set>/...` path resolves from
  the child's working directory (§2.1) with no per-config edits. It is a path key
  like the roots (the `<data-root>` token expands, so `"dataset_base":
  "<data-root>/data"` is valid) and is made absolute before the link is created —
  a relative value would point *inside* the run dir, not at your data. Unset
  disables it: no link is created and configs must use absolute dataset paths. The
  link is refreshed on each submit and never clobbers a real `data` file or dir.

Config-file-only knobs with no CLI flag: `build_concurrency`,
`max_build_failures`, `audit_git_timeout_seconds`,
`relaunch_backoff_base_seconds`, `relaunch_backoff_max_seconds`,
`min_free_disk_bytes`, `log_compress_after_seconds`, `log_delete_after_seconds`,
`log_sweep_period_seconds`, `build_log_max_bytes`, and the
client-edge bounds `comms_max_connections` (128), `comms_max_request_bytes`
(10 MiB), `comms_max_config_bytes` (4 MiB), `comms_first_request_timeout_seconds`,
`comms_line_assembly_timeout_seconds`, `comms_send_timeout_seconds`. The comms
bounds default to values suited to a trusted lab LAN; raise the timeouts for
slow links, raise `max_connections` for many clients.

### 6.3 Ports

- **TCP 51900** — the command port (JSONL request/reply).
- **UDP 51901** — discovery: a client broadcasts a probe, the daemon replies with
  its `machine_info`.

Change with `tcp_port` / `udp_port` in the config, or `--tcp-port` / `--udp-port`.
A TCP bind failure is fatal at startup (you would otherwise have a daemon nobody
can reach); a UDP bind failure is logged and discovery is simply disabled.

Once both ports are open the daemon writes `<data-root>/coastd.ports`, e.g.
`{"pid":1234,"tcp_port":51900,"udp_port":51901}` (`udp_port` 0 = no discovery
responder), and removes it on an orderly shutdown. A file left by a crashed daemon
is removed at the next start; the `pid` lets a reader tell a leftover from the
live daemon's file.

`--tcp-port 0` / `--udp-port 0` let the kernel pick free ports; `coastd.ports`,
`machine_info` and discovery replies then report the ports actually bound. This is
**CLI-only** — the config file keeps `1..65535` — and meant for tests and ad-hoc
second daemons: a service whose port changes on every start is unreachable through
any `hosts.json` that names it. The test suite uses it so that no test port can
collide with anything (a fixed port inside the ephemeral range, 32768–60999 on
Linux, is blocked whenever an outbound connection on the host has just used it as
its source port — `SO_REUSEADDR` does not help against a client socket).

**Discovery knobs.** The UDP responder answers an unauthenticated probe, and a UDP
source address is trivially forged — so its reply can be aimed at a host that never
contacted this daemon. Four settings bound that:

| Key | Default | Meaning |
|---|---|---|
| `discovery_enabled` | `true` | Turn the responder off entirely. The TCP `machine_info` command still answers, so clients configured with an address (`hosts.json`, `$COASTD_HOSTS`) lose nothing — only LAN auto-discovery goes away. |
| `discovery_bind` | *(empty = all interfaces)* | Bind one interface, e.g. `10.0.0.7`. An unparseable value **disables** discovery rather than falling back to all interfaces. |
| `discovery_max_replies_per_sec` | `20` | Global reply cap: the amplification bound. |
| `discovery_max_replies_per_source_per_sec` | `2` | Per-source cap: what one (possibly forged, i.e. victim) address can be made to receive. |

The reply deliberately **omits** `runs_root` and `repo_root`; the TCP
`machine_info` command still returns them. Turn `discovery_enabled` off on any host
whose UDP port is reachable by untrusted peers.

### 6.4 Output locations — nothing to configure, but worth knowing

At boot `coastd` writes its own COAST *app config* to
`<data-root>/coast_app_config.json` with `config_base = "<config_dir>/"`, and
passes it to every child via `--app-config`. Because each child is launched with
`-c <run_dir>/config.json`, `<config_dir>` expands to that job's run directory,
so every app-config-governed output lands flatly inside it.

Being an explicit `--app-config`, it has the highest discovery precedence and
therefore **shadows** any inherited `$COAST_APP_CONFIG` or stray `./coast.json`
that would otherwise silently redirect a job's outputs out of its run directory.
You do not need to configure this, but you should not fight it: do not set
`COAST_APP_CONFIG` in the service environment expecting it to take effect.

---

## 7. Install as a systemd user service

### 7.1 The built-in installer

```sh
~/.local/bin/coastd --install-service \
    --data-root ~/.local/share/coastd \
    --repo ~/src/coast
```

This writes `~/.config/systemd/user/coastd.service` pointing at the *currently
running* binary (resolved via `/proc/self/exe`) and the roots you passed, then
prints the enable steps. It computes `TimeoutStopSec` from your resolved config
as `build_timeout + term_grace + 30`, because the longest graceful shutdown is
dominated by an in-flight build step, not by the child's checkpoint grace.

Then:

```sh
systemctl --user daemon-reload
systemctl --user enable --now coastd
sudo loginctl enable-linger "$USER"
```

- **Why a *user* service, not a system one** — it inherits your git credentials
  (needed to build refs from a private remote) and your environment, without a
  root-owned unit.
- **Why `enable-linger`** — without it a user service only runs while that user
  has an active login session. Lingering lets it start at boot on a headless box.
  This is the step people forget; skip it and the daemon quietly disappears after
  you log out.

### 7.2 Do not override `KillMode`

The unit keeps systemd's stock `KillMode=control-group`. On stop, restart, or
crash, systemd reaps the entire cgroup, so **every child `coast` process dies
with the daemon.**

That is not incidental — it *is* the recovery contract. "No live orphans, ever"
is what makes it safe for each restart to relaunch interrupted jobs from their
last checkpoint: two children can never end up writing the same run directory.
Setting `KillMode=process` or `mixed` would leave orphans and break this.

There is a second, independent guard: at boot `coastd` checks each active job's
recorded pid against a start-time token read from `/proc/<pid>/stat` (on macOS,
from `sysctl(KERN_PROC_PID)`), and kills a genuine survivor before relaunching.
That is the fallback for environments without cgroup reaping (see Appendix A).

### 7.3 Verify

```sh
systemctl --user status coastd
journalctl --user -u coastd -f
```

You should see a startup line naming the version, data root, repo root and slot
count, then `booted: N active job(s)`.

A useful sanity check: the daemon takes an exclusive `flock` on
`<data-root>/coastd.lock` for its whole lifetime. Two daemons on the same roots
would both load the store and spawn children into the same run directories, so a
second instance refuses to start with a clear message. If you see that message
unexpectedly, you have a stray instance — find it before doing anything else.

---

## 8. Verify end to end

### 8.1 Talk to the daemon

The wire protocol is JSONL: one minified JSON object per line, `\n`-terminated.
Every request is `{"cmd":"<name>", ...}` and every reply is
`{"ok":true, ...}` or `{"ok":false,"error":"..."}`. That makes it speakable with
`nc`, which is the fastest way to confirm a live install:

```sh
printf '{"cmd":"machine_info"}\n' | nc -q1 localhost 51900
```

Expect something like:

```json
{"ok":true,"proto_version":1,"hostname":"box","os":"Linux",
 "coastd_version":"0.1.0-dev","slots":2,
 "runs_root":"/home/coast/.local/share/coastd/runs",
 "repo_root":"/home/coast/src/coast","tcp_port":51900,"data_link_ok":true}
```

`proto_version` is the wire-protocol version this daemon speaks (also carried in
the discovery datagram), so a client can check compatibility *before* submitting
work. An absent field means a pre-versioning daemon.

### 8.2 Submit a real job

The request must be **one line** — the daemon reads `\n`-delimited records, so do
not pretty-print it. Because the config travels as *content* (§5.4), the simplest
way to build the request by hand is to let `jq` inline the file:

```sh
jq -cn --arg ref main --rawfile cfg /home/coast/configs/mnist_abs.json \
   '{cmd:"submit", ref:$ref, config_text:$cfg, config_name:"mnist_abs.json",
     steps_target:1000000, label:"smoke", message:"install verification"}' \
  | nc -q1 localhost 51900
```

Three things to get right here:

- **`config_text` carries the config's content, not a path** (§5.4). The file is
  read on the machine you run this from — it does not need to exist on the
  server. There is deliberately no path field; a `config` key is rejected with a
  migration message.
- **Make datasets reachable** (§2.1): either set `dataset_base` in `coastd.json`
  so a `data` symlink is placed in the run dir (then the stock relative
  `data/<set>/...` paths work), or use absolute dataset paths inside the config.
  The child runs with its working directory set to the run directory.
- **`steps_target` of 0 means unbounded.** A job submitted without a step target
  runs forever. This is intentional, and it is the single easiest way to burn a
  machine's CPU for days by accident. Always set it unless you truly mean
  "until cancelled". Such a job is launched as `coast --unlimited …` rather than
  with `-n` (there is no "0 means forever" sentinel any more), and it still stops
  cleanly on cancel or daemon shutdown: `--save-on-term` is always on the child's
  command line, so the SIGTERM writes a final checkpoint, the state file and the
  run manifest. You will see both flags if you inspect the child with `ps`.

The daemon rejects a config that names a path resolving inside `state_root`,
`worktrees_root`, or `runs_root` — that guard stops a config from clobbering
another job's artifacts or the daemon's own state. `repo_root` is deliberately
*not* forbidden, since configs legitimately reference datasets that live there.

The reply is `{"ok":true,"id":"<job-id>"}`.

### 8.3 Watch it

```sh
printf '{"cmd":"list"}\n' | nc -q1 localhost 51900
```

`status` walks `queued → resolving → building → ready → running → done`. The
first job on a fresh SHA spends a long time in `building` — that is a full COAST
build. Follow it:

The `steps` shown for a **running** job is the *live* step, parsed from the
child's progress output as it runs — so it advances continuously, not only at
checkpoint boundaries. This is display-only; the *durable, resumable* point is
still the newest checkpoint (`steps_done`), which is what a restart resumes from
(§8.4). A job that is running but has not yet reached its first checkpoint shows a
rising `steps` with a durable checkpoint of 0 — that is expected, not a stall.

```sh
tail -f ~/.local/share/coastd/worktrees/<sha>/coastd-build.log
```

Once running, the job's own output is under
`~/.local/share/coastd/runs/<job-id>/`:

```sh
ls -R ~/.local/share/coastd/runs/<job-id>/
tail -f ~/.local/share/coastd/runs/<job-id>/logs/run_0_0_0.log
```

There is one log file **per launch attempt**, named
`run_<steps_done>_<launch_failures>_<no_checkpoint_launches>.log`, so a
crash-loop leaves every attempt's log intact rather than overwriting.

### 8.4 Test recovery

This is the property worth verifying explicitly on a new install, because it is
the whole point of the daemon:

```sh
systemctl --user restart coastd
printf '{"cmd":"list"}\n' | nc -q1 localhost 51900
```

The job should come back as `running` with `steps_done` at or after its last
checkpoint, and a new log file in `logs/`. If `steps_done` reset to 0, the child
never checkpointed — check that `checkpoint_interval` is below the job's step
target.

Then verify cancellation:

```sh
printf '{"cmd":"cancel","id":"<job-id>"}\n' | nc -q1 localhost 51900
```

The child gets SIGTERM, `term_grace` seconds to write a final checkpoint, then
SIGKILL, and the job's file is renamed into `cancelled/`.

### 8.5 Other commands

| Command | Request | Effect |
|---|---|---|
| `submit` | `ref`, `config_text` required; `config_name`, `steps_target`, `label`, `message`, `priority` optional | queue a job |
| `list` | — | `{"ok":true,"jobs":[…]}` for all jobs, all buckets |
| `cancel` | `id` | SIGTERM → grace → SIGKILL, move to `cancelled/`. Accepted immediately; the job then reports `cancel_requested:true` while still `running` until the child stops |
| `reorder` | `id`, `pos` | move a *pending* job to 0-based position `pos` |
| `machine_info` | — | static self-description, plus `data_link_ok` |

Note on `reorder`: it renumbers the whole pending queue positionally, which
**overwrites any explicit `priority` the jobs were submitted with**. That is the
deliberate price of expressing a drag-to-position with a single integer.

For a richer client than `nc`, link `coastd_client` (the thin protocol library —
sockets + json, no simulator) or use the `coast::coastd` console façade from the
COAST REPL. Clients can find daemons by name via `~/.config/coastd/hosts.json`
(`[{"name":"gpu-box","host":"10.0.0.5","port":51900}, …]`; override the location
with `$COASTD_HOSTS`), or by UDP broadcast discovery.

The console façade mirrors the `simc` workflow: **load a base config, edit it in
the console, then submit the edited object** — no need to stage a file per
variant. It shares the same editable `run_config` type as `simc`, so a config you
built for a local run submits to a daemon unchanged:

```cpp
auto s = coastd::session::connect("captncrunch");   // or connect_to("gpu-box")
auto c = coastd::config("configs/mnist.json");       // editable run_config
c.set("/ngm2/dendrites/default/default_primary_learning_rate", 0.03);
auto id = s.submit("main", c, "sweep-lr", "", 1'000'000);  // sends c.dump() as content

// A sweep is just a loop over edited copies — nothing staged on the server:
for (double lr : {0.01, 0.02, 0.04}) {
    auto v = coastd::config("configs/mnist.json");
    v.set("/ngm2/dendrites/default/default_primary_learning_rate", lr);
    s.submit("main", v, "lr", "", 1'000'000);
}
```

`submit(ref, run_config, …)` sends the edited config's current bytes as
`config_text`; the daemon assigns the run directory and rewrites `snapshot_dir`
itself, so the config's `run_dir()`/`seed()` staging hints (meaningful to `simc`)
are ignored here. A `submit(ref, path, …)` overload remains for firing off a
finished config file straight from disk.

---

## 9. Security — read this before exposing the port

**`coastd` binds `INADDR_ANY` and has no authentication whatsoever.**

Submitting a job means asking the daemon to check out an arbitrary git ref,
build it, and execute the result. **Anyone who can reach TCP 51900 has remote
code execution as the daemon user.** This exposure is an accepted trade-off for
the intended trusted-lab-network deployment; it is not acceptable on any network
reachable by untrusted peers.

Mitigations, in order of preference:

1. **Firewall the port to known clients.** The minimum acceptable step.

   ```sh
   sudo ufw allow from 10.0.0.0/24 to any port 51900 proto tcp
   sudo ufw allow from 10.0.0.0/24 to any port 51901 proto udp
   ```

2. **Bind to loopback and reach it over SSH.** There is no loopback-bind option
   today, so this means firewalling everything except `lo` and having clients
   forward the port:

   ```sh
   ssh -L 51900:localhost:51900 coast@server
   ```

3. **Run it on an isolated network segment or inside a VPN.**

### 9.1 Code execution is per-submit; persistence is what we harden against

The RCE above is by design — you asked the daemon to run that commit. What is
*not* acceptable is one submitted job's code outliving its own job and running
again inside later, unrelated ones. The build/run child executes as the daemon
user, so without care it can leave something behind that the daemon itself
re-executes:

- **git hooks.** A child can write `<repo_root>/.git/hooks/post-checkout`. The
  daemon runs `git worktree add` for every new SHA, and that fires the hook — as
  the daemon, inside someone else's job, for as long as `.git` stays dirty.
- **git config.** A child can write `<repo_root>/.git/config` or the daemon
  user's `~/.gitconfig`. Keys like `core.sshCommand` and `core.fsmonitor` hold
  commands that git executes on later invocations.

Every git call the daemon makes therefore goes through one hardened seam
(`coastd::run_git`), which pins `core.hooksPath=/dev/null` and `core.fsmonitor=`
via `-c` (command-line `-c` outranks every config file, so relocating
`hooksPath` in local config does not dodge it), and the spawn env pins
`GIT_CONFIG_GLOBAL`/`GIT_CONFIG_SYSTEM` to `/dev/null`. `unit.coastd.proc_util`
tests this against real git, including a negative control that confirms an
unhardened call *does* fire the hook.

**Residual — and it is broader than the two flags close.** An adversarial review
of this hardening demonstrated two further working implants, both from
**repo-local** config, needing no remote and no network:

- **A `filter` (smudge) driver.** `[filter "x"] smudge = <cmd>` in
  `<repo_root>/.git/config` plus `* filter=x` in `<repo_root>/.git/info/attributes`
  executes `<cmd>` during `git worktree add` — for any future SHA, since
  `.git/info/attributes` is repo-scoped and independent of that SHA's tree.
  Measured firing against the daemon's exact hardened argv.
- **`submodule.<name>.update = !<cmd>`.** Git ignores `!cmd` from `.gitmodules`
  but honours it from `.git/config`, which the job can write; it fires during
  `git submodule update --init --recursive`. Also measured.

Note `core.attributesFile=/dev/null` does **not** help — that setting governs the
*user-level* attributes file, not repo-scoped `.git/info/attributes` (verified).

So the honest statement is: **local git config is an execution surface, and no
combination of flags closes it.** `core.sshCommand` and `url.*.insteadOf` are two
more examples, not the list. Closing this needs the build to run under a uid that
cannot write `<repo_root>/.git` — a deployment choice, not a flag. If you want the
stronger boundary, give the daemon a read-only clone that a separate privileged
path fetches into.

**Operational consequences of the config pinning.** Three, all with remedies:

1. A *global* git credential helper no longer applies, so `git fetch` against an
   **https** remote needing auth fails — non-fatally; the builder warns and
   resolves refs against the local clone, which may then be stale. Use an ssh
   remote (keys come from `~/.ssh`, not git config) or put credentials in the
   repo-local config.
2. Global `http.proxy` / `http.sslCAInfo` no longer apply, which also affects the
   fresh repos cmake's FetchContent clones during a build — those have no local
   config to fall back on. **Remedy: set them in the daemon's environment
   instead** (`http_proxy`, `https_proxy`, `GIT_SSL_CAINFO`), e.g. via
   `Environment=` in the systemd unit. Only `GIT_CONFIG_*` is pinned; these are
   untouched.
3. Because both protected config scopes are pinned, `safe.directory` cannot be set
   from a config file — git honours that key only in system, global, or
   command-line scope, never repo-local. The daemon therefore passes
   `safe.directory=*` itself on every git invocation. Without it, a `repo_root`
   owned by a **different uid** makes every `git rev-parse` fail with "dubious
   ownership", every ref resolution fail, and every submitted job be filed to
   `failed/` immediately and permanently — which would have broken exactly the
   read-only-clone deployment recommended above. It is permissive by intent: the
   ownership check exists to stop you running code from a repo you do not control,
   and this daemon builds and runs such a repo by design.

Additionally:

- Run as an unprivileged, dedicated user (§1.3). A compromise then costs you that
  user's data and git credentials — bad, but bounded.
- The daemon does defend against *accidental* and *malformed* input: request and
  config size caps, a JSON nesting-depth pre-scan (a deeply nested line would
  otherwise overflow the parser's stack), connection caps, and per-phase
  timeouts against slowloris-style stalls. None of that is a substitute for
  network access control.

---

## 10. Operating it

### 10.1 Logs

| What | Where |
|---|---|
| Daemon | `journalctl --user -u coastd` |
| Per-SHA build | `<worktrees_root>/<sha>/coastd-build.log` |
| Per-job, per-launch | `<runs_root>/<id>/logs/run_*.log` (older ones are `.log.gz` — §10.4) |
| Job state history | `git -C <state_root> log` |

Raise daemon verbosity with `-v` in `ExecStart` (debug level) or quiet it with
`-q` (warnings only).

### 10.2 Upgrading

```sh
cd ~/src/coast && git pull --recurse-submodules
cmake --build build-coastd -j --target coastd
install -m 755 build-coastd/coastd/coastd ~/.local/bin/coastd   # replaces the installed binary
systemctl --user restart coastd
```

Do not use `cp` for the third step. The daemon is running the installed
binary, and Linux refuses to open a running executable for writing, so
`cp build-coastd/coastd/coastd ~/.local/bin/coastd` fails with
`cp: cannot create regular file '…/coastd': Text file busy` and installs
nothing. `install` writes the new binary as a new file under the same name
instead (GNU coreutils removes the old name first); the running daemon keeps
executing the old image until the restart replaces it. Copying to
`~/.local/bin/coastd.new` and `mv`-ing it over `coastd` works for the same
reason.

The restart is safe by construction: systemd reaps the children, and the boot
path relaunches every interrupted job from its newest checkpoint. Jobs already
running keep their frozen SHA, so upgrading the daemon does not change what a
running job is computing.

If the upgrade changes the job-file schema, files with a `schema_version` this
daemon cannot interpret are moved to `quarantine/` **untouched** — never
mangled, never able to crash-loop the queue. Check `quarantine/` after any
upgrade that touches the store.

### 10.3 Reading the state directly

The store is plain JSON on a filesystem; you can inspect it without the daemon:

```sh
ls ~/.local/share/coastd/state/jobs/*/
jq . ~/.local/share/coastd/state/jobs/active/*.json
```

Do not *edit* those files while the daemon is running — it is the single writer,
and your write will be overwritten or lost.

### 10.4 Garbage collection

Worktrees idle beyond `worktree_cooloff` are removed automatically (`git worktree
remove` with a raw-sweep fallback), and stale `*.tmp` files are swept at boot.

**Logs age out on their own.** Every `log_sweep_period` (6 h) the daemon walks
`runs_root` and `worktrees_root` and applies the standard Linux shape: run logs
and per-SHA build logs untouched for `log_compress_after` (7 d) are gzipped via
`gzip(1)`, and `.gz` files older than `log_delete_after` (90 d) are deleted.
Selection is by mtime, so a file currently being written is never a candidate.
Every step is best-effort: a log that cannot be compressed is reported once and
skipped, never retried into a loop and never able to fail a job. Raise the
thresholds (§6.2) if run logs are part of your provenance record.

Everything *else* in a run directory — checkpoints, `final_state.h5`,
`coast_runtime.json`, the configs — is **never** removed automatically. Prune them yourself on
whatever retention policy you choose; each is self-contained, so deleting one is
safe as long as its job is terminal.

---

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `another coastd is already running against <root>` | The lockfile is held. Find the other instance (`pgrep -af coastd`) — do not delete the lock. |
| `could not resolve ref 'X'` | Neither `<remote>/X` nor `X` exists after the fetch. Check the name; if you run `--no-fetch`, fetch the clone yourself (§5.2). |
| `git fetch … failed — resolving against the local clone, which may be stale` | Network, credentials, or a lock race. Benign once; persistent means the daemon user cannot reach the remote — check its SSH keys / credential helper. |
| A job ran code you did not expect | Ref-resolution precedence (§5.2). A plain branch name resolves to `<remote>/<name>`; pin exactly with `refs/heads/<name>` or a raw SHA. |
| Job stuck in `building`, then `failed` | Read `<worktrees_root>/<sha>/coastd-build.log`. Usual causes: missing dev packages, submodule fetch failure, or a `build_cmake_args` mismatch. |
| Build times out | A cold submodule clone exceeded `build_timeout_seconds`. Each build worktree clones its submodules from their URLs (§1.2). Raise the timeout, or point the submodule URLs at local mirrors. |
| `coastd-build.log` shows `fatal: transport 'file' not allowed` | The submodule URLs point at local mirrors, and the daemon's environment does not set `protocol.file.allow=always`. Set it there, as described in §1.2. |
| Job fails with a dataset path error | The config used relative dataset paths with no `dataset_base` set. Either set `dataset_base` in `coastd.json` (a `data` symlink is then placed in each run dir) or use absolute paths. Children run with cwd = run dir (§2.1, §6.2). |
| Job failed after several relaunches | Crash-loop cap (`max_launch_failures`). The child is dying with no forward progress — read `logs/run_*.log`, oldest first. |
| Job failed with a "no checkpoint" reason | The child ran real steps but never checkpointed. `checkpoint_interval` is larger than the job will ever run — lower it. |
| Queue is full of `queued` jobs and nothing starts; log says `DISK PRESSURE` | Free space fell below `min_free_disk_bytes` (§6.2). No build or launch starts until it recovers; running jobs are untouched. Free space (old run dirs under `runs_root` are usually the bulk) or lower the floor. The daemon logs a `disk pressure cleared` line when it resumes. |
| Daemon vanishes after logout | `loginctl enable-linger` was not run (§7.1). |
| `coastd: this platform provides no process start-token source` | You are starting the daemon on a platform other than Linux or macOS. It has no recycle-safe start token there, so it refuses to run (Appendix A); run the daemon on Linux or macOS and point your clients at it. |
| `bind port 51900` fatal at startup | Port in use. Change `tcp_port`, or find the squatter. |
| Discovery does not work, daemon otherwise fine | UDP 51901 bind failed (logged as a warning) or the broadcast is blocked. Use the fixed-list `hosts.json` instead — it is the deterministic path; discovery is only a convenience. |
| Jobs land somewhere unexpected | Something is overriding output locations. Check that no `COAST_APP_CONFIG` is set in the service environment (§6.4). |

---

## Appendix A — macOS

**The daemon runs on macOS when you start it by hand. It is not a macOS service.**
Until 2026-11-03 it refused to start there (ADR-16 in `docs/content/coastd/`);
ADR-17 lifted that.

The reason it used to refuse is one specific guarantee. launchd does not reap a
process group the way systemd's cgroup kill does, so "no live orphans after a
crash" is not automatic on macOS. The fallback is the boot-time pid
duplicate-guard, which identifies a survivor by its process start time. On Linux
that comes from `/proc/<pid>/stat`, which macOS does not have. Without it the
guard would read every orphan as *dead*, and the daemon would relaunch a job
**alongside its still-running child**, both writing the same run directory. The
daemon now reads the start time on macOS from
`sysctl({CTL_KERN, KERN_PROC, KERN_PROC_PID, pid})`, so the guard works there as
it does on Linux run outside systemd. On any platform with neither source the
daemon still refuses to start, with no override flag.

Run it in a terminal, with the same flags as on Linux:

```sh
cmake --preset headless && cmake --build build-headless -j --target coastd
./build-headless/coastd/coastd --data-root <dir> --repo <checkout>
```

Stop it with Ctrl-C or SIGTERM.

Still out of scope on macOS:

- **`--install-service`**: it writes a systemd user unit, and exits 1 on macOS.
- **Unattended operation under launchd.** The template at
  `coastd/service/com.coast.coastd.plist` has never been run. Start at boot,
  restart on failure and git credentials for a private remote are unverified
  there.

---

## Appendix B — quick reference

```sh
# Build
cmake -B build-coastd -DCMAKE_BUILD_TYPE=Release -DCOAST_WITH_GUI=OFF \
      -DCOAST_GPU_BACKEND=none -DCOAST_WITH_CONSOLE=OFF
cmake --build build-coastd -j --target coastd

# Test
cmake -B build-tests -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DCOAST_WITH_GUI=OFF -DCOAST_GPU_BACKEND=none -DCOAST_WITH_CONSOLE=OFF
cmake --build build-tests -j
cd build-tests && ctest --output-on-failure -R '^unit\.coastd\.'   # serial — no -j

# Install
mkdir -p ~/.local/bin && install -m 755 build-coastd/coastd/coastd ~/.local/bin/coastd
~/.local/bin/coastd --install-service --data-root ~/.local/share/coastd --repo ~/src/coast
systemctl --user daemon-reload && systemctl --user enable --now coastd
sudo loginctl enable-linger "$USER"

# Operate
systemctl --user status coastd
journalctl --user -u coastd -f
printf '{"cmd":"list"}\n' | nc -q1 localhost 51900
```

Layout recap:

```
<data-root>/
├── coastd.json                 daemon config (optional; auto-discovered)
├── coastd.lock                 single-instance flock
├── coast_app_config.json       written at boot; pins each job's output locations
├── state/                      git-versioned job files (BACK THIS UP)
│   └── jobs/{active,done,failed,cancelled,quarantine}/<id>.json
├── runs/<id>/                  per-job artifacts
│   ├── config.json  snapshots/  logs/  final_state.h5  coast_runtime.json
└── worktrees/<sha>/            per-SHA source + build (regenerable)
    ├── build/coast             the binary jobs on this SHA run
    ├── .coastd-ready           build-completion sentinel
    └── coastd-build.log
```
