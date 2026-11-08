+++
title = "Changelog"
weight = 70
+++

# Documentation Changelog

A log of substantive documentation changes, newest first.

## 2026-11-08 — `machine_info` no longer reports `core_passes` (#282)

The daemon's heartbeat `machine_summary::core_passes` and `core::passes()` are
gone. No test waits on it since the tests wait on a job's own state, and the
latency bench now does the same (`rig::fingerprint`). A client that still reads
the field gets 0, the value it always had for discovery replies and older
daemons. See [Protocol]({{% relref "/coastd/protocol" %}}).

## 2026-11-08 — Downloaded dependencies come from a machine-wide cache (#282)

Every new build tree downloaded the wgpu-native zip from GitHub again: a new
branch in CI, the same branch on another runner, every coastd worktree. On
2026-11-08 GitHub's release downloads answered 503 for minutes, and CI configures
failed that way. The zip now comes from `COAST_DEPS_CACHE` (default
`~/.cache/coast/deps`) when the cache holds a copy with the pinned SHA256, and is
downloaded there once otherwise. Each tree still extracts it into its own `_deps`,
so trees with different generators do not collide. See *Build Integration* in
[core/gpu]({{% relref "/architecture/modules/core_gpu" %}}).

## 2026-11-07 — `frozen` and `run_id` are the run's; a frozen trial resumes

`frozen` moves from the neuron groups to the top of the config: a frozen run keeps
all learning off, in every group. Whether a checkpoint load continues the datasets'
sequences is decided by a new top-level `run_id`: every checkpoint records the id
of the run that wrote it (`metadata/run_id`, checkpoint format 6), and a dataset
continues only when the loading run has the same id. `"auto"` (the default) means a
fresh id — coastd fixes it once per job when it writes the run config, so a resume
continues, while a frozen trial or training on a new subset starts fresh; name a
checkpoint's id to continue it in a new run. `--resume` of an `"auto"` config adopts
the checkpoint's id. Before, `restore_sequence_from_checkpoint=false` restarted a
resumed frozen trial's sequence. Each dataset records its resolved params, and a
restoring load warns on a difference. The per-group `frozen` and
`restore_sequence_from_checkpoint` are refused. The run manifest records `frozen`
and `run_id`; the group config records no longer carry `frozen`. A resumed run also
continues its observe files (they record `run_id`): the rows from the checkpoint's
step on are dropped and appended again, where a relaunch used to recreate them and
lose everything before the checkpoint. See
[io]({{% relref "/architecture/modules/io" %}}) and SPEC.md S.5.14.

## 2026-11-07 — `coastd` hashes its inputs; manifests record the toolchain

A new command, `hash_inputs {paths: [rel…]}`, returns the SHA-256 of files under
the daemon's `dataset_base`, computed on the daemon's filesystem, so a client on
another machine can prove which bytes a run's `./data/<rel>` reads. Paths must be
relative and stay inside `dataset_base` after symlinks; the count and total bytes
are capped. `machine_info` now also reports `dataset_base` itself (discovery
replies omit it). Both are additive: no `proto_version` bump. `coast_runtime.json`
gains `coast_compiler` and `coast_target_cpu` (what `-march=native` resolved to),
because results were measured to be bit-identical across hosts only for the same
compiler and target CPU. See [Protocol]({{% relref "/coastd/protocol" %}}).

In a burst of submissions for a commit the daemon's clone has not fetched yet, a
job could fail with "could not resolve ref" if it looked before the window's one
fetch had finished; resolutions that arrive during that fetch now wait for it. See
[Persistence]({{% relref "/coastd/persistence" %}}).

## 2026-11-04 — Hook removal is synchronous (#235)

`env::remove_*_hook()` now returns only once the hook is not running and can
never run again. Removal used to be queued to the next step boundary, so a
visualization destroyed while the sim thread was inside `swap_io()` (closing its
window, loading another config, shutdown) could still have its `this`-capturing
post-swap or barrier hook called on freed memory. The sim thread holds a new
`active_hooks_mutex_` for each hook phase (post-swap and barrier hooks are one
phase); a removal waits for at most that phase, and not at all when the sim is
not stepping. Removal from inside a hook stays deferred. While a removal waits,
`env::hook_removal_waiting()` is set and the visualizations' barrier hooks stop
waiting for the GUI, so the remover (usually the GUI thread) is not held for
their 100 ms timeouts. The visualizations' destructors no longer spin on their
sync state. See [Hook System]({{% relref "/key-abstractions/hook-system" %}}).

## 2026-11-04 — An exception in a headless run ends its feed on `phase=failed` (#233)

An exception after `phase=start` (in the run loop, extracting or writing
`--save-state`) made `run_headless` exit 1 with no terminal progress line. The
catch now stops the sim thread and writes `phase=failed`, so no `checkpoint`
line can follow it. A non-`std::exception` is now caught too: it used to escape
`run_headless`, which is documented not to throw, and abort the process. See
*Run progress* in [diag]({{% relref "/architecture/modules/diag" %}}).

## 2026-11-04 — ngm2 depth capped at 4; depth-15 hang fixed (#237)

ngm2 now refuses `max_branch_level` above 4 (it accepted up to 15), the same
upper bound as ngm3 and the ngm25 GPU path; every shipped config uses 3. At
depth 15 a `u16` loop counter in `get_leaf_mask()` wrapped and `process()` never
returned, and a neuron's representation count wrapped to 0 above 65535; those
loops and counts are now `std::size_t`. See
[hd_ngm2]({{% relref "/architecture/modules/hd_ngm2" %}}).

## 2026-11-04 — ngm2 checkpoints record dendrite types, tree depths and their seed (#236, #239)

Checkpoint format version 5. An ngm2 dendrite now records its type (`apical`)
and tree depth (`max_branch_level`), and the structural pre-pass refuses a
checkpoint whose types or depths differ, before anything is assigned. Before
this, a proximal dendrite's checkpoint loaded into an apical one of the same
width, and a depth-3 tree into a depth-4 one, because every segment id fitted.
An older checkpoint is checked for depth against its configuration record; its
types cannot be checked, which one notice per load reports. ngm2 also writes its
group's `effective_seed`, and a load restores it with the RNG streams, so
`effective_seed()`, the run manifest and the next checkpoint name the seed of the
restored state. See [hd_ngm2]({{% relref "/architecture/modules/hd_ngm2" %}}) and
[logging]({{% relref "/architecture/modules/logging" %}}).

## 2026-11-04 — GPU checkpoints fail instead of saving zeros (#234)

The ngm3 and ngm25 GPU groups discarded a failed `gpu_device::readback` and used
the zero-initialized result, so a failed transfer during `save_state` produced a
"successful" checkpoint of zeroed learned state that replaced the last good file;
in ngm25 a failed growth read branched on zeros and uploaded them to the device.
Required reads now throw (the save fails, nothing is published), ngm25 stages its
reads before committing, and the vis/probe paths keep or skip rather than show
zeros. See [hd_ngm3]({{% relref "/architecture/modules/hd_ngm3" %}}), Observability.

## 2026-11-03 — coastd runs on macOS

The daemon now starts on macOS. Its boot pid-guard reads the process start time
there from `sysctl(KERN_PROC_PID)`, so the startup gate that kept it Linux-only
opens. The accept loop also wakes itself every 200 ms, because on macOS
`shutdown()` does not wake a blocked `accept()` and SIGTERM never stopped the
daemon. `--install-service` and unattended operation under launchd stay out of
scope. See ADR-17 in [decisions]({{% relref "/coastd/decisions" %}}) and
[boundaries]({{% relref "/coastd/boundaries" %}}).

## 2026-11-01 — Save State refuses a halted run and survives a failed extraction (#229)

The Logging window's Save State, like auto-save on pause, now refuses a halted
(faulted or partly loaded) run. Both go through the new
`logging_manager::try_extract_state`, which checks the halt inside the quiesced
task, before the env is read, and returns a throw from the extraction (a failed
GPU readback) as an error. Such a throw used to reach the GUI thread uncaught
and end the process. See [logging]({{% relref "/architecture/modules/logging" %}}).

## 2026-10-31 — Buffer logger reports write failures (#203)

A failed HDF5 write in the buffer logger (disk full, file size limit) used to be
dropped: the log was truncated or unreadable while the GUI and the headless run
reported success. The first failure now trips a sticky `write_ok()` flag, stops
further writes, and makes `stop()` return an error. A headless run then exits 1,
and the Logging window shows the failure. `hdf5::dataset::extend/write`,
`dataspace::select_hyperslab` and `file::flush/close` are `[[nodiscard]]`; the
GUI's plot save (`plot_h5`) now also checks its dataset writes and close. See
*Write failures* in [logging]({{% relref "/architecture/modules/logging" %}}).

The headless run's terminal progress event is emitted last, after buffer logging
has stopped and `--save-state` has been written. A run that exits 1 for either
reason now ends its feed with `phase=failed` instead of `phase=done`. Before
this, a failed `--save-state` also ended with `done`. `logged=` on that line,
and the *Logging summary*, now count every frame: `combined_stats()` keeps a
stopped session's final counts. Before this the summary never printed, and the
terminal line missed the frames flushed at stop.

## 2026-10-31 — An exception during a step halts the run instead of aborting it (#198)

An exception thrown by an entity's `process()`, a step hook or `swap_io()` used
to unwind out of the sim thread into `std::terminate` (exit 134, nothing flushed).
It now halts the run like a returned fault, with the new `error_kind::exception`:
headless logs the exception's message and exits 1. A throw from a CPU entity is
stamped with the entity's id. See *Hot-Path Fault Channel* in
[Threading]({{% relref "/threading" %}}).

## 2026-10-31 — Dataset loaders check their input files; one loader per format (#200, #208, #201)

The IDX reader (`mnist_db.cpp`) rejects a truncated header, a size that
overflows and data that does not match the header, for label files as well as
image files. It reads only unsigned-byte IDX data (type 0x08), the only type
any loader consumes; files of the other IDX types are rejected (they used to be
read with the wrong byte order). A dataset with no images, or images of size 0,
is rejected at load instead of aborting the run on its first step. A truncated or oversized CIFAR batch file throws instead
of turning into black samples, and `color_mnist` rejects image files that are
not 28×28 instead of writing past its output. EMNIST and KMNIST are
instantiations of one `dataset::idx_gray_io<Tag>` with MNIST; `cifar10_io` and
`cifar100_io` are only constructors over `cifar_io_base`. See *Shared Format
Bases* in [io]({{% relref "/architecture/modules/io" %}}). Simulation output is
unchanged.

A Step request left pending while the sim runs no longer fires on the next
pause or steps past the step limit, and no Step or Run request steps past
the limit while paused at it (`sim_thread_controller` checks the limit in
`do_step()`).

## 2026-10-31 — Node-set copies share their template's seed, by design (#191)

Documented, not changed: a node-set copy takes its template node's
`random_seed_offset`, so every copy runs the template's random streams and the
copies differ only in wiring. Replicas with independent randomness are separate
direct nodes with their own offsets. See *sim_node_set_t* in
[json_cfg]({{% relref "/architecture/modules/json_cfg" %}}).

## 2026-10-31 — random_index's ri_seed takes the seed offsets (#190)

`ri_seed` now composes like every other seed: config value + `--seed-offset` +
the node's `random_seed_offset`, modulo 2³². It used to be taken alone, so a seed
sweep varied the groups' own streams but kept every random_index inhibition matrix
fixed. Runs with a nonzero offset and a random_index inhibitor get a different
matrix than before; with zero offsets nothing changes. The manifest's `node_seeds`
records the matrix seed as `inhibition_seed`. See
[inhibition]({{% relref "/architecture/modules/inhibition" %}}).

## 2026-10-31 — Config validation (#199)

A config is checked against itself bound and serialised again: a number that
does not fit its C++ type (wrapped, truncated, overflowed to inf) is refused by
its JSON path, an unknown key is warned about by its path — see *json_loader* in
[json_cfg]({{% relref "/architecture/modules/json_cfg" %}}). Seeds and seed
offsets, in configs and `--seed-offset`, take [0, 2³² − 1]; a negative value is
refused instead of taken as its two's complement (#167). Also refused at load or
build: a sigmoid `steepness` outside [0, 1) (1 made every weight-filter output
NaN), a duplicate node id, and in ngm2/ngm3 (CPU and GPU) a reversed neuron
noise range or a `stochastic_win_thres` outside [0, 1].

## 2026-10-31 — Buffer logger: quiesced stop, real steps, safe stats (#197, #202, #214)

`logging_manager::stop_buffer_logging` now takes the controller and detaches the
logger's post-swap hook inside `run_quiesced`. Hook removal is only queued, so
before this fix the hook could still run, from a GUI stop, a config reload or
the headless end of run, while `stop()` finished the file and freed the logger.
`~sim_instance` joins the sim thread before destroying the logging manager.
Buffer logs record simulation steps (not frame counts) and a real `@start_step`,
and their timestamps start at each session's `start()`. `buffer_logger::stats()`
is a snapshot of atomics, safe to poll from the GUI or the headless main thread.
See *Buffer Logger* in [logging]({{% relref "/architecture/modules/logging" %}}).

## 2026-10-31 — HDF5 probe logger removed (#222)

`probe_logger` and `file_observer` are gone, with `logging_manager`'s probe
API, the Logging window's probe section and the Probe Browser's logging
callback. No GUI or CLI path had reached them since 392c6e28 replaced
them with the per-probe CSV observer (*CSV File...*). The HDF5 Browser no
longer knows a `probe_log` family: a file of that type is now reported as
not a recognized COAST log. `log_stats::samples_written` is gone too, so the
headless *Logging summary* and the Logging window no longer print sample
counts. See [logging]({{% relref "/architecture/modules/logging" %}}).

## 2026-10-30 — Random seeds are 32-bit strong types (#167)

Every seed and seed offset is now `seed_t` / `seed_offset_t` (`core/seed.h`), a
strong 32-bit type, composed modulo 2³². `effective_seed()`, the manifest's
`node_seeds` and the checkpoint's configuration records report exactly the value
an RNG was seeded with (a negative dataset seed used to be reported
sign-extended to 64 bits), and seed + offset can no longer overflow. Config seeds
and `--seed-offset` accept integers in [−2³¹, 2³² − 1]; a negative value is its
two's complement; a refused config seed is named by its JSON path. The retina's `[0, INT_MAX]` refusal is gone, as is
`random_index`'s 64-bit `ri_seed`, of which the RNG only ever used 32 bits. See
*Random seeds* in [core]({{% relref "/architecture/modules/core" %}}).

ngm2 hashes its neuron and dendrite stream seeds zero-extended now (they were
sign-extended through an `int`), so ngm2 runs differ from earlier ones and the
ngm2 goldens were re-baselined.

## 2026-10-28 — 1 − Hoyer noise gate; soft_wta and the NSE removed

The dendrite noise gate of HD-NGM2, HD-NGM2.5 and HD-NGM3 (CPU and GPU) judges input
by **1 − Hoyer sparsity** instead of the normalized Shannon entropy, with fixed
response and learning windows shared by all three models — see the *Noise gate*
section of [hd_ngm2]({{% relref "/architecture/modules/hd_ngm2" %}}). The response gate's min
over a dendrite's or bank's channels leaves out channels without shape evidence (zero or
non-finite sum, or a single entry). The NSE is gone
from the code base: `normalized_shannon_entropy`, `input_stats_t::nse`, the dataset
probes (`NSE`/`MAX_VAL`/`AVG` on mnist and cifar10, `nse`/`max`/`avg` and their
reduction kernel on `mnist_io_gpu`) and the replay viewer's NSE statistic.

The entropy-modulated `soft_wta` inhibition went with it.
[inhibition]({{% relref "/architecture/modules/inhibition" %}}) now defaults to
`max_softmax`: an empty `inhibition_cfg` synthesizes one with beta =
`local_inhibition_strength`, whose default is now 10 (the repo configs were moved
from 5 to 10, and their named `soft_wta` entries to `max_softmax`). A config naming
`"soft_wta"` is rejected. A named `inhibitions` entry that omits `type` or `strength` now gets
max_softmax and 10 (it used to get soft_wta and 5).

## 2026-07-23 — coastd, the headless-job daemon (feat/coastd)

The repository now builds a **second executable**, documented in a new top-level
[coastd]({{% relref "/coastd" %}}) section. `coastd` is a resident per-machine
service that runs COAST simulations headless on stable — possibly remote —
machines so long batch runs survive the loss of the client that started them. A
client submits a job (a committed **git ref** plus the config's **content**) and
may then disconnect entirely; the daemon resolves and freezes the SHA, builds that
version natively in a per-SHA git worktree, schedules against fixed slots,
supervises the one-shot `coast` child, and **auto-resumes it from its newest
checkpoint after a crash, reboot or daemon restart**.

It is deliberately *not* filed under `architecture/modules/` — it is a separate
process with its own target tree (`coastd/`), it depends on `coast_lib` rather
than being part of it, and `coast_lib`'s `include/` carries no coastd headers.

The design spine is a single **pure function**,
`reconcile(jobs, reality, config) → actions`, which *is* the scheduling and
recovery state machine. Its most valuable property is that the **recovery path is
identical to the steady-state path** — so crash recovery is not rarely-exercised
code, it runs about once a second on every healthy daemon. Everything a crash
could desync (slot availability, SHA refcount, worktree presence and idle age) is
**derived every pass, never stored**. The recovery model (R1) then dissolves the
hardest problem rather than solving it: children die with the daemon under
systemd's stock `KillMode=control-group`, so there are no live orphans and the
duplicate-run corruption is impossible by construction.

- [coastd]({{% relref "/coastd" %}}) — new section: overview, [architecture]({{% relref "/coastd/architecture" %}}) (targets, the single-writer core loop, the coupling invariant, dispatch idempotence and the pid skew guard), [reconcile & recovery]({{% relref "/coastd/reconcile" %}}) (the rule table, the three failure counters, R1, boot), [persistence & the child contract]({{% relref "/coastd/persistence" %}}) (two-tree layout, FS-is-truth, output ownership by construction, the build side, ref freshness), [protocol & clients]({{% relref "/coastd/protocol" %}}) (JSONL wire, edge hardening, discovery, the console and GUI frontends), [boundaries]({{% relref "/coastd/boundaries" %}}), and [15 ADRs]({{% relref "/coastd/decisions" %}})
- [Architecture]({{% relref "/architecture" %}}) — scoping notice (that page is the `coast` exe / `coast_lib`); new `coastd_console` module row; `coastd_panel.cpp` added to `gui_tools`
- [Decisions (ADR)]({{% relref "/architecture/decisions" %}}) — notice scoping that page to the *ngm3-gpu-port* cycle and pointing at the coastd record
- [Threading]({{% relref "/threading" %}}) — notice that coastd is a separate process and not on the thread map
- [Testing]({{% relref "/testing" %}}) — a `coastd` tier row plus a new subsection explaining that the `unit.coastd.*` tests span the whole realism range, ending at the opt-in `run_real_gate.sh` bit-identity gate
- [File Reference]({{% relref "/reference" %}}) — a full `coastd/` file map and a `tests/unit/coastd/` row

**Four decisions supersede the design document** and are recorded as such rather
than narrated as if they had always been the plan. The consequential one: the wire
carries the config's **bytes**, not a path. The design specified a path, having
never confronted that the client and daemon are in general *different machines* —
across which a path is meaningless, and worse, a path existing on both hosts with
different content runs the wrong config **successfully and undetectably**. The path
form was removed outright rather than kept as a fallback. The others: path-safety
now resolves relative paths against the **run-dir** frame the child actually uses
(the old config-parent frame left relative paths effectively unchecked); a run
directory carries **two** config files (`config.json`, which coastd rewrites, and
`config.submitted.json`, the verbatim provenance record); and ref resolution now
**fetches first** and prefers the remote's branch, because a stale clone would
otherwise silently run old code under a current-sounding name.

**Rebased onto the headless-build branch**, which removed the `-n 0` sentinel —
and with it the command line coastd used for every unbounded job. `plan_launch`
now says `--unlimited`, and because `--unlimited` refuses the end-of-run outputs
coastd depends on, `--save-on-term` (this branch's SIGTERM handler) is what makes
them honest and lifts that refusal. The two flags are now a pair: dropping
`--save-on-term` would turn every unbounded launch into a parse error, not merely
lose a shutdown checkpoint. Verified by measurement — a SIGTERM'd unlimited run
exits 0 with a readable buffer log, a final state file, and a manifest reading
`exit_reason: user_signal`.

One stale comment noted but deliberately left in the source: `reconcile.h`'s
rule-table lists `cancel_requested` before `failure_reason` under "first match
wins", while `reconcile.cpp` checks failure **first** — intentionally, so a real
failure is not relabelled `cancelled/` and its `failure_reason` filed where no
operator will look. The implementation is correct; the header comment is not.
[Reconcile & Recovery]({{% relref "/coastd/reconcile" %}}) documents the actual
order and flags the discrepancy.

Documented honestly as **not** verified: the GUI panel is compile-verified only
(its runtime test needs an installed, running daemon and was deferred by
decision); nothing has been run on macOS, which additionally has a real
orphan-detection gap because the pid guard's start-time token comes from
`/proc/<pid>/stat`; and the checkpoint fence's power-loss leg rests on settled
POSIX `rename`/`fsync` semantics as *reasoning*, not measurement.
## 2026-07-22 — Review follow-ups: `--unlimited` honesty, and a `design/` cull

Third review round on `refactor/headless-build`, addressing what the reviewers
found rather than what was convenient.

**`--unlimited` no longer accepts outputs it would destroy.** Measured on a
6-second run: `--save-state` and `--emit-run-manifest` produced *no file at all*
(both are written on the way out, which never happens for a run that ends only by
being killed), and `--log-buffers` left a 48 MB HDF5 file that `h5ls` reports as
`**NOT FOUND**` — never closed, unreadable. Accepting those silently would have
recreated, in a new place, the exact footgun that removing the `-n 0` sentinel
closed: a long run that quietly yields nothing usable. All three are now a parse
error. `--checkpoint` is deliberately exempt and is the supported way to get data
out of an unbounded run — its snapshots are completed *during* the run.

Also: the `-n`/`--unlimited` conflict is now reported *before* the value is
judged (`--unlimited -n 0` used to answer "pass --unlimited", advising the flag
already on the command line), and the headless build's help no longer advertises
`-r, --run`, which is inert there.

**The branch's headline safety property is now tested.** That a headless-built
`coast config.json` is *rejected* rather than started as an unbounded run lived
in `main()` and was asserted by nothing — while a unit test asserted the same
command line *parses* fine, which is correct but the opposite claim. B-tier now
carries `#ifndef COAST_WITH_GUI` cases for the rejection and for the
notice/`-H`-suppression pair, plus cases for the `--unlimited` refusals.

**`design/` cull.** Three documents were removed as implemented and superseded:
`location_config_design.md` (self-marked IMPLEMENTED), `location_config_scan.md`
(a decision scan, superseded by it) and `path_resolution_refactor_plan.md`
(self-marked IMPLEMENTED). Nothing outside those three referenced them.

The rest of `design/` stays, deliberately. Most of it is not stale plans but the
**design record cited from the code itself** — `include/core/step_error.h` points
at `error_handling_plan.md`, `include/hd_ngm25/*` at the ngm2.5 documents,
`include/console/*` at `console_api_design.md`, and so on across roughly thirty
references. Deleting those would strip the "why" from the headers that point at
them. Two documents (`remote_run_dispatch.md`, `ngm3_improvement.md`) are
unimplemented proposals and were kept for the opposite reason.

## 2026-07-22 — A clean headless build (refactor/headless-build)

`coast` used to be a single binary with the GUI compiled in whether or not it was
ever opened: one `coast_lib` archive held all 47 GUI translation units plus
ImGui/ImPlot/ImNodeFlow and PUBLIC-linked SDL2 and OpenGL, so every headless run,
every unit test and every GPU probe dragged the graphics stack along. The archive
is now **split in two** — `coast_lib` (headless core, TBB + HDF5 only) and
`coast_gui` (everything touching ImGui/ImPlot/ImNodeFlow/SDL2/OpenGL) — with the
dependency pointing one way only, `coast_gui → coast_lib`. That makes the boundary
a **link-time property** rather than a convention: a GUI include that creeps into
a core TU now produces an undefined symbol instead of quietly re-coupling the
halves.

`-DCOAST_WITH_GUI=OFF` (default `ON`) drops `coast_gui` entirely. SDL2 and OpenGL
are never even searched for, and the embedded clang-repl console is forced off —
`console_panel` *is* an ImGui window. New `headless` / `headless-tests` presets.
Measured on one Linux box (Release, gcc): full 114 s / 15.5 MB / 30 shared libs
versus headless 21 s / 4.1 MB / 13 libs; with `COAST_GPU_BACKEND=none` as well,
20 s / 3.8 MB / 10 libs. A headless binary produces **byte-identical** simulation
state to the full build, so removing the GUI does not perturb the simulation.

Three headers turned out to be misfiled rather than GUI-coupled and moved:
`gui_vis/core/vis_hints.h` → `core/vis_hints.h` (a config POD, parsed headlessly
and written into the HDF5 buffer-log topology), its serializer
`json_cfg/gui_vis_json.h` → `json_cfg/vis_hints_json.h`, and
`gui_core/demand_published.h` → `core/demand_published.h` (plain atomics). No core
header now includes anything under `gui_*`. Genuinely GUI-side and moved into
`coast_gui`: `run_gui`, the four probe observers (all render through ImGui — the
probe *recording* side stays in the core), and `sim_console`.

A headless binary runs headless implicitly and says so once, unless `-H/--headless`
was given; `--headless` is still accepted and now only suppresses that notice.

**Run length became explicit, and `-n 0` no longer means "forever".** Making
headless implicit exposed a latent foot-gun: `coast config.json` — the command
that opens a window in a GUI build — would have started an *unbounded* run on a
headless binary, announced only by an INFO line that `-q` hides. Rather than
special-case that path, the sentinel is gone. `cli::options` now carries a
`run_length` mode (`unset`/`finite`/`unlimited`) and `max_steps` is read only
through it; `-n/--steps` requires a count ≥ 1, an unbounded run must say
`--unlimited`, the two are mutually exclusive, and a headless run with neither is
rejected. The rule is split so that `cli::parse` stays **build-independent** — it
rejects an explicit `-H` with no length, while `main()` applies the same rule
(and the same message, shared via `cli::headless_needs_length_msg`) at the point
where a headless build promotes an unmarked run. Putting the build-dependent half
in the parser would have made its unit tests mean different things in the two
configurations. The run manifest gains a schema-additive `unlimited` boolean so
`n_steps_requested: 0` is never ambiguous. The `@coast/progress` v1 wire format
keeps `target=0` for an unbounded run — that sentinel is a compatibility contract
with existing parsers (consim, `sim_console::parse_log_progress`), and
`diag/progress.h` now says so and points at the additive way to disambiguate it.

- [Architecture]({{% relref "/architecture" %}}) — new "Two archives" section; module map gated on `COAST_WITH_GUI`; GUI-layer table gains `sim_console` and `probe/observers`
- [console]({{% relref "/architecture/modules/console" %}}) — the headless build forces the console off; `coast_console` links `coast_gui`
- [gui_plot]({{% relref "/architecture/modules/gui_plot" %}}) — lives in `coast_gui`; its unit suite is skipped headless
- [Reference]({{% relref "/reference" %}}) — moved-header rows, `CMakePresets.json`
- Tests: 102/102 GUI, 92/92 headless. The 10 the headless build drops are exactly the GUI-layer suites (7 `unit.gui_plot`, 3 `unit.console`), plus the 10 `b_tier.simc_*` cases compiled out of `test_b_tier.cpp`

## 2026-07-18 — Run simulations from the console (feat/console-simctrl)

The embedded C++ console can now **drive the simulation**, not just read its
output. A new `coast::simc` façade (`include/sim_console/sim_console_api.h`, in
`coast_lib`, preamble-included as `simc::…`) loads a config, edits it by JSON
Pointer, and runs the sim in chunks for parameter search / characterization —
each run self-documenting in one directory (`config.json`, `observe/`,
`snapshots/`, one immutable `state_<NNN>.h5` per chunk as a rolling history,
per-`step()` timestamped logs). Crucially it **shells
out to a child `coast -n` process** rather than building a `sim_instance`
in-process: this reuses the whole tested headless binary and isolates native sim
faults to the child, closing the one containment hole the in-process console
cannot. Runs are fresh-sim-per-config because group hyperparameters are baked into
the neuron groups at build time. `step()` blocks the console thread but the GUI
stays responsive; `tail -f h->last_log()` is a live monitor.

Runs can also be **non-blocking**: `step_async()` launches a chunk and returns
immediately, opening a **"Simulation Runs"** status table that collects every
concurrent run (progress bar, `cur/target` steps, status, elapsed) — the sweep
path. `render_runs_window()` (driven each frame beside `coast::plot::render_all()`)
polls the children and chains each completed run's state back into its handle.

- [console]({{% relref "/architecture/modules/console" %}}) — new "Running Simulations" section (surface, subprocess rationale, run-dir layout, blocking + async monitoring, the status dialog)
- `design/sim_console_api.md` — the design record (subprocess-vs-in-process, the baked-hyperparameter fact, the `const char*`/`bool` overload trap, `COAST_SIMC_BINARY`, the async registry + one-way lock order)
- Test: `tests/integration_b/test_b_tier.cpp` gains `b_tier.simc_facade` (config edit + run + chunked state-chaining + logs) and `b_tier.simc_async` (two concurrent `step_async` runs, registry tracking, `wait_all_runs`, in-flight guard), against the built `coast`

Post-review hardening (four independent review rounds): the `fd_capture` pipe is close-on-exec so `step_async` is genuinely non-blocking; each chunk saves an immutable per-chunk `state_<NNN>.h5` (rolling history + survivable writes); colliding run dirs get a `_NNN` suffix; `run()` writes **absolute** observe/snapshot paths (a relative `snapshot_dir` would double-nest under `--checkpoint`, since coast resolves it against the config's parent); an `int` `set()` overload fixes an ambiguity on `set(ptr, 42)`; and `sim_run`'s constructor is passkey-restricted to `run()` so a non-`shared_ptr` instance can't trip `bad_weak_ptr` in `step_async`. Round three: the HDF5 browser now binds datasets through a new `console_iface::bind_dataset`, which escapes the dataset name as a C++ string literal — a crafted name (quotes/backslashes) could otherwise inject code into the JIT console (regression-tested); the in-flight guard (`step_in_flight`) now covers blocking `step()` too, not just `step_async`; and the resumed-chunk progress bar no longer underflows to 100%. Round four: `set()`'s integer handling is now a single integral template, so `set(ptr, vec.size())` and other `unsigned`/`size_t` args compile (the `int`-only overload was an incomplete fix); plus a regression test for the failed-`step_async` path (guard cleared, nothing chained) and `save_state`'s no-state throw. Round four also closes the broader fd-inheritance gap: `spawn_child` adds a `closefrom` file action (glibc `posix_spawn_file_actions_addclosefrom_np`, i.e. `close_range(3, ~0U, 0)` in the child before `exec`) so a spawned `coast` inherits only fds 0/1/2 — not the console's open HDF5 handles, the fd-capture saved dups, or GPU/SDL descriptors it never uses. The macOS-portable `FD_CLOEXEC`-on-the-pipe fix stays as the fallback where `closefrom` isn't available. Round five: a run dir whose name starts with `-` (e.g. `run_dir("-baseline")`) no longer fails every step with a confusing `"Unknown option"` — the façade prefixes a relative config path with `./` so coast's CLI parser doesn't read it as a flag; plus a Linux-only regression test that a spawned child does not inherit a parent fd (proving the `closefrom` action), a leading-dash regression test, and two stale-doc fixes (`running()` help text, and the design-doc example that used a non-existent `learning_rate()` sugar). Round six (2026-07-19): the fd-inheritance close is now **implemented on macOS too** (the earlier round left it Linux-only and called the residual "benign") — `spawn_child` sets the Apple spawnattr `POSIX_SPAWN_CLOEXEC_DEFAULT`, closing every inherited fd in the child except 0/1/2, which are preserved with the `dup2(fd,fd)` idiom (**needs macOS verification** — authored and tested on Linux only). `step_async` gains a **process pool**: a global cap (`max_concurrent_runs()`, default 4, via `COAST_SIMC_MAX_CONCURRENT` / `set_max_concurrent_runs()`) bounds concurrent children, and runs past the cap **queue** and drain automatically inside `poll_runs()` as children exit — so a sweep never fans out an unbounded number of full `coast` processes, and the call still never blocks. Plus robustness: `run()` reports the real cause when the run-dir parent can't be created (was a discarded `error_code`), `spawn_child` checks every `posix_spawn_*` setup return code (RAII-cleaned), and new regression tests (`b_tier.simc_pool` for queue-and-drain, `b_tier.simc_async_capture_eof` — a cross-platform guard that a `step_async` child doesn't pin the caller's stdout pipe). Two more stale-doc fixes (`console.md` "six virtuals" → seven with `bind_dataset`; `sim_console_api.md` status banner design → implemented). **Run cancellation + window reopen**: async runs can now be stopped — `sim_run::cancel()`, `cancel_all_runs()`, and per-row **Cancel** / **Cancel all** buttons in the Simulation Runs table. A running child is SIGTERM'd (coast installs no handler, so it terminates) and reaped as `cancelled` (amber bar, chains nothing); a still-queued run is dropped before it spawns; either way the handle's in-flight guard clears. Closing the dialog's X now only hides it — `simc::show_runs_window()` / `hide_runs_window()` toggle it back (previously a closed window could only be reopened by launching another run). Regression test `b_tier.simc_cancel` (queued-cancel, running-cancel, `cancel_all`). **Post-round-six review follow-ups** (dual Opus + Sonnet 5 review): the per-row Cancel button now receives clicks (`ImGuiSelectableFlags_AllowOverlap` — a span-all-columns row selectable was swallowing them); `spawn_child`'s error messages use thread-safe `std::system_category().message()` (it can now spawn from both the console thread and the GUI thread's queue-drain, so `std::strerror`'s shared buffer would race); `run()` now checks the `config.json` write, not just the open (a truncated write no longer surfaces only as an opaque child parse failure); the fd-close comment now states its scope explicitly (Linux glibc ≥2.34 and macOS only — older/musl toolchains are unsupported, not silently degraded); and `step()`'s deliberate bypass of the async pool (a foreground call runs immediately rather than stalling behind a saturated sweep) is now documented. This branch was rebased onto current `main` (PR #118, the independent-file console): the h5 browser binds datasets through both `reserve_var` (main's name dedup) and `bind_dataset` (this branch's injection escaping), and `bind_dataset` uses main's extracted `console::detail::quote_cpp`.

## 2026-07-18 — The console is no longer bound to one open file (feat/independent-console)

The embedded C++ console used to require an open HDF5 file: opening one **rebuilt the whole interpreter** and injected a single global `h5` reference (`repl::bind_h5`), so there was no interpreter at all until a file was chosen, switching files threw away every user definition, and only one file could be inspected at a time. That coupling is removed. The interpreter is now built **once**, independent of any file, and is usable from the first prompt; its preamble already pulls in `console_api.h`, so a real free function `open("path.h5")` — returning `shared_ptr<file_console>` — is in scope immediately. **Several files are open at once**, each in its own session variable. The Open button and the browser inject `auto <var> = open("<path>")` (the handle name derived from the file's stem via `var_name_for`, disambiguated on collision), and dataset binds go through that handle (`<var>->dataset(...)`) instead of a global `h5`. `session::reset()` rebuilds and re-opens the open files; handle cleanup is automatic — interpreter teardown runs its globals' destructors (measured under ASan/LSan: no `file_console` leak on `~session` or `reset()`), and `file_console`'s deleter is compiled into `coast_lib`, so every open handle is closed cleanly (this is the opposite of the plot case, whose JIT-emitted deleter needs the retaining arena). `open()`'s executor is registered in the `session` constructor, mirroring `coast::plot::set_executor`.

- [console]({{% relref "/architecture/modules/console" %}}) — intro, "What the User Types", entry points, the two file-coupling boundary bullets, and the tests table all rewritten for the handle-based, multi-file model; the `h5`-binding mentions in error-recovery removed
- [File Reference]({{% relref "/reference" %}}) — `console_api`, `repl`, `session`, `console_panel`, and the console test rows updated
- `design/console_api_design.md` — dated update note at the top; the "how does the GUI bind the current file" open item marked resolved

## 2026-07-15 — `.L`/`.x` run through the guarded path (fix/exec)

Closes the gap the console page had just finished documenting as open, and rewrites that documentation to match. `repl::load()` — the `.L`/`.x` path, which runs **user-supplied files** — called the raw `exec()`, so macro files had neither an exception guard nor `#include` isolation. Both were measured: a throwing file-scope initializer **aborted the process** (a typo suffices, since `h5.dataset()` throws on an unknown name), and a file with a fresh `#include` plus a parse error **poisoned the interpreter** — the exact failure `split_into_units` exists to defuse, on the likeliest macro-file shape there is.

The cause is worth remembering, because it is a documentation failure as much as a code one. The original design reasoned: statements must be wrapped in a `try` → declarations *cannot* be wrapped → therefore declarations cannot be guarded → so `load()` must run raw. Reversing the wrapper in favour of a **host-side** catch killed that inference — you do not need to wrap code to guard it — but `load()` was never moved over, and the stale `repl.h` comment still describing the wrapper kept the gap invisible to review. The design record even listed a throwing static initializer as an *uncontainable* process-killer, a claim true only of the wrapping design and never re-tested; it is now refuted by measurement and superseded in place. Layered staleness like that is the actual defect.

`load()` now calls `exec_guarded()`. That alone would have regressed one shape — `split_into_units` is line-based and would cut a block comment in half, turning code clang accepts into a parse error — so it now tracks comment state and isolates only lines lying wholly outside a `/* */`. The tracker is deliberately not a full lexer; the asymmetry is safe by construction, since a false "in a comment" only skips isolation while it can never split one apart.

- [console]({{% relref "/architecture/modules/console" %}}) — error-recovery section rewritten: the guard rule stated as an invariant ("you do not need to wrap code to guard it; the declaration/statement split is not a guarding boundary"), the history kept as a cautionary box, comment-aware splitting documented, and the two now-false boundary bullets replaced — one with the residual native-fault limit, which no in-process catch closes
- [File Reference]({{% relref "/reference" %}}) — `repl.cpp` row updated for the guarded `load()` and `scan_comment_state`

## 2026-07-15 — The embedded C++ console and its plot module (doc/console-gui-plot)

Documents `console` and `gui_plot`, a pair that had shipped with **no coverage on the site at all**.

The console page frames the feature around its governing idea — a **compiled/interpreted split** where an `-O0` clang-repl layer only orchestrates `-O3` compiled COAST code, which is why an interpreted loop over a multi-GB dataset runs at compiled speed — then covers what a maintainer can break: the auto-detecting LLVM build gate and its four non-optional flags (`-rdynamic`, `-resource-dir`, `-std=c++23`, `-I`), the six-virtual `console_iface` seam that keeps libLLVM out of `coast_lib`, the interpreter thread with its fd-capture-plus-reader-thread output path, the bound `h5` surface and meta-commands, and the boundaries (`console_api.h` as a parse boundary, the baked-in absolute `COAST_INCLUDE_DIR`, the `row_view` span-dangling trap). The **error-recovery** section is the load-bearing one: `exec_guarded` runs code *unwrapped* (the interpreter-side try/catch wrapper was implemented, then reversed — it scoped away every user declaration), `Undo(1)` fires only on Execute failure (undoing after a *parse* error rolls back the previous good input and eventually unwinds past the preamble into a crash), and `split_into_units` isolates each `#include` because a unit that enters a source file *and* fails to parse poisons every later `Parse()`. All three are counter-intuitive, pinned by regression tests, and must not be "simplified" back.

The gui_plot page leads with the constraint that explains the module's whole shape: **every macro-reachable header must be ImGui-free**, because clang-repl parses it at runtime — hence the `region` POD instead of `ImVec2`, ImPlot confined to the `.cpp` painters, and a deliberately flat design with no factory or type erasure (the opposite of `gui_vis`, and the reason is parse cost, not taste). It documents the `painter` closure that pins its figure via `shared_from_this`, the registry's exact reaping rule (closed **and** `use_count()==1`), `deinit()`'s process-wide tombstone and the host-less-manager-at-atexit failure it exists to prevent, the load-bearing teardown order in `run_gui`, and persistence — including two things that read as bugs and are not: `flush_captures` must run outside the ImGui frame (each export runs its own frame pass), and the capture builds **two** frames because ImPlot AutoFits the frame *after* data submission. Dependency direction is stated explicitly: gui_plot knows nothing of the console; `COAST_WITH_CONSOLE=OFF` leaves it intact.

The threading page's thread count was **wrong** — four, not counting the console's interpreter thread — and is corrected to five, with the two tool threads called out as such.

- [console]({{% relref "/architecture/modules/console" %}}) — new page: build gate, seam, threading, user surface, error recovery, plot bridge, boundaries
- [gui_plot]({{% relref "/architecture/modules/gui_plot" %}}) — new page: the ImGui-free constraint, figure/painter seam, the six kinds, plot_manager, persistence, themes, boundaries
- [Threading]({{% relref "/threading" %}}) — four → five threads; console thread in the map and three new `What Runs Where` rows
- [File Reference]({{% relref "/reference" %}}) — new `console/` and `gui_plot/` sections, the missing `console_api.h` row under h5browse/, console + gui_plot test rows

Known gaps left open, all documented in place rather than silently inherited: **`.L`/`.x` are not exception-safe** — `repl::load()` takes the raw `exec()` path, so a throwing file-scope initializer in a user macro aborts the whole process (measured: SIGABRT; only `exec_guarded` is covered by a test); `design/console_api_design.md` is partly stale against the shipped code (it still describes the reversed try/catch wrapper); and a saved plot has **no GUI load path** — `load()` exists as API but nothing outside the console calls it, so a saved plot is unreachable from the HDF5 Browser.

> The first two were closed by `fix/exec` (next entry), which this branch is the base of. The plot GUI load path remains open.

## 2026-07-07 — Index-manifest subsets & frozen inference mode (feat/dataset-eval-contract)

Documents the branch's in-coast mechanisms for an external evaluation harness, with eval policy (split roles, config validation, manifest generation) left to that harness. `sample_sequencer` gains an optional **index manifest** — a plain-text file of sample indices restricting presentation to an auditable subset of the canonical dataset files (manifest order when `shuffle` is false, per-epoch reshuffle *within* the manifest when true) — plus the shared `build_sequencer`/`restore_sequenced_state` helpers every loader funnels through. `dataset::sequence_params_t` gains `index_manifest` and `restore_sequence_from_checkpoint` (false = start the sequence fresh rather than resume a checkpoint cursor that has no position on a different subset; restoring under a mismatched manifest now throws). The `frozen` flag on ngm2 and ngm3 groups gates winner selection, adaptation, branching, and the running-activity-max update — mechanisms shown empirically to mutate persisted state even at lr=0, so the flag (not zeroed rates) is the isolation primitive. On the GPU, `build_step_plan` omits the learning kernels and a repurposed `Params` pad carries the frozen flag into the mx_act gate; frozen=false numerics stay bit-identical. The run manifest mirrors active dataset configs (`datasets` object). New tests: sequencer manifest units + resume parity, the mnist_io seam test, and frozen-inert units plus self-comparing A-tier tests (CPU and GPU, no goldens).

- [io]({{% relref "/architecture/modules/io" %}}) — index manifests, sequence_params fields, loader wiring helpers, state-persistence gating; the `mnist_ngm3_frozen_eval` example config wires them together
- [sim_core]({{% relref "/architecture/modules/sim_core" %}}) — run_manifest datasets mirror + `node_seeds` (effective per-node RNG seed)
- [hd_ngm2]({{% relref "/architecture/modules/hd_ngm2" %}}) — frozen inference mode: mechanism, load_state behavior, lr-vs-frozen distinction
- [hd_ngm3]({{% relref "/architecture/modules/hd_ngm3" %}}) — frozen inference mode for CPU and GPU; GPU uniform gating, plan omission, step_/input_cnt distinction
- [Tier Model]({{% relref "/testing/tier-model" %}}) — new unit rows, self-comparing A-tier pattern
- [File Reference]({{% relref "/reference" %}}) — new/updated rows in io/ and tests/

## 2026-07-06 — Collapse dataset serialisers onto derived-type macros (refactor/loader-params, commit 3 of 3)

Follow-up simplification of the two refactor commits below. The per-family hand-written `to_json`/`from_json` blocks in `dataset_json.h` are replaced by `NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_WITH_DEFAULT` over a single `sequence_params_t` base macro (nlohmann 3.12.0), so the shared field list is defined once and a new sequencing field still costs 1x. The `to_json_sequence_fields`/`from_json_sequence_fields` helpers are removed. This drops the earlier byte-identical-output constraint: the derived macro emits base fields before family fields, so serialised key order changes — safe because dataset config JSON is parsed order-independently and nothing compares serialised config text (only model values). Null-tolerance is preserved automatically by the macros' `WITH_DEFAULT` variant. dsprites stays hand-written for its `randomize_readout` deprecation warning; chairs3d stays flat (no sequencer). Separately, `cifar_io_base::rebuild_sequencer` now reuses the sequencing params captured at construction (`rebuild_sequencer(count)`), removing the risk of a subclass re-supplying divergent values.

- [File Reference]({{% relref "/reference" %}}) — `dataset_json.h` row updated to the derived-macro description.

## 2026-07-06 — Fold clone dataset loaders into shared format bases (refactor/loader-params, commit 2 of 2)

Pure refactor, zero behavior change. Introduces two shared base classes that eliminate the near-identical method bodies across the six folded loaders:

- **`dataset::idx_io_base`** (`include/io/idx_io_base.h` / `src/io/idx_io_base.cpp`) — shared base for `mnist_io`, `emnist_io`, `kmnist_io`, and `color_mnist_io`. Holds `output_buf`, `id`, `seq_`, `rgen`, `dis` (all protected); inherits `sim::io_entity` and `mnist::mnist_db`. Implements `set_output_buffer`, `get_outp_id`, `get_outp_size` (returns `get_image_size()`), and `save_state`/`load_state` (seq + rng). `color_mnist_io` overrides `get_outp_size` (RGB) and `save_state`/`load_state` (adds `current_color`). `mnist_io::process()` retains its NSE/MAX_VAL/AVG probes; `emnist_io`/`kmnist_io` do not gain them.
- **`dataset::cifar_io_base`** (`include/io/cifar_io_base.h` / `src/io/cifar_io_base.cpp`) — shared base for `cifar10_io` and `cifar100_io`. Holds image buffer, image_count, sequencer, RNG; implements the same five common methods plus `rebuild_sequencer()` called after file loading.
- **Standalone** (not folded): `dsprites_io` (HDF5 format, no shared structure); `chairs3d_io` (no sample_sequencer); `mnist_io_gpu` (GPU sibling of mnist_io — diverges entirely in output production, documented in the header as intentionally flat).
- **Null-guard fix in `dataset_json.h`**: all hand-written `from_json` functions now guard on `j.is_null()` before calling `j.value()`, matching the `!nlohmann_json_j.is_null()` check that `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT` always emits. Without the guard, a literal JSON `null` value would throw `type_error.304` instead of silently defaulting (the old macro behavior). Added to all seven hand-written `from_json` functions and to `from_json_sequence_fields`.
- **Verification**: all 53 tests pass (ctest outside sandbox, including all A-tier golden tests and GPU tests); save-state HDF5 structure confirmed via h5dump — field layout (`ci_cnt`, `cur_epoch`, `cur_idx`, `rng/*`, `seq_perm`, `seq_pos`, `seq_rng/*`) matches pre-refactor.

- [io]({{% relref "/architecture/modules/io" %}}) — shared base classes documented; dataset specifics table updated with base column.
- [File Reference]({{% relref "/reference" %}}) — `idx_io_base.h/cpp` and `cifar_io_base.h/cpp` rows added; per-loader rows updated to show inheritance.

## 2026-07-06 — Unified loader sequencing params (refactor/loader-params, commit 1 of 2)

Pure refactor, zero behavior change. The four sequencing fields shared by all seven index-dataset loaders (`change_interval`, `noise_range`, `random_seed`, `shuffle`) are now consolidated into `dataset::sequence_params_t` in `include/io/sequence_params.h`; each loader's `*_param_t` struct inherits publicly from it, keeping field access sites unchanged (`params.change_interval` etc.). JSON serialisation is now hand-written per family instead of NLOHMANN macros; shared fields delegate to `dataset::to_json_sequence_fields`/`from_json_sequence_fields`; byte-identical ordered_json output verified by a pre/post diff. `chairs3d` is excluded (no sample_sequencer). A follow-up commit (commit 2) will fold the loader implementations.

- [io]({{% relref "/architecture/modules/io" %}}) — directory layout updated; `sequence_params.h` and `chairs3d` exclusion noted.
- [File Reference]({{% relref "/reference" %}}) — `include/io/sequence_params.h` row added; `dataset_json.h` description updated.

## 2026-07-03 — HDF5 Browser (feat/h5-browser)

Documented the *h5-browser* cycle, which gave COAST an in-tool HDF5 log-file triage
tool (Analysis menu): an offline, read-only browser that opens one COAST-written
observe/buffer-log/probe-log file at a time and answers "is this run's data healthy
or garbage?" through presence/volume, per-dataset sanity stats, and distribution-shape
(column histograms + pairwise-distance/similarity histograms) tiers, plus a raw-data
table, structural search, composable row filtering, and a popup-free diagnostic log
panel. The domain layer is GUI-free and introduces the codebase's fourth thread.

**New pages**

- [h5browse]({{% relref "/architecture/modules/h5browse" %}}) — the GUI-free domain
  module: execution model (one scan thread funneling ≥4 MB chunk jobs through the
  shared `hdf5::executor`), the four measured production rules (shape-matched
  memspaces, multi-lane accumulation, selective chunk-cache disabling, depth-2
  pipelining), key structures, and the native-dtype policy.

**Corrected / extended pages**

- [gui_tools]({{% relref "/architecture/modules/gui_tools" %}}) — added the
  `h5_browser_window` panel alongside Metric Explorer (written during Develop,
  verified accurate this pass).
- [observe]({{% relref "/architecture/modules/observe" %}}) — the writer now stamps
  `type`/`schema_version` root attributes on new files (the Discover-recommended
  side task); files written before this stamp carry no `/metadata@type` and are
  rejected by the browser (no readable-fallback path).
- [tools]({{% relref "/architecture/modules/tools" %}}) — added `metric_registry.h`
  (D9b): enumerable metric metadata shared between the browser's pairwise view and
  a planned Metric Explorer refactor.
- [gui_core]({{% relref "/architecture/modules/gui_core" %}}) — Analysis menu now
  lists "HDF5 Browser..."; `h5_browser_window_` documented as a second persistent,
  sim-independent tool alongside `metric_explorer`.
- [Architecture overview]({{% relref "/architecture" %}}) — added `h5browse` and
  `hdf5` to the module map and category tables. Also closed a **pre-existing gap**
  from the prior cycle: `gui_tools` (Metric Explorer's home) was never added to this
  map either; both landed together here.
- [Threading]({{% relref "/threading" %}}) — revised from three to **four** threads;
  added the scan thread to the Thread Map and "What Runs Where", and a new §7
  covering two-lane command servicing and the `latest_value` mailbox (a deliberate
  divergence from `demand_published`, §4b — that primitive needs a producer that
  keeps re-offering after consumption, which an event-driven scan does not provide).
- [Testing / Tier Model]({{% relref "/testing/tier-model" %}}) — added the 11
  `tests/unit/h5browse/` executables (incl. the retained G6 malformed-file corpus and
  the kernel throughput regression pin) and `tests/unit/tools/test_metric_registry`.
- [File Reference]({{% relref "/reference" %}}) — added `hdf5/` (previously
  undocumented — pre-existing gap, now closed as a side effect of h5browse's direct
  dependency on it) and `h5browse/`; updated `tools/` and `gui_tools/` file rows.

**Known follow-ups (not this pass — see project memory / debrief)**: the `tests/`
file-reference table still omits `hd_ngm3`, `inhibition`, `gui_core`, and (partially)
`tools` unit-test subdirectories — a broader pre-existing gap, not introduced by this
cycle. `architecture/decisions.md` was left untouched: per the precedent set by the
(ADR-less) metric-explorer cycle, h5-browser's design decisions stay embedded in its
module pages rather than becoming new ADRs.

## 2026-06-14 — GPU abstraction layer (refactor/gpu_abstraction_layer)

Documented the *gpu-abstraction-layer* refactor, which removed WebGPU from the simulation's agnostic core. `env`, the GPU entities, and the `io_buffer` base now describe their per-step work as a backend-agnostic `step_plan` (POD) and drive it through a virtual `gpu_device` façade exposing only opaque handles (`buffer_id`/`kernel_id`/`plan_id`); a single translation unit, `backend_webgpu.cpp`, is the only code that sees `<webgpu/webgpu.h>`. The GPU work plan is built **once** per entity and is static across steps; per-step variation is uniforms-only plus the ping-pong generation. WebGPU remains the sole backend and behavior is unchanged (CPU and GPU goldens hold); the seam is shaped so native Metal/Vulkan backends can slot in without touching the agnostic side.

- [GPU Dispatch]({{% relref "/data-flow/gpu-dispatch" %}}) — rewritten around the `gpu_device` façade: `begin_step`/`enqueue(plan)`/`submit_step` (no wait) / CPU∥GPU overlap / `rendezvous_step` (the one mandatory per-step coherence wait), and `swap_io`'s `upload`/`advance_generation`/`device_to_cpu_readback`. The gated-branch pass replaces the old host-side conditional append.
- [core/gpu]({{% relref "/architecture/modules/core_gpu" %}}) — the agnostic `gpu_device.h` seam (`step_plan` POD + opaque handles) and the `backend_webgpu.cpp` backend that owns `gpu_context`; corrected ownership, `gpu_fwd.h` scope (the header has since been removed), and the TU/probe lists.
- [Heterogeneous Execution]({{% relref "/key-abstractions/heterogeneous-execution" %}}), [IO Buffer]({{% relref "/key-abstractions/io-buffer" %}}), [IO Entity]({{% relref "/key-abstractions/io-entity" %}}), [sim_core]({{% relref "/architecture/modules/sim_core" %}}) — `build_step_plan()`/`plan()`/`graph_epoch()` replace the per-step `record(encoder)`; the `io_buffer` base now carries an opaque `buffer_id`/`topology` seam (`gpu_register`/`gpu_page_id`/`device_to_cpu_readback`) instead of WGPU-typed virtuals.
- [Decisions (ADR)]({{% relref "/architecture/decisions" %}}) — ADR-3 annotated (record→build_step_plan), ADR-4 marked superseded (the two-submit encoder mechanism → façade). [File Reference]({{% relref "/reference" %}}) updated.

## 2026-06-13 — Quiesced state access (fix/state-io-sim-pause)

Documented the quiesced-state-access redesign (`design/quiesced-state-access.md`), which unified all cross-thread access to sim state onto two complementary mechanisms at the post-swap boundary: `sim_thread_controller::run_quiesced` for one-shot blocking operations (state save/load/checkpoint, `set_step_count`) and the existing post_swap hook system for periodic publication — now including the buffer-browser stats and status-dialog strings via `demand_published<T>`. The short-lived `pause_guard` quiescence hold was retired.

- [Threading]({{% relref "/threading" %}}) — rewritten around the three-thread model (GUI / sim / HDF5 executor), `run_quiesced` (§1b), and `demand_published` (§4b); removed `pause_guard`.
- [Logging Pipeline]({{% relref "/data-flow/logging-pipeline" %}}) and [logging]({{% relref "/architecture/modules/logging" %}}) — the `extract_state` / `serialize_state` / `parse_state` / `apply_state` split, the in-memory `mem_group` tree (`mem_state_io.h`), and the `build_env` startup-load exemption.
- [IO Buffer]({{% relref "/key-abstractions/io-buffer" %}}) and [sim_core]({{% relref "/architecture/modules/sim_core" %}}) — the `snapshot()`/`restore()` read-page pair and the write-page restore bug it eliminated.
- [Hook System]({{% relref "/key-abstractions/hook-system" %}}), [gui_core]({{% relref "/architecture/modules/gui_core" %}}), [Startup]({{% relref "/data-flow/startup" %}}), and the [File Reference]({{% relref "/reference" %}}) — updated accordingly.

## 2026-06-06 — Heterogeneous CPU/GPU execution (ngm3-gpu-port)

Documented the *ngm3-gpu-port* development cycle, which gave COAST heterogeneous CPU/GPU execution and the flat-SoA HD-NGM3 model. The site previously documented only HD-NGM2 and the CPU-only framework, so this pass spanned three tiers: correcting pages invalidated by the cycle, documenting the new GPU capability, and introducing the previously-undocumented CPU ngm3 model.

**New pages**

- [Heterogeneous Execution]({{% relref "/key-abstractions/heterogeneous-execution" %}}) — the conceptual anchor: entity-device rule, the four `io_buffer` variants, the two-encoder-phase dispatch, `gpu_io_entity`, `gpu_context`.
- [core/gpu]({{% relref "/architecture/modules/core_gpu" %}}) — wgpu lifecycle, WGSL loader, the stateless PCG PRNG, and the GPU build integration (then `COAST_WITH_WGPU`; later reworked into the `COAST_GPU_BACKEND` backend switch).
- [hd_ngm3]({{% relref "/architecture/modules/hd_ngm3" %}}) — the flat-SoA model, covering both the CPU `neuron_group_t` (full per-step algorithm) and the GPU `neuron_group_gpu_t`.
- [GPU Dispatch]({{% relref "/data-flow/gpu-dispatch" %}}) — a heterogeneous step traced end to end.
- [Decisions (ADR)]({{% relref "/architecture/decisions" %}}) — eight ADRs covering framework choice, the seam, the parity reframe, the S0 refactors, and the retracted alternatives.

**Corrected pages** (invalidated by the S0 refactor and the S3/S4 device-seam work)

- [IO Buffer]({{% relref "/key-abstractions/io-buffer" %}}) — rewritten: abstract base + four variants; removed the closure-access and per-buffer-stats descriptions.
- [IO Entity]({{% relref "/key-abstractions/io-entity" %}}) — `set_output_buffer`/`set_input_buffer`; added `gpu_io_entity`.
- [sim_core]({{% relref "/architecture/modules/sim_core" %}}), [hd_ngm2]({{% relref "/architecture/modules/hd_ngm2" %}}) (Refactor A), [io]({{% relref "/architecture/modules/io" %}}) (`mnist_io_gpu`), [tools]({{% relref "/architecture/modules/tools" %}}), [Architecture overview]({{% relref "/architecture" %}}), [Simulation Loop]({{% relref "/data-flow/simulation-loop" %}}), [Threading]({{% relref "/threading" %}}), [Golden Protocol]({{% relref "/testing/golden-protocol" %}}) (aggregate compare mode + the four ngm3 A-tier tests), and the [File Reference]({{% relref "/reference" %}}).

**Known limitation recorded:** on Strix Halo the GPU ngm3 mean runs ~16–25 % below the CPU reference (stateless-hash vs. mt19937 divergence) — documented and accepted, see `benchmarks/ngm3_aggregate_noise_floor.md`.
