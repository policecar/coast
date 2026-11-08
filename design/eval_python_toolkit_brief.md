# Brief: COAST evaluation toolkit in Python (issue #82)

**Status on 2026-11-08: merged.** The toolkit is on `main` — package `eval/src/coast_eval/`, tests `eval/tests/`, configs `eval/configs/`, drivers `scripts/eval_*.py` (from e207e557; latest #285, 0bcd5c14). `--pre-206-keys` was removed in 0f5338e3. The body below is the pre-merge brief; its planned layout and status sections are historical.

Written 2026-11-03 for the agent that implements this. Replaces the Julia harness
(`roast/`, PRs #131, #136, #242, #243 — all closed unmerged). Read this whole file,
then `CLAUDE.md`, before writing code.

## 1. Decision and why

Language: **Python**, replacing the Julia harness. It matches the repo's existing analysis
tooling (`scripts/*.py`: uv script headers, h5py/numpy/matplotlib) and plays to Python's
strengths for this workload: subprocess glue, HDF5 reads and small-matrix statistics.

Size: the priority is an **excellent evaluation toolkit for COAST** — correct, reliable,
readable by its reviewer. There is no hard line cap. The constraint is that every line
must be one the owner cares about: it computes a measurement, defines the protocol,
reads or writes a run, or tests one of those against an independent reference. The
Julia version was 10k lines for ~550 lines of statistics because it spent the rest
re-verifying what `coast` already guarantees and testing its own provenance machinery.
The ~1.5k-line figure below is an expectation for milestone 1, not a budget: if the
code is well past it, check what the extra lines are for before continuing.

Division of labour (see §3 for the open C++ question):

| Concern | Owner |
|---|---|
| Running a simulation, checkpoints, run manifest, observe output | `coast` (C++, exists) |
| Running many simulations unattended, from a git ref, with resume | `coastd` (C++, exists) |
| Finding and reading a directory of runs | `campaign` loader (C++, exists) + `h5py` |
| Expanding an experiment into runs (seeds × conditions) | Python, thin |
| Metrics, references/controls, report | Python |
| Reports | generated, **not committed here**: kept in the private `policecar/roast` |
| Plots | Python/matplotlib, **generated, not committed** |

## 2. What `coast` already provides (verified on `main` 97c235df)

CLI (`./build/coast --help`): `-n <steps>`, `--seed-offset <int>`, `--load-state
<h5>`, `--save-state <h5>`, `--checkpoint <N>`, `--emit-run-manifest <path>` (writes
`coast_runtime.json`: git SHA + dirty flag, seed offset, steps requested/completed,
wall clock, hostname, observe config + output list), `--log-buffers`.

Config (`include/json_cfg/dataset_json.h`): datasets take `image_file`, `label_file`,
`index_manifest` (explicit presentation order), `change_interval` (steps per image),
`noise_range`. Neuron groups take `"frozen": true` (inference only). `observe.groups`
selects sites and step ranges; output is `observe/<group>.observe.h5`.

Observe sites for a CPU population (after #241): `population/in`, `population/out`,
`population/out_pre_inhibition` (float32 per neuron) in the `population` group, and
`population_structure/representation_count` (uint64) in its own `population_structure`
group, written only when a run enables that group (the count is a serial scan of every
segment id, too costly to pay for in every run observing `population`). The
producer-owned list is `tests/integration_observe/observe_contract.json`.
`scripts/load_observe.py` shows how to read the HDF5 layouts.

Do **not** re-verify in Python what `coast` already guarantees. Provenance is: read
`coast_runtime.json`, check `coast_git_sha`, `coast_git_dirty == false`,
`steps_completed == steps_requested`. That is ~10 lines, not a module.

## 3. Open question: how much orchestration goes into C++

Owner's preference: as much as sensibly possible. What exists and what is missing:

**`coastd`** (`coastd/`, docs `docs/content/coastd/`): a per-machine daemon. `submit`
takes `ref` (git ref, resolved to a SHA and built natively) + `config_text` +
`steps_target` + `priority`; it queues, launches `coast` with `--checkpoint`,
`--save-state <final_state.h5>`, `--emit-run-manifest`, and re-queues/resumes from the
newest checkpoint after a crash. One directory per run (`config.json`,
`coast_runtime.json`, `observe/`, `final_state.h5`).

**`campaign`** (`include/campaign/campaign_api.h`): opens such a directory tree, finds
the swept parameters by diffing configs, hands out run handles (`info()` = manifest,
`observe_path(group)`), never opens HDF5 until asked. Reachable from the REPL and GUI.

**What the frozen protocol needs that `coastd` does not have (verified by grep of
`coastd/src`, 2026-11-03):**

1. **A job that starts from another job's final state.** `--load-state` is only used on
   resume from a job's *own* checkpoint (`child_cmd.cpp:245`). The frozen trials must
   load the training job's `final_state.h5`. First check whether the *config* can name an
   initial state file; if not, the minimal C++ change is a `load_state` field on `submit`
   (path, validated inside the daemon's data/run roots like the existing path-safety
   rules) — estimated ~50–100 lines plus a `test_comms` case. One small PR.
2. **Job dependencies** ("run B after A succeeds"). Not present. Doing it properly
   touches `reconcile`, the persistence model and the protocol together — the owner's
   stated principle is that these are one state machine, so this is a designed change
   (est. 300–500 lines + tests), not a bolt-on. **Recommendation: do not build this
   now.** The client (Python) submits training jobs, polls `list` until they are `done`,
   then submits the frozen jobs. ~30 lines of Python; revisit if a protocol needs
   hundreds of dependent jobs.
3. **Matrix expansion** (seeds × corruptions × conditions → N configs). Keep in Python:
   it is the experiment definition and changes per protocol. ~40 lines.
4. **Seed per job**: `--seed-offset` is a CLI flag; check whether `coastd` exposes it or
   whether the seed must go into `config_text`. If neither, same small PR as item 1.
5. **Data on the daemon machine**: datasets are read by path; `machine_info` reports
   `data_link_ok`. MNIST-C must be provisioned there (currently in the main checkout's
   `data/mnist_c/`, 15 corruptions, hashes pinned in `roast/configs/mnistc_ab_sha256.json`
   at `477f346e`).
6. **Local use without a daemon**: not needed once PR 1 below lands — `coastd` runs on
   the macOS dev box too. Keep the Python side free of subprocess spawning.

**Owner's decision (2026-11-03): use `coastd` from the start.** Two C++ PRs precede
the Python runner, in this order:

- **C++ PR 1 — `coastd` on macOS.** The daemon is gated off macOS by ADR-16
  (`docs/content/coastd/boundaries.md` §"macOS"), not by a technical blocker: the boot
  pid-guard needs a recycle-safe process start token and only `/proc/<pid>/stat` is
  implemented. Port = implement `read_start_token()` for Darwin
  (`proc_pidinfo(pid, PROC_PIDTBSDINFO, …)` → `pbi_start_tvsec`/`tvusec`, or
  `sysctl({CTL_KERN, KERN_PROC, KERN_PROC_PID, pid})` → `kp_proc.p_starttime`) and add
  `__APPLE__` to `start_token_supported()`, both in `coastd/include/daemon/child_proc.h`
  / `coastd/src/child_proc.cpp` — the single platform seam, as the header says. The
  startup gate opens by itself. Verification: `unit.coastd.recovery`,
  `fault.coastd.matrix`, `unit.coastd.console` currently skip on macOS; they must run
  and pass. Update the boundaries page and add an ADR superseding ADR-16's macOS
  clause. `--install` (systemd unit) and unattended launchd operation stay out of scope.
- **C++ PR 2 — `submit` accepts `load_state` (+ `seed_offset` if not already
  expressible).** Item 1/4 above. Path validated against the daemon's existing
  path-safety rules; a `test_comms` case; protocol doc updated.

Then the Python runner talks only to `coastd` (`eval/daemon.py`). The local-spawn path
(item 6) is dropped unless a concrete need appears.

What **not** to push into C++: metric computation, controls, the report. Those are
numpy one-liners that will change as #82 evolves; a C++ version would be slower to
iterate and would not be reviewed faster.

## 4. The protocol to implement (milestone 1; training rate and length, test images and settle lead revised 2026-11-06)

One ngm25 CPU population, proximal MNIST input (template:
`roast/configs/templates/mnist_ngm25_frozen_eval.json` at `477f346e`, 50 neurons —
copy it to the new location; later protocols will need the shipped config sizes).

- **Train**: seed s ∈ S, `-n` 1M steps, `index_manifest` reshuffled every pass
  (`shuffle`, as in the shipped configs; the order differs per seed),
  `change_interval` = 1 step/image (the shipped MNIST configs' rate), `--save-state`.
  1M steps pass over the 60k training images about 16.7 times; the report says so. The
  length comes from the saturation runs (`261106-1827-saturation-pixel-shuffled` in
  `policecar/roast`): 50k is badly under-trained, and the measures nearly flatten between
  700k and 1M.
- **Frozen trials**: for each (seed, condition): `--load-state <seed checkpoint>`,
  `"frozen": true`, 1000 held-out test images (`protocol.evaluation_sample`: a fixed random sample below 9000), 15 steps/image, observe
  `population/out` (and `in`), drop the settle lead (the hop count from the dataset to
  the group: 1 for pixels, 2 with a retina) and average the remaining steps of the
  presentation (2..15 for pixels), code = relative winner rule `amp >= 0.3 · max(amp)`;
  non-positive max → empty code.
- **Conditions**: clean (identity), all 15 MNIST-C corruptions, all-zero input (chance
  control). The corruption files must have the pinned SHA256 before any run.
- **Primary metrics**: (a) Kendall τ-a between pixel-cosine and code-cosine pairwise
  distances over clean images (ties count against agreement); (b) per corruption,
  cosine AUC of matched (clean code vs own corrupted code) against mismatched (vs other
  images' corrupted codes); 0.5 = chance; summarised as mean-of-15 and worst-of-15.
- **Secondary** (reported, not judged): clean separation (mean cosine between codes of
  different images), never-wins fraction, domination (max presentation share of one
  unit). Separation and domination rise with code size alone, so they are read against
  their chance levels: each presentation's units shuffled, code size and amplitudes kept
  (`261106-1923-collapse-retina-1m` in `policecar/roast`).
- **Controls**: all-zero input; raw-pixel reference on the same AUC; random WTA
  projection of the pixels with code sizes matched *per training seed* to that seed's
  clean code sizes, projection seeds 1..5 (nested in the training seed — never counted
  as replicates).
- **Reading rule for A/B comparisons**: a measure "moved" only if all seeds move in the
  same strict direction. No thresholds, no aggregate verdict, no significance claim.

The port's cross-check, passed 2026-11-05 (PR #243, before `f5f1d360`, after `52463b64`,
5 seeds, 50k steps at 5 steps/image, lead 2, indices 0..199): τ-a down in 5/5; mean-of-15 AUC down in 5/5; model
below its matched random reference (0.77 vs 0.78 → 0.74 vs 0.79); pixel reference 0.89.
Same binaries, same template and same indices gave the Julia numbers to float tolerance.
This shows that the port computes what the Julia code computed. It does not show that
#243's protocol was right: #243 is not a golden result, and the before/after comparison
runs again under the current protocol (§10). `eval_ab.py` no longer selects those indices,
so the exact reproduction needs the code before the sampled test images.

## 5. Port sources (pinned; do not import Julia, re-implement and keep the reference tests)

At `origin/feat/eval-foundation` `2bdf3aa0`:
- `roast/src/metrics/ngm_eval/second_order.jl` — τ-a with O(n log n) merge count; port
  as `scipy.stats.kendalltau(variant="a")`-equivalent and keep the independent
  quadratic reference test.
- `roast/src/metrics/ngm_eval/metrics/similarity.jl` — matched/mismatched AUC, cosine
  over the winner union (zero-norm convention: similarity 0; empty union: 1 for
  separation). Port the conventions exactly; they are what makes the #243 numbers
  reproducible.
- `roast/src/metrics/ngm_eval/metrics/constraints.jl` — structural bound for
  `representation_count` from the config.
- `roast/src/metrics/ngm_eval/codes.jl` + `frames.jl` — window/reduction/winner rule.
- `roast/test/upstream/ngmeval_second_order.jl`, `metrics.jl` — the reference tests.
- `roast/README.md` — the "establishes / does not establish" table; copy into the docs.

Do not port: `provenance/`, `orchestration/run_spec.jl` (run-id hashing),
`paths.jl`, `cached.jl`, `_code_digest` — `coast_runtime.json` and the run directory
are the provenance.

## 6. Deliverables, as small PRs in this order

1. `eval/` package skeleton: `pyproject.toml` (deps and tooling per §8), `eval/metrics.py` with the ported kernels + reference tests
   (`tests/eval/`), `eval/observe.py` (read codes from an observe file; reuse
   `scripts/load_observe.py` layout knowledge). No spawning yet. Expect ~500 lines.
2. `eval/runs.py`: run-directory layout (the one `coastd` writes), manifest check (≈10 lines), MNIST/MNIST-C IDX readers with SHA pins.
   `eval/protocol.py`: matrix expansion for milestone 1. Expect ~400 lines. No
   subprocess spawning here — runs go through `coastd`.
3. C++ PR 1 (`coastd` on macOS) and C++ PR 2 (`load_state` on `submit`) from §3, then
   `eval/daemon.py` (submit / poll `list` / locate run dirs) ~100 lines. PRs 1–2 of
   this list and the two C++ PRs are independent and can run in parallel.
4. `eval/report.py` + `scripts/eval_ab.py`: per-seed table, deltas, reading rule,
   JSON + Markdown. Reproduce the #243 result as the port's cross-check; reports and
   figures are generated, never committed here (§7). ~300 lines.
5. Docs page `docs/content/evaluation/frozen-evaluation.md` (a new page in a new Evaluation section; `main` has no Julia one),
   `cd docs && hugo` zero warnings.

## 7. Rules (non-negotiable for this work)

- Code rules, language-independent: pass context explicitly (no module-level state, no
  globals mutated by tests); a test covers a term only if its inputs make that term
  vary; a bad input raises — never coerce a bad file or reply into a default; small PRs.
- **No absolute or machine-specific paths in any committed file** — the Julia report
  JSON embedded `/Users/p/...` 30 times. All paths relative to a data root passed in.
  `git grep -n '/Users/\|/home/'` must be empty before every commit.
- **Reports are not committed here** (decided 2026-11-06). A report is generated output:
  nothing here reads it, and a two-state comparison is about 2,500 lines. The full JSON
  and Markdown go to `reports/<experiment>/` in the private `policecar/roast`, which also
  holds the findings and the planned experiments. This brief holds only how the toolkit
  and protocol work: a rule chosen from a measurement goes here, with a pointer to roast.
- **No `.svg`, `.png` or `.pdf`
  anywhere in the repo** — `main` tracks no image files outside the Hugo theme and the
  docs use Mermaid; keep it that way. Figures are regenerated from a report's numbers
  by `scripts/eval_plot.py` into the run's output directory (ignored). The first toolkit
  PR adds `*.svg`, `*.png`, `*.pdf` (with `!docs/static/**`) to `.gitignore` and says so
  in one sentence of its body — it is a repo-wide policy, so the reviewer sees it.
- PR bodies: `### Context / ### Change / ### Risk / ### Validation`, short; no
  nanosecond-mtime theatre. State what was run and the result.
- `tmp/` for scratch, never `/tmp/`. Data via symlink to the main checkout's `data/`.
- Python style: functional over object-oriented — plain functions over immutable records
  (`msgspec.Struct`); a class only where state and the operations on it genuinely belong
  together (the coastd client). Type hints everywhere, pyright strict; `argparse` for
  CLIs; numpy/h5py; `uv` for environments; pytest for tests; no mocks except at the
  `coast` subprocess boundary (a fake binary that writes a run directory is fine for
  runner tests; metric tests use synthetic matrices with known answers).
- Validation of every PR: `pytest`, and for PRs 2+ one real short run against
  `build-headless/coast` (3 seeds × 2k steps × 24 indices is enough to prove the
  pipeline; it is not a result).

## 8. Kickoff (decided 2026-11-03)

- **Method**: Plan → Build → Verify per PR (not the six-stage dew). One worktree and one
  agent per PR; the agent reads this brief and a one-screen task brief, writes a short
  plan for the medium-clarity PRs (C++ 2, Py 2–4) before building.
- **Branches / worktrees** (`.worktrees/<handle>`, branch `feat/<handle>`, off `main`):
  `coastd-macos` (C++ PR 1), `coastd-load-state` (C++ PR 2), `eval-python` (Py PRs; this
  brief is its first commit). C++ 1, C++ 2 and Py 1 are independent — run in parallel.
  Py 2–4 follow Py 1 and C++ 2. The runner's real-run validation needs #241 in the binary.
- **Review**: every PR gets `coast-review-lean` from a different model than its author
  before it is opened; that review also reports the PR's line count against the §6
  expectation and explains any excess.
- **Python packaging and libraries** (decided 2026-11-03): `eval/pyproject.toml` as
  its own `uv` project (the repo root stays CMake-only); `requires-python = ">=3.13"`;
  CLIs as `uv run` entry points; tests under `eval/tests/`, `uv run pytest`.
  Runtime deps, exactly these five: `numpy` (2.x), `h5py`, `scipy` (`rankdata` /
  Mann-Whitney AUC with ties — no scikit-learn), `msgspec` (all manifests, report
  schema, config structs: strict by default, unknown fields and wrong types raise —
  no pydantic), `matplotlib` (figures, never committed). Kendall τ-a is hand-ported
  (scipy has only τ-b/τ-c). stdlib `json` with `sort_keys` for canonical hashing;
  stdlib `socket` for the coastd JSON-lines client; `argparse` for CLIs. No dataframe
  library in milestone 1; if report logic outgrows numpy + msgspec structs, add
  `polars`, not pandas. Dev deps: `pytest`, `hypothesis` (property tests of the metric
  kernels against quadratic references, with ties), `ruff` (lint + format), `pyright`
  strict.
- **Historical binaries for the #243 reproduction**: archived at
  `~/src/_cimc/coast-archive/october-ab-binaries/coast-{before,after}` with `SHA256SUMS`
  (matching the #243 body: before `14cea2a6…`, after `4715dc4f…`). Rebuilding "before" =
  detached worktree at `7ec9e0b4` + the ngm25 observe hunk of `b292b5b4` (recipe in the
  #243 body).
- **MNIST-C provisioning**: `scripts/fetch_mnist_c.py` (#247) downloads MNIST-C,
  converts it to IDX and verifies the pins from `roast/configs/mnistc_ab_sha256.json`
  at `477f346e`.
- **C++ test runs**: `unit.coastd.*` and anything `_gpu` run outside the Claude Code
  sandbox (CLAUDE.md). C++ PR 1's oracle is the three coastd tests that currently skip
  on macOS.

## 9. Carry-forwards for Py 4 (report and the #243 reproduction)

Findings from PRs 1–3 and the retina templates that the report PR has to act on. Each
item says what holds and what the report does about it.

1. **Settling window per template.** The settle lead is the hop count, the longest
   dataset → group path in the config's node graph (`protocol.settle_lead`: 1 for the
   pixel template, 2 for the retina templates), and the window is the remaining steps
   of the presentation. Each hop is one buffered step, so the group sees an image `hops`
   steps after the dataset presents it (a retina adds one, measured on `population/in`,
   #258), and the settling curves (`reports/*-settling-*` in `policecar/roast`) show no settling beyond the
   hop for this model. `scripts/eval_settling.py --template` measures any template, with
   its reference window derived from the template's lead (steps h+2..15 for h hops).
   Re-measure after the model or a template gains temporal state.
2. **`activity_learning_window` in the template.** Since #171 coast tracks the window
   and ignores this key with a warning. Done: the template dropped it in #247; the #243
   binaries' outputs are unchanged without it (#268). A run on a post-#171 binary is a
   different model and is not expected to match #243.
3. **The zero-input control gives full codes.** With all-zero input the peak amplitude
   is baseline neuron noise (≈ 0.045), so under the relative rule every unit wins every
   presentation. Report the control as what it is, not as an empty code.
4. **Random-WTA reference.** #243's 0.78 / 0.79 came from Julia's MersenneTwister and
   `randn`, which numpy cannot replay; it matches only statistically. Every other
   kernel follows the Julia conventions, so "match to float tolerance" excludes this
   one reference.
5. **Float differences against Julia.** The winner threshold is computed in float64,
   as in Julia. The settle mean is averaged in float64 and rounded to float32, where
   Julia averaged in Float32: differences of about 1 ulp, which can flip a unit that
   sits exactly at the threshold.
6. **The #243 "before" binary is not a git ref.** It is `7ec9e0b4` plus one hunk of
   `b292b5b4` (§8). coastd builds only refs, so the reproduction runs both archived
   binaries through a driver script outside `eval/`; the library still spawns nothing.
7. **Seeds of frozen trials.** After `--load-state` the restored random state wins and
   `random_seed_offset` does not apply to restored entities (#239, #249 §4), although
   the manifest reports the configured seed. Frozen trials from one `final_state.h5`
   therefore differ by condition only; replication comes from the training seeds. Do
   not read a frozen run's manifest seed as its effective seed.
8. **`representation_count` and `population_structure`.** The site is in its own group
   since #241 and no protocol config requests it. Done: no report measure uses it, so
   `check_representation_counts` and `ngm25_representation_count_bounds` and their
   tests were deleted in #268.
9. **τ-a reference across templates.** Pixel cosine stays the common reference for all
   three templates, so their numbers are comparable. Add a retina-output column only if
   the pixel-referenced numbers diverge.
10. **Before Py 4 starts.** Py 3's real run used a daemon built from a local merge of
    #244 and #245. Done: with both merged, its recipe was re-run on `main` at `f5cde7e5`.

## 10. After the stack lands (decided 2026-11-06, for jk to see)

Findings and planned experiments are in `policecar/roast`; its README is the index. The
comparisons wait for the stack. Until it lands, a sample comparison only checks that the
pipeline works, and its numbers are not results.

**Recycled training images.** At 1M steps every training image repeats about 16.7 times, so
training reshuffles every pass, as the shipped configs do; one fixed order would be the same
for every seed. Frozen trials keep the manifest's order. The saturation curve is the same
under reshuffling (`261106-1827-saturation-pixel-shuffled` in `policecar/roast`), so 1M stands. The shipped configs also add
input noise, which the protocol leaves at zero. A retina's output noise stays on in frozen
trials: it is the retina's designed null signal, coast refuses `[0, 0]`, and it accounts for
about half of retina's larger codes (`261106-2102-settling-retina-1m` in `policecar/roast`).
A pixel-vs-retina comparison therefore also compares input without and with a noise floor.

**Rerun the length sweep once the stack has merged**, on the pixel, retina ON/OFF and divisive
normalisation templates and past 1M. The 1M default comes from the pixel template alone, and
the flattening from 700k to 1M is not shown to be a plateau: τ-a was also flat from 200k to
300k, then rose by 0.08.

**Several daemons** (`design/eval_multi_daemon_plan.md`). `scripts/eval_ab.py --ref …` runs on
every `--daemon HOST[:PORT][=LOCAL_RUNS_ROOT]` (port default 51900) instead of the one
`--ports-file` daemon; `--discover` lists the daemons that answer on the local network and runs
nothing. Seed `s` runs on daemon `s mod N` for both states, the daemons sorted by hostname and
port, so within a seed both states share a host (seeds should spread evenly mod N, as
consecutive seeds do; `0 2 4` on two daemons all land on one, which is correct but slower); the report names the host per seed when
there is more than one, and warns when their toolchains differ (results are bit-identical
across hosts only for the same compiler and `-march=native` target, plan §3 A1). Each daemon
must run a `coastd` with `hash_inputs`, and its `dataset_base` must hold every pinned file:
`scripts/deploy_eval_data.sh <dataset_base>` puts them there. The eval host must read each
daemon's `runs_root` (a shared filesystem), and `--dataset-base` is where it ships the inputs,
which every daemon then has to report with the expected hashes. An interrupted run is resumed
by running the same command again: `<out>/ledger.json` names the run, and its jobs are
re-attached instead of submitted anew; the same command with `--cancel` stops every active
job of the evaluation, ledger or not (plan D8).

**`--pre-206-keys` stays** until the October pair has run under the current protocol, because
the "before" binary silently ignores the current key name and would run at the wrong tree
depth.

**Status on 2026-11-06.** Merged to `main`: #244 (`coastd` on macOS) and #245 (`load_state_from`
on `submit`). Nothing of the Python stack has merged. Open for review, in landing order: #246
(toolkit), #247 (runs, protocol), #248 (`coastd` client), #268 (report, A/B driver) and #270
(settling, protocol). Review is requested from jk on #246–#248, #268 and #270, and #246 needs
its approval before anything can land. #258 (retina templates) and #274 (this follow-up) are
drafts.

**To merge.** #258 is not an ancestor of #268. The combined stack, #270 with #258 merged in,
passed CI on 2026-11-06 at `47fe3051`, before #258 cleared its templates' observe windows.
#258 and #270 conflict in one small hunk of `eval/tests/test_protocol.py`,
to resolve when the second of them lands. Once both are on `main`, #268's synthetic retina
graph test becomes the two file-based parametrize entries.
