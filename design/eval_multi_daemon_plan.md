# Plan: the eval suite on several `coastd` daemons

Branch `eval/upgrade-multi-daemon`. Companion to `design/eval_python_toolkit_brief.md`,
whose rules (§7) apply: no absolute or machine-specific paths in committed files, a bad
input raises, `pytest` plus one real short run per step. Except: this feature lands as one
PR, not the brief's small PRs (§5).

## 1. Goal

`scripts/eval_ab.py --ref …` runs one comparison on **N ≥ 1** daemons, with the same
guarantees it gives today on one local daemon:

- every input file a run read is the pinned one (SHA256), and
- every run directory the report reads is the one the daemon wrote for that job.

The number of daemons is a throughput setting. It must not change a result. §4 D4 says
what "not change" means and how it is measured.

## 2. What prevents it today

| # | Where | What |
|---|---|---|
| 1 | `scripts/eval_ab.py:251` | One `Client` from one `--ports-file`; every ref, seed and condition goes to it. |
| 2 | `daemon.py` `Client.from_ports_file` | Host defaults to `127.0.0.1`; the CLI passes none. |
| 3 | `runner.py:91-98` | `run_milestone1` refuses a daemon whose `hostname` differs from the client's. This is a guarantee, not an accident: input verification and `check_data_link` only prove anything on the daemon's own filesystem. The error names the intended fix, "daemon-reported hashes". |
| 4 | `runner.py` `ship_inputs`, `check_data_link`; `eval_ab.py:253` | `dataset_base` and `runs_root` are used as local paths. |
| 5 | protocol | `load_state_from` names a job id known only to the daemon that ran it: a training job and its frozen trials cannot be split across daemons. |
| 6 | `eval_ab.py:254-271` | Refs run one after another; even one daemon idles between a ref's last frozen job and the next ref's first training job. |
| 7 | ops | A daemon of an older build writes no `<data-root>/coastd.ports`; the TCP port may be firewalled from the eval host. |

## 3. Assumptions, and how each is checked before it is relied on

| | Assumption | Matters for | Check (phase 0) |
|---|---|---|---|
| A1 | The same job (ref, config, seed, steps) gives **bit-identical** output on every daemon host. | D4: whether daemon assignment can bias an A/B result. | Run one training job and one frozen trial from it, same ref and config, on two daemons; `h5diff` `final_state.h5` and every `*.observe.h5`; compare the report's measures computed from each. |
| A2 | The eval host can read every daemon's `runs_root` and `dataset_base` through a shared filesystem. | D2, D3. | From the eval host, list a remote daemon's run dir of a known job id and byte-compare its `config.submitted.json` with the submitted text. |
| A3 | Hashing the full pin set (train + test + 15 MNIST-C pairs, ≈ 180 MB) on the daemon takes seconds, not minutes. | D2: timeouts; whether to cache. | Time `sha256` of the pinned files on the daemon host, cold and warm cache, on the shared filesystem. |
| A4 | The eval host can reach each daemon's TCP port (directly or through `ssh -L`). | D1. | `machine_info` from the eval host to each daemon. |
| A5 | Builds per SHA on each daemon are independent: N daemons build the same ref N times, ≈ 4 min each. | Wall-clock expectations only. | Observed in the phase 0 runs. |

If A1 fails, D4 still keeps the comparison valid (the design does not depend on A1), but
the report must then say that seeds ran on different hosts, and pooling across daemons
needs a host term. If A2 fails, D3's fallback applies.

**Phase 0 results (2026-11-07, eval host `cachyos-desktop`, daemons majorcrunch and
captncrunch, ref `cbd1f27b`).** Precondition found missing: the daemons' `dataset_base`
(`/mnt/NAS/datasets`) held only the MNIST train split; `scripts/deploy_eval_data.sh`
(this branch) now deploys all 34 pinned files, and running it is an ops step per
`dataset_base`.

- **A1 holds for this pair, conditionally.** Seed 0, 2 000 steps, then all 17 frozen
  conditions on 24 test images, on each daemon: all 1 163 datasets in the 52 HDF5 files
  are identical (`h5diff`, no tolerance); only the `/metadata` `timestamp`/`created`
  attributes differ. Both hosts build with GCC 16.2.1 and `-march=native` (top-level
  `CMakeLists.txt`), which resolves to `znver5` on both CPUs. The two `coast` binaries
  are **not** byte-identical (their `.text` differs too, cause not established), so the
  identical output was observed, not guaranteed by construction. A daemon with another
  compiler version or `-march` target must be expected to diverge (A-tier goldens differ
  ~99 % across toolchains). Consequence: D7 must record the compiler and the resolved
  `-march` per daemon, and D4 stays as decided.
- **A2 holds.** Every run dir, read through the NAS on the eval host, has the
  `config.submitted.json` text submitted for it, a `data` link to `/mnt/NAS/datasets`, and
  a manifest passing `check_provenance`. The resolved `config.json` differs between hosts
  only in `snapshot_dir`, an absolute path into the run's own dir.
- **A3 holds.** `sha256sum` of the 34 files (172.7 MB) on the daemon host: 0.32 s /
  0.52 s with `O_DIRECT` (no client cache; the NAS's cache was warm), 0.11 s / 0.14 s
  from the page cache (majorcrunch 10 GbE / captncrunch 5 GbE; both CPUs have SHA-NI, so a
  portable vendored SHA256 will be slower — measure in step 1).
- **A4 holds.** `machine_info` from the eval host to both daemons, after the `ufw` rules
  for `192.168.100.0/24`.
- **A5:** a new ref costs majorcrunch ≈ 45 s of build (training 2 000 steps: 5 s total on
  captncrunch, which had the build), not the ≈ 4 min estimated.

## 4. Decisions (decided by the owner 2026-11-07: D1, D2-impl, D4)

**D1 — naming the daemons.** `--daemon HOST[:PORT][=LOCAL_RUNS_ROOT]`, repeatable; without
a port, coastd's default (51900); `--ports-file` stays as the one-local-daemon shorthand.
The daemon index used by D4 does **not** depend on the command line (owner, 2026-11-07):
the daemons are sorted by what each reports about itself, `(hostname, tcp_port)`
(`runner.canonical`), and a daemon named twice is refused. `--discover` lists the daemons
that answer a UDP probe, sorted, with the matching `--daemon` arguments; it does not run
anything, so a run's daemons are always named explicitly (owner, 2026-11-07). *Alternative:* the coastd client's existing `hosts.json` list
(`docs/content/coastd/protocol.md` §Discovery) via `--hosts NAME…`. Reuse is attractive,
but that file has no order and silently falls back to localhost when malformed, which the
eval must not do. **Decided:** `--daemon`.

**D2 — proving which input files a remote run read.** Replace the hostname check
(§2 #3) by evidence from the daemon itself:

1. *coastd:* `machine_info` reports `dataset_base` (the path string, additive field).
2. *coastd:* a new command `hash_inputs {paths: [rel…]}` returns the SHA256 of each path
   **relative to `dataset_base`**, computed by the daemon on its filesystem. Paths that
   are absolute, contain `..`, or resolve (after symlinks) outside the canonical
   `dataset_base` are refused. The count and the total bytes are capped.
3. *eval:* before submitting, `hash_inputs` on every daemon for the pinned files and the
   shipped inputs (`eval/<name>/…`); every hash must equal its pin or the shipped file's
   local hash. After the last job settles, the same call again: equal hashes before and
   after bound the window in which a file could have changed under a run.
4. *eval:* `check_data_link` compares the run dir's `data` link target with the
   **reported** `dataset_base` string (not a local `resolve()`).

*Alternatives considered:* (a) coast records the SHA256 of every dataset file it opens in
`coast_runtime.json` — the strongest form (it is what the run read, with no window), but a
change to every loader in the core; worth doing later, and D2 would then shrink to a
manifest check. (b) Trust that the same NFS export at the same path is the same data —
not evidence; rejected. (c) Hash over `ssh` from the eval host — adds a second access
path besides coastd, contrary to the brief's "the runner talks only to coastd"; rejected.

**D2-impl — SHA256 in coastd. Decided: a vendored single-file implementation**, tested
against the NIST vectors (≈ 150 lines, no dependency). The C++ code has none today; the
eval side keeps Python's `hashlib` (`datasets.sha256_file`) and compares the daemon's
hashes with the same `PINS` table. *Rejected:* OpenSSL `libcrypto` (a new build
dependency for every configuration); spawning `sha256sum` / `shasum -a 256` (a
per-platform command and its output to parse).

**D3 — reading remote run dirs.** The eval host reads run dirs through the shared
filesystem at the path the daemon reports (`machine_info.runs_root`), optionally mapped:
`--daemon HOST:PORT=LOCAL_RUNS_ROOT` when the mount point differs. Identity is checked per
run, not assumed: the directory's name is the job id, and its `config.submitted.json`
must equal, byte for byte, the `config_text` the eval submitted. *Fallback if A2 fails:*
a coastd file-fetch command, out of scope here.

**D4 — assigning work to daemons. Decided.** This is placement only: the seeds
themselves — their values, their number, how each enters its job's config — are
unchanged; a seed runs the same jobs on whichever daemon it is placed on. The unit of
placement is the **chain** (ref, seed): one training job and all frozen trials from it
(forced by §2 #5). N is the number of daemons, daemon `i` the `i`-th in `canonical` order (D1).
Chain (ref, seed) goes to daemon `seed mod N`, the **same daemon for every ref**.
Example: seeds 0–4 on N = 2 place seeds 0, 2, 4 on daemon 0 and 1, 3 on daemon 1, for
both refs. So within a seed, the two
states of an A/B comparison ran on the same host, whatever A1's outcome, and a host
effect cannot masquerade as a ref effect. Cost: load balance is per seed, coarse when
seeds < N or not divisible by N. *Alternative:* dynamic least-loaded assignment balances
better but makes the host a function of timing, so the ref/host pairing is no longer
controlled. Rejected unless A1 holds and is recorded as a protocol rule.
Placement expects the seeds to spread evenly over the residues mod N, as consecutive
seeds do (the default 0–4, or any run of N or more): seeds that share a residue (`0 2 4` on
two daemons) all land on one daemon. That stays correct, only slower; `eval_ab.py` prints
the placement before it submits.

**D5 — orchestration.** One loop over all chains of all refs on all daemons: submit every
training job, poll each daemon's `list`, submit a chain's frozen trials the moment its
training job is `done` (with the provenance and data-link checks of today), finish when
every job is `done`. It replaces the per-ref `run_milestone1` sequence and also removes
§2 #6 for a single daemon. Plain functions over immutable records; the state of the loop
is one value passed through, no module state (brief §7).

**D6 — failure.** Fail fast, as today, with one addition: a `list` that fails to connect
is retried for a bounded time (default 60 s) before it is fatal, because a poll lasting
hours meets transient network errors that a one-host run did not. Any other error, or the
retry budget running out, raises with the daemon, the job ids still active there, and the
`cancel` command to stop them, or to re-attach by running the same evaluation again.

**D6, requests (owner, 2026-11-07): no deadline on a reply.** How long coastd takes to
answer depends on its load, its disk and the NAS, so a per-request timeout (it was 30 s)
guessed. A request now waits on TCP keepalive instead: the daemon host's kernel answers the
probes while coastd works, and on an idle connection a host that goes down or is cut off
fails the request after 10 s + 3 x 5 s of silence (`daemon.Client`). Two cases keepalive
does not cover, both bounded by the OS instead: a connect to a host that is already down
waits for the SYN retries (~2 min on Linux defaults), and a request sent but not yet
acknowledged waits for TCP retransmission (~15 min, `tcp_retries2`). A connection the daemon closes without a reply
(it stopped or restarted) counts as unreachable, which `list` retries. A daemon that is
alive but never answers is waited for: interrupt and resume (D8). Not measured: the
dead-host case, which needs a firewall rule or a pulled cable to provoke.

**D8 — resume (owner, 2026-11-07).** An evaluation that stops for any reason (Ctrl-C, a
timeout, a crash, a failed job) is resumed by running the same command again.
`<out>/ledger.json` names the run: a run id, drawn and written (atomically) before the
first submit, and a fingerprint of what decides the jobs (states and templates, seeds,
steps, conditions, the inputs and their hashes, the daemons, and the config texts made from
them, so that an edit of the toolkit between invocations is refused rather than adopted:
review 4). Every job is tagged on its
daemon with `coast-eval <fingerprint> <run id> <label/seed/condition>`; the job ids live
only on the daemons, which keep a job's message across restarts (review 3: a job list in
the ledger as well was redundant). Run again with the same ledger, the runner re-attaches:
a job whose tag a daemon lists as active or done when this invocation starts is not
submitted again; a done one is checked and used, an active one waited for. One that had
failed or been cancelled by then is submitted afresh (so running again is also the retry);
a job that fails from then on stops the run, as in a fresh one. A ledger with another
fingerprint is refused, and an exclusive lock on `<ledger>.lock` refuses a second
evaluation on the same ledger. Removing the ledger (or another `--out`) starts the
evaluation afresh: its jobs carry the old run id. The cost: a lost ledger cannot be resumed
(review 2, findings 1 and 2). The same command with `--cancel` cancels every active job of
the evaluation, of any run, with or without the ledger (owner, 2026-11-07). The inputs are
verified again at every start, so the window D2 bounds runs from the first start to the
last job of the last invocation. The aim: no timeout or failure costs the compute already
done. A done job whose run dir fails a check would be adopted again at every start and
refused again (review 4). Every refusal therefore names the job's key (`label/seed/
condition`), in the runner's checks and in the report's, and the same command with
`--redo KEY` submits that job afresh; the newest job of a key is the one adopted, so later
runs use the redone one (owner, 2026-11-08: "a crash is a crash"; no attempt to make a run
crash-proof). Known causes: a wrong `LOCAL_RUNS_ROOT` (corrected on the command line, no
redo needed), and a crash (`kill -9`, power loss) during a frozen trial, which leaves its
observe files unreadable or behind (HDF5 is not crash-safe and the writer flushes only on
exit; measured in the live kill test). A clean interruption, SIGTERM through coastd's
`--save-on-term`, needs no redo: the trial resumes from its final checkpoint and continues
its observe files (SPEC.md S.7.19). A frozen trial resumes like training because
`restore_sequence_from_checkpoint`, which stood in for "this run starts afresh", is replaced
by the top-level `run_id` (coastd fixes "auto" once per job), recorded in each checkpoint
(SPEC.md S.5.14). Every run must record the `frozen` its job asked for, so a ref that
predates it is refused (no backward compatibility, owner 2026-11-07).

**D7 — provenance in the report.** Each state records, per seed, the daemon's hostname and
the coastd version that ran it (from `machine_info` and the manifest's hostname, which
must agree). Additive fields in `report.State`; the Markdown names the host per seed only
when there is more than one.

## 5. Steps, in order

**One branch, one PR** (owner, 2026-11-07): every step is a commit series on this branch
(`eval/upgrade-multi-daemon`), and the PR is opened only once the feature is finished.
Each step: Plan → Build → Verify; full test suite before every push.

0. **Phase 0, no code committed: measurements (done, §3).** A1–A5, results into the PR
   body and a pointer in `policecar/roast` (brief §7: reports are not committed here).
1. **C++ — `coastd`: `dataset_base` in `machine_info`, `hash_inputs` (done).** SHA256 per D2-impl;
   path safety reusing the existing path-safety helpers; caps. Tests: NIST vectors;
   `test_comms` cases for a good path, a missing file, `..`, an absolute path, a symlink
   escaping `dataset_base`, cap exceeded; protocol doc. Additive: no `proto_version` bump;
   an old daemon answers the unknown command with `ok:false`, which the eval reports as
   "daemon too old for multi-daemon".
2. **Py — `daemon.py` (done):** `Client.hash_inputs`, `MachineInfo.dataset_base`, parsing of the
   `--daemon` spec, `verify_daemon_inputs(client, expected)`. Tests against a fake daemon
   socket (the existing `test_daemon.py` pattern).
3. **Py — `runner.py` (done; `eval_ab.py` adapted to it with one daemon).** Real check: two
   coastd instances of this branch on one host, 2 states × seeds 0–2 × 2 000 steps × 17
   conditions on 24 images, 108 runs in 73 s; seeds 0, 2 on daemon 0 and 1 on daemon 1; the
   two states (same ref) bit-identical per seed; seed 0 bit-identical to captncrunch's phase
   0 runs (a third znver5 / GCC 16.2.1 host). Not decided by the plan and chosen here:
   `--timeout-s` now bounds the whole run (default raised from 4 h to 24 h); `connect`
   requires the runs_root to exist on the eval host before any submit. Planned: chains, D4 assignment, D5 loop, D3 identity check, D6 retry;
   the hostname check goes, its guarantee now carried by D2. Tests: assignment table for
   N ∈ {1, 2, 3} and seeds not divisible by N; a chain never spans daemons; frozen jobs go
   to the training job's daemon; a `config.submitted.json` mismatch raises; the retry
   budget; two fake daemons end to end.
4. **Py — `eval_ab.py` + report + docs:** `--daemon`, D7 provenance, brief and
   `docs/content` updated. Validation: the brief's real short run (3 seeds × 2k steps × 24
   indices) on **two** daemons, and the same on one daemon; with A1 holding, the two
   reports' measures are identical.

**Step 4 (done).** `--daemon`, `--discover`, per-seed hosts and toolchains in the report,
the brief updated. Validation: before = `main` d91ea215, after = c153832c, 3 seeds × 2 000
steps × 24 images × 17 conditions, on captncrunch alone and on majorcrunch + captncrunch:
all 822 measure values, the deltas and the directions identical, every delta 0. Found on
the way and fixed in coastd: a burst of submissions for a commit the daemon's clone had not
fetched could fail a job with "could not resolve ref" (the fetch throttle did not make the
burst wait for its one fetch). The review (coast-review-lean) then found that waiters
fetched again one after another once a fetch outlasted the window; fixed in 20191da3. Both
daemons run a coastd whose C++ is that of 20191da3.

*Hashes in these notes:* c153832c (the validation's "after") and the daemons' build commit
753d400a are commits of this branch before its rebase onto `main` bbcfda85; the rebase
changed no C++ line.

**Ops, before step 4's validation** (not in the repo): every daemon the eval uses runs a
build of this branch (it needs step 1); the eval host reaches each daemon's port (firewall rule for the eval host,
or `ssh -L`); `dataset_base` and `runs_root` of each daemon visible from the eval host (A2).

## 6. Out of scope

Fetching run dirs over the
protocol; daemons without a filesystem shared with the eval host; daemons that build
with different `build_cmake_args` (e.g. GPU backend). Identical `build_cmake_args` on
every daemon is an ops precondition: the manifest records `coast_build_type` but not
the CMake arguments, so the eval cannot check it from the run dirs today.
