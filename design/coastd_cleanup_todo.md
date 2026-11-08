# coastd cleanup TODO — deferred low-severity findings

Deferred items from the three-model review + four-agent verification of `feat/coastd`
(range `aab0401^..HEAD`). Tier 1–3 were fixed in this branch; the items below were
judged LOW (or explicitly excluded) and left for a later pass. Each is CONFIRMED
against the code with a file anchor and a concrete fix direction.

Severity uses the same normalization as the review: coastd's TCP command port and
UDP discovery port are **intentionally unauthenticated** for a trusted-LAN
deployment (`submit` is effectively RCE, a documented+accepted decision), which
**caps** the severity of network-facing DoS/disclosure findings — a caller who can
reach the port can already RCE, so a DoS there grants no new capability.

---

## Re-assessment (against commit 11b8bf8, after the Tier 1-3 fixes)

All items below were re-verified against the committed code and remain valid — none
became moot. Changes worth knowing:

- **Line drift** (my edits moved two anchors): #23/#24 client `discover()` is now
  `coastd_client.cpp:295-321` (recvfrom `:318`, `n <= 0` `:320`); #26 `event_queue`
  deque is `event_queue.h:85`. All other anchors are unchanged.
- **#20 is now a one-liner** — the `no_checkpoint_launches` field added by the #12
  fix (`types.h`) just needs to join the log-name key in `child_cmd.cpp:102`.
- **#29 signal-disposition is now cheaper** — `run_capture` owns a
  `posix_spawnattr_t` (added for #9's process group), so `POSIX_SPAWN_SETSIGDEF`
  there is a two-line add; `child_proc.cpp`'s coast spawn still needs its own attr.
- **#23 "client freeze" softened** — the #17 background poll moved the periodic
  refresh off the render thread, so `discover()` only blocks on an explicit Scan
  click now, never idle rendering.

**Recommended to promote out of "deferred":**
1. **#29 regression tests for the DoS mitigations** — highest value: a coverage gap
   for shipped, regression-prone timing logic (not a bug, but the cheapest way to
   keep the hardening honest). Do next.
2. ~~**#19 doc-comment half**~~ — **#19 is now DONE IN FULL** (2026-08-02), promoted
   out of "deferred" after the PR #130 review made the third-party-victim angle
   concrete. See below.

**Newly noted minor interaction (from #8 + #17):** if the background poller connects
**by hostname to a host whose DNS resolver is unreachable**, each ~2 s reconnect
spawns a detached `getaddrinfo` thread that lives until the resolver gives up
(~10-30 s), so up to ~15 can coexist transiently. Bounded, self-limiting, and only
on the broken-DNS-hostname path (localhost/IP resolve instantly) — negligible, but
recorded. Mitigation if ever needed: short-circuit repeated resolve failures in the
poll reconnect factory (`coastd_panel.cpp`).

---

## Excluded from this pass by request

### #19 — UDP discovery reflection/amplification + path leak — **DONE 2026-08-02**
`coastd/src/discovery_server.cpp`. The responder bound `INADDR_ANY`, matched any
datagram beginning with the fixed public token `coastd-probe-v1`, and replied with
the full `machine_summary` (hostname, OS, real `runs_root`/`repo_root`, port) to the
datagram's **source** address with no origin check and no rate limit — a
spoofed-source flood reflects the JSON blob at a third-party victim (~17×
amplification).

Originally trust-capped to MEDIUM because reaching the port already implies
`submit`-RCE reach. The PR #130 review is what moved it: that reasoning covers the
*prober*, not the *victim*. This is the one channel that reaches a host which never
contacted the daemon, so "they could have submitted anyway" does not apply to the
party actually receiving the traffic. Closed rather than deferred on that basis.

- **(a) DONE** — accepted-risk comment in `discovery_server.h`.
- **(b) DONE** — fixed-window reply caps, both global
  (`discovery_max_replies_per_sec`, default 20) and per-source
  (`discovery_max_replies_per_source_per_sec`, default 2). Both orders of magnitude
  above real use; the per-source window clear is also what bounds the map's memory
  under a forged-address flood.
- **(c) DONE** — `discovery_bind` selects an interface. An unparseable value
  DISABLES discovery rather than falling back to `INADDR_ANY`, which would silently
  do the opposite of what the operator asked.
- **(d) DONE, not originally listed** — `runs_root`/`repo_root` are stripped from
  the discovery reply. The TCP `machine_info` command still returns them; that path
  at least required a handshake. No consumer regressed: the console's discover
  printout never displayed them (only the configured-host branch did).
- **(e) DONE** — `discovery_enabled` turns the responder off entirely. Default stays
  **on**: with (b) and (d) in place the reply is just "a coastd exists here, at this
  port, speaking this version", and the GUI's discover button depends on it. Flip it
  off wherever the port is reachable by untrusted peers.
- Covered by `unit.coastd.discovery`, which asserts against a real socket and was
  mutation-checked (removing the strip and the caps fails it).

---

## Tier 4 (LOW) — deferred

### #20 — Log-file `O_TRUNC` collision on the "did work, no checkpoint" relaunch — DONE
`coastd/src/child_cmd.cpp:102`. The log name now keys on the full triple
`run_<steps_done>_<launch_failures>_<no_checkpoint_launches>.log`, so the
`did_work && !progressed` relaunch (which advances none of `steps_done`/
`launch_failures`) no longer collides and O_TRUNCs the prior attempt's log.

### #22 — Send-side slowloris: `SO_SNDTIMEO` resets per `send()`, no total-reply deadline
`coastd/src/comms_server.cpp:253-261` (`write_all`), `:364` (`SO_SNDTIMEO` set once).
A slow reader draining a few bytes per timeout window holds a connection slot
indefinitely — the read side was fixed with a total `line_assembly_timeout` via
`poll()`; the write side has no equivalent. Trust-capped; bounded by `max_conns=128`.
- **Fix:** track a `steady_clock` deadline across the `write_all` loop (mirror
  `line_reader::next`'s shrinking-budget `poll()`), treat overrun as a dead connection.

### #23 — `discover()` has no overall collection deadline
`coastd/src/client/coastd_client.cpp:225-260`. `SO_RCVTIMEO` is set once but resets
on every received packet, so a chatty/hostile LAN peer keeps `discover()` (and the
GUI "Scan" click that calls it) from ever returning. Trust-capped, but also a
client-freeze on the render thread.
- **Fix:** track a hard wall-clock deadline for the whole collection loop,
  independent of per-packet arrival.

### #24 — `discover()` treats a valid 0-byte datagram as EOF
`coastd/src/client/coastd_client.cpp:250` — `if (n <= 0) break;`. For `SOCK_DGRAM`,
`recvfrom() == 0` is a legitimate empty datagram; the server side correctly uses
`< 0` (`discovery_server.cpp:65`). A single spoofed 0-byte packet truncates results.
- **Fix:** change to `n < 0`; treat `n == 0` / unparseable as "ignore, keep waiting."

### #25 — `close()`-from-another-thread to unblock `accept()`/`recvfrom()` is macOS-unreliable
`coastd/src/comms_server.cpp:463-489`, `coastd/src/discovery_server.cpp:76-89`.
Reliable on Linux, a documented weak spot on BSD/macOS kernels; no self-pipe/eventfd
fallback. Worst case on macOS is a hang on daemon shutdown. macOS is already flagged
project-wide as secondary/unverified for coastd (`coastd/service/README.md`).
- **Fix (only if macOS support is promoted):** self-pipe/`eventfd` wakeup, or bound
  the blocking accept/recv with `poll()` + a `running_` re-check loop.

### #26 — No per-connection RPC rate limit + unbounded event queue (PARTIAL / overstated)
`coastd/src/comms_server.cpp:430-451`, `coastd/include/daemon/event_queue.h:81`.
Originally raised as a CRITICAL queue-flood; verification downgraded it: the comms
loop is **synchronous** (one command in flight per connection), so reachable queue
depth is bounded by `max_conns=128`, not unbounded. Trust-capped to LOW.
- **Fix (optional):** a small per-connection token-bucket on RPC-issuing commands,
  and/or a soft depth cap on `event_queue` with backpressure — only worth it if the
  port is ever exposed beyond the trusted LAN.

### #28 — `stop()` runs one full rescan/reconcile/dispatch pass — **DONE**
`coastd/src/core.cpp` `run()` loop. Closed: the loop re-checks the shutdown latch
after the `pop_until` wake, before rescan/reconcile/dispatch. (The same change set
also replaced `running_` with a latched `stop_requested_`, because a `stop()` that
landed BEFORE `run()` was scheduled used to be overwritten by `run()`'s
unconditional `running_.store(true)` — leaving the daemon spinning on a closed
queue, spawning children, and never joinable.)

### #29 — Assorted small items
- **Dead `task_kind::commit`** — `coastd/include/daemon/events.h:36`,
  `coastd/src/core.cpp` `on_worker_result` `case task_kind::commit: break;`. No
  producer (audit commits go through `job_store`'s `committer_` jthread). Drop the
  enum value + case.
- **CLOEXEC set-after-create race** — **DONE for the daemon.** `cloexec_pipe` /
  `cloexec_accept` / `cloexec_socket` in `proc_util` use `pipe2(O_CLOEXEC)` /
  `accept4` / `SOCK_CLOEXEC` on Linux with the set-after fallback elsewhere; the
  daemon's pipe, listen socket, accepted sockets and discovery socket all go
  through them. The macOS-portability rationale for deferring was retired by
  `1a96f434` (Linux-only daemon). `coastd_client.cpp` still sets-after — it is a
  client that spawns few children, and it must build on macOS.
- **`posix_spawn` doesn't reset signal disposition** — `coastd/src/child_proc.cpp:47`,
  `coastd/src/proc_util.cpp`. Children inherit coastd's `SIGPIPE=SIG_IGN`
  (`main.cpp`). Benign for git/cmake/coast. Add `POSIX_SPAWN_SETSIGDEF` for hygiene.
- **`int64_t`→`pid_t` narrowing at boot** — `coastd/src/core.cpp` `recover_at_boot`.
  Unvalidated `static_cast<pid_t>(j.pid)` from on-disk state; mitigated in practice
  by the `start_token` start-time cross-check in `is_same_process`. Add a range check.
- **Config-size TOCTOU** — `coastd/src/comms_server.cpp:161-173`. `fs::file_size()`
  checked, then `ss << in.rdbuf()` reads the whole file regardless if it grew.
  Needs local write to that exact path (≈ already-compromised). Re-cap during read.
- **`is_under` fails open on `weakly_canonical` error** — **DONE.** Now tri-state
  (`yes`/`no`/`unknown`): a plain bool could not fail closed, because the two call
  sites read "inside" in opposite senses (one PERMITS the job's own run dir, one
  REJECTS the daemon roots), so whichever value meant "could not tell" would open
  one of them. `unknown` neither exempts nor clears. (Still open: forbidding all of `runs_root` blocks the
  legitimate pattern of chaining a new job's input from a prior run's artifact —
  worth a carve-out for read-only input references.)
- **Regression tests for the DoS mitigations** — `tests/unit/coastd/test_comms.cpp`.
  **DONE for four of five.** The comms bounds were promoted to `daemon_config`
  (config-file tunable), which unlocked fast timeout tests. Gates now cover:
  `max_config_bytes` (oversized config rejected before it is read), `max_conns`
  (a >128 flood is refused and the daemon survives), `first_request_timeout` (a
  connect-then-silent client is reaped), and `line_assembly_timeout` (a slowloris
  partial line is dropped) — the last two run against a test coastd.json that sets
  both timeouts to 1 s. **Still uncovered:** `send_timeout` — even though it is now
  configurable, the daemon's replies are too small to fill a kernel send buffer, so
  a non-draining-reader test cannot force the send() to block; it stays review-only.

---

## Note — items already handled while fixing Tier 1–3

Two nominally-Tier-4 items were fixed in passing because they were coupled to
in-scope work (both verified by the passing test suite):
- **`write_attr(string_view)` 1-byte OOB read** (F14/L5) — fixed alongside #16
  (`include/logging/hdf5_utils.h`): the value is now copied into NUL-terminated
  storage before `H5Awrite`.
- **Core scan deadline on `system_clock`** (L2) — fixed alongside the review of the
  event loop: `event_queue::pop_until` now takes a `steady_clock` deadline, so a
  backward NTP step no longer stretches the ~1 Hz recovery backstop.


---

## Review round 2026-07-26 (two-model review of `feat/coastd`) — resolved

Every finding below was fixed on this branch, one commit each; the notes record
what was measured rather than assumed.

- `--save-on-term` restored `SIG_DFL` before the checkpoint was actually written
  (`cp->write` only submits; the drain, the log close and the manifest all follow).
  With coastd re-emitting SIGTERM ~1 Hz for a cancelled job, the second signal
  killed the process mid-drain. The restore is now an RAII guard that outlives the
  whole teardown.
- Client-side: the two parses reading REMOTE bytes (TCP reply, UDP discovery
  datagram) skipped `proto::json_depth_ok`; `type_error` could escape
  `list()`/`machine_info()`/`host_config::load` despite the header's
  "returned, never thrown" contract; `connect_with_timeout` re-applied the full
  timeout per resolved address.
- Job children were spawned into coastd's process group, so a terminal Ctrl-C
  killed every running job with no `--save-on-term` checkpoint.
- `job_store::load_all` renamed unreadable files out of the directory it was
  iterating (the duplicate-stray path already deferred this, with a comment saying
  why); discarded `directory_iterator` error_codes made an unreadable bucket read
  as "0 jobs" and an unreadable snapshots dir read as "no checkpoints" — the latter
  restarting a long run from step 0.
- `run_git`'s timeout branch used a blocking `waitpid` after SIGKILL, undoing its
  own bound; `run_capture` buffered child output without a cap.
- The build self-heal never pruned a stale worktree registration. Reproduced on a
  scratch repo: `worktree remove --force` fails rc=128 when the tree's `.git` file
  is missing, the registration survives `remove_all`, and every later `worktree add`
  is refused rc=128 — permanently for that SHA. `prune` clears it.
- GUI panel: `PushID(i)` let a background refresh re-target Cancel at a different
  job; `adopt()` committed the session before the throwing handshake.
- Tests: the real-coast gate passed with ZERO cases run when its filter matched
  nothing (doctest 2.4.11 exits 0 with "SUCCESS" — verified), and had no build edge
  to `coast`; a `WIFEXITED(0)`-is-true fallback leaked a daemon; bare `std::thread`
  + `REQUIRE` aborted the binary on first failure (measured: SIGABRT/exit 134, 2 of
  6 cases reached, vs a clean failure with an RAII owner); `atomic_write` gained
  its first direct test, mutation-proofed against the old fixed tmp name.

### Deliberately NOT changed
- **No idle-session timeout** on comms (`comms_server.cpp`). 128 sockets that send
  one valid request then go silent still hold their slots. The tradeoff is
  documented in the header and an idle cap would break long-lived console sessions;
  raising it is a design decision, not a bug fix.
- **Submit-time symlink TOCTOU** between config validation and child launch. Within
  the documented trusted-LAN threat model, and the config-bytes TOCTOU it is often
  confused with is already closed by construction (configs travel as content).
- **Loggers warn where `state_logger` aborts** on a failed attribute write. A
  severity judgement (an unresumable checkpoint vs a log that still has its data),
  now stated at each of the three sites.
