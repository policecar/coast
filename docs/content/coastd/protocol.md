+++
title = "Protocol & Clients"
weight = 40
+++

# Protocol & Clients

## The wire

**JSONL over TCP.** One minified JSON object per line, `\n`-terminated, on port
51900. `--tcp-port 0` lets the kernel choose instead (CLI only, for tests); the
daemon writes the ports it actually bound to `<data-root>/coastd.ports` —
`{"pid":…,"tcp_port":…,"udp_port":…}` — once it accepts connections, and
`machine_info` and discovery replies advertise that bound port.

The daemon listens on **IPv4 only**, and the client library resolves host names to
IPv4 only to match. With both address families, `localhost` resolves to `::1`
first, and the client keeps the first address that accepts — so any IPv6-only
service on the daemon's port number (rpc-statd, the NFS lock manager) would take
the connection instead of the daemon.

```
request   {"cmd":"<name>", ...args}
reply     {"ok":true,  ...data}
          {"ok":false, "error":"<human message>"}
```

That is the entire envelope. `daemon/protocol.h` is the single source of truth for
it, and — decisively — it is **header-only and depends on nothing but
nlohmann/json and the standard library**. Both the server (inside `coastd_core`)
and the client (`coastd_client`) link the same header, so the two ends cannot
drift out of a compiler-checked contract, and a frontend that wants to talk to a
daemon pays for sockets and JSON rather than for the entire simulator.

Line orientation is a deliberate feature: the whole surface is
`printf … | nc`-speakable, which makes both ad-hoc operation and scripted B-tier
tests trivial.

```sh
printf '{"cmd":"machine_info"}\n' | nc -q1 localhost 51900
printf '{"cmd":"list"}\n'         | nc -q1 localhost 51900
```

### Commands

| Command | Arguments | Reply |
|---|---|---|
| `submit` | `ref`, `config_text` **required**; `config_name`, `label`, `message`, `steps_target`, `priority`, `load_state_from` optional | `{"ok":true,"id":"<job-id>"}` |
| `list` | — | `{"ok":true,"jobs":[…]}` — all jobs, all buckets |
| `cancel` | `id` | SIGTERM → grace → SIGKILL, move to `cancelled/` |
| `reorder` | `id` + `pos` **or** `delta` | move a *pending* job: `pos` = absolute 0-based slot; `delta` = relative to its live position (`delta` wins if both are sent) |
| `machine_info` | — | self-description, incl. `data_link_ok` and `dataset_base` |
| `hash_inputs` | `paths`: array of paths relative to `dataset_base` | `{"ok":true,"sha256":{"<path>":"<hex>",…}}` — see below |

Two notes with teeth:

**`steps_target: 0` means unbounded.** A job submitted without a step target runs
forever. This is intentional, and it is the easiest way to burn a machine's CPU
for days by accident. Always set it unless you truly mean "until cancelled".

**`reorder` overwrites explicit priorities.** It renumbers the whole pending queue
positionally (`N..1`, using the same comparator `reconcile` schedules with), which
is the deliberate price of expressing an exact drag-to-position with a single
integer. A job that must always jump the queue should be re-reordered, not given a
large submit-time priority and expected to survive an unrelated reorder.

### DTOs are a projection, not a state dump

`job_summary` and `machine_summary` are the **client's view** — a deliberately
reduced projection of the internal `job`. They omit `pid`, `start_token`,
`run_dir`, and `launch_failures` on purpose: the wire is a *status surface*, not a
state dump. `status` and `bucket` cross as plain string tokens so the header stays
free of the daemon's internal vocabulary.

One field is client-populated only: `machine_summary::address`, the source address
a discovery reply arrived from. The daemon cannot know how a client sees it, so it
is never sent on the wire — `discover()` fills it from `recvfrom()`.

#### Two fields that answer questions a client cannot answer itself

**`job_summary::cancel_requested`** — a cancel has been accepted, the child has
been SIGTERMed, and the daemon is waiting out `term_grace` so it can write a final
checkpoint. Without this the job reads plain `running` for up to 10 s and then
jumps to `cancelled`, so the operator's click looks like it did nothing — which is
the Discover §5 cancel criterion ("near-instant *or* show it in progress")
failing. It is a **flag, not a `cancelling` status token**: `status` is the job's
real, persisted status, and there is no `cancelling` state in the store (reconcile
derives the kill from this very flag). A wire-only token would invite clients to
match on a value the store never holds. Clients render it themselves — the console
prints `cancelling` in the STATUS column, the GUI panel does the same and disables
the button.

**`machine_summary::data_link_ok`** — will a config's conventional relative
`data/…` path resolve on that machine? True iff the operator set `dataset_base`
*and* its target exists (a configured-but-missing target yields a dangling link,
i.e. a job that still dies at dataset load, so configuration alone is not
enough). This is the "well-known data location" the client actually needs: not
the path — which is useless to a client, since the config travels as content and
runs on the daemon's filesystem — but the yes/no that lets it warn *before*
submitting instead of after a build and a failed load. Not cached: the rest of the
description is static, but this bit flips when the operator mounts the share.

`machine_summary::dataset_base` is the target every run dir's `data` link gets: the
configured path, made absolute against the daemon's working directory exactly as the
link site does (`""` if unset). A client still cannot open it, but it can compare a run dir's `data` link
against it: that is how a client proves a run read the directory whose files
`hash_inputs` vouched for. Discovery replies omit it, like `runs_root` and
`repo_root`.

All three are additive — an older client ignores the keys and behaves exactly as before.

### `hash_inputs`: which bytes will a run read here?

A client on another machine cannot hash the daemon's datasets itself, so the daemon
does: `hash_inputs` returns the SHA-256 of each named file under `dataset_base`,
read on the daemon's own filesystem (`daemon/hash_inputs.h`, vendored FIPS 180-4
implementation in `daemon/sha256.h`). The eval suite calls it before submitting and
again after the last job settles, and compares the hashes with its pins.

Every path is relative to `dataset_base` — the frame in which a run's `./data/<rel>`
resolves. The whole request is refused, naming the path, for an absolute path, a
`..` component, a path that resolves after symlinks outside the canonical
`dataset_base` (a symlink that stays inside is followed), anything that is not a
regular file, more than 256 paths, or more than 4 GiB in total; both caps are
checked before any file is read. There are no partial answers. An older daemon
answers `unknown command 'hash_inputs'`.

The hashing runs on the connection's handler thread, so it blocks only that
connection. Measured on Zen 5: 0.65 s for the eval's 34 files (173 MB) from the page
cache, about 265 MB/s — so the 4 GiB cap is ~16 s, which a client's reply timeout
must cover.

## Config travels as content

{{% callout info %}}
**`submit` carries the config's bytes in `config_text`. There is no "path on the
daemon host" field, and deliberately no fallback to one.**
{{% /callout %}}

This is the most consequential change made during implementation, and it is a
**documented deviation** from the design document, which specified `submit{ref,
config, …}` where `config` was a path.

The design never confronted the fact that the client and the daemon are, in the
general case, **different machines** — even though remote operation is the
daemon's entire premise. Across that boundary a path is meaningless. And it is
worse than meaningless: a path that happens to exist on *both* hosts with
*different content* submits a job that **silently runs the wrong config,
successfully**. Nothing on either side could detect it. Two developer boxes with a
`~/work/configs/mnist.json` each is not a contrived scenario; it is the expected
one.

A wrong answer that looks right is worse than a failure. So the path form is not
offered at all, not even as a fallback — *an option to reintroduce a defect is the
defect.* The server rejects a legacy `config` key with an explicit migration
message rather than a bare "field required".

There was also a second, independent forcing reason: **the GUI could not submit at
all** under the path model. Its config editor holds an edited buffer that may
correspond to no saved file, so there is no path it could ever have sent.

### Why a JSON string, not a nested object

The bytes travel as a JSON *string*. Two reasons, both measured:

**Byte fidelity end to end.** The daemon archives exactly what it received as
`config.submitted.json`. Carrying a nested object would apply nlohmann's
reserialization — reordered keys, normalized indentation — to the wire as well,
leaving no copy of the user's actual file anywhere. (Recall that
`config.json` is *already* a rewrite, since `snapshot_dir` is injected.)

**String escaping is lossless and JSONL-safe.** Newlines become `\n`, so the
request stays on one line while indentation and layout survive exactly. Measured
on a real 8 KB config with 243 newlines and 4,312 indent spaces: single line, yes;
round-trip byte-exact, yes; **~8.4 % escape overhead** — versus ~33 % for base64,
which would buy nothing that is not already true.

One adjacent hazard: nlohmann's `dump()` **throws on invalid UTF-8** inside a
string. Such a file is not valid JSON either, so the outcome is a rejection
regardless — but clients pre-flight-parse the config locally so the user sees a
clean "not valid JSON" instead of a raw `type_error.316`. Clients always send the
**original bytes**, never a re-dump of what they parsed.

The cost, accepted knowingly: `submit` is no longer comfortably hand-typeable. An
ad-hoc caller inlines the file with `jq`:

```sh
jq -cn --arg ref main --rawfile cfg ./configs/mnist.json \
   '{cmd:"submit", ref:$ref, config_text:$cfg, config_name:"mnist.json", steps_target:500000}' \
  | nc -q1 <server> 51900
```

`--rawfile` is what keeps the formatting intact. The other four commands remain
trivially `nc`-speakable.

## Path-safety validation

Before a submission is accepted, the connection edge recursively scans the config
JSON for any string denoting a path that lands inside a daemon-critical directory
(`state_root`, `worktrees_root`, or another job's run directory). The scan is
**key-agnostic on purpose** — it does not trust field names, it inspects values.

`repo_root` is deliberately *not* forbidden, since configs legitimately reference
datasets that live there.

The **relative-path frame** is a second documented deviation from the design. The
design resolved relative paths against the *config file's parent directory*. But a
relative path in a config is resolved by the **child**, and the child runs with its
working directory set to the run directory. Under the old frame, a relative path
could never land under a forbidden root — meaning relative paths were effectively
**unchecked**. Resolving against the run-dir frame (via a `runs_root/__pending__`
sentinel, because the real job id is not minted until the core creates the job)
closes that hole: paths inside the job's own run directory are legitimate, and one
that escapes it into another job's directory is rejected.

There is no submit-config TOCTOU, by construction: the exact bytes the edge
validated are carried through to `prepare_run_config`. Nothing is re-read between
validation and use.

## Starting from another job's final state (`load_state_from`)

`load_state_from` names a **job id**; the new job's **first launch** loads that
job's `final_state.h5` (a frozen evaluation starts from a trained model). It is an
id, not a path, for the reason configs travel as content
([ADR-10]({{% relref "decisions" %}}#adr-10--ports-are-not-the-wire-the-configs-content-is)):
the daemon resolves it to the file itself.

The connection edge checks it **before the job exists**, and any failure is the
usual `{"ok":false,"error":…}` reply; a present `load_state_from` is never dropped.
It must be a non-empty string naming a job **in `done/`**, whose run directory holds
a regular `final_state.h5`.

The job file records the source id (`load_state_from`), the resolved canonical
path of that file (`load_state`) and the step it holds (`load_state_step`), so a
restarted daemon reproduces the command line without resolving anything again.
The step is the source job's recorded `steps_done`. For a `done` job that is its
manifest's `n_steps_completed`, and `--save-state` wrote the final state at that
same step: both are coast's step count at exit.

**`steps_target` counts from the loaded state.** coast continues the loaded
state's step count, so the job's checkpoints and its manifest's
`n_steps_completed` count from `load_state_step`. The daemon therefore stores
`steps_target` as `load_state_step + steps_target`: a job submitted with
`steps_target: 3000` from a job that finished at step 50 000 shows
`steps_target: 53000` in `list`, and its `steps_done` starts at 50 000 and counts
in the same space.
Every completion and resume decision compares those two numbers, so they must
share one step space. `steps_target: 0` stays 0 (unbounded).

**Only the first launch loads it.** A resume from the job's own checkpoint wins,
exactly as for any job ([Launch and resume]({{% relref "persistence" %}}#launch-and-resume)).
If the file is gone by the first launch, the job is filed under `failed/` with a
`failure_reason` naming it — starting fresh would run a different experiment under
the same job id.

The seed needs no field: each `sim.nodes` entry's `random_seed_offset` in the
config is added to coast's `--seed-offset`, for every entity type, so a per-job
seed travels in `config_text`. The run manifest's `seed_offset` then reads 0; the
effective offsets are in `config.submitted.json`.

This applies to a job that starts fresh. A `load_state_from` job continues the
random state saved in the source's final state, so `random_seed_offset` has no
effect on it: two such jobs that differ only in the seed produce identical runs.
Trials from one final state must differ in their config, such as their input. The
run manifest's `node_seeds` can still report the config's seed for a dataset node
of such a job (#249). The config may change settings such as `frozen`, but the
model structure must match the source job.

## Edge hardening

The daemon binds `INADDR_ANY` with **no authentication** (see
[Boundaries]({{% relref "boundaries" %}}#no-authentication-at-all)). The bounds
below defend against *accidental and malformed* input — they are not a substitute
for network access control.

| Bound | Default | Closes |
|---|---|---|
| `max_connections` | 128 | Connection flood; beyond it new fds are closed until handlers drain. |
| `max_request_bytes` | 10 MiB | An unbounded/binary stream buffered without limit. Sized to clear the `2 × config_cap + 64 KiB` floor below with headroom. |
| `max_config_bytes` | 4 MiB | A config rejected *before* it is read into memory. |
| `first_request_timeout` | 30 s | Connect-then-never-send pinning a slot forever. Applied only to the first read, so an idle persistent session survives. |
| `line_assembly_timeout` | 30 s | Slowloris. `SO_RCVTIMEO` resets *per byte*, so a one-byte-at-a-time dribble would otherwise hold a slot indefinitely; this bounds the whole in-progress line. |
| `send_timeout` | 30 s | A non-reading peer parking a handler thread in `send()` — times `max_connections`, that starves every real client. |
| `max_json_depth` | 64 | **Stack overflow.** nlohmann's DOM parser recurses once per nesting level, so an adversarial `[[[[…]]]]` line — still within the byte cap — could `SIGSEGV`, which no `try`/`catch` can defend against. An O(n), string-aware pre-scan rejects it *before* the parser ever recurses. |

Two ordering constraints are enforced rather than documented-and-hoped:

**`max_request_bytes` must exceed `max_config_bytes`**, because a submit carries
the config inside its request line. If the line cap were smaller, a config passing
the config cap would be killed by the line cap first — a legal input failing at
the wrong layer with the wrong message. `clamp_edge_bounds()` raises the line cap
to `2 × config_cap + 64 KiB` (the 2× covers JSON escaping plus the envelope) and
**warns**, because a silent fixup would hide the operator's mistake and the
symptom would get misdiagnosed as a daemon fault.

**TCP keepalive is on** (60 s idle, 10 s interval, 3 probes) so a peer that *dies*
while connected is detected in ~90 s and its slot freed — without disconnecting one
that is merely idle, since a live peer's kernel answers probes even when its
application sends nothing.

A client error — malformed JSON, an unknown command, a mid-write disconnect — is
answered with an `{"ok":false,…}` line or a dropped connection, **never a daemon
crash**. `SIGPIPE` is ignored process-wide for the same reason.

## Discovery

Two layers, and the priority between them is deliberate.

**Fixed-list is the deterministic primary.** A client-side JSON file names known
daemons, so a frontend reaches `gpu-box` by name with no broadcast at all:

```json
[{"name":"gpu-box","host":"10.0.0.5","port":51900}]
```

Located via `$COASTD_HOSTS`, else `$XDG_CONFIG_HOME/coastd/hosts.json`, else
`~/.config/coastd/hosts.json`. A **missing file is not an error** — it yields a
config containing just the built-in `localhost` entry. A malformed one produces a
warning and still returns the localhost default, so a frontend stays usable. A
`localhost` entry is always present, synthesized if the file omits one.

**UDP is soft convenience.** Model P — the *client* broadcasts a probe
(`coastd-probe-v1`) on port 51901 and daemons answer; the daemon is quiet by
default. It is convenience because broadcast is routinely blocked across VPN
segments, so it can never be the reliable floor. A UDP bind failure at startup is
logged and discovery is simply disabled; a TCP bind failure is fatal, because a
daemon nobody can reach is useless.

## The client library

`coastd_client` is what a frontend links. One blocking TCP socket per
`connection`; each call is a single synchronous JSONL round-trip.

**Errors are returned, never thrown.** Every call takes a `std::string& err`;
`nullopt`/`false` means "see err".

**There is no client-side subscriber state.** A frontend stays fresh by **polling
`list()`** on a ~2 s timer. This is the stateless-resync model, and it is what
makes the client a genuinely detachable controller: quit the GUI, come back, and a
fresh `list()` re-syncs the complete world. The daemon is the sole source of truth
and keeps no per-client state to leak or desync.

{{% callout warning %}}
**A transport error poisons the connection.** A timed-out reply is still in
flight, so the next round-trip on that socket would read the *previous* command's
reply — and the protocol has no request-id with which to detect the desync.
Poisoning turns a latent mis-attribution into a clean "connection closed" that the
frontend recovers from by reconnecting. Check `alive()`; on failure, drop and
reconnect.
{{% /callout %}}

## Frontends

### The console façade

`coast::coastd` (`include/coastd_console/coastd_console_api.h`) is the
REPL-ergonomic front onto the same client library, usable from the embedded
[C++ console]({{% relref "/architecture/modules/console" %}}):

```cpp
auto s = coastd::session::connect();               // localhost:51900
auto id = s.submit("main", "configs/mnist.json", "sweep", "", 500000);
for (auto j : s.jobs()) if (j.status == "failed") s.cancel(j.id);
auto h = s.info(); std::printf("%s: %d slots\n", h.hostname.c_str(), h.slots);
```

`submit`'s `config` argument is a path on **this** machine: it is read here and
its *content* is sent, so submitting to a remote daemon needs nothing staged on
that host.

The header is an interpreter **parse boundary**, so — like the other console
façades — it exposes neither nlohmann/json nor the socket-bearing client header.
Wire types are re-projected as plain `job_info`/`host_info` structs and the live
connection is held by pimpl.

The `job_list` / `host_list` / `host_config_list` handles exist for a specific
JIT reason: the clang-repl JIT resolves symbols against the host's `-rdynamic`
table and **cannot materialize the inline STL members of a `std::vector`
instantiated over a coastd-specific type** when that vector is the return value
crossing the JIT boundary. So every accessor is compiled out-of-line and the
underlying vector stays hidden inside compiled code — exactly the technique the
HDF5 façade uses with `row_view`/`row_range`. A range-based `for` works through the
compiled iterator.

Errors mirror `sim_console`: every failure throws `std::runtime_error` carrying the
daemon's own message.

### The GUI panel

`src/gui_tools/coastd_panel.cpp` provides queue/job viewing, submission, and
control through the same client library. Because a GUI config editor holds an
edited buffer rather than a file path, it is the frontend that made the
content-not-path wire change **necessary** rather than merely better.

{{% callout info %}}
The GUI panel is currently **compile-verified only**. Its runtime behaviour was
not exercised, because doing so requires an installed and running daemon; that
test was deferred by explicit decision. Everything beneath it — the client
library, the protocol, the daemon — is covered by the test suite.
{{% /callout %}}

## Tests

| Test | Covers |
|---|---|
| `tests/unit/coastd/test_comms.cpp` | Launches the **real** `coastd` binary and drives the wire through the thin client |
| `tests/unit/coastd/test_console.cpp` | The `coast::coastd` façade against a live daemon |
| `tests/unit/console/test_console_repl.cpp` | The façade **interpreted**, end to end against a live daemon — which is what proves the JIT handle technique actually works |
