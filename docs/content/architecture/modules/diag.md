+++
title = "diag"
weight = 66
+++

# diag — Diagnostic Logging & Run Progress

The `diag` module is COAST's **text diagnostics** facility: levelled, category-tagged, thread-safe log records, plus a separate machine-readable run-progress contract. It replaced the ~120 ad-hoc `std::cout` / `std::cerr` / `fprintf(stderr, …)` sites that previously served as an informal logging layer.

{{% callout note %}}
Not to be confused with [logging]({{% relref "/architecture/modules/logging" %}}). That module (`logging::`, `include/logging/`) persists **simulation data** — IO buffers and state checkpoints — to HDF5. `diag` (`coast::diag`, `include/diag/`) is **text diagnostics** for humans and supervisors. The names and namespaces are deliberately disjoint to avoid collision.
{{% /callout %}}

## Directory Layout

```
include/diag/
├── level.h       # coast::diag::level enum + level_name/level_from (lightweight)
├── diag.h        # coast::diag — sinks + COAST_* macros (includes level.h)
└── progress.h    # coast::diag::progress — structured run-progress contract
```

All three are **header-only** (inline globals + inline functions). There is no `src/diag/`. `level.h` is split out so `cli/options.h` — which only stores a resolved `diag::level` — needn't pull in `<format>`, `<mutex>`, `<functional>`, the sink state, and the macros.

## Why hand-rolled

The needs here are modest — a few severity levels, a category tag, one or two sinks, one mutex — and C++23's `std::format` already provides type-checked formatting. A logging *library* (spdlog, Quill) mainly buys async lock-free queues and deferred formatting, which target a hot-path logging load COAST does not have: the simulation hot path barely logs (only the per-step GPU device-fault check, and only on fault). The sink is a narrow functional seam, so a library-backed sink can replace the default later without touching a single call site.

{{% callout warning %}}
`diag` renders with `std::format` into a `std::string` and writes that — it never uses `std::print` / `std::println`. `std::print` needs a runtime symbol in the standard-library shared object (libc++ 18+ / GCC 14+) and adds a link dependency; `std::format` is header-only and available on GCC 13+ / libc++ 17+, under both macOS toolchains COAST supports. `std::filesystem::path` has no `std::format` formatter until C++26 — call `.string()` on paths.
{{% /callout %}}

## Levels and macros

Severity, ascending: `trace < debug < info < warn < error < off`. A record shows when its level `>=` the runtime threshold.

```cpp
#include <diag/diag.h>

COAST_INFO ("gpu",     "adapter: {}", name);
COAST_WARN ("observe", "dtype lock mismatch on {}", path);
COAST_ERROR("io",      "read failed: {} ({} bytes)", err, n);
COAST_DEBUG("io",      "dSprites load: decode={}ms", ms);
```

The category is a short string literal (`"gpu"`, `"observe"`, `"io"`, `"headless"`, `"gui"`, …) — it formalizes the ad-hoc `[gpu]` / `[console_plot]` prefixes that were scattered through the old code. The default sink formats each record as `[LEVEL][category] message`.

### Two-stage gating

| Stage | Mechanism | Effect |
|-------|-----------|--------|
| Compile-time | `COAST_DIAG_COMPILE_LEVEL` (preprocessor) | Records below the floor expand to `((void)0)` — the `std::format` call is never compiled in. Default: strip `trace` in Release (`NDEBUG`), keep everything in Debug. |
| Runtime | `diag::level_enabled()` | Records compiled in but below the current runtime level short-circuit before `std::format` runs (one relaxed atomic load). |

Preprocessor stripping is used deliberately instead of `if constexpr`, which in a non-template context would still compile the discarded format call.

### Exception safety — a `COAST_*` call never throws

Every macro routes format-and-dispatch through `detail::emit_guarded`, a `noexcept` helper that runs the `std::format` **and** the sink call inside a single `try` and swallows anything that escapes. The only realistic throw is `std::bad_alloc` — from `std::format` building the message string (the pre-`diag` `fprintf` path could not throw; this is the guarantee centralizing it here restores) or from a stateful sink under OOM. For a diagnostics logger, dropping the line is strictly better than unwinding the caller.

The payoff is that a bare `COAST_*` call is safe **by construction** in the contexts that previously had to opt back in: a `noexcept` function (e.g. `write_run_manifest`), a `std::thread` entry point (the HDF5 executor's `worker_main_`), and a value-firewall catch block (the WebGPU `check()` device-fault path). None of these wrap their diagnostics — the macro carries the guarantee. The `diag::log_nothrow(fn)` helper remains, but only for guarding a *multi-statement* block or a *non-diagnostic* throwing call in such a context; a lone `COAST_*` no longer needs it.

## Sinks

A sink is `std::function<void(level, string_view category, string_view message)>` — no inheritance. The dispatcher holds a mutex **across** the sink call, so records from the GUI thread, the sim thread, and the HDF5 executor thread are serialized and cannot interleave. A custom sink therefore need not synchronize internally.

| Function | Target |
|----------|--------|
| *(default)* | stderr, flushed per record |
| `set_stream_sink(FILE*)` | any open stream (e.g. `stdout`) |
| `set_log_file(path)` | append to a file (owned for the program lifetime) |
| `set_sink(fn)` | arbitrary callable — e.g. mirror into the GUI logging window; empty restores the default |

## CLI control

`cli::parse` resolves these into `cli::options`; `main()` applies them via `diag::set_level` / sink routing **before anything else can log**.

| Flag | Effect |
|------|--------|
| `-v`, `--verbose` | Lower the threshold one level (repeatable; clamps at `trace`) |
| `-q`, `--quiet` | Raise the threshold one level (repeatable; **clamps at `error`** — see below) |
| `--log-level <name>` | Set explicitly: `trace\|debug\|info\|warn\|error\|off` (default `info`) |
| `--log-file <target>` | `stderr` (default) \| `stdout` \| `<path>` to append to — a sink route that avoids relying on shell `2>` redirection |

`-q` deliberately cannot reach `off`: stacking `-q` clamps at `error`, so the one line that explains a non-zero exit is never silenced by accident. Total silence is reserved for an explicit `--log-level off` — an "I mean it" choice, not a side effect of raising verbosity. Symmetrically, an explicit `off` is **sticky**: a later `-q` keeps it `off` (order-independent), and only `-v` (louder) steps it back into range, to `error`. So neither direction of verbosity stacking can accidentally cross the `off` boundary.

**One exception to `off`:** a failed `--log-file` open still prints a one-line warning to stderr, even under `--log-level off`. It is deliberately un-gated because it reports the *diagnostics sink's own* configuration failure — honoring `off` here would leave the user with a silently empty log file and no explanation, and the message cannot go to the sink that just failed to open.

## Run progress (`progress.h`)

Machine-readable progress is **not** a log-record stream, so it lives in its own header with its own contract — it is a small state feed an external supervisor consumes. Concretely: `cimc-coast-consim` launches `coast` child processes and drives a progress table from the headless output. Previously it `sscanf`'d run_headless's human prose (`"from step A to B"`, `"Steps: N"`), so any reword silently broke the bar.

The fix is one line grammar with both ends in this header — `format_progress()` (emitter, in `run_headless`) and `parse_progress()` (consumer) — so a field rename is a compile-time break on both sides, never a silent drift:

```
@coast/progress v1 phase=<phase> step=<u64> start=<u64> target=<u64> sps=<f64> logged=<u64>
```

- `phase` ∈ `start | step | checkpoint | done | failed`
- Emitted **alongside** the human stdout narration (one extra line per progress point), never instead of it — humans still read headless output.
- Once `phase=start` is out, the run's last progress line is exactly one of: `done` (target reached, exit 0), `failed` (exit 1: an in-run fault, a failed buffer log or `--save-state`, or an exception — #233), or a `step` line after a `--save-on-term` SIGTERM stop. On an exception the sim thread is stopped before the `failed` line, so no `checkpoint` line follows it. But not every exit emits a terminal phase: an *early setup* failure (missing thread controller, buffer-logging start failure, a config-load error) exits non-zero before any progress line, and a killed process (SIGKILL, SIGTERM without `--save-on-term`, SIGINT) writes nothing more. A supervisor must therefore also honor the process **exit code**, not rely on a `failed` line alone (consim already tracks child exit codes).
- There is deliberately **no flag to suppress** the progress lines. They are additive and few (one per progress point, sentinel-prefixed), the feed is the whole point for a supervisor, and a human at the terminal can ignore or `grep -v @coast/progress` them. `--log-file` does not affect them (they are not diagnostics — they go to stdout). If a genuine need for an opt-out appears, add it then rather than pre-emptively.
- `emit_progress()` writes the line flushed and under a dedicated mutex (so the sim-thread checkpoint hook and the main-thread poll loop don't tear).
- Parsers must ignore unknown trailing `key=value` tokens, so fields can be appended without breaking older consumers.

{{% callout info %}}
The consim consumer still parses the legacy prose today; it switches to `coast::diag::parse_progress` once both branches merge to `main`. Until then the structured line is additive and non-breaking.
{{% /callout %}}

## Migration policy: diagnostics vs. program output

Not every print is logging. The split applied across the codebase:

| Goes through `diag` | Stays raw on stdout (program output) |
|---------------------|--------------------------------------|
| Errors, warnings, debug traces (`cerr`/`fprintf(stderr)`, unsolicited stdout) | `run_headless` human run narration |
| Startup/config notices (`build_env`, `run_gui`) | `_main.cpp` probe/benchmark executables |
| Subsystem faults (`observe`, `hdf5` executor, GPU device fault) | `mnist_db::print_info()` — an explicit dump method |
| — | `coast_compare` report, `cli::print_usage`, `console/session.cpp` fd-redirection |

`main()`'s pre-config parse errors also stay raw on stderr — the options aren't trusted yet, so diagnostics aren't configured. `run_headless` additionally sets `std::cout << std::unitbuf` so its stdout narration line-flushes under redirection (the structured progress lines self-flush).

## Threading

See [Threading]({{% relref "/threading" %}}). The mutex-across-sink dispatch is the serialization point: `std::cout`/`std::cerr` were previously shared unsynchronized across the main/GUI, sim, and HDF5 executor threads, so interleaved/torn lines were possible. Routing diagnostics through `diag` removes that **for diag-routed records**. The headless human narration is deliberately *not* routed (it stays raw `std::cout` — see the migration-policy table above), so those stdout lines can still interleave when the sim-thread checkpoint hook and the main poll loop write concurrently; they are data-race-free but not atomically ordered.
