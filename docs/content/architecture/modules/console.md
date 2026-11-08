+++
title = "console"
weight = 68
+++

# console — Embedded C++ Interpreter

The `console` module embeds a real C++ interpreter (LLVM's `clang-repl`) into the
GUI so a user can write **C++23 against COAST's own headers, at runtime** — no
recompile, no separate analysis language. It is the interactive counterpart to
the [HDF5 Browser]({{% relref "h5browse" %}}): the browser answers *"is this run's
data healthy?"* with fixed views; the console answers *"what about **this**
question?"* with arbitrary code.

The console is usable **the moment it is opened** — no file needs to be loaded
first. HDF5 files are opened from code via `auto f = open("path.h5")`, which
returns a `shared_ptr<file_console>`; **several files can be open at once**, each
in its own variable. (The Open button and the browser just inject that same
`open(...)` statement for you.)

The governing idea is a **compiled/interpreted split**. The interpreter is a
thin, `-O0` orchestration layer; every byte of data it touches is moved and
crunched by `-O3` code that was compiled into the host binary. The interpreter
calls into COAST — it does not reimplement it. This is why an interpreted loop
over a multi-GB dataset runs at compiled speed: the loop body is a call.

## Directory Layout

```
include/console/
├── repl.h               # clang-repl wrapper (LLVM-free pimpl)
├── session.h            # threaded wrapper: command queue + output capture
└── console_panel.h      # ImGui panel; implements coast::console_iface

src/console/             # matching implementations

include/gui_core/console_iface.h   # the abstract seam (a coast_gui header)
include/h5browse/console_api.h     # the `h5` surface the interpreter parses
```

Two files outside `console/` are load-bearing and belong to this picture:
`console_iface.h` is the seam that keeps LLVM out of `coast_gui`/`coast_lib`, and
`console_api.h` is the **parse boundary** — the header the interpreter actually
reads (see [Boundaries](#boundaries-read-these-before-extending)).

## Build Gate — an Optional Dependency

The console is the only part of COAST that needs LLVM/Clang development
libraries, and the build is **auto-detecting**: `find_package(LLVM CONFIG)`
succeeds → the console is built; it fails → a `message(WARNING)` fires and the
rest of COAST builds normally. A machine without `llvm-dev` still gets a working
`coast`. There is deliberately **no FetchContent fallback** — building LLVM from
source costs hours and gigabytes. `-DCOAST_WITH_CONSOLE=OFF` skips it silently.

A **headless build (`-DCOAST_WITH_GUI=OFF`) forces the console off** regardless of
`COAST_WITH_CONSOLE`, and says so at configure time
(`Console: disabled — COAST_WITH_GUI=OFF`). This is not a policy choice but a
structural one: `console_panel` *is* an ImGui window and the console's plot bridge
is `gui_plot`/ImPlot, so there is nothing left to attach the interpreter to.
CMake derives an internal `_coast_build_console` for this rather than overwriting
the cached `COAST_WITH_CONSOLE`, so your setting survives turning the GUI back on.

Developed against **LLVM 22**; CMake pins no minimum, it just prints
`Console: clang-repl via LLVM ${LLVM_PACKAGE_VERSION}`. Install `llvm-dev
libclang-cpp-dev` (Debian/Ubuntu), `clang llvm` (Arch), or `brew install llvm`
plus `-DCMAKE_PREFIX_PATH="$(brew --prefix llvm)"` (macOS).

On **macOS** the console additionally dictates the compiler: Homebrew's libLLVM
is built against **libc++**, so the console must be compiled with Homebrew's LLVM
Clang (libc++) — a GCC/libstdc++ build cannot resolve libLLVM's `std::__1`-mangled
symbols and skips the console. The `if(APPLE)` branch in `CMakeLists.txt` applies
`-stdlib=libstdc++` only under GCC, leaving Clang on its default libc++. A
console-enabled macOS build is therefore:

```bash
brew install llvm
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
  -DCMAKE_PREFIX_PATH="$(brew --prefix llvm)"
```

That clang/libc++ build compiles *and passes* the whole project — GUI, sim, GPU,
console, and the test suite. Exceptions thrown from JIT-compiled code (including a
*file-scope initializer* in a `.L`/`.x`-loaded unit) unwind back into
`exec_guarded()`'s host `try`/`catch` and are reported, not fatal. On macOS that
requires the console's custom object-linking layer to register eh-frames itself via
`EHFrameRegistrationPlugin` — the stock LLJIT layer does this by default, but the
custom layer (installed for the retaining JIT arena) replaces it and sets up no
MachO Platform to register frames, so without the explicit plugin a JIT-thrown
exception would `std::terminate` before reaching the host handler. See
`repl.cpp`'s `setObjectLinkingLayerCreator`.

`coast_console` is a **separate STATIC target** linking `coast_gui` (not
`coast_lib` — it needs ImGui and `gui_plot`) plus `clang-cpp` + `LLVM`. Nothing
else pulls in libLLVM — neither `coast_lib`, nor `coast_gui`, nor the unit tests.
Four build details are not optional and each fails loudly if dropped:

| Setting | Why |
|---|---|
| `-rdynamic` on every exe that runs the interpreter | the JIT resolves the compiled `console_api` façade symbols against the **host's dynamic symbol table**; without it, lookups fail |
| `-resource-dir ${COAST_CLANG_RESOURCE_DIR}` | an embedded Clang cannot self-locate its builtin headers — `argv[0]` is not `clang`. Baked in at configure time via `clang -print-resource-dir` |
| `-std=c++23` | the parsed headers are C++23 |
| `-I ${COAST_INCLUDE_DIR}` | so `#include "h5browse/console_api.h"` resolves |

`COAST_WITH_CONSOLE` is a **PUBLIC** compile definition on `coast_console` —
that is how it reaches `main.cpp`. The console links *after* (outside) the
`--whole-archive` wrap of `coast_gui`/`coast_lib`, and its wiring is gated on
`if(TARGET coast_console)` rather than on the option, so a missing-LLVM build
(or a headless one) still links cleanly.

## The Seam — Why the Libraries Carry No LLVM

`coast::console_iface` (`include/gui_core/console_iface.h`) is seven virtuals:
`menu_label()`, `visible()`, `set_visible()`, `render()`, `open_file()`,
`submit()`, and `bind_dataset()`. That is the entire contract.

`gui_manager` owns a `console_iface`, menus it, renders it per frame, and hands
the raw pointer to the HDF5 browser. It never learns what backs it.
`main.cpp` constructs the concrete `console::console_panel` under
`#ifdef COAST_WITH_CONSOLE` and injects it through `run_gui`. Consequence: the
GUI layer compiles identically with or without LLVM present, and the LLVM-free
`repl.h`/`session.h` pimpl headers are safe to include from anywhere.

## Threading

The interpreter runs on its **own thread**, created by `console::session`. This
is not decoration: a macro scanning a multi-GB dataset blocks on the
`hdf5::executor`, and running that on the GUI thread would freeze the window for
the duration.

```
GUI thread  ──commands (non-blocking enqueue)──▶  console thread  ──read jobs──▶  hdf5::executor
GUI thread  ◀──drain_output() ─── output_line ───  (fd capture + reader thread)
```

Every `session` command (`start`, `open_file`, `submit`, `reset`, `load`,
`run_macro`, `add_include_path`, `load_library`) only enqueues — `open_file`
additionally assigns the file's variable name synchronously and returns it. The
GUI thread polls `drain_output()`, `busy()`, `current_file()`, and
`open_files()`. `wait_idle()` blocks until the queue drains — for tests and
shutdown only.

**Output capture** is an fd redirect: `session` dups the process `stdout`/`stderr`
onto a pipe around each execution, so interpreted `printf`, `std::cout`, *and*
Clang's own diagnostics all land in the same `output_line` stream, tagged
`input | output | error | info`. A **background reader thread** drains the pipe —
without it a chatty macro deadlocks once the pipe buffer fills. Teardown flushes
both C stdio and the `std::cout`/`cerr`/`clog` streams, which buffer separately.

See [Threading]({{% relref "/threading" %}}) for how this thread sits beside the
main, sim, executor, and h5browse-scan threads.

## What the User Types

Code runs immediately — **no file is required**. The interpreter is built once
per session (pre-warmed the first time the console is shown, off the GUI thread),
and its preamble is exactly:

```cpp
#include <cstdio>
#include <cstdint>
#include <string>
#include "h5browse/console_api.h"
using namespace h5b;
using namespace h5b::console;
```

Because the preamble pulls in `console_api.h`, `open()` is in scope from the
first prompt. Files are opened as ordinary variables — as many as you like:

```cpp
auto f = open("run.h5");        // shared_ptr<file_console>; several may coexist
auto g = open("baseline.h5");
```

Everything else needs an explicit `#include` — COAST's `include/` is already on
the search path. The surface reached through a file handle, in `.? ` cheatsheet
order:

```cpp
open("path.h5")      // shared_ptr<file_console>  (throws on open failure)

f->list()            // vector<string>  dataset display names
f->count()           // size_t
f->family()          // string_view     "observe" | "buffer log" | ...
f->dataset("name")   // dataset_handle  (throws if unknown; also takes an index)

d.rows() / d.width() / d.dtype()
d.scan()             // range           stream all rows, single-pass
d.scan(first, n)     // range           stream a sub-range
d.row(i)             // row_view        random access
d.column(c)          // vector<double>  one column across all rows
d.steps()            // vector<uint64>  sim step per row (if present)

r.at(col)            // double          any dtype, widened
r.as<float>()        // span<const float>  zero-copy (throws on dtype mismatch)
r.as_copy<float>()   // vector<float>      owning copy
r.bytes()            // span<const byte>
```

Errors are `std::runtime_error` **deliberately**: its typeinfo already lives in
libstdc++, so it crosses the JIT boundary without any extra export machinery.

Meta-commands are handled by the panel before the interpreter sees the line:
`.L <file>` (load declarations), `.x <file>` (load + call the function named
after the file, ROOT-style), `.I <dir>` (add a header search dir), `.load <lib.so>`
(load a shared library so its symbols resolve), and `.? [topic]` (built-in help;
topics: `h5`, `dataset`, `row`, `metric`, `plot`). An unrecognized `.command`
falls through to the interpreter, which reports it clearly.

Reaching it: the **Tools → "C++ Console..."** menu item, the HDF5 Browser's
**Console** toolbar button (opens the browsed file into a handle variable), or a
**right-click on a dataset** in the browser tree, which opens the file into its
handle (e.g. `f_run`) and submits `auto <name> = f_run->dataset("<display>");`,
then shows the panel. The handle variable name is derived from the file's stem
(`var_name_for`), so repeated opens of one file dedupe to the same variable.
There is **no CLI flag** — the console is GUI-only.

The panel is a **notebook**: every echoed input line renders as a re-runnable
cell with Run / Edit / Copy. Ctrl+Enter runs. The log is capped at 5000 lines.

## Error Recovery — the Load-Bearing Part

Naive REPL embedding produces several failure modes that silently destroy user
work. All are closed here, and each fix is **counter-intuitive enough that the
code must not be "simplified" back**.

**1. `exec_guarded()` runs code UNWRAPPED.** The obvious design — wrap each
submission in an interpreter-side `try { ... } catch` — was implemented, then
**reversed**: the wrapper is a scope, so every `auto x = ...` the user typed was
destroyed at the closing brace and declaration persistence died. The interpreter
now runs input unwrapped and the throw is caught **host-side**, around
`Execute()`. Same protection, no scope.

The consequence is the rule that governs the whole module:

> You do not need to **wrap** code in order to **guard** it. A host-side catch
> guards declarations and statements alike. The declaration/statement split is
> therefore **not** a guarding boundary — do not treat it as one.

So `exec_guarded()` is the path for **all** untrusted input, cells and macro
files alike. `exec()` is raw — no exception guard, no `#include` isolation — and
is reserved for code this class controls (the preamble), where a throw would be a
COAST bug.

{{% callout info %}}
**This was learned the hard way.** The original design reasoned: statements must
be wrapped → declarations *cannot* be wrapped → therefore declarations cannot be
guarded → so `load()` (`.L`/`.x`) must run raw. When the wrapper was reversed
that inference died, but `load()` was never moved over, and the stale comment
describing the wrapper kept the gap invisible. Two measured consequences before
it was fixed: a throwing file-scope initializer in a `.L`'d macro **aborted the
process** (SIGABRT — a typo was enough, since `h5.dataset()` throws on an unknown
name), and a macro file with a fresh `#include` plus a parse error **poisoned the
interpreter** because `load()` also bypassed point 3.

Both are pinned by regression tests. If you find yourself reintroducing a
`try`-wrapper, or routing user input through `exec()`, re-read this box.
{{% /callout %}}

**2. `Undo(1)` runs ONLY on `Execute()` failure — never after a parse error.**
`Parse()` and `Execute()` are split deliberately. A failed `Parse()` never built
a translation unit, so undoing after one rolls back the *previous good* input,
silently wiping the user's declarations; repeated, it unwinds past the preamble
into a corrupted interpreter (observed to crash).

**3. `split_into_units()` isolates every `#include` into its own `Parse()` unit.**
Measured on LLVM 22: a unit that both enters a new source file *and* then fails
to parse corrupts the preprocessor's source-file state and poisons **every later
`Parse()`** — the symptom being an unrelated later cell reporting
`use of undeclared identifier 'std'` as the whole preamble effectively vanishes.
Upstream clang-repl dodges this by consuming one line at a time; COAST consumes
blocks, so it must isolate includes itself.

Detection is line-based, so it tracks block-comment state and isolates a line
only when it lies **wholly outside** a `/* */` comment. Otherwise two ordinary
shapes — an `#include` commented out, and `#include <x>  /* why ...` — would be
cut in half into an unterminated comment: a hard parse error on code clang
accepts. `scan_comment_state()` is deliberately **not** a full lexer (it does not
model string literals), and that asymmetry is safe by construction: a false "in a
comment" only *skips* isolation, degrading to the old whole-block behaviour,
while it can never split a comment apart. The failure mode points one way only.

Regression tests in `tests/unit/console/test_console_repl.cpp` pin every one of
these — declarations surviving a parse error, a fresh `#include` in a failing
block, both block-comment shapes, a `.L`'d throwing file-scope initializer, and
`.L` with `#include` + parse error. Each was verified to **fail** against the
unfixed code; a green test nobody has seen fail proves nothing here.

When the interpreter *is* poisoned (ORC `Failed to materialize symbols`, which is
inherent to clang-repl and not fully avoidable), `session::do_reset()` rebuilds
the whole interpreter. It is the only reliable escape, and it costs every user
definition.

## The Plot Bridge

`#include "gui_plot/console_plot.h"` gives the console live ImPlot windows. The
module is documented in full at [gui_plot]({{% relref "gui_plot" %}}); what
follows is what a console user needs.

```cpp
#include "gui_plot/console_plot.h"
auto f = open("run.h5");
auto h = std::make_shared<coast::plot::histogram>("activity[0]");
h->show();                                   // nothing appears until show()
h->update(f->dataset("act").column(0));
```

Kinds are `histogram`, `line`, `histogram2d`, `pixels`, `image`, and `canvas`
(a grid you `place()` figures into). The design is deliberately flat — **no
factory, no type erasure** — so a macro constructs a concrete type and reaches
its typed config directly.

The cross-thread seam is a `painter` (`std::function<void(region)>`) where
`region` is a plain `{w,h}` POD, keeping `figure.h` ImGui-free and cheap for
clang-repl to parse. The registry **owns** the `shared_ptr`, so a figure shown
without a retained handle stays alive; it is reaped only when closed *and*
`use_count() == 1`. `update()` sets data only — visibility is explicit, and a
closed handle persists until `show()` (not `update()`) brings it back.

`save_png()` / `save_h5()` only **enqueue** from the console thread; the GUI
thread renders each figure into an off-screen FBO (exports may exceed the
on-screen window) and hands HDF5 writes to the executor. The flush must run
*outside* an active ImGui frame — `run_gui` registers it as a state func.

{{% callout info %}}
Objects a macro creates can outlive the interpreter that built them, and their
**code** does not automatically follow. COAST closes that crash class
structurally with a retaining JIT code arena in `src/console/repl.cpp`. Read
**Key Abstractions → JIT Object Lifetime** before touching interpreter teardown
or "fixing" what looks like a leak there.
{{% /callout %}}

## Running Simulations

The `coast::simc` façade (`include/sim_console/sim_console_api.h`, in `coast_gui`
— its run dashboard and log viewers are ImGui windows —
preamble-included as `simc::…`) drives the simulation *from* the console for
parameter search and characterization. It is documented in full by
`design/sim_console_api.md`; the shape a user needs:

```cpp
auto c = simc::config("configs/mnist.json");          // editable config
c.set("/ngm2/dendrites/default/default_primary_learning_rate", 0.03);
c.run_dir("runs/lr_003").seed(0);
auto h = simc::run(c);                                 // writes config.json, paused
h->step(50000);                                        // run a chunk (blocks)
h->step(50000);                                        // continue — chains state
auto f = open(h->observe_dir() + "/...");              // inspect with the h5 side
```

The **editable config itself** — `run_config` and its loader `config()` — is not
part of the GUI layer. It lives in `coast_lib` as `coast::cfg`
(`include/run_config/run_config.h`), GUI-free (nlohmann/json only), and both
consoles re-export it into their own namespace via `using`. Only `run()` and the
subprocess-backed `sim_run` (plus the ImGui dashboard) are `coast_gui`. This is
what lets the **coastd** console offer the *same* edit-then-dispatch workflow — see
*Controlling a coastd daemon* below and `design/run_config_extraction.md`.

**It shells out, it does not build a sim in-process.** `step(n)` spawns a child
`coast -n n --load-state <prev> --save-state <next> --seed-offset <s>
<run_dir>/config.json` and blocks on it. This reuses the entire tested headless
binary, and — decisively — a native sim fault kills only the child, closing the
one containment hole an in-process console cannot (see *Boundaries*). Runs are
**fresh-sim-per-config**, which is correct because group hyperparameters (learning
rate, …) are baked into the neuron groups at build time, not read live per step;
for a parameter sweep that independence is the point.

Everything for a run lands in one directory — `config.json` (the exact input),
`observe/`, `snapshots/`, one immutable `state_<NNN>.h5` per chunk (a rolling
history: `state_001.h5`, `state_002.h5`, …), and per-`step()` logs at
`logs/step_<timestamp>_<NNN>.log` (state and log share `<NNN>`) — so a run is
self-documenting and every intermediate state is retained for introspection.
`step()` blocks
the console thread (the GUI stays responsive); "finished" is the child exiting,
and a nonzero exit is thrown. `coast` prints periodic progress, so `tail -f` on
`h->last_log()` is a live monitor while a long step runs.

`step_async(n)` runs the chunk **without blocking** and (by default) opens a
**"Simulation Runs"** dialog — a table of every concurrent + queued run with a
progress bar, `cur/target` steps, status, and elapsed time. This is the sweep
path: fire `step_async` on several handles and watch them all in one table. A
**process pool** caps how many run at once (`max_concurrent_runs()`, default 4,
overridable via `COAST_SIMC_MAX_CONCURRENT` or `set_max_concurrent_runs()`); runs
past the cap show as **queued** and start automatically as slots free — so a big
sweep never spawns an unbounded pile of full `coast` processes, yet the call never
blocks. A failed run's
bar turns **red** (`failed (<exit>)`), a cancelled one **amber**; **each row has a
Cancel button** (and the header a **Cancel all**) — `sim_run::cancel()` /
`cancel_all_runs()` SIGTERM a running child (reaped as `cancelled`, chaining
nothing) and drop a still-queued run before it spawns. Closing the dialog's **X**
just hides it; `simc::show_runs_window()` reopens it. **Double-clicking a row**
opens a live log viewer that tails that run's log — the quick path to a child's
error output.
(`run_headless` sets `std::cout << std::unitbuf` so the child's progress lines
flush to the log incrementally rather than only at exit — without that the bars
would jump 0→100%.)
`render_runs_window()` — called each frame by `gui_manager` right beside
`coast::plot::render_all()` — polls the child processes (`waitpid(WNOHANG)`,
progress parsed from the log) and draws the table. On completion each run's saved
state chains back into its handle, exactly as a blocking `step()` would;
`wait_all_runs()` blocks until a whole sweep finishes (draining the queue as it
goes — it counts queued runs as active, so it never returns early). `sim_run` is a
`shared_ptr`-held, `enable_shared_from_this` handle, and the one-way lock order
(a handle locks itself *then* the registry; the poller fires completion callbacks
*after* dropping the registry lock) keeps the console thread and the GUI frame
free of a lock-order inversion. Because the runs are independent processes,
closing the GUI leaves a sweep running to completion on disk.

The header stays lean — `nlohmann/json` is held by pointer, so the config is
edited through a JSON-Pointer `set()`/`get_*` API rather than exposing nlohmann
across the interpreter's parse boundary. `COAST_SIMC_BINARY` overrides the spawned
binary (default `/proc/self/exe`), used by the B-tier test to point at the built
`coast`. A string-literal `set()` needs an explicit `const char*` overload, or
`const char* → bool` would win overload resolution and silently store a boolean.

## Controlling a coastd daemon

The `coast::coastd` façade (`include/coastd_console/coastd_console_api.h`, in
`coast_lib` — no GUI, so it works headless too) drives a
[coastd]({{% relref "/coastd" %}}) daemon from the same console. It is the REPL-ergonomic
front over `coastd_client`, and it deliberately mirrors the `simc` workflow:
**load a base config, edit it, then dispatch it** — the daemon just replaces the
"spawn a local child" transport with "queue a job on a machine". The requirement
is recorded in `design/remote_run_dispatch.md` (§1/§5: *same workflow, no
console-facing API change*); the shared `run_config` (above) is what makes it
literal — a config edited for a local `simc::run()` submits to a daemon unchanged.

```cpp
auto s  = coastd::session::connect("captncrunch");   // or connect_to("gpu-box")
auto c  = coastd::config("configs/mnist.json");       // the SAME editable run_config
c.set("/ngm2/dendrites/default/default_primary_learning_rate", 0.03);
auto id = s.submit("main", c, "sweep-lr", "", 1'000'000);   // sends c.dump() as content
for (auto j : s.jobs()) if (j.status == "failed") s.cancel(j.id);
```

`submit(ref, run_config, …)` sends the edited config's current bytes as the job's
`config_text` — the daemon receives config *content, never a path*, so an
in-memory config is exactly what the wire wants and nothing is staged on the
server. The queue view (`s.list()` and the panel table) shows each **running**
job's *live* step, parsed from its log tail by the daemon — so a job advances
between checkpoints rather than sitting frozen; the durable resumable point stays
`steps_done` (see `design/live_step_in_status.md`). A `submit(ref, path, …)` overload remains for firing a finished file
straight from disk. The daemon owns the run directory and rewrites `snapshot_dir`
itself, so `run_config`'s `run_dir()`/`seed()` staging hints (used by `simc::run`)
are ignored on this path. Discovery mirrors the client library: `connect_to(name)`
against the fixed-list `~/.config/coastd/hosts.json`, or `discover()` by UDP
broadcast. The queue and discovery results cross the JIT boundary as compiled
`job_list`/`host_list` handles (never a bare `std::vector` of a coastd type), the
same technique the h5 façade uses with `row_range`. The full extraction that put
`run_config` in `coast_lib` is `design/run_config_extraction.md`.

## Boundaries (read these before extending)

- **`console_api.h` is a parse boundary, not just a header.** Every declaration
  it pulls in is parsed by clang-repl at *every* session start. Keep it lean:
  what you add here, every user pays for on every open.
- **`COAST_INCLUDE_DIR` is baked in as an absolute source-tree path.** The built
  binary resolves console `#include`s against the machine's *source* checkout, so
  a relocated or shipped binary loses them. Fine for an internal tool; a blocker
  for distribution.
- **The interpreter is built once, not per file** (~2.3 s, on the console thread
  when the console is first shown). Opening a file is just an injected
  `auto <var> = open("...")` statement, so it is cheap and non-destructive;
  `session` dedupes repeated opens of one path to the same variable. `reset()`
  rebuilds the interpreter and re-opens the files that were open.
- **`file_console` is non-copyable *and* non-movable** — it owns a read-only
  HDF5 handle and an LRU `chunk_cache`, and shares the process executor. It is
  handed out only via `shared_ptr` (from `open()`); `make_shared` runs in
  compiled `coast_lib`, so the control block and the `~file_console` deleter live
  in the host image. That means the handle is released cleanly whenever the
  owning variable is destroyed — including on interpreter teardown, which *does*
  run its globals' destructors (measured under ASan/LSan: no `file_console` leak
  on `~session` or `reset()`, for GUI-opened and user-typed opens alike). This is
  the opposite of the plot case, whose `make_shared` runs in the JIT: there the
  deleter code lives in the interpreter's module and must be kept mapped past
  teardown by the retaining arena (see **JIT Object Lifetime**). `file_console`
  needs no such protection.
- **`open()` needs its executor registered** via `h5b::console::set_executor`,
  done once in the `session` constructor. Same shape as `coast::plot::set_executor`.
- **Spans never outlive their `row_view`.** `auto ref = d.row(i).as<float>();`
  dangles at the `;` — and reads *another row* once the cache evicts the block,
  silently and without a crash. `as_copy<T>()` exists precisely to close this.
- **A handle kept from a temporary file dangles.** `dataset_handle`/`row_range`/
  `row_view` hold a raw `const file_console*`, so `auto d = open("x.h5")->dataset("y");`
  drops the only `shared_ptr` at the `;`, leaving `d` pointing at a freed file.
  Bind the file first: `auto f = open("x.h5"); auto d = f->dataset("y");` — `f`
  keeps it alive. (Using the temporary within a single expression, e.g.
  `open("x.h5")->dataset("y").rows()`, is fine — the temporary lives to the `;`.)
- **Never route user input through `exec()`.** It has no exception guard and no
  `#include` isolation; `exec_guarded()` is the only path for anything a user
  wrote. See [Error Recovery](#error-recovery--the-load-bearing-part) for what
  happened the last time this rule was broken.
- **A native fault still kills the process.** The host-side catch handles C++
  throws, including from file-scope initializers, but a null deref in JIT'd code
  is contained by no in-process `try/catch` — only an out-of-process ORC executor
  would close that, which `design/console_api_design.md` weighs and defers.

## Tests

| Test | Covers |
|---|---|
| `tests/unit/console/test_console_repl.cpp` | interpreter lifecycle, the façade over real data, and every recovery regression above (parse-error/Undo, `#include` isolation, both block-comment shapes, `.L` throwing init, `.L` include+parse-error) |
| `tests/unit/console/test_console_session.cpp` | running with no file open, reset (rebuild + re-open), **several files open at once**, `.I`, `.x`, declaration persistence |
| `tests/unit/h5browse/test_console_api.cpp` | the file façade + `open()`/`var_name_for` (no LLVM — links `coast_lib`) |
| `tests/integration_b/test_b_tier.cpp` (`b_tier.simc_facade`) | the `coast::simc` sim-control façade end-to-end: config edit, `run()`, chunked `step()` state-chaining, per-step logs, bad-config → throw (spawns the built `coast`) |

The console test directory is gated by `if(TARGET coast_console)`; both console
exes link `-rdynamic`. No test covers ImGui interaction — Ctrl+Enter and the
splitter drag are verified by hand.

## See Also

- `design/console_api_design.md` — the design record and validation spikes. Its
  exception model carries in-place supersession notes (the wrapping design, and a
  "throwing static initializer is uncontainable" claim since refuted by
  measurement); read those before trusting the surrounding prose. Residual
  staleness: it still sketches a `row_view::step()` that does not exist.
- `design/sim_console_api.md` — the sim-control façade (`coast::simc`): design record, subprocess rationale, run-directory layout, monitoring
- [h5browse]({{% relref "h5browse" %}}) — the domain layer behind `h5`
- [gui_plot]({{% relref "gui_plot" %}}) — the figures a macro creates
- [gui_tools]({{% relref "gui_tools" %}}) — the browser panel that hosts the entry points
- [Threading]({{% relref "/threading" %}}) — where the console thread sits
