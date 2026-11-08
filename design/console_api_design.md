# HDF5 Browser console: an embedded C++ interpreter over the h5 read layer

Status: **feature-complete (v1)** — all three increments implemented and validated
2026-07-05 on branch `feat/h5-console`. Data-access façade + interpreter engine are
unit-tested; the GUI panel builds into `coast` and awaits visual verification on a
machine with a display (the CI sandbox has none).

> **UPDATE (2026-07-18, branch `feat/independent-console`) — the console is no
> longer bound to one open file.** The single global `h5` reference (injected by
> the now-removed `repl::bind_h5`) is gone. The interpreter is built **once**,
> independent of any file, and is usable from the first prompt. Files are opened
> through a real free function
> `std::shared_ptr<file_console> h5b::console::open(const std::filesystem::path&)`
> — reintroducing the by-value `open()` the v1 sketch had dropped, now returning a
> **shared handle** (which sidesteps `file_console` being non-movable). Multiple
> files are open at once, each in its own session variable; the Open button and
> the browser inject `auto <var> = open("<path>")` (`<var>` = `var_name_for(path)`,
> a stem-derived identifier, collision-disambiguated by `session`). `open()` gets
> the process executor via `h5b::console::set_executor`, registered in the
> `session` constructor (mirrors `coast::plot::set_executor`). `session::reset()`
> now rebuilds *and re-opens the open files*. Handle cleanup needs no special
> care: interpreter teardown runs its globals' destructors (measured under
> ASan/LSan — no `file_console` leak on `~session` or `reset()`), and
> `file_console`'s deleter is compiled into `coast_lib`, so every open handle
> (GUI-opened or user-typed) is closed cleanly on teardown. (The retaining JIT
> arena is for the *plot* case, whose deleter is JIT-emitted; `file_console`,
> built by a compiled `make_shared`, does not depend on it.) This
> resolves the "how does the GUI bind the current file / reflect a file-switch"
> open item below: there is no current file to bind — access is by handle.

## Implementation status

- **Increment 1 — data-access façade: DONE.** `include/h5browse/console_api.h` +
  `src/h5browse/console_api.cpp`, built into `coast_lib`; unit test
  `tests/unit/h5browse/test_console_api.cpp` (`unit.h5browse.console_api`) green
  against a real executor + temp HDF5 file (discovery, streaming scan, random
  `row(i)`, `column()`/`steps()`, `at()`→double, `as<T>()`, throwing error surface).
  Deviations from the sketch below, made during implementation:
  - `family()` returns `std::string_view` (the family name), not `file_family` —
    avoids pulling `file_model.h`'s enum into the interpreter-parsed header
    (parse-light). `file_family` stays an internal type.
  - `row_view` has **no** per-row `step()`/`has_step()` in v1 (it would couple the
    view back to the file to read the step-index sibling). Step data is available via
    `dataset_handle::steps()` (whole step-index column). May add per-row `step()`
    later if analysis wants it.
  - `file_console` is **non-copyable and non-movable** (owns an HDF5 handle + a
    mutex-bearing `chunk_cache`); constructed in place as `file_console(exec, path)`.
    The by-value free `open()` from the sketch is dropped — the console subsystem
    constructs the `h5` instance directly.
- **Increment 2 — interpreter/REPL engine: DONE.** `include/console/repl.h` (pimpl,
  LLVM-free header) + `src/console/repl.cpp` (embeds `clang::Interpreter`), built as a
  separate `coast_console` static lib so the LLVM/Clang dependency stays OUT of
  `coast_lib` and the small unit tests. Gated by CMake option `COAST_WITH_CONSOLE`
  (**default ON**; turn OFF for an LLVM-free build), using **system LLVM** via
  `find_package(LLVM/Clang CONFIG)`. `repl` binds a live `file_console` as `h5`,
  runs the preamble that pulls in `console_api.h`, and offers `exec()` (raw —
  preamble/binding only), `exec_guarded()` (host-side catch + `#include` isolation
  + `Undo(1)` recovery; the path for ALL untrusted input, statements and
  declarations alike), and `load()`/`run_macro()` for `.L`/`.x`. Any exe
  running it links with `-rdynamic`. Test `unit.console.repl` drives the facade over
  real data through the interpreter (sum/count), confirms an uncaught macro throw is
  reported not fatal, and runs a `.x` macro file. Main `coast` exe still builds.
  - Error/persistence model (revised 2026-07-05 after GUI testing): `exec_guarded`
    now runs input UNWRAPPED and catches runtime throws on the HOST side — verified
    that a JIT-thrown exception unwinds into a host `try/catch` rather than aborting.
    This preserves top-level declaration persistence across submits (the earlier
    interpreter-side `try{}` wrapper scoped every `auto x = ...` away) while keeping
    crash-safety. On a failed compile/materialization the bad input is dropped with
    `Undo(1)`. Diagnostics use `-fno-color-diagnostics` (ANSI escapes would render as
    garbage in the text panel). stdout capture flushes C++ iostreams (std::cout
    buffers separately from printf), not just C stdio.
- **Increment 3 — GUI panel + wiring: DONE.** Seam: abstract `coast::console_iface`
  (`include/gui_core/console_iface.h`, LLVM-free) with `menu_label/visible/set_visible/
  render/open_file/submit`. `gui_manager` menus + renders it (owns its own
  visibility); `run_gui` gained an optional `unique_ptr<console_iface>` param;
  `main.cpp` (guarded by `COAST_WITH_CONSOLE`) constructs the concrete panel and
  passes it. The panel `console::console_panel` (`console/console_panel.{h,cpp}`, in
  `coast_console`) owns a `console::session`, renders an output log (colored by line
  kind) + resizable code input + `.h5` chooser, Ctrl+Enter to run. `coast` links
  `coast_console` with `-rdynamic`; `coast_lib` carries no LLVM dependency.
  - Chosen approach (b): explicit `main.cpp` wiring (no static-init registrar) +
    own console thread + fd-based stdout/stderr capture.
  - **Browser integration (2026-07-05 usability pass):** `gui_manager::set_console`
    also hands the console pointer to `h5_browser_window`. The browser gains a
    "Console" toolbar button (opens the browsed file in the console) and a dataset
    right-click → "Create console variable" (`auto <sanitized_name> =
    h5.dataset("<display>");`). `console_panel::open_file` dedupes repeated opens of
    the same path (avoids the ~2.3s interpreter reconstruct).
  - **Still not visually verified** (sandbox has no display). Two ImGui bits to
    confirm on a real display: Ctrl+Enter-to-run key detection, and the log/input
    splitter drag feel.

- **Post-v1 usability/extensions (2026-07-05/06, all confirmed by spikes/tests):**
  - Notebook-style input cells (Run / Edit / Copy, selectable code) in the output.
  - `.? [topic]` curated cheatsheet (h5 / dataset / row / metric / plot).
  - **CRITICAL fix** — `exec_guarded` splits `Parse()`/`Execute()`; `Undo(1)` ONLY on
    Execute failure. Undo-after-parse-error was rolling back the *previous* good
    input (wiped variables) and, repeated, corrupted the interpreter → crash.
  - **External libraries**: `.I <dir>` adds a header search path at runtime
    (`HeaderSearchOpts::AddPath` + `ApplyHeaderSearchOptions`); `.load <lib.so>` loads
    a shared lib (`Interpreter::LoadDynamicLibrary`). Header-only libs need only `.I`.
    Both verified (spike_extlib). Caveat: a dylib-load then interpreter-teardown
    crashed in the spike — watch load-then-close-file.
  - **Using compiled coast functions**: header-only (`inline`/template, e.g.
    `tools/metric.h`) just needs `#include`; compiled `coast_lib` symbols resolve via
    the coast exe's `--whole-archive` + `-rdynamic`. Don't call live GUI/GPU/sim state.
  - **Plot bridge** (`gui_plot/console_plot.{h,cpp}`, in coast_gui — ImGui/ImPlot only,
    NO LLVM): one flat hierarchy of plain, named figure objects. The macro constructs a
    concrete kind directly — `auto h = std::make_shared<coast::plot::histogram2d>(title, xb, yb)` —
    there is NO creation factory and NO type erasure, so kind-specific config is reached
    by typed methods on the concrete object (`h->set_axis_visible(plot_axis::x, false)`,
    `set_palette`, `set_log_density`, `set_candle`). Kinds: `histogram`, `line`,
    `histogram2d`, `pixels`, `image`, `canvas`, all deriving the base `figure`.
    - **Threading seam = a painter.** The base `figure` carries the common state (title,
      theme, visibility) and hands the GUI thread a `painter` — `std::function<void(region)>`
      (`region` is a plain `{w,h}` POD; the header stays ImGui-free for clang-repl). The
      painter pins the figure, locks it, applies its theme, and calls the kind's
      `paint_body`; all ImPlot/GL lives inside it, in the .cpp. The macro thread only holds
      DATA (thread-safe `update()`), the GUI thread draws.
    - **`show()` registers.** Nothing appears until `h->show()`, which adds the figure to
      the registry (`gui_manager` calls `coast::plot::render_all()` each frame). The
      registry OWNS the shared_ptr, so a handle-less shown figure stays alive; it is reaped
      only when closed AND unheld (`use_count()==1`). A canvas cell is `place()`d, never
      shown. Data/registry layer unit-tested (`unit.gui_plot.console_plot`, incl. the
      programmatic axis-visibility surface); rendering verified by hand.
    - `image` → `set_png(path)` / `set_rgba(pixels,w,h)`: a generic picture output channel
      for plot libraries that write images. Macro provides the source; GUI thread decodes
      (stb_image) + uploads (reused `gui_vis` `gl_texture`) + draws (`ImGui::Image`).
      `gui_manager::free_resources()` calls `coast::plot::free_resources()` to release GL
      textures before context teardown. float overloads on `update()` (row_view::as<float>()
      drops straight in). A concrete kind's draw/controls/snapshot live in `plot_kinds.cpp`.

## Motivation

The HDF5 Browser (`h5browse/`, `gui_tools/h5_browser_window`) gives a fixed,
GUI-driven view of a COAST log file: scan stats, histograms, a filtered table,
pairwise plots. Every question it can answer is one someone anticipated and wired
a panel for. Real data analysis is open-ended — a user wants to iterate over a
dataset, correlate two columns their own way, compute a statistic no panel offers,
or load a small script that does all three.

The goal: an **interactive C++ console inside the browser** that can inspect the
open file programmatically and run ROOT-style "macro" files (`.L` to load, `.x` to
execute), *without* linking the ROOT framework. We embed an interpreter and expose
the browser's existing read machinery through a thin, ergonomic façade bound as a
global `h5`.

Explicit non-goal for v1: plotting. ROOT's strength is visualization, but that is
the bulk of its dependency weight. A quick view can reuse the existing ImPlot
gadgets; publication-grade output is a later, separate decision.

## Decision: clang-repl, not Cling

Cling (CERN/ROOT) and clang-repl (mainline LLVM) share one core — Clang-as-a-library
+ LLVM ORC JIT + incremental compilation. clang-repl is Cling upstreamed; ROOT
itself is transitioning to it.

We choose **clang-repl** because:
- **C++23-native on the system LLVM.** clang-repl is just our system Clang built as
  a library, so its standard support is whatever that Clang supports (LLVM 22 here).
  Cling historically rides a *patched fork* pinned to older LLVM — painful for a
  C++23 codebase. Validated: it parses and runs our `std::expected` / concepts /
  `if consteval` (spike T1).
- **We route console input ourselves**, so Cling's `.L`/`.x` meta-commands give no
  advantage — a leading-`.` match in our input handler dispatches load/exec in a few
  lines. The HEP-specific Cling features (ROOT-file name lookup, implicit `auto`
  injection) are things we do not want.
- It is the direction of travel.

Kept as fallback if clang-repl's error-recovery maturity or a parse edge case bites:
Cling, or the `CppInterOp` backend-agnostic layer (works over either).

## Architecture

### Threading — the HDF5 single-funnel is the load-bearing constraint

`hdf5::executor` exists because the linked libhdf5 is built **Threadsafety: OFF**:
exactly one thread may be inside the library process-wide (not per-file). The
executor owns that one worker; everyone else submits jobs to it.

The console is therefore **another client of the executor, on its own thread**:

```
Cling console thread ──► console façade ──► executor::submit_sync ──► the ONE hdf5 worker
                                    ├─ read_pipeline  (streaming, depth-2 prefetch)
                                    └─ read_block + chunk_cache (random access)
```

- The interpreter runs on a **dedicated console thread** — never the GUI thread (a
  long macro would freeze the UI) and never the scan thread (would stall its
  two-lane servicing).
- Every read goes through `executor::submit_sync`, which **blocks the console
  thread** and returns the result (or rethrows). For imperative macro code, blocking
  is the feature, not a limitation — `for (auto row : d.scan())` reads synchronously.
- The console does **not** go through `scan_engine`'s async command/snapshot layer;
  that model is built for fire-and-poll GUI interaction and is the wrong shape for a
  synchronous macro. The console reuses the *read layer and the file model* directly.
- Trade-off (accepted): console reads and GUI scan/table reads serialize on the one
  worker and can delay each other. This is the existing single-funnel property; the
  console is just one more client. `submit_sync` blocks the console thread, not the
  GUI.

### Ownership — decouple the console's file from the GUI's

The façade holds its **own** read-only file handle (`open_file_readonly`) plus a
`shared_ptr<const file_model>` (the immutable, already-parsed structural model).

Rationale: a user closing the file in the GUI mid-macro must not dangle the
console. HDF5 permits multiple read-only opens of the same file, and in-process live
opens are coherent (verified during the browser's own development). Separate handles,
shared immutable model.

### The compiled/interpreted split

The interpreter runs at **`-O0`** (fast turnaround, correctness over codegen speed)
and *orchestrates* the **`-O3`-compiled** `coast_lib` façade, which does the heavy
data lifting (the read pipeline, the dtype dispatch, cache gathers). The macro is
glue; the real work is compiled. This is the right performance model — an interpreted
inner loop over millions of synapses would be unacceptable, but the inner loop lives
in compiled code.

## The `console_api.h` surface

Design rule: **the console header is the parse boundary.** The interpreter parses
*this* header; the implementations are already compiled by the host and reached via
the JIT's process-symbol generator. So the header is kept deliberately lean —
forward-declare the coast internals (`executor`, `file_model`, `chunk_cache`) and
hold them by pointer in the `.cpp`; include only `<span> <cstdint> <string> <vector>
<memory>` and the tiny, pure `scalar_type.h`. The interpreter never has to parse
`file_model.h` / `executor.h`, minimizing parse cost and C++23 surface.

```cpp
namespace h5b::console {

// One row's native bytes. Streaming views are valid until the next ++;
// random-access rows pin their cache block and stay valid while held.
class row_view {
public:
    std::uint64_t index()  const;                 // absolute row
    std::uint64_t width()  const;                 // row_width (flattened)
    std::span<const std::uint64_t> shape() const; // per-row logical shape
    scalar_type   dtype()  const;

    double at(std::uint64_t col) const;               // native → double (always ok)
    template <class T> std::span<const T> as() const; // zero-copy; throws on dtype mismatch
    std::span<const std::byte> bytes() const;

    bool          has_step() const;               // dataset has a step-index sibling
    std::uint64_t step()    const;                // sim step of this row (throws if none)
};

class row_range {                 // single-pass input range; drives read_pipeline
public:
    class iterator;
    iterator begin();
    iterator end();
};

class dataset_handle {
public:
    std::string_view name()  const;
    std::uint64_t    rows()  const;
    std::uint64_t    width() const;
    scalar_type      dtype() const;

    row_range scan() const;                            // whole dataset, streamed+prefetched
    row_range scan(std::uint64_t first, std::uint64_t n) const;
    row_view  row(std::uint64_t i) const;              // random access via chunk_cache

    std::vector<double>        column(std::uint64_t col) const;  // strided, materialized
    std::vector<std::uint64_t> steps() const;                    // step-index column, if any
};

class file_console {              // bound into the interpreter as the global `h5`
public:
    std::vector<std::string> list() const;                 // dataset display names (discovery)
    dataset_handle dataset(std::string_view name) const;   // throws if unknown
    dataset_handle dataset(std::uint32_t index) const;
    file_family    family() const;
};

// Opened on the console thread; or the GUI hands you the currently-open file.
file_console open(const std::filesystem::path& path);      // throws on open failure

} // namespace h5b::console
```

Example macro (`.x mean_activity.cpp`):

```cpp
void mean_activity() {
    auto d = h5.dataset("neuron[3]/activity");
    double s = 0; std::uint64_t n = 0;
    for (auto row : d.scan()) { s += row.at(0); ++n; }   // streamed 4 MB batches, prefetched
    printf("mean=%g over %llu rows\n", s / n, n);
}
```

### Mapping to existing machinery (no reimplementation)

| Façade element            | Backed by                                            |
|---------------------------|------------------------------------------------------|
| `scan()` streaming range  | `read_pipeline` (depth-2 prefetch, ≥4 MB batches)    |
| `row(i)` random access    | `read_block` + `chunk_cache` (block-aligned, pinned) |
| `at(col)` native → double | the existing `dispatch()` in `scalar_type.h`         |
| `column()` / `steps()`    | strided gather over cache blocks                     |
| every read                | `executor::submit_sync`                              |

### Range vs generator

The core is a **hand-rolled single-pass input iterator**, not `std::generator`:
- Single-pass is *honest* about the streaming reality (data arrives in prefetched
  batches); pretending random-access invites accidental O(n²) macros.
- `std::generator` is C++23 + coroutines — whether clang-repl JITs coroutine frames
  cleanly is unverified. A plain iterator avoids that risk and is
  `std::ranges`-compatible anyway. Random access is the separate `row(i)` path.

## Error handling

The read layer returns `result<T>` (`std::expected`) internally. The façade converts
failures to **exceptions** at the boundary (unknown dataset, read failure, `as<T>()`
dtype mismatch). For interactive macros, throw-and-report is the right ergonomics.

Validated (spike `spike_exc`):
- **A `throw` from compiled façade code IS catchable by an interpreted `try/catch`**,
  including by `const std::exception&` base — typeinfo is unified across the JIT
  boundary. So the exception surface is viable.
- **An *uncaught* cross-boundary throw calls `std::terminate` and aborts the
  process.** `ParseAndExecute` does not wrap execution in a catch-all. This is
  general — any macro-level `std::vector::at` / `std::stoi` / `json::parse` throw
  aborts the GUI, not just façade calls.

**Therefore the console REPL MUST wrap executed statements in an interpreter-side
`try/catch` before running them:**

```cpp
try { /* user statement */ }
catch (const std::exception& e) { report(e.what()); }
catch (...)                     { report("unknown exception"); }
```

This is the declaration/statement split ROOT already does: **statements** are
wrapped; **declarations** (function/type definitions) are not — you cannot wrap a
definition in a `try`. A `.x macro.cpp` file defines functions (declarations) and
then invokes one; the *invocation* is the wrapped statement, and an uncaught throw
inside the function propagates to that wrapper.

> **SUPERSEDED (2026-07-15, branch `fix/exec`).** The two paragraphs above are the
> original *wrapping* design, and their central inference — "declarations cannot be
> wrapped, therefore declarations cannot be guarded" — is **false** under the
> host-side catch adopted on 2026-07-05 (see the revision note in Increment 3).
> You do not need to wrap code in order to guard it: catching around `Execute()`
> on the host side guards declarations *and* statements, with no scope introduced
> and so no loss of declaration persistence.
>
> The inference was never revisited when the wrapper was reversed, and it is what
> left `repl::load()` — the `.L`/`.x` path, which runs **user-supplied files** — on
> the raw `exec()`. Measured consequences before the fix: a throwing file-scope
> initializer aborted the process, and a file with a fresh `#include` plus a parse
> error poisoned the interpreter (`load()` also bypassed `split_into_units`).
> `load()` now calls `exec_guarded()`; the declaration/statement split is **not** a
> guarding boundary and must not be treated as one.

Façade refinement: throw **standard** exception types (`std::runtime_error`, …),
whose typeinfo already lives in libstdc++, rather than a custom exception type that
would need its typeinfo explicitly exported via `-rdynamic`.

## Interpreter lifecycle & robustness (validated, spikes `spike_err`/`spike_err2`)

- **Configuration is mandatory**, not optional: pass `-resource-dir
  $(clang -print-resource-dir)` (the embedded driver cannot self-locate Clang's
  builtin headers because `argv[0]` isn't the clang binary — every libc `#include`
  fails without it), and link the host exe with `-rdynamic` / `-Wl,--export-dynamic`
  (so the JIT resolves façade symbols against the process). The original "fatal
  errors segfault" scare was a *misconfigured* interpreter (no resource-dir), not a
  fatal-diagnostic hazard.
- **Ordinary user errors are recoverable in-process**: undeclared identifier, syntax
  error, type error, redefinition, and missing-`#include` all return a clean error
  and leave the interpreter usable.
- **One transient poisoning class**: a JIT materialization failure (calling an
  unlinked symbol) breaks the *immediately next* input, then self-heals. Recipe:
  after any failed submission, call `Undo(1)` to guarantee the next user input starts
  clean. `getSymbolAddress(name)` can pre-validate a symbol without executing.

## Startup cost

`#include <nlohmann/json.hpp>` (header-only, fully JIT-compiled) cost ~1 s
one-time parse+prepare in the spike (single-run, `-O0`). Heavy header-only libraries
should be **pre-included at console startup** (off the GUI thread) so the user never
waits mid-session. The lean `console_api.h` itself is cheap; this only matters if we
expose json/other heavy headers into the session.

## Scope

**v1**: discovery (`list`/`dataset`), streaming `scan()` + random `row(i)`,
`column()`/`steps()` extraction, dtype→double, `.L`/`.x` load/exec, the REPL
host-side exception catch + `#include` isolation + `Undo(1)` recovery.

**Deferred, not blocking**:
- Kernels as façade methods (`histogram`/`pairwise`/`series`) — the GUI has them; the
  console can gain synchronous wrappers over the same `stats_kernel`/`base_histogram`/
  `pairwise` code later.
- `where(row_filter)` filtered ranges — `row_filter` already exists; trivial to add.
- Plotting.
- **Out-of-process execution.** One residual process-killer survives the in-process
  recipe: genuine native faults (null deref) from executing bad JIT'd code, which no
  in-process `try/catch` contains. Only running the interpreter behind an
  out-of-process ORC executor fully protects the GUI. Not worth it for v1; revisit if
  the console is exposed to untrusted macros.

  > **CORRECTED (2026-07-15, branch `fix/exec`).** This item originally listed a
  > throwing *static initializer* as a second uncontainable killer. That was true of
  > the **wrapping** design only — you cannot wrap a declaration in a `try`, so a
  > static init ran unguarded. Under the host-side catch it is **measured as
  > contained**: `int g = boom();` in a `.L`'d macro now returns
  > `uncaught exception: ...` and the process survives (regression test:
  > "console::repl: .L a throwing file-scope init is reported, not fatal"). The claim
  > was not re-tested when the catch moved host-side on 2026-07-05. Native faults
  > remain genuinely uncontainable — that half of the item stands.

## Validation record

All feasibility gates were checked empirically (scratchpad spikes, 2026-07-05,
LLVM/clang-repl 22.1.6):

| Spike        | Question                                              | Result |
|--------------|-------------------------------------------------------|--------|
| `spike`      | embed + C++23 parse + host-symbol call + json stress  | green  |
| `spike_real` | real coast headers parse + real compiled symbol call  | green  |
| `spike_err`  | error-recovery taxonomy (compile/syntax/type/fatal)   | recoverable |
| `spike_err2` | JIT materialization-failure poisoning + Undo recovery | transient, Undo fixes |
| `spike_exc`  | exception unwinding across the JIT boundary           | catchable; uncaught aborts → REPL guard required *(the guard is a HOST-side catch around `Execute()`, not the interpreter-side wrapper this row originally implied — see the 2026-07-05 revision and the 2026-07-15 supersession)* |

## Open items before implementation

- Confirm the `Undo(1)`-immediately-after-failure ordering actually spares the *next*
  user input the collateral materialization error (spike ran probes before the Undo).
- ~~Decide how the GUI binds "the currently open file" into the console session vs. the
  console's own `open(path)` — and how a GUI file-switch is reflected (rebind `h5`).~~
  **RESOLVED (2026-07-18):** no single bound file — the GUI/browser inject
  `auto <var> = open("<path>")` and access is by handle; many files coexist. See
  the update note at the top.
- Confirm a custom façade exception type's typeinfo export if we ever want one beyond
  the standard hierarchy (current plan avoids this by throwing `std::runtime_error`).
```
