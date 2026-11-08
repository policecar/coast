+++
title = "Campaign meta-loader"
weight = 66
+++

# campaign — querying a directory of runs

The **campaign meta-loader** (`coast::campaign`, `include/campaign/campaign_api.h`)
turns a directory tree of simulation runs into something you can ask questions of:

```cpp
auto c   = camp::open("/mnt/NAS/coast-experiments/basic/ngm25_gpu/LRI_sweep");
auto sel = c->where("primary_learning_rate", 0.01)
             .where("local_inhibition_strength", 5.0);
```

It exists because campaigns are now produced in bulk. A single `coastd` sweep
writes one directory per run, and the reference campaign — an 81-point 3⁴
factorial — has no navigable structure beyond 81 sibling directories with an
11 kB `config.json` each. Finding "the runs where the primary learning rate was
0.01" meant reading those files by hand.

It lives in **`coast_lib`**, not the GUI layer: the REPL reaches it, the GUI panel
reaches it, and a headless build still has it. `campaign_api.h` is an interpreter
**parse boundary**, so it follows the same rules as `run_config.h` and
`coastd_console_api.h` — no `nlohmann/json` in the header, internals held by
pimpl, failures reported as `std::runtime_error`.

## What a campaign is

A directory is a **run** if and only if it directly contains `config.json`. The
scan walks the root recursively and does *not* descend into a directory it has
recognised, so both a flat sweep and a whole experiment tree several levels deep
scan correctly, and non-run files sitting beside the runs are ignored.

```
LRI_sweep/
  LRI_sweep.md                        ← ignored
  20260731T190909Z_lrsweep_00406d/
    config.json                       ← this is what makes it a run
    coast_runtime.json                ← manifest: exit reason, steps, wall clock, git sha
    observe/population.observe.h5     ← 2.0 GB
    snapshots/  final_state.h5  logs/
  … 80 more
```

## Axes are discovered, not declared

The swept parameters are found by **diffing the run configs against each other**:
any leaf whose value differs across the campaign is an axis. Nothing has to be
registered, and the loader needs no knowledge of what a sweep was *about*.

This is also what makes short names usable. A coast config carries defaults for
**every** model family, so the leaf `primary_learning_rate` occurs twice — under
`ngm3` and under `ngm25` — in a file where only the `ngm25` copy was ever swept.
(`ngm2` spells its own as `default_primary_learning_rate`, so it does not
collide.)
Because only differing leaves become axes, the dead copies never enter the
namespace and the ambiguity dissolves on its own. Where a genuine collision
remains, names are qualified from the **root section** down —
`ngm25/primary_learning_rate` — and a full JSON Pointer is always accepted in
place of a short name.

A leaf that is missing from *some* runs (which happens when one root spans
several campaigns) contributes `kind::absent` as a distinct value rather than
failing the scan.

### Output locations are denied, not discovered

`/snapshot_dir` differs in every run — `coastd` writes each run's own directory
into its config — but it is an output location, not a hyperparameter. It and
`/observe/output_dir` are excluded by a built-in **deny-list**, extensible via
`scan_options::ignore_pointers`.

This is deliberately *not* the heuristic "as many distinct values as runs implies
not an axis". That rule would misfire on the most ordinary sweep there is: one
parameter, N values, N runs. The deny-list is instead a fact about coast's own
config schema, which is knowledge the loader legitimately has.

## Querying

`where()` does **not** compare with `==`. Before filtering, the requested value is
checked against the axis's *discovered* value set with a relative tolerance
(default `1e-6`), and a value that matches nothing **throws, naming the values
that exist**:

```
no value of axis 'primary_learning_rate' within 1e-06 (relative) of 0.02;
this campaign has {0.01, 0.05, 0.1}
```

Two reasons for this. First, config numbers can arrive float32-round-tripped — a
stored `0.01` reads back as `0.009999999776482582`, a relative error of 2.2e-8 —
so exact equality silently returns nothing. The `1e-6` default sits ~45× above
that measured error (~17× above the worst case, `2^-24`) while decade-spaced
sweep values stay ~10⁵ tolerance units apart. Second, the precheck turns a
mistyped value into a message instead of an empty result.

Runs then match **every** discovered value within tolerance, not just the nearest,
so a campaign that stored one logical value in two representations does not split
into two groups. And the precheck sees the *campaign's* value set, not the current
selection: it guards the first `where()` in a chain. A later `where()` that
contradicts an earlier one, or a cross-axis combination a non-factorial campaign
never ran, still yields an empty selection — which is the honest answer, because
those values do exist.

`selection` is chainable and cheap (indices plus a shared pointer that keeps the
campaign alive). Beyond `where()` it offers `sort_by`, `group_by`, `values`,
`labels`, `ids`, `dirs`, and a `where(predicate)` escape hatch for conditions the
axis filters cannot express.

## Strict laziness

The scan reads **configs and manifests only**. The reference campaign holds
~160 GB of observe data; opening any of it during a scan would make the loader
unusable. Measured on the real campaign over NFS: **81 runs scanned in 0.04 s**.

HDF5 is opened only when a `run` handle is explicitly asked, and the handle is
not cached — so a loop over 81 runs holds one file open at a time:

```cpp
auto s = c->where("local_inhibition_strength", 5.0).sort_by("primary_learning_rate");
std::vector<double> x = s.values("primary_learning_rate"), y;
for (const auto& r : s) y.push_back(my_metric(r.observe()));
```

`run::observe()` returns an `h5b::console::file_console` — the same handle
`open()` gives the REPL — so every existing `scan()` / `row()` / `column()`
facility applies unchanged and no second HDF5 reader exists.

`values()` is aligned element-for-element with iteration order, which is what
makes "x from an axis, y from a metric" correct without bookkeeping. That
pairing is the loader's reason for existing: derive one number per run, then plot
it against the hyperparameter the runs differ in.

## A bad file is reported, never absorbed

Nothing is coerced to a default to make a malformed file look healthy. A field that
is **present with the wrong JSON type** is an error; it is recorded, with the field
name and the offending value, in `campaign::issues()`:

```
coast_runtime.json: field "n_steps_completed" is a float (-1.5); expected an unsigned integer
config.json: parse error: syntax error while parsing object key
```

A field that is merely **absent** is not an error and keeps its default — the manifest
schema is additive (`sim_core/run_manifest.h`), so an older or newer writer
legitimately omits keys, and rejecting that would break compatibility both ways.

The two bad states are now **distinguishable**, which they were not when both
degraded to `present == false`:

| | meaning |
|---|---|
| `present == false`, `error` empty | no manifest yet — a run in flight. Normal. |
| `error` non-empty (`malformed()`) | the manifest is damaged; its fields are not to be trusted. |

Conflating them meant a corrupted provenance record rendered exactly like a healthy
run that simply had not finished.

**A run is dropped only when its `config.json` is unusable** — without it there are no
parameters, so the run cannot enter the diff at all — and even then it is reported with
`run_dropped == true` rather than quietly disappearing. A damaged
`coast_runtime.json` never costs the run: its parameters are still perfectly readable
and only its provenance is in doubt, so discarding it would throw away good
scientific data over a broken metadata file.

Issues surface in three places: `summary()` prints each one in full (a count the
reader has to go and investigate is the same as hiding them), the GUI status line
turns red and lists them on hover, and the run table shows `manifest error` in red
with the message on hover instead of a neutral grey `no manifest`.

This is also what removed an undefined behaviour. The step counts used to be guarded
with `is_number()`, which admits a float and then hands it to nlohmann's
`get<uint64_t>()` — a `static_cast<uint64_t>(double)` that is UB out of range
(measured: `-1.5` yielded `18446744073709551615`, `1e300` yielded `0`). The writer
types both fields as `uint64_t`, so a float there never came from coast; it means the
file is not ours, which is exactly what the loader now says.

**A `null` is not a wrong type.** It is the writer's own encoding of *not
applicable*: `write_run_manifest` emits `gpu_adapter: null` for every CPU-only run
precisely so a consumer can tell "CPU" from "unknown". The first strict version of
this reader rejected it, so every run of a CPU campaign reported a malformed
manifest — 19 false positives out of 19 real runs, measured. It stayed hidden
because the test fixture hand-built a manifest string that *omitted* the field,
the one shape the reader accepted; a DTO written by the test can only ever agree
with the test. The fixture now comes from `write_run_manifest` itself.

### What the walk could not see is an issue too

The axes come from diffing *whatever the walk found*, so a subtree skipped in
silence does not merely lose runs — it changes the answer. Measured before this
was fixed: three runs with `a` = 1, 2, 3 where the third sat under an unreadable
parent reported **"1 run, no swept parameters (all configs identical)"**. Every
directory the walk cannot enter, refuses to follow (a symlink resolving outside
the root), or leaves beyond `max_depth` is now reported as an issue with `file`
empty. Such an issue leaves `run_dropped == false`: how many runs are missing
below it is unknown, and an unknown must not be added to a count presented as
exact.

## Surfaces

**REPL.** The preamble aliases the namespace as `camp` (`src/console/repl.cpp`).
It is an alias rather than a `using namespace` because the free function `open()`
would otherwise collide with `h5b::console::open()`, which is pulled in
unqualified and opens a single file rather than a campaign.

**GUI.** *Analysis → Campaign Browser* (`gui_tools/campaign_panel`) is a thin
front over the same façade: a run table with one column per discovered axis,
value filters per axis, manifest status, "open this run's observe file in the
HDF5 browser", and handing runs to the console as a variable. The **scan runs on
a background thread** — a campaign root is typically on network storage, where a
stat storm or a vanished mount would otherwise block the ImGui loop.

### Two ways to say "these runs"

The panel deliberately carries **two** notions of a run set, and they answer
different questions:

- the **axis filters** (left pane) — *which runs match these parameter values*;
- the **picked rows** (table) — *these specific ones*, via ImGui's
  `BeginMultiSelect`: click, Ctrl+click to add, Shift+click for a range, Ctrl+A
  for all, Escape to clear, box-select by dragging.

`Ctrl+A` is handled by the panel rather than left to `BeginMultiSelect`'s own
shortcut (`Escape` is handled both ways — the scope also sets
`ImGuiMultiSelectFlags_ClearOnEscape`). Measured: those never fire for this table —
with `io.KeyCtrl == 1` and `IsKeyDown(A) == 1` on the frame and keyboard capture
on, the selection did not change. ImGui routes them to the window owning the
multi-select scope, and the scope here lives inside a `BeginChild` the routing
does not reach. Mouse click / Ctrl+click / Shift+range go through the normal path
and are left to it.

Picked rows **win** when both are in play: picking a row is a more specific act
than setting a filter. Rather than leave that precedence to be guessed at, the
button *names what it will send* — `12 picked rows -> Console` versus
`81 filtered runs -> Console`.

Picked rows are keyed by row index, which is only stable while the filter is, so
changing a filter **clears them** instead of leaving a selection that silently
refers to different runs than when it was made.

What reaches the console is **exact** either way. With rows picked, or with a
filter that no `where()` chain can express, the panel emits `select({ids…})`;
when every axis is pinned to a single value it emits the `where()` chain the user
would have typed, which stays readable in the console history.

### Handing work to another panel also raises it

Both hand-offs make their target visible first. Loading a file into a hidden HDF5
browser, or submitting a variable into a hidden console, puts the result somewhere
the user cannot see and reads as a button that did nothing.

- **Console**: `console_iface::set_visible(true)` before `submit()`, which also
  pre-warms the interpreter so the statement is not waiting on the one-time build.
  Same convention as `h5_browser_window::bind_console_variable`.
- **HDF5 browser**: `open_path()` raises the window with `SetNextWindowFocus` (so an
  already-open but buried browser surfaces) and records a visibility request that
  `gui_manager` consumes via `take_visible_request()` — the show flag is owned there,
  since the browser is rendered as `render(bool* open)`. The request is consumed at
  most once, so it cannot fight a user who closes the window afterwards.

## Tests

`unit.campaign.meta_loader` (`tests/unit/campaign/`) covers axis discovery, the
deny-list, name disambiguation, the float32 tolerance, the
value-matches-nothing message, incomplete and corrupt runs, nested roots, and
extraction alignment — all against fixtures synthesized into a temp directory. A
final case runs against the real campaign on the NAS and **skips when it is not
mounted**.

`unit.console.repl` drives the loader through the interpreter, which is what
proves the façade symbols resolve across the JIT boundary and that a
`shared_ptr<campaign>` and a by-value `selection` survive being held across
cells.
