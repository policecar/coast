+++
title = "Application config"
weight = 51
+++

# Application config — where things go

The **application config** (`coast::paths::app_config`, `include/core/app_config.h`)
is a single, app-wide file that controls **where COAST puts its outputs** and
**where it finds its own assets**. It lives in the low-level `core/` module (next
to the pure `coast::paths` output resolver), not in `app/`: it is leaf-level
configuration data — depending only on the standard library, `nlohmann/json`, and
`diag` — that is loaded *sideways* into several layers (`app/`, `gui_core/`,
`sim_core/`), so it must sit **below** all of them. It is deliberately **distinct** from the
per-experiment simulation config (`configs/*.json`, `loader::global_cfg_t`): the
experiment config describes *one simulation*; the app config describes *file
locations*, app-wide, across every config you run.

It is loaded **once at startup** in `main()` (before `build_env`) and threaded
explicitly — no singleton — into `build_env`, `run_headless`, and `run_gui`.

## The file is optional

Every field has a built-in default that reproduces the intended layout, so **no
file is required**: with none present, the defaults apply. A file only *overrides*
the fields it names (nlohmann `_WITH_DEFAULT`, the same pattern as the experiment
config).

To start from a template rather than the docs, run **`coast --init`** — it writes
a `./coast.json` with every field at its default (a pure CLI action that then
exits) and **never overwrites** an existing `coast.json`.

### Discovery precedence

1. `--app-config <path>` (explicit, always wins)
2. `$COAST_APP_CONFIG`
3. `./coast.json` (the CWD default path)
4. built-in defaults

A file named on the command line via **`--app-config`** that is missing or
malformed is a **hard error** — an explicit, in-the-moment request must be honored,
not silently ignored. Both auto-discovered sources (**`$COAST_APP_CONFIG`** and
**`./coast.json`**) are **soft**: a missing or malformed file logs a warning and
falls through to the next source / built-in defaults (the app must still run). An
exported `$COAST_APP_CONFIG` can outlive the file it named, so treating a stale one as
fatal would brick every invocation for a cause far from the command being run.

## `config_base` — the per-config output template

`config_base` is a template string that governs where a config's per-experiment
outputs go. Two tokens are expanded against the **config path as supplied on the
CLI**:

| Token | Expands to | Where it may appear |
|---|---|---|
| `<config_dir>` | the directory holding the `.json` (or `.` when the config was given as a bare filename) | **whole path component only** (`<config_dir>/…`) |
| `<config>` | the config file stem (e.g. `mnist` for `configs/mnist.json`) | anywhere in a component (`<config>`, `run_<config>`, `<config>_out`) |

The default is `config_base = "<config_dir>/<config>/"`, so `configs/mnist.json`
writes its outputs under `configs/mnist/…` — next to the config, isolated per
config.

`<config_dir>` expands **only when it is a whole path component**, because it can
be empty, multi-segment, or absolute — splicing such a value into the middle of a
segment (`x<config_dir>y`) is ill-defined, so a glued `<config_dir>` is left
literal and logs a warning. `<config>` is always a single filename stem, so it
substitutes cleanly anywhere within a component.

### Resolution algorithm

For a per-config output with leaf `L` (e.g. the config's `snapshot_dir`):

```
if is_absolute(L):            # an explicit absolute leaf always wins
    return L
base = expand(config_base)    # tokens -> path components; empty <config_dir> -> "."
return base / L               # std::filesystem::path::operator/
```

The path is **composed component-wise** (`coast::paths::expand_config_base`): the
template is walked one path segment at a time and each is joined with
`std::filesystem::path::operator/`. Tokens are substituted *within* a segment
(`<config>`) or *as* a segment (`<config_dir>`), but the segments are never
concatenated as a raw string. This is what avoids the collapse a naive whole-string
`"<config_dir>/<config>"` substitution would cause: a config given as a bare
filename (`coast mnist.json`) has an empty `parent_path()`, and string substitution
would yield the leading-slash **absolute** `/mnist/…`. Composing per segment (with
an empty `<config_dir>` mapped to `.`) keeps a relative template relative.

A leaf is either a plain relative name (placed under the base) or an absolute path
(a sanctioned explicit escape that wins). A **relative** leaf containing a `..`
component is neither — it would climb *above* `config_base` and drop outputs
outside the resolved tree. That is rejected at **config load** (both the app config
and the experiment config), so the pure per-frame resolver never has to guard it:
`coast::paths::leaf_escapes_config_base` is the predicate, and `snapshot_dir` /
`observe.output_dir` / the `paths.*` defaults carrying a `..` are a hard error. To
point outputs at a sibling tree, give an **absolute** path instead.

Worked examples (`config configs/mnist.json`, run from the repo root):

| `config_base` | leaf `snapshot_dir` | result |
|---|---|---|
| `<config_dir>/<config>/` (default) | `snapshots` | `configs/mnist/snapshots/` |
| `<config_dir>/<config>/` | `/tmp/s` (absolute) | `/tmp/s/` (leaf wins) |
| `./runs/<config>/` | `snapshots` | `<cwd>/runs/mnist/snapshots/` |
| `/data/<config>/` | `snapshots` | `/data/mnist/snapshots/` |
| `<config_dir>/<config>/` | `../out` (relative `..`) | **rejected at load** |

### The hard invariant

`config_base` is applied **only** to values that come from a config file (the
experiment config's `snapshot_dir` / `observe.output_dir`) or from a built-in GUI
default. A path a user **types on the command line** (`--save-state`,
`--load-state`, `--log-buffers`, `--emit-run-manifest`, `--log-file`, the config
path) is **never** rebased under `config_base` — it stays verbatim (Tier-1
semantics, see [json_cfg]({{% relref "/architecture/modules/json_cfg" %}})).

### Layering with the experiment config

For `snapshot_dir` / `observe.output_dir`, the **experiment config's value wins
when non-empty**; an empty leaf (`""`) defers to the app-config default leaf
(`paths.snapshot_dir` / `paths.observe_dir`). Both then resolve under
`config_base`. So the experiment config chooses the *leaf name*; the app config
chooses the *base* and supplies the default leaf.

## Schema

```json
{
  "paths": {
    "config_base":  "<config_dir>/<config>/",
    "snapshot_dir": "snapshots",   // default leaf when the experiment config omits it
    "observe_dir":  "observe",
    "log_dir":      "logs"
  },
  "assets": {
    "font": ""                     // empty -> compile-time COAST_FONT_PATH
  },
  "state": {
    "plot_themes": "coast_plot_themes.json",
    "imgui_ini":   "imgui.ini"
  },
  "output": {
    "plot_dir": "."
  }
}
```

## What each field governs

| Field | Consumer | Default behavior |
|---|---|---|
| `paths.config_base` | the output resolvers (`resolve_snapshot_output_dir`; `sim_instance` for observe) | outputs land under `<config_dir>/<config>/` |
| `paths.snapshot_dir` | headless resume + `--checkpoint`, GUI auto-save / resume / auto-checkpoint | `<base>/snapshots/` |
| `paths.observe_dir` | `observe_registry` attach (`sim_instance`) | `<base>/observe/` |
| `paths.log_dir` | GUI logging window default **buffer log** name | `<base>/logs/` (the state save/load default goes under `paths.snapshot_dir`, not here) |
| `assets.font` | GUI font (`sdl_app`) | build-copied absolute `COAST_FONT_PATH` (run-anywhere) |
| `state.imgui_ini` | ImGui `io.IniFilename` (`sdl_app`) | `imgui.ini` (CWD) |
| `state.plot_themes` | plot theme persistence (`plot_tools`) | `coast_plot_themes.json` (CWD) |
| `output.plot_dir` | save-PNG / save-HDF5 dialog default dir (`plot_tools`) | `.` (CWD) |

The font is the one field whose default is **not** CWD-relative: the tracked asset
`assets/DejaVuSans.ttf` is copied into the build tree at build time and pointed to
by an absolute `COAST_FONT_PATH` compile-def, so the GUI finds its font from any
working directory (`assets.font` overrides it).

## Consistency with `simc`

The console sim-control layer (`coast::simc`) stages a run directory and writes
**absolute** `snapshot_dir` / `observe.output_dir` into the child config. Because
an absolute leaf always wins, those bypass `config_base` entirely — simc and the
app-config default coexist with no conflict.
