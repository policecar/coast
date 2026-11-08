# run_config extraction — coastd console gets the simc workflow

**Status:** implemented (2026-07-23) — all steps below done except the two
explicitly deferred items. Validated: full + headless builds clean, simc B-tier
green (extraction behavior-preserving), coastd console edit-then-submit test
green in both configs and **mutation-proofed** (sending the base value instead of
`cfg.dump()` fails exactly the three edit-content assertions).
**Goal:** the coastd REPL console must offer the **same workflow as `simc`** —
load a base config, **edit it in the console**, then dispatch it — not
`submit(ref, path)`. Requirement origin: `design/remote_run_dispatch.md` §1/§5
("the *same* workflow to target a different machine … no console-facing API
change"). The daemon replaced the SSH transport; the console contract did not.

## Root cause

`submit(ref, path)` was written to a file-path model, never to the simc
requirement. The editable config type (`simc::run_config`) lives in `coast_gui`
(the sim_console façade), and `coastd_console` lives in `coast_lib`; `coast_lib`
cannot link `coast_gui`, so the object was simply unreachable from where submit
is. (This library split arrived *after* the daemon, by rebase — it is not why the
shortcut was taken, it just makes the fix a genuine extraction.)

## Design

Extract the **editable config** into `coast_lib` as a standalone, GUI-free value
type shared by both façades. `sim_run` and the ImGui run dashboard stay in
`coast_gui`. The cut is exactly along the boundary the `sim_console` CMake
comment already anticipated ("split … and move the façade to coast_lib").

- New module `coast::cfg` in **coast_lib**: `include/run_config/run_config.h`,
  `src/run_config/run_config.cpp`. Parse-boundary-safe header (pimpl, **no
  nlohmann/json exposed** — same discipline as the two console headers).
- `run_config` becomes fully public-interface (no friends): add read accessors
  `base_path()`, `run_dir_hint()`, `seed_value()`. `config()` (the factory) moves
  with the type and stays its friend.
- `simc` re-exports: `namespace coast::simc { using cfg::run_config; using
  cfg::config; }`. `run()` is rewritten to use only the public surface (copy +
  `set()` the two output pointers + `dump()`), so it needs no internal access.
  All existing simc code/tests compile unchanged.
- `coastd` gains the same surface: `using cfg::run_config; using cfg::config;`
  plus a new overload `session::submit(ref, const run_config& cfg, …)` that sends
  `cfg.dump()` as `config_text`. The path overload stays as sugar (no ambiguity —
  `run_config` is not constructible from a string literal).

Target usage (REPL):
```cpp
auto c = coastd::config("base.json");
c.set("/ngm2/dendrites/default/default_primary_learning_rate", 0.03);
auto id = s.submit("main", c, "sweep-lr", "", 1'000'000);
```

`run_dir()`/`seed()` remain on the shared type (simc staging hints); coastd
ignores them — the daemon assigns the run dir and rewrites `snapshot_dir`. Documented, not silently surprising.

## Steps

1. [x] `include/run_config/run_config.h` — class moved verbatim (drop `run()`
       friend; add the 3 getters), `config()` decl, namespace `coast::cfg`.
2. [x] `src/run_config/run_config.cpp` — method impls + `config()` +
       `to_pointer`/`cfg_json` helpers moved out of sim_console_api.cpp.
3. [x] CMake — add the two files to `coast_lib`.
4. [x] `sim_console_api.h` — drop the class + `config()` decl; include the new
       header; `using` re-export into `coast::simc`.
5. [x] `sim_console_api.cpp` — remove moved code; rewrite `run()` via public
       accessors; drop now-unused json include if applicable.
6. [x] `coastd_console_api.h` — include header; `using` into `coast::coastd`;
       declare `submit(ref, const run_config&, …)`.
7. [x] `coastd_console_api.cpp` — implement object submit (`config_text = dump()`).
8. [x] Build full + headless; run simc and coastd console tests.
9. [x] Test: edit-then-submit against a fake/real daemon (mutation-proof it).
10. [x] Docs: INSTALL.md §8.2, coastd console docs, header workflow comments.

## Deferred (explicit)

- GUI `coastd_panel` in-place config editor. The panel keeps its path field and
  the retained path overload, so it is unaffected; an edit UI is a follow-up.
- Plumbing a seed into coastd `submit_args` (no seed field on the wire today).
