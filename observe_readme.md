# OBSERVE — offline data capture for COAST

The OBSERVE system is now documented in the Hugo docs site under `docs/`. It is the
single source of truth — keep it in sync when changing the `observe` module.

| Topic | Page |
|---|---|
| Module reference (macros, supported types, config schema, output format, lifecycle) | `docs/content/architecture/modules/observe.md` |
| GUI panel (Observer Capture) | `docs/content/architecture/modules/gui_observe.md` |
| Data flow (capture pipeline, gate-vs-commit, HDF5 executor) | `docs/content/data-flow/observe-pipeline.md` |
| File reference | `docs/content/reference/_index.md` (`observe/` and `gui_observe/` sections) |

Build and browse locally:

```bash
cd docs && hugo server   # then open the printed localhost URL
```
