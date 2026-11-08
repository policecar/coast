+++
title = "logging"
weight = 65
+++

# logging — Data Persistence

The `logging` module provides two persistence capabilities: buffer logging and state checkpointing. Both use HDF5.

Per-probe values are written to CSV by `probe::csv_file_observer` (Probe Browser → *CSV File...*), not by this module — see [probe]({{% relref "/architecture/modules/probe" %}}).

{{% callout note %}}
This is **simulation-data** persistence, not text diagnostics. For levelled log records (`COAST_INFO`/`WARN`/`ERROR`) and run-progress output, see [diag]({{% relref "/architecture/modules/diag" %}}) — a deliberately separate module (`coast::diag`, `include/diag/`).
{{% /callout %}}

## Directory Layout

```
include/logging/
├── logging.h                  # Convenience include
├── logging_common.h           # Shared types: log_result, compression_config, etc.
├── logging_manager.h          # Central coordinator
├── buffer_logger.h            # Per-step IO buffer capture
├── state_logger.h             # Full state save/load (extract/serialize/parse/apply)
├── state_io.h                 # state_writer/state_reader interfaces
├── state_checks.h             # check_shape: refuse a checkpoint of another shape
├── mem_state_io.h             # In-memory state tree (mem_group) + mem_state_writer/reader
└── hdf5_utils.h               # RAII wrappers for HDF5 C API

src/logging/
├── logging_manager.cpp
├── buffer_logger.cpp          # Hook-based capture, chunked writes
├── state_logger.cpp           # hdf5_state_writer/reader + the state-IO split
├── mem_state_io.cpp           # Tree navigation, HDF5 replay/parse
├── state_checks.cpp
└── hdf5_utils.cpp
```

## HDF5 Utilities (hdf5_utils.h)

RAII wrappers for all HDF5 handle types:

| Wrapper | C API | Purpose |
|---------|-------|---------|
| `hdf5::file` | `H5Fopen/create/close` | File access (read, readwrite, create) |
| `hdf5::group` | `H5Gcreate/open/close` | Hierarchical groups (recursive creation) |
| `hdf5::dataspace` | `H5Screate*` | Shape descriptors (scalar, simple, extensible) |
| `hdf5::plist` | `H5Pcreate` | Property lists (chunking, compression) |
| `hdf5::dataset` | `H5Dcreate/open/close` | Data containers with extend/write/read |

Plus free functions: `write_attr()`, `read_attr_*()`, `link_exists()`, `append_rows()`.

These wrappers call `H5*close` directly, so they are used inside an executor job (`hdf5::executor::submit_sync`). `hdf5/handle.h` is the second RAII layer: its handle *posts* the close to the executor and may be destroyed on any thread (observe's per-site datasets). Both obey the same rule — libhdf5 is entered only from the executor thread — and are kept apart on purpose: the writers here batch a whole save or frame flush into one job and must check `H5Fclose`'s status synchronously. A wrapper held as a member of a longer-lived object (`buffer_logger`, `buffer_log_reader`) is reset by its owner inside a job, and the owner's move-assignment is deleted or does the same (a defaulted one would close on the caller's thread).

### Reading files COAST did not write

`H5Dread` with `H5S_ALL`, and every `H5Aread`, transfers as many elements as the
**file** declares, whatever buffer the caller allocated. A reader that sizes its
buffer from part of the shape therefore overflows it with a length and content
chosen by the file (issue #150). Every reader of a checkpoint or buffer log goes
through these guards instead, which throw `std::runtime_error` naming the object:

| Helper | Checks |
|--------|--------|
| `require_rank(ds, rank)` | exact rank; rank 0 must be a scalar dataspace (`H5S_NULL` also reports rank 0 but holds nothing, and a read from it "succeeds" without writing) |
| `element_count(ds, dims)` | product of the dims fits `std::size_t` |
| `require_stored(ds, n)` | the file stores the `n` elements it declares (#159): dims are metadata, so a small file can declare a 64 GiB dataset with no storage behind it. Contiguous or compact: the bytes stored; chunked: the allocated chunks at their uncompressed size, so a fully written compressed dataset passes. A dataset with unwritten regions (fill value) beyond its allocated storage is refused, though valid HDF5 (no COAST writer leaves one), as are external and virtual storage. Data that is really stored but compresses well (deflate ~1000:1) can still ask for far more memory than the file's size |
| `read_all(ds, type, dst, n)` | reads through a memory dataspace of `n` elements, so HDF5 itself refuses any other count; a failed read or type conversion throws |
| `read_array<T>(ds, type, rank)` | `require_rank` → `element_count` → `require_stored` → `read_all`: the one place a buffer is sized from a dataset's shape; returns the values and the dims |
| `read_fixed_strings(ds)` | a 1-D dataset of fixed-length strings, each bounded by its slot; a variable-length or non-string type throws |
| `read_attr_*(loc, name)` | `nullopt` only when absent; otherwise exactly one element, a successful read, and for strings one fixed-length string (a variable-length one would store a `char*`) |

#### Both state-load paths refuse the same files

A state file may hold only what `state_writer` produces: datasets of rank 0–2 with
an integer or float type, a non-null dataspace and the storage they declare, attributes holding one
fixed-length string, and hard links forming a tree (each object reached by one
link, at most 64 groups deep). One walk (`detail::parse_group`) enforces that over
the **whole** file, read by a load or not, before any entity is loaded. The tree
rule also protects the walk itself: a link back to an ancestor would recurse until
the stack overflows, shared groups would be walked exponentially often, and a soft
or external link could point it anywhere, including another file.

Both paths first pass the format-version gate (`metadata/schema_version`, one
attribute read), so a newer format that holds objects this build refuses is
reported as newer rather than as malformed. Then:

- **Runtime** (GUI *Load state* / *Resume*): `parse_state` → `parse_hdf5_to_tree`
  parses with the walk and refuses before `apply_state`.
- **Startup** (`--load-state`, `--resume`): `load_state` is the same
  `parse_hdf5_to_tree` + `apply_state`, run back to back in one executor job
  (the startup exemption), so the two paths cannot drift apart.

What only the read can know — whether a field has the rank its reader asks for —
is checked by the read, the same way in both readers: `hdf5_state_reader` and
`mem_state_reader` throw on a field of another rank, or on a name that is a
group. At startup the process exits; at runtime the throw comes from inside
`apply_state`, so the env may be partly loaded and the sim halts with
`partial_load`, like an entity's own `load_state` refusal.

Both paths turn a throw into `state_load_result::error_message`.

## Buffer Logger

Captures all IO buffer contents at each simulation step.

**Mechanism**: Registers a `post_swap_hook` that runs on the simulation thread after each `swap_io()`. Copies buffer data into a frame buffer; when `chunk_size` frames accumulate (default 1000), batch-writes to HDF5.

**Steps and timestamps**: each frame records the *simulation* step (`probe::get_current_step()`, the counter that probe samples and OBSERVE output also carry), so a log started mid-run or after `--load-state` lines up with probe and OBSERVE output. Step *k* is the 0-based index of the step whose post-swap state the frame holds. Checkpoints are labelled with the number of *completed* steps, so frame *k* holds the state of the checkpoint labelled *k+1* (after loading checkpoint *N*, the first frame is step *N*). `@start_step` is the step of the first logged frame, written with the first batch (`start()` cannot know it, because the GUI can start logging while the sim runs); an empty log has none. `timestamps` are seconds since this session's `start()`.

**Stopping while the sim runs**: the hook must not be running while `stop()` finishes the file. `logging_manager::stop_buffer_logging(ctrl)` detaches the hook inside `ctrl.run_quiesced`, between steps (#197), so after that `stop()` writes the last frames and closes the file with no hook running. Since #235 env hook removal is synchronous from any thread, which makes the quiesce redundant; it is kept so the #197 path is unchanged. `buffer_logger::stop()` on its own requires that no step can run: the hook was already detached at a quiesced boundary, or the sim thread has been joined (`~sim_instance` destroys the controller before the logging manager).

**Stats**: `stats()` returns a snapshot of two atomics published after each written batch. It is safe to call from any thread; the GUI and the headless progress line poll it while the executor writes.

**Write failures** (#203): every HDF5 result in the write path is checked (`extend`, `write`, `select_hyperslab`, `flush` and `close` are `[[nodiscard]]` in `hdf5_utils.h`). The first failure (disk full, file size limit, I/O error) is logged with `COAST_ERROR` and trips a sticky flag, `write_ok()`, that any thread may read. After it, no further batch is written, because row offsets and dataset extents may no longer agree, and the failed batch is not counted in `stats()`. `stop()` then returns an error. The headless run exits 1 (after `--save-state`, which is still written) and ends its progress feed with `phase=failed`, and the Logging window shows the failure while recording and the error after Stop. A failure in the topology datasets fails `start()`; the metadata attributes only warn, as before.

A full disk usually does not fail `H5Dwrite`: HDF5 caches chunks (8 MiB per dataset by default in HDF5 2.2), so the error first appears when they are written out. `stop()` therefore calls `H5Fflush` and checks it before releasing the dataset handles, whose `H5Dclose` would otherwise take the error and drop it in the RAII destructor, and then checks `H5Fclose`. `frames_written` counts frames HDF5 *accepted*, not frames that can be read back: a log that hit a full disk usually cannot be opened at all, because its metadata was never written. HDF5 keeps such a file open after the failed close.

`logging_manager::combined_stats()` keeps the final counts of a stopped session — including the frames `stop()` itself flushed — until the next `start_buffer_logging()`. The headless run relies on this: it emits its terminal progress event (`done`, `failed`, or `step` after SIGTERM) only after logging has stopped and `--save-state` has been written, so the event reports the run's real outcome and its full `logged=` count.

**HDF5 Schema**:
```
/metadata/
    @type, @schema_version, @start_step, @config, @created
    topology/
        buffer_ids [N], buffer_sizes [N], buffer_names [N]
        connectivity [E x 2], vis_hints [N x 3]
/steps/
    step_numbers [T], timestamps [T]
    buffers/<id>/
        data [T x S]    (float32, chunked + compressed)
```

## State Logger

Full simulation state save/load using abstract `state_writer` / `state_reader` interfaces.

### Interfaces

**`state_writer`**: `begin_group/end_group`, `write(name, scalar/vector)`, `write_2d()`, `write_attr()`

**`state_reader`**: `enter_group/leave_group`, `has_field()`, `read_*(name, default)` with graceful fallbacks, and `read_scalars()` — every scalar field and string attribute of the current group with its written type, for entries a reader cannot name in advance

Both have two implementations each: the HDF5 pair (`hdf5_state_writer`, `hdf5_state_reader`, group stack for nesting) and the in-memory pair (`mem_state_writer` builds a `mem_group` tree; `mem_state_reader` navigates one — `mem_state_io.h`). Entities only ever see the abstract interface, so they are oblivious to which backend is in play.

### The extract / serialize / parse / apply split

The runtime save/load path is split into four static, stateless functions that respect the thread boundaries (see [Threading]({{% relref "/threading" %}}) and [Logging Pipeline]({{% relref "/data-flow/logging-pipeline" %}})):

| Function | Thread | Does |
|----------|--------|------|
| `extract_state(env, step) → mem_group` | Sim (in a quiesced task) | Buffer `snapshot()` (incl. GPU readback) + `entity.save_state(mem_writer)` into a tree. No HDF5. An entity's throw (e.g. a GPU group's failed readback of learned state, #234) propagates and no file is written, so the last good checkpoint stays. |
| `serialize_state(tree, cfg, exec) → future<log_result>` | Executor (async) | Adds save-event metadata (config, timestamp), **moves** the tree into the job, replays it through `hdf5_state_writer`. Returns the write future. |
| `parse_state(path, exec) → {result, tree}` | Executor (blocking) | Generic HDF5 → tree parse + metadata/schema check. |
| `apply_state(env, tree) → state_load_result` | Sim (in a quiesced task) | Buffer `restore()` into the **read page** (no swap) + `entity.load_state(mem_reader)`. Returns `restored_step` for the caller's `set_step_count`. |

Because serialize replays the exact `state_writer` call sequence the entities originally made, it reproduces the established HDF5 structure by construction (`coast_compare`-clean; HDF5 is not byte-reproducible). The moved-in tree is the race-freedom invariant: after handoff the sim thread never touches it, and the executor never touches live sim state.

The monolithic `save_state` / `load_state` members remain for two narrow roles: `save_state` as a single-thread convenience for tests, and `load_state` as the **startup-load exemption** — `build_env` loads before the sim thread has ever stepped, when the main thread is the sole user of the wgpu queue, so it may parse and apply inline (`design/quiesced-state-access.md` §5.2). Any load after stepping begins must use the split.

**HDF5 Schema**:
```
/metadata/
    @schema_version, @type, @config, @timestamp
    step_count
/io_buffers/<id>/data [S]
/entities/<id>/ ... (entity-specific via save_state/load_state)
/entities/<id>/config/  (neuron groups, retina nodes: configuration record, below)
/entities/<id>/{apical,proximal}_channels/   (ngm25, ngm3: channel record, below)
/entities/<id>/neurons/<n>/dendrites/<d>/input_channels/   (ngm2: channel record)
```

**Configuration record** (`config_record.h`, SPEC.md M.7.8). On load the configuration wins: a neuron group's settings come from the configuration it was built with, never from the checkpoint. So that a changed configuration is not continued blindly, each group writes the resolved parameters it runs with — numbers as datasets, strings as attributes — into a `config` subgroup, and `load_state` compares that record with its current parameters (`check_config_record`). Every difference is reported in one `COAST_WARN("checkpoint", …)` per group, e.g. `common_learning_rate 0.001 -> 0.003; effective_seed 0 -> 1 (not applied: the random state comes from the checkpoint)`, including entries only the record has (read with `read_scalars()`), e.g. `neuron[n]#5.branch_interval 7000 -> (not in the current configuration)`; seeds are tagged random state, because the checkpoint's RNG state (and `random_index`'s matrix) wins so a resume stays exact. A record that matches reports nothing. A checkpoint written before records existed loads with one notice: `check_config_record` notes each group without a record on the reader it was loaded from (`state_reader::note_unrecorded_config`), and `apply_state` / `load_state` report them all in a single notice after the entity loop (`report_unrecorded_configs`). ngm3 and ngm25 share one record function between their CPU and GPU groups, so a checkpoint compares the same way across backends, and the two models build it from one field list (`ngm/common_config_record.h`; ngm25 adds `penalty_policy`). Retina nodes write one too (their filter parameters, noise range and seed); dataset nodes do not.

**Channel record** (`config_record.h`, SPEC.md M.7.5). Each group records the channels its synapses are laid out by, by name and width in order: ngm2 one `input_channels` per dendrite, ngm25 and ngm3 `apical_channels` and `proximal_channels`. ngm2's record names the group-level channel each of a dendrite's channels resolves to (the name at the group's own `init_io` input map), not the dendrite's own, local channel key — a channel_mappings hop can rename a channel again on its way from the group to the dendrite without the dendrite's own declaration changing, so the dendrite's local key alone cannot tell two differently-wired configurations apart. A load whose record differs in name, order or width is refused, naming both layouts; it does not cover which underlying source feeds a name, so swapping the sources of two equal-width channels under unchanged names still loads. Checkpoints before format version 4 load unchecked; each such group is noted on the reader (`state_reader::note_unrecorded_channels`), and the load reports them in one notice after its entity loop (`report_unrecorded_channels`).

**A checkpoint that does not fit is refused** (`state_checks.h`). Unlike the record, which compares settings and warns, these checks throw. Every neuron group's `load_state` runs, in order: `check_shape` on its structural fields (neuron, dendrite and bank counts — ngm25 also the apical/proximal split from format version 4 on, since two groups of the same total dendrite count can still disagree on how it splits between the banks — branch depth, input sizes — whichever its layout depends on), which throws `<group>: checkpoint does not fit this group — <field> <saved> (checkpoint) vs <built> (configured)` (or `<field> missing`), listing every differing field (a caller whose field set depends on the checkpoint's format version builds it into one list and makes one `check_shape` call, so every mismatch is reported together rather than only the first); then `check_channel_record` on each channel record; then `check_config_record`; then the payload, by one rule (written once in `state_checks.h`): a field **absent** from the checkpoint is a *pre-field* checkpoint, written before the format gained it, and the group keeps its constructed value — this holds only for fields added after checkpoints already existed, such as ngm3's `neuron_mx_act` and the RNG streams (both added 2026-06-21, in `dd12721f` and `98122325`); a field **present but mis-sized**, or absent although the format has always carried it, is corrupt and throws `<group>: corrupt checkpoint — <what>` (ngm2 names the neuron and dendrite, e.g. `neuron 5: dendrite 2:`). An RNG group is all or nothing: `mt`, `tempered` and `mti` have been written together since the group exists, so `logging::read_rng` (`rng_state_io.h`) keeps the seeded stream only when the whole group is absent, and for every entity — neuron groups, datasets, retina — throws `<who>: corrupt checkpoint — rng <name>: <what>` when it is present but incomplete (`mt has 623 words, expected 624`, `mti missing`, `mti 625 is past the batch of 624`). Only then does it adopt the state. One deliberate exception: a `random_index` inhibitor whose checkpoint claims a matrix but holds none, or a non-square one, warns and regenerates it from the checkpoint's seed instead of refusing the load (a matrix of the wrong size for the group is refused). A refused checkpoint used to be skipped without a word while the step counter and the io buffers were still restored, so a run claimed a resume it never did.

ngm2 runs its own checks a level deeper than `check_shape`'s scalar fields: a read-only pre-pass walks every neuron and dendrite in the checkpoint before anything in the group is assigned, checking each neuron's dendrite count and each dendrite's own shape — from format version 5 on its type (`apical`) and tree depth (`max_branch_level`), then its synapse arrays against its tree depth, read via the same `dendrite_t::check_shape` that `load_state` uses to adopt them — and, from format version 4 on, each dendrite's channel record. An older checkpoint has no type or depth field: its depths are checked against the configuration record's `dendrite[<cfg>].max_branch_level` (refused on a difference, before anything is assigned), and its types cannot be checked; each such group is noted on the reader (`state_reader::note_unchecked_dendrite_types`) and the load reports them in one notice after its entity loop (`report_unchecked_dendrite_types`, `state_checks.h`), naming any dendrite cfg whose depth the record lacks too. Without the pre-pass, a checkpoint whose tree is too deep for a *later* neuron's dendrite was refused only once the load loop reached it, after the group's rng and every earlier neuron had already been assigned.

What a throw leaves behind depends on the caller. At startup (`build_env`, `--load-state` / `--resume`) the half-built sim is discarded and `coast` exits. At runtime `logging_manager::load_state` returns `success=false` with the message and halts the run (`ctrl.halt`, fault kind `partial_load`, inside the same quiesced task, so no step runs on the partly loaded env): the apply is **not atomic across the env**, so the io buffers and the entities before the refusing one stay restored, and the step counter is not set. Within one group, ngm3 and ngm25 (CPU and GPU) stage their whole payload and are unchanged on a throw (ngm25: `stage_checkpoint`, `hd_ngm25_host.h`, which runs `check_arena_consistency` on the staged arena; the inhibitor loads its own part, all or nothing and checked against the group's neuron count, before the group commits); ngm2 checks each piece before assigning it but can be left partly loaded.

**Format version.** `@schema_version` holds the checkpoint format version, an integer (`logging::k_format_version` in `state_io.h`, currently 5: version 2 added every model group's `activity_window_tracking/` edges, version 3 ngm2's declared channel order, version 4 the channel record and ngm25's apical/proximal split, version 5 ngm2's per-dendrite type and tree depth and its group `effective_seed`); checkpoints written before it was an integer say `"1.0"`, which reads as version 1. Every load path (`parse_state`, `apply_state`, the startup `load_state`) refuses a checkpoint without a `metadata` group or whose version is missing, malformed or newer than the build, before any entity reads it, since an older build cannot know what a newer format changed. An older version loads without a warning: the version is read from the checkpoint before the reader is built, and every `state_reader` takes it as a constructor argument (there is no setter, so no load path can leave it at a default), so it reaches every entity's `load_state` as `state_reader::format_version()`, and a check that needs a newer format is gated on it, e.g. `if (r.format_version() >= 2)`, so older checkpoints still load. Bump `k_format_version` with each change to what a checkpoint holds that a load must tell apart, and add a line to its history comment. One version covers the whole file, not one per model.

## Logging Manager

Central coordinator owning the buffer and state loggers:

- `start_buffer_logging(config)` / `stop_buffer_logging(ctrl)` — create/destroy `buffer_logger`; the stop detaches its hook via `ctrl.run_quiesced` (see Buffer Logger). If the quiesce is refused (controller stopping), the logger is kept and an error returned
- `try_extract_state(ctrl)` — `extract_state` for the GUI save paths: refuses a halted controller (checked inside the quiesced task, so the faulted env is never read) and returns an extraction throw (e.g. a failed GPU readback) as `std::expected`'s error instead of propagating it to a GUI thread with no catch above it
- `extract_state(ctrl)` / `serialize_state(tree, cfg)` — the save halves: extraction via `ctrl.run_quiesced`, serialization on the executor (returns the write future — headless awaits it for correct manifest sizes; the GUI may not)
- `load_state(ctrl, path)` — runtime load: parse on the executor, apply + `set_step_count` via `ctrl.run_quiesced`, in one call. An apply that fails partway halts the run (`ctrl.halt`, inside the same task) and returns a failed result with the message (see above for what stays applied)
- `load_state_startup(path)` — the `build_env` startup exemption (synchronous, no `run_quiesced`; valid only before stepping begins)
- `flush_all()` / `stop_all()` — lifecycle management; `stop_all()` (run by `~logging_manager`) does not quiesce, so it requires that the sim thread has been joined
