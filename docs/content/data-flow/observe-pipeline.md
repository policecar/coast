+++
title = "Observe Pipeline"
weight = 35
+++

# Observe Pipeline

How user-instrumented values flow from simulation state to per-group HDF5 files. This is the offline-capture counterpart to the live [Probe Pipeline]({{% relref "/data-flow/probe-pipeline" %}}); see the [observe]({{% relref "/architecture/modules/observe" %}}) module for the static structure.

Observe runs as a **single post-process pass**: once per step, after every entity's `process()` and (on the GPU path) after the device is idle, the sim thread walks the entity tree and each element emits the values it has already computed for the step. There is no inline hot-path emission and no mid-step machinery.

## End-to-End Flow

```mermaid
sequenceDiagram
    participant Sim as Sim Loop<br/>(sim thread)
    participant Post as Post-process Hook<br/>run_observation_pass()
    participant Reg as observe_registry
    participant Entity as Entity Tree<br/>perform_observations()
    participant Acc as observe_accessor
    participant Exec as hdf5::executor<br/>(single HDF5 thread)
    participant H5 as group.observe.h5

    Note over Sim: All process() done, device idle
    Sim->>Post: post-process hook fires
    Note over Post: groups configured?<br/>→ NO: return (zero cost)
    Post->>Reg: recompute active_this_step per group<br/>(range_hit OR/AND prog_active, under mutex)

    Post->>Entity: for each root: perform_observations(acc)
    Note over Entity: override emits own values,<br/>then recurses into children
    Entity->>Acc: acc.group_active("group")?
    Note over Acc: false → skip expensive work
    Entity->>Acc: acc.observe(*this, group, id, value)
    Note over Acc: resolve site (lazy);<br/>group active? → enqueue value
    Acc->>Exec: enqueue (copy into owned buffer)

    Post->>Reg: consume prog_single_pending pulses

    Note over Exec: ─── HDF5 thread ───
    Exec->>H5: lazy-create dataset + __step_index<br/>append (value, step) row
```

## Cost Profile

There is **no inline cost in entity `process()` code** — the OBSERVE macro is gone. The only observe work is in the post-process pass:

- **No group configured** → the pass returns at the top (`group_index_.empty()`); true zero cost.
- **Configured but all groups dormant this step** → the pass still walks the entity tree so any `trigger()` call inside an override is evaluated; each override short-circuits on `group_active()`. Measured overhead ≈ a few µs/step (the tree walk + per-element `active_this_step` reads). No production override currently calls `trigger()`, so this walk is, today, the price of a retained-but-unused capability — a cheap `any_active` guard could skip it if it ever matters.
- **One or more groups active** → the walk additionally resolves sites and enqueues values for the active groups.

## The Single Commit Decision

Each group carries one atomic `active_this_step`, recomputed at the start of every pass from `(ranges, combine, prog_active, prog_single_pending)` and re-evaluated live by `trigger()`. `group_active(group)` is a lock-free read of it. This replaces the old two-flag gate/commit split and sticky bit: because emission is single-threaded and reads retained end-of-step state, the mid-step-race defenses that machinery existed for are unnecessary.

## Site Resolution

`acc.observe(e, group, id, value)` resolves the site for `(e, id)` lazily in the group's site map (a TBB `concurrent_hash_map`, kept for the shared resolve path though the pass is single-threaded). The first lookup for a `(group, full_id)` pair locks the dtype and per-row shape (the **shape lock**). If the group is active this step, the value is enqueued directly to the writer, which copies it into an owned buffer — there is no per-site scratch buffer or `filled_this_step` flag.

## The HDF5 Executor

HDF5 on this platform is built `Threadsafety: OFF`, so **all** HDF5 calls for the observe files funnel through a single shared `hdf5::executor` thread (the same executor the logging subsystem uses). Each per-group `observe_writer`:

- opens its file via `submit_sync` (blocking) on first commit,
- appends rows via `submit` (fire-and-forget) so the sim thread is never blocked on I/O,
- on `detach`, issues a final `submit_sync` barrier that flushes (`H5Fflush`) and closes all dataset handles and the file in order.

Datasets are extensible along axis 0 with chunk shape `[chunk_rows, per-row-shape...]`. `chunk_rows` is sized from a byte budget (`pick_chunk_rows()` in `observe_chunk.h`): it targets ~1 MiB per chunk, clamped to at most 1024 rows. Scalar / small-row sites keep the historical 1024 rows; array-shaped sites whose row exceeds ~1 KiB get fewer rows per chunk so one chunk stays near 1 MiB and fits the HDF5 chunk cache (each dataset also gets a per-dataset chunk cache via `H5Pset_chunk_cache`). This avoids the cache-bypass read-modify-write that a fixed 1024-row chunk caused for wide rows — a ~1.9–2.6× writer-throughput win on the large-row path (it does not change captured values, only on-disk layout).

## Lifecycle Touchpoints

| When | What |
|---|---|
| `sim_instance` construction | `observe_registry::attach()` installs the single post-process hook, seeds groups from config, calls `H5open()` on the executor thread |
| Every step (post) | Return if no group configured; else recompute `active_this_step` per group, walk `perform_observations`, enqueue active-group values, consume `prog_single_pending` |
| `sim_instance` destruction | `observe_registry::detach()` flushes + closes writers, removes the hook, clears groups |
