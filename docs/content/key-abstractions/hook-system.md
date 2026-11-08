+++
title = "Hook System"
weight = 30
+++

# Hook System

The hook system enables the GUI thread to register callbacks that execute on the simulation thread, without blocking either thread in the steady state.

## The Problem

The GUI needs data from the simulation, but the simulation thread can't be interrupted. A contended lock on every step would slow the simulation. COAST's approach: **deferred registration**, with a lock-free check for pending changes, and **synchronous removal** guarded by one uncontended lock per hook phase.

## Four Hook Points

| Hook | When It Fires | Typical Use |
|------|--------------|-------------|
| `pre_process` | Before entity processing | (rarely used) |
| `post_process` | After all entities processed | (rarely used) |
| `pre_swap` | Before buffer swaps | (rarely used) |
| `post_swap` | After all buffers swapped | **Visualization sync**, **buffer logging**, **buffer-browser stats / status-dialog publish** |

`post_swap` is by far the most used — it's the moment when fresh data is available in the read buffers.

Hooks are the **periodic** half of COAST's quiescence story: they fire every step at the post-swap boundary, asynchronously, with no waiting requester — right for continuous publication (visualizers, buffer logging, the `demand_published` GUI displays). Its one-shot complement is `sim_thread_controller::run_quiesced`, which runs a single blocking task at the same boundary (state save/load, `set_step_count`). See [Threading]({{% relref "/threading" %}}).

## Deferred Registration Pattern

```
GUI Thread                          Sim Thread
─────────                          ──────────
set_post_swap_hook(fn):
  lock(pending_mutex)
  push to pending_post_swap_hooks
  has_pending_hooks = true (release)
  unlock
  return hook_id
                                   process() / swap_io():
                                     if has_pending_hooks (acquire):
                                       lock(active_mutex, pending_mutex)
                                       move pending → active maps
                                       clear pending vectors
                                       has_pending_hooks = false (release)
                                       unlock
                                     for each hook phase:
                                       lock(active_mutex)
                                       for each active hook: hook()
                                       unlock
```

**Key insight**: In the steady state (no hooks being added/removed), the sim thread does one atomic load that returns `false` and skips the pending mutex entirely; that mutex is only acquired while hooks are actually changing. The active mutex is taken once per hook phase and is contended only by a removal in progress (see below).

## Hook IDs and Synchronous Removal

Each `set_*_hook()` returns a unique `size_t` ID. The owner later calls `remove_*_hook(id)` to unregister. Unlike registration, **removal is synchronous** with respect to invocation (#235): when `remove_*_hook()` returns, the hook is not running and will never run again — whether it was already active or still pending — so the caller may immediately free whatever the hook captures (the visualizations capture `this`).

```
GUI Thread                          Sim Thread
─────────                          ──────────
                                   swap_io():
                                     apply pending changes
                                     lock(active_mutex)  ─┐ post_swap phase:
                                       post_swap hooks    │ signal hooks, then
                                       barrier hooks      │ barrier hooks, one
remove_post_swap_hook(id):           unlock(active_mutex) ┘ locked region
  lock(active_mutex)  ← waits here while a phase runs
  lock(pending_mutex)
  erase id from active map / pending adds
  unlock both; destroy the removed closure
  return  → hook can no longer run
```

The sim thread holds `active_hooks_mutex_` for the whole of each hook phase (`pre_process`; `post_process`; `pre_swap`; `post_swap` together with the barrier hooks) and while it applies pending changes. The lock is uncontended except while a removal is in progress; the steady state never touches `pending_hooks_mutex_`. The post-swap and barrier hooks share one phase so a removal can never fall between a two-phase visualization's signal and its wait.

What a removal can wait for, by controller state:

| Sim state | `remove_*_hook()` blocks for |
|---|---|
| paused / halted / parked in `run_quiesced` / between steps / controller destroyed | nothing (no phase runs) |
| stepping (throttled, unlimited, single step) | at most the hook phase in progress |

A phase includes the barrier hooks, which wait up to 100 ms each for the GUI to service a requested update. The GUI thread that is removing a hook cannot service them, so each such wait would run out its full timeout: destroying K visualizations (a config reload) measured 1.9 s for K = 5. `remove_*_hook()` therefore announces itself through `env::hook_removal_waiting()` while it waits for the lock, and the visualizations' barrier loops cancel their request as soon as it is set (the same `requested → idle` CAS as the timeout), bringing the K = 5 case to ~0.2–3 ms; the other visualizations skip one update. A hook that waits for another thread must honour `hook_removal_waiting()`; one that waited on the removing thread **without** a timeout would deadlock it. No current hook does.

A removal made **from inside a hook** of the same env (on the sim thread) cannot wait for its own phase. It is deferred to the next `process()`/`swap_io()` boundary as before; the hook may still run later in the current phase.

## Lifecycle

1. **Visualization creation**: Registers a `post_swap_hook` (signal) and a `post_swap_barrier_hook` (wait), both capturing `this`
2. **Every sim step**: Hooks fire, data is copied (or an atomic flag is set)
3. **Visualization destruction**: Calls `remove_post_swap_hook(id)` then `remove_post_swap_barrier_hook(id)`; each returns once the hook can no longer run
4. **Destructor returns**: nothing on the sim thread references the object any more

Before #235 removal was deferred to the next step boundary, so a hook could still run once after its owner was destroyed — a use-after-free for the `this`-capturing visualizations (closing a window, loading another config, shutdown). The `demand_published` hooks (buffer browser, status dialog) and the GUI checkpoint hook capture `shared_ptr` state by value, which was already safe and remains so. Regression: `unit.sim_core.hook_removal`.
