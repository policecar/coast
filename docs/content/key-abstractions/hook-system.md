+++
title = "Hook System"
weight = 30
+++

# Hook System

The hook system enables the GUI thread to register callbacks that execute on the simulation thread, without blocking either thread in the steady state.

## The Problem

The GUI needs data from the simulation, but the simulation thread can't be interrupted. Solutions like mutexes would slow the simulation. COAST's approach: **deferred registration with lock-free fast path**.

## Four Hook Points

| Hook | When It Fires | Typical Use |
|------|--------------|-------------|
| `pre_process` | Before entity processing | (rarely used) |
| `post_process` | After all entities processed | (rarely used) |
| `pre_swap` | Before buffer swaps | (rarely used) |
| `post_swap` | After all buffers swapped | **Visualization sync**, **buffer logging** |

`post_swap` is by far the most used — it's the moment when fresh data is available in the read buffers.

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
                                       lock(pending_mutex)
                                       move pending → active maps
                                       clear pending vectors
                                       has_pending_hooks = false (release)
                                       unlock
                                     for each active hook: hook()
```

**Key insight**: In the steady state (no hooks being added/removed), the sim thread does one atomic load that returns `false` and skips the mutex entirely. The mutex is only acquired during the brief moment when hooks are actually changing.

## Hook IDs

Each `set_*_hook()` returns a unique `size_t` ID. The GUI can later call `remove_*_hook(id)` to unregister. Removal is also deferred — the ID is pushed to a pending removal vector and processed by the sim thread.

## Lifecycle

1. **Visualization creation**: Registers a `post_swap_hook` to copy entity data
2. **Every sim step**: Hook fires, data is copied (or an atomic flag is set)
3. **Visualization destruction**: Calls `remove_post_swap_hook(id)` — deferred removal
4. **Next sim step**: Sim thread processes the removal, hook no longer fires

This means there's a brief window where a hook might fire after the visualization has been destroyed. The visualization adapters handle this by checking if they're still valid before accessing data.
