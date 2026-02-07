+++
title = "Threading"
weight = 30
+++

# Threading Model

COAST uses **two threads** with lock-free communication. Understanding the thread boundaries is critical for working with this codebase.

## Thread Map

{{< mermaid >}}
graph LR
    subgraph MainThread["Main Thread (GUI)"]
        SDL[SDL Event Loop]
        ImGui[ImGui Rendering]
        Browsers[Browsers & Editors]
        ProbeUpdate[Probe Observer Update<br/><i>drain SPSC queues</i>]
        VisUpdate[Visualization Update<br/><i>copy sim data</i>]
        LoggingWin[Logging Window<br/><i>auto-checkpoint</i>]
    end

    subgraph SimThread["Simulation Thread"]
        Process[entity.process()]
        SwapIO[buffer.swap_buffer()]
        Hooks[Hook callbacks]
        ProbeRecord[probe() recording]
        BufCapture[Buffer logger capture]
    end

    ProbeRecord -.->|SPSC queue| ProbeUpdate
    Hooks -.->|atomic FSM| VisUpdate
    Process -.->|atomics| SDL

    style MainThread fill:#2d5a27,color:#fff
    style SimThread fill:#1a3a5c,color:#fff
{{< /mermaid >}}

## What Runs Where

| Component | Thread | Notes |
|-----------|--------|-------|
| SDL event polling | Main | `SDL_PollEvent()` |
| ImGui frame lifecycle | Main | NewFrame → Render → Draw |
| `gui_manager` rendering | Main | Menu bar, dialogs, status bar |
| Browser windows | Main | Entity, buffer, dataset browsers |
| Probe observer drain | Main | `probe_browser.update()` |
| Visualization update | Main | Copy data during `update()` |
| `env::process()` | Sim | All entity `process()` calls |
| `env::swap_io()` | Sim | All buffer swaps |
| `probe()` calls | Sim | SPSC queue push |
| Post-swap hooks | Sim | Visualization sync, buffer capture |
| Buffer logger capture | Sim | Via post_swap hook |
| State save/load | Main | Must be paused |

## Communication Mechanisms

### 1. Lock-Free Atomics

`sim_thread_controller` uses atomics for all control:

| Atomic | Direction | Ordering | Purpose |
|--------|-----------|----------|---------|
| `mode_` | GUI → Sim | release/acquire | Mode control |
| `target_sps_` | GUI → Sim | release/acquire | Speed setting |
| `step_count_` | Sim → GUI | relaxed (write), acquire (read) | Step counter |
| `actual_sps_` | Sim → GUI | relaxed (write), acquire (read) | Measured speed |
| `single_step_pending_` | GUI → Sim | release/acquire | Single step request |
| `step_limit_` | GUI → Sim | release/acquire | Auto-pause target |

### 2. SPSC Queues (Probe Data)

Each active probe has a `spsc_queue<probe_sample>` with capacity 4096:
- **Producer** (sim thread): `try_push()` with release on head
- **Consumer** (GUI thread): `drain()` with acquire on head
- **Drop policy**: If full, values are silently dropped

### 3. Deferred Hook System

Hooks are registered by the GUI thread but execute on the sim thread. The deferred pattern avoids locks on the hot path:

1. **GUI thread**: Acquires `pending_hooks_mutex_`, pushes hook into pending vector, sets `has_pending_hooks_ = true` (release)
2. **Sim thread**: At the top of `process()` and `swap_io()`, checks `has_pending_hooks_` (acquire). If true, acquires the mutex, moves pending hooks to active maps, clears pending vectors
3. **Sim thread**: Iterates active hooks (no lock needed — only the sim thread touches active maps)

This means the mutex is only acquired when hooks are actually being added/removed — in the steady state, the sim thread does a single atomic load (fast path).

### 4. Visualization Atomic FSM

Visualizations (ngm_flat_vis, ngm_tree_vis) use a 3-state atomic:

```
idle → requested → updating → idle
```

- **Sim thread** (post_swap hook): CAS `idle → requested`, yields briefly if GUI is slow
- **GUI thread** (`update()`): CAS `requested → updating`, copies data, CAS `updating → idle`

### 5. TBB Concurrent Hash Map (Probes)

`entity_probe_set` uses TBB's `concurrent_hash_map`:
- **Sim thread**: `const_accessor` (reader lock) for `record()` — concurrent reads are non-blocking
- **GUI thread**: `accessor` (writer lock) for `discover()`, `activate()`, `deactivate()`

## Thread Safety Rules

1. **Never access entity internal state from the GUI thread** — always copy via hooks or SPSC queues
2. **State save/load requires the sim to be paused** — not safe while running
3. **Visualization data is copied, never shared** — the FSM ensures exclusive access during copy
4. **Probe observers are GUI-thread-only** — `push()`, `render()`, and observer management all happen on the main thread
5. **Hook registration is safe from any thread** — the deferred system handles the synchronization

## Shutdown Sequence

```
1. User closes window → running = false
2. Main loop exits
3. Shutdown callbacks:
   a. gui.set_sim_instance(nullptr) → frees GL resources
   b. sim.reset() → destroys sim_instance:
      i.   logging_manager destroyed
      ii.  thread_controller: request_stop() + notify_all() + join
      iii. sim_builder, sim_env, json_loader destroyed
4. sdl_app destructor: cleanup ImGui, SDL
```

The key insight: `thread_controller` is destroyed before `sim_env`, so the simulation thread is guaranteed to be joined before any entities or buffers are destroyed.
