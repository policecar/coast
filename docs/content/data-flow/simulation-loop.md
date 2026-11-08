+++
title = "Simulation Loop"
weight = 20
+++

# Simulation Loop

What happens every simulation step and every GUI frame.

## The Simulation Step

Each step executes on the **simulation thread** inside `sim_thread_controller::do_step()`:

```mermaid
sequenceDiagram
    participant TC as thread_controller
    participant Env as sim_env
    participant Entity as io_entity
    participant Buffer as io_buffer

    TC->>TC: set_current_step(step_count)
    TC->>Env: process()

    activate Env
    Env->>Env: process_pending_hooks()
    Note over Env: Fast path: atomic check,<br/>skip if no pending hooks

    loop pre_process hooks
        Env->>Env: hook()
    end

    loop each entity (TBB parallel_for, affinity-pinned)
        Env->>Entity: process()
        Note over Entity: Entities are mutually independent within a step<br/>(one-step buffer latency) → run concurrently.<br/>Reads input buffers (previous step)<br/>Writes output buffer (current step)<br/>Calls probe() for instrumented values
    end

    loop post_process hooks
        Env->>Env: hook()
    end
    deactivate Env

    TC->>Env: swap_io()
    activate Env
    Env->>Env: process_pending_hooks()

    loop pre_swap hooks
        Env->>Env: hook()
    end

    loop each io_buffer
        Env->>Buffer: swap_buffer()
        Note over Buffer: write_idx ↔ read_idx<br/>(pure index flip — no stats)
    end

    loop post_swap hooks
        Env->>Env: hook()
        Note over Env: Visualization updates,<br/>buffer logging capture
    end
    deactivate Env

    TC->>TC: step_count++
```

## What Happens Inside Entity process()

### Dataset Entity (e.g., mnist_io)

```
1. Get write buffer: output_mem()
2. Copy current sample data into buffer
3. Add noise per element: clamp(val + uniform_noise, 0, 1)
4. Advance sample counter
   → When change_interval reached: next sample
   → When all samples seen: next epoch, wrap
```

### Neuron Group (neuron_group_t)

```
1. Get write buffer: output_mem()
2. For each neuron (parallel):
   activity = neuron.get_response()
   out[neuron.id] = activity
3. Local inhibition: soft winner-take-all
4. Find max activity
5. Stochastic winner selection (within top 20%)
6. Winner adaptation (primary, strong)
7. Common adaptation (secondary, weak, all neurons)
```

See [hd_ngm2 module]({{% relref "/architecture/modules/hd_ngm2" %}}) for the full algorithm.

## The IO Buffer Swap

The swap creates a **one-step delay**:

```
Before swap:
  Write buffer: current step's output (just produced)
  Read buffer:  previous step's output (being read by consumers)

After swap:
  Read buffer:  current step's output → available to consumers next step
  Write buffer: recycled for next step's output
```

This means consumers always see data from the previous step. This avoids read-write races entirely — no locks needed between entities. It is also what makes the entities mutually independent within a `process()` pass, so `env::process()` dispatches them with a TBB `parallel_for` (CPU∥CPU) rather than a serial loop. A *persistent* `affinity_partitioner` keeps each entity on the same worker core across steps, preserving cache locality and the thread placement the neuron groups' inner `static_partitioner` loops rely on (without it, nesting an outer parallel loop over the inner loops regressed the entity-sparse mnist config ~7%; with it the dispatch is a net win over the old serial loop). Reordering independent entities is provably bit-identical (verified against the golden suite). When a GPU context is present, the recorded GPU batch is submitted *without* an immediate wait so it executes concurrently with these CPU entities.

> **GPU entities** add a second encoder phase to each of `process()` and `swap_io()`, producing two GPU submits per step. The diagram above is the CPU-only path; the GPU-augmented step is traced in [GPU Dispatch]({{% relref "/data-flow/gpu-dispatch" %}}).

## The GUI Frame

Each frame of the main loop runs on the **main thread**:

```
1. Poll SDL events (keyboard, mouse, window)
2. State update phase:
   - Check if config load is pending → if yes, construct sim
3. ImGui::NewFrame()
4. GUI callback phase:
   - render_menu_bar()
   - render_dialogs():
     - File browser
     - Control dialog (play/pause/speed)
     - Status dialog
     - Config editor
     - Entity/Buffer/Dataset browsers
     - probe_browser.update() → drain all SPSC queues
     - probe_browser.render() → show observers
     - logging_window.update() → reap completed async checkpoint writes
       (the auto-checkpoint cadence itself runs on the sim thread via a post_swap hook)
     - Buffer replay dialog
     - check_auto_save() → detect pause transition
   - render_status_bar() (FPS, SPS, config path)
5. ImGui::Render()
6. OpenGL draw + swap
```

## Thread Modes

| Mode | Behavior | Pacing |
|------|----------|--------|
| `paused` | No stepping, waits on CV | Wakes for single-step requests |
| `throttled` | Target steps/second | Sleep-based, adjustable 1-250/s |
| `unlimited` | As fast as possible | `yield()` every 1000 steps |

The control dialog's speed slider maps 1-250 to throttled mode, 251 to unlimited.

## SPS Measurement

Measured via exponential moving average over 500ms windows:

```
smoothed_sps = 0.7 * measured + 0.3 * previous
```

Displayed in the status bar as actual vs. target (throttled) or just actual (unlimited).
