+++
title = "COAST Documentation"
+++

# COAST — Mental Model Documentation

**Coherent Active State Dynamics in Self-Organizing Systems**

COAST is a C++23 neural simulation framework with an interactive SDL2/OpenGL3 GUI. It implements the **HD-NGM2** (Hierarchical Dendritic - Neuron Group Model 2), a biologically-inspired neural model featuring permanence-based synapses, dendritic branching, and competitive learning.

---

## How to Read This Documentation

This documentation is structured to build your mental model from **coarse to fine**. Start with the sections in order:

1. **[Architecture]({{% relref "architecture" %}})** — The static structure. What modules exist, what each one does, and how they depend on each other.

2. **[Code Flow]({{% relref "data-flow" %}})** — The dynamic behavior. What happens at runtime, step by step, from startup through the simulation loop.

3. **[Threading]({{% relref "threading" %}})** — The concurrency model. Which thread does what, and how they communicate without locks.

4. **[Key Abstractions]({{% relref "key-abstractions" %}})** — The core concepts. The design patterns and interfaces you need to "think in" when working with COAST.

5. **[Reference]({{% relref "reference" %}})** — Per-file quick lookup. Every header and source file, what it contains, and where to find it.

---

## Quick Orientation

```text
┌─────────────────────────────────┐
│   GUI Layer (gui_core)          │  SDL2 + ImGui event loop, dialogs, browsers
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Visualization (gui_vis)       │  OpenGL renderers, vis_registry factory
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Simulation Core (sim_core)    │  env container, sim_thread_controller
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Neural Model (hd_ngm2)        │  neuron_group, dendrite components
└─────────────────────────────────┘
```

Two threads run concurrently:
- **Main thread** — SDL event loop, ImGui rendering, probe observer updates
- **Simulation thread** — `process()` and `swap_io()` each step, probe recording

They communicate via lock-free atomics, SPSC queues, and a deferred hook system.

---

## Building

```bash
# Linux (requires GCC with C++23 support)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/coast

# Headless benchmark
./build/coast -n 500000 configs/mnist.json
```

Always use Release builds — Debug is ~10x slower due to missing `-O3`.
