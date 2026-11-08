+++
title = "COAST Documentation"
+++

# COAST — Mental Model Documentation

**Coherent Active State Dynamics in Self-Organizing Systems**

COAST is a C++23 neural simulation framework with an interactive SDL2/OpenGL3 GUI. It implements biologically-inspired neural models featuring permanence-based synapses, dendritic branching, and competitive learning: the original **HD-NGM2** (Hierarchical Dendritic - Neuron Group Model 2), the arena-based **HD-NGM2.5** (ngm2's semantics in a new implementation) and the flat-SoA **HD-NGM3**.

Since the *ngm3-gpu-port* cycle, COAST also supports **heterogeneous CPU/GPU execution** — individual entities can run on the GPU (via wgpu-native compute kernels), meeting CPU entities at the `io_buffer` seam. The value is *capability, not speed*: it unlocks GPU-resident topologies (and platforms where GPU memory bandwidth dominates) rather than promising a speed-up on any given machine. The whole GPU layer is gated behind the `COAST_GPU_BACKEND` build option (`none`/`webgpu`/`null`; default `webgpu`) and is absent from `=none` builds. Start with [Heterogeneous Execution]({{% relref "/key-abstractions/heterogeneous-execution" %}}) to understand it.

---

## How to Read This Documentation

This documentation is structured to build your mental model from **coarse to fine**. Start with the sections in order:

1. **[Architecture]({{% relref "architecture" %}})** — The static structure. What modules exist, what each one does, and how they depend on each other.

2. **[Code Flow]({{% relref "data-flow" %}})** — The dynamic behavior. What happens at runtime, step by step, from startup through the simulation loop.

3. **[Threading]({{% relref "threading" %}})** — The concurrency model. Which thread does what, and how they communicate without locks.

4. **[Key Abstractions]({{% relref "key-abstractions" %}})** — The core concepts. The design patterns and interfaces you need to "think in" when working with COAST.

5. **[coastd]({{% relref "coastd" %}})** — The headless-job daemon. A **separate process** that builds a git ref natively, schedules jobs against fixed slots, supervises `coast` children, and auto-resumes them after a crash or reboot. Read this only when you care about running simulations on a remote machine; it is orthogonal to everything above.

6. **[Reference]({{% relref "reference" %}})** — Per-file quick lookup. Every header and source file, what it contains, and where to find it.

7. **[Testing]({{% relref "testing" %}})** — The regression-prevention infrastructure. Test tiers, golden-file protocol, static and dynamic analysis, and the design decisions behind them.

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
│   Neural Models (hd_ngm2/25/3)  │  neuron groups; CPU or GPU entities
└────────────┬────────────────────┘
             │ (GPU entities only)
┌────────────▼────────────────────┐
│   core/gpu (backends/*)        │  GPU seam + backends, COAST_GPU_BACKEND
└─────────────────────────────────┘
```

Two threads carry the simulation (the [Threading]({{% relref "threading" %}}) section covers the others, such as the HDF5 executor):
- **Main thread** — SDL event loop, ImGui rendering, probe observer updates
- **Simulation thread** — `process()` and `swap_io()` each step, probe recording

They communicate via lock-free atomics, SPSC queues, and a deferred hook system.

Alongside this executable the repository builds a second, independent program:
**[coastd]({{% relref "coastd" %}})**, a resident daemon that runs `coast`
headless on a (possibly remote) machine and keeps long batch runs alive across
client disconnects, crashes and reboots. It shares `coast_lib` but is a separate
process with a separate architecture.

---

## Building

```bash
# Linux (GCC 15+, HDF5 1.12+; see the README's Dependencies)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/coast

# macOS (Homebrew LLVM Clang — builds the full project incl. the C++ console)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$(brew --prefix llvm)/bin/clang \
  -DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++ \
  -DCMAKE_CXX_FLAGS=-stdlib=libc++ \
  -DCMAKE_PREFIX_PATH=$(brew --prefix llvm)
cmake --build build -j

# Headless benchmark
./build/coast -n 500000 configs/mnist.json
```

Always use Release builds — Debug is ~10x slower due to missing `-O3`.

On macOS the default Homebrew-LLVM Clang build includes the embedded C++ console; a lighter GCC 15 build works too but skips it (libLLVM is libc++-built, so a GCC/libstdc++ build can't link it). See the [console module]({{% relref "/architecture/modules/console" %}}) page.
