# macOS / Apple Metal Validation Guide

**Status:** validation procedure — *not yet executed on hardware*. This document
exists so a collaborator with an Apple Silicon Mac (or a PR reviewer) can build
and exercise the ngm3 GPU port on the Metal backend and confirm it behaves
identically to the Linux/Vulkan path. It corresponds to dew node
`develop.s13.macos-build-run` (collaborator-gated) and surfaces acceptance
criterion **A-II1** (wgpu boots and runs on the target platform) on Apple Metal.

The Strix Halo (Linux/Vulkan) path is the development reference; everything below
is the *prediction* of what the Metal path should do. Where a result is expected
to differ from Linux, it is called out explicitly. **Please record the actual
observed output** (see [Recording results](#recording-results)) so the prediction
can be confirmed or corrected.

---

## 1. Why this is expected to "just work"

The GPU port talks to the GPU exclusively through **wgpu-native** (the Rust
`wgpu` core exposed as a C library). COAST contains **no** Metal-, Vulkan-, or
Objective-C-specific code — the only platform branch is a single backend-type
constant:

```cpp
// src/core/gpu/gpu_context.cpp:14-18
// Backend is fixed per platform: Strix Halo / Linux -> Vulkan, Apple -> Metal.
#ifdef __APPLE__
constexpr WGPUBackendType kBackend = WGPUBackendType_Metal;
#else
constexpr WGPUBackendType kBackend = WGPUBackendType_Vulkan;
#endif
```

All WGSL kernels are portable: they use no platform intrinsics, and the PRNG is a
stateless PCG-1D32 hash (`core/gpu/shared_prng.wgsl`, ~6 integer ops). wgpu's
WGSL→MSL translation (via naga) is the same code path the Linux build uses for
WGSL→SPIR-V. The CMake build already knows the macOS prebuilt assets:

```cmake
# CMakeLists.txt:65-70
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_wgpu_asset "wgpu-macos-aarch64-release.zip")
    set(_wgpu_sha   "e3e3f6144f9b197b8199681e956391631cfc6d1cef057a1e8c5a1c6aedef5054")
else()  # APPLE x86_64
    set(_wgpu_asset "wgpu-macos-x86_64-release.zip")
    set(_wgpu_sha   "8201b3b9c05c395b9d3c8bb37d965c0c47c5e53a2e251cf6af04356f01e6c81a")
```

So the open question is **not** "does it compile" — it is "does naga's MSL
translation of our kernels produce numerically equivalent results, and does the
unified-memory / submission model behave the same on Metal."

---

## 2. Prerequisites

- **Apple Silicon Mac** (M1 or newer) — the `wgpu-macos-aarch64` asset.
  Intel Macs use the `x86_64` asset; the procedure is identical but untested and
  lower priority (Apple Silicon is the relevant target).
- **macOS 12+** (Metal 3-capable; any recent macOS is fine).
- **GCC 15** via Homebrew. (AppleClang also builds a headless binary — verified
  with AppleClang 21.0.0, 2026-07-23, see CLAUDE.md.) This headless GPU validation needs no GUI or console, so GCC 15 is the
  right choice here (only the optional embedded C++ console, out of scope below,
  instead needs a Homebrew-LLVM Clang build — see the project README):
  ```bash
  brew install gcc@15     # pin the versioned keg; brew's `gcc` rolls forward (16+)
  ```
- **CMake ≥ 3.18** and a network connection on first configure (FetchContent
  downloads the pinned wgpu-native zip; the SHA256 is hard-verified).
- The MNIST dataset in `data/` (see `scripts/download_datasets.sh`) — required
  only for the `mnist_io_gpu` and ngm3 config runs, not for the bare smoke/kernel
  probes.

> **GUI is out of scope for this validation.** The SDL2/OpenGL GUI on macOS is a
> separate concern (Apple deprecated OpenGL; SDL2 comes from brew). All GPU
> validation below is **headless** — the probe executables and `ctest` need no
> window. Build the GUI separately if desired, but a green headless run is the
> A-II1 evidence we need.

---

## 3. Build

From the repository root:

```bash
cmake -B build-tests -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_C_COMPILER=$(brew --prefix gcc@15)/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=$(brew --prefix gcc@15)/bin/g++-15
cmake --build build-tests -j
```

> Pin the versioned `gcc@15` keg: Homebrew's `gcc` formula rolls forward (16 as
> of writing), so `$(brew --prefix gcc)/bin/g++-15` no longer resolves.

`-DCMAKE_BUILD_TYPE=Release` is mandatory (`-O3`; Debug is ~10× slower and A-tier
FP tolerances assume `-O3` codegen). `COAST_WITH_WGPU` defaults **ON**, so the
macOS wgpu asset is fetched automatically.

**Configure-time check.** Confirm the build picked the Metal asset:

```
-- wgpu-native: wgpu-macos-aarch64-release.zip -> .../build-tests/_deps/wgpu_native-src
```

If you instead see a `FATAL_ERROR` about "no prebuilt wgpu-native asset," the
`CMAKE_SYSTEM_PROCESSOR` did not match `arm64|aarch64` — report the value.

---

## 4. What to run, and what to observe

Run from the **repository root** (datasets and config-relative paths resolve
against the current working directory). The probes are listed cheapest-first;
each is an independent A-II1 data point. Stop and report at the first failure.

### 4.1 wgpu bootstrap smoke — `coast_gpu_smoke`

```bash
./build-tests/coast_gpu_smoke
```

**Expect:** an adapter line naming an Apple GPU and the Metal backend, a block of
device limits, and a clean teardown with no faults. The Linux reference prints
`... [Vulkan]`; on Metal you should see **`[Metal]`**:

```
[coast-gpu-smoke] constructing gpu_context...
adapter: Apple M_ [Metal]
device limits:
  maxStorageBufferBindingSize     : ... bytes (... MiB)
  ...
[coast-gpu-smoke] OK — tearing down (reverse order)...
[coast-gpu-smoke] teardown complete, no faults.
```

The exact adapter name and limit values will differ from Strix Halo — that is
expected and fine. The **only** failure signal here is a `FAILED:` line or a
non-zero exit.

### 4.2 minimal compute kernel — `coast_gpu_min_kernel`

```bash
./build-tests/coast_gpu_min_kernel
```

End-to-end "add 1.0 to every element + verify readback" plus the bad-WGSL error
path. **Expect** a PASS / OK line and exit 0. A failure here means naga's MSL
translation or the dispatch/readback round-trip is broken on Metal — the most
fundamental possible problem.

### 4.3 io_buffer GPU variants — `coast_io_buffer_gpu_probe`

```bash
./build-tests/coast_io_buffer_gpu_probe
```

Round-trips all three GPU `io_buffer` variants (`cpu_to_gpu`, `gpu_to_cpu`,
`gpu`) in isolation from entity logic. **Expect** all PASS. This exercises
`wgpuQueueWriteBuffer`, `CopyBufferToBuffer`, and the map-back readback path —
the unified-memory / submission mechanics that A-II validates.

### 4.4 mnist_io_gpu — `coast_mnist_io_gpu_probe`

```bash
./build-tests/coast_mnist_io_gpu_probe
```

Uploads the dataset to the GPU and runs the read-by-index + noise + clamp +
reduction kernels. **Expect** output values in range, no NaN, images advancing,
PASS. (Needs `data/` populated.)

### 4.5 ngm3 GPU model — `coast_ngm3_gpu_probe`  ← the important one

```bash
./build-tests/coast_ngm3_gpu_probe
```

This is the core model probe. It checks, each as a `PASS`/`FAIL` line:

1. **range** — output in [0,1];
2. **non-degenerate** — output max is not pinned;
3. **aggregate** — (mean, max) roughly match CPU ngm3;
4. **learning** — permanences drift from the frozen init over the run and stay
   valid (finite, in [0,1]);
5. **branch** — distinct segment-id count grows (branching happens);
6. **probes** — probe-output readback fans out correctly;
7. **vis** — the per-data-view CPU mirror refreshes.

**Expect** a final `[coast-ngm3-gpu-probe] ALL PASS`. If any sub-check prints
`FAIL`, copy the full line — the printed numbers (drift counts, mean/max,
distinct seg-ids) tell us *which* kernel diverged on Metal.

### 4.6 A-tier integration suite (the 4 ngm3 device-topology tests)

```bash
cd build-tests && ctest --output-on-failure -R 'integration\.mnist_ngm3'
```

Runs the four S12 tests — `mnist_ngm3_philox` (CPU, strict), `mnist_ngm3_cpu_to_gpu`,
`mnist_ngm3_gpu_to_cpu`, `mnist_ngm3_heterogeneous` (GPU, aggregate). **Expect 4/4
passing.** These compare GPU output against goldens captured on the Linux/Vulkan
run:

- The GPU tests use **aggregate** compare mode at the default `rel=1e-6`, because
  GPU ngm3 is deterministic against *its own* golden — but those goldens were
  produced on **Vulkan**. If naga's MSL translation differs numerically from its
  SPIR-V translation, these are the tests that will catch it.
- A failure here is **the most likely** failure on Metal, and is *informative,
  not necessarily a defect*: it would mean Metal and Vulkan disagree past
  `rel=1e-6` on the same kernels. If that happens, capture the `coast_compare`
  report (`FAIL <path> golden=X new=Y Δ=% tol=…`) — we then decide whether to
  (a) widen the aggregate tolerance for cross-backend goldens, or (b) retain a
  separate per-backend golden set.

Sanity-check that the CPU-only suite is green too (should be platform-invariant):

```bash
ctest --output-on-failure -LE gpu      # ~13 non-GPU tests
```

---

## 5. Expected differences from the Linux/Vulkan reference

| Aspect | Linux (reference) | macOS (expected) |
|--------|-------------------|------------------|
| Adapter name | `... [Vulkan]` | `Apple M_ [Metal]` |
| Device limits | Strix Halo values | Apple-GPU values (differ; non-fatal) |
| wgpu asset | `wgpu-linux-x86_64` | `wgpu-macos-aarch64` |
| Probe verdicts | ALL PASS | **expected** ALL PASS |
| ngm3 numerics | golden source | **may differ ≤ a few % from Vulkan golden** — see §4.6 |

**Known limitation carried from Develop (not Metal-specific):** GPU ngm3's mean
output runs systematically ~16–25% *lower* than CPU ngm3 at the same input
(directional, sub-noise-floor; documented in
`benchmarks/ngm3_aggregate_noise_floor.md`). This is present on Vulkan already
and is **not** a Metal-validation failure. The Metal aggregate tests compare GPU
against the **GPU** (Vulkan-captured) golden, so this CPU/GPU bias does not enter
the pass/fail decision — only a Metal-vs-Vulkan disagreement would.

---

## 6. Recording results

Please append observed results here (or in the PR) for each section run:

```
Hardware:        <e.g. MacBook Pro M3 Pro, macOS 14.5>
GCC:             <g++-15 --version first line>
Configure:       wgpu asset line = <paste>
4.1 smoke:       <PASS/FAIL + adapter line>
4.2 min_kernel:  <PASS/FAIL>
4.3 io_buffer:   <PASS/FAIL>
4.4 mnist_io:    <PASS/FAIL>
4.5 ngm3 probe:  <ALL PASS / which sub-check FAILed + numbers>
4.6 ctest:       <N/4 ngm3 + N non-gpu; paste any coast_compare FAIL block>
```

A fully green run closes A-II1 on Apple Metal. Any failure — especially a §4.6
cross-backend golden mismatch — is a useful result: capture the numbers and we
will decide tolerance vs. per-backend-golden together.
