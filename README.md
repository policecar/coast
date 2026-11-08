# coast
Coherent active state dynamics in self-organizing systems

## Dependencies

- CMake 3.24+
- A C++23 compiler and standard library: **GCC 15+**, or Clang with a standard
  library that has `std::vector::append_range`, `std::jthread` and floating-point
  `std::from_chars` (libstdc++ 15+; libc++ 19 lacks the last two). GCC 13/14 and
  libstdc++ 14 are too old. The GPU backends additionally need `#embed` (GCC 15+,
  Clang 19+). Configure checks all of this and stops with a message if it fails.
- Intel TBB (oneTBB)
- HDF5 **1.12+**
- SDL2 + OpenGL — **only** for the default GUI build; a headless build never
  searches for them (see [Headless build](#headless-build))
- LLVM/Clang **22+** development libraries — **only** for the optional embedded
  C++ console (GUI builds). Without them, or with an older LLVM, configure skips
  the console with a warning and builds everything else.
- [Hugo](https://gohugo.io/) (extended edition, for building documentation)

These minimums were measured with fresh-clone builds (2026-11-08). Ubuntu 24.04's
own packages are too old (GCC 13/14, HDF5 1.10); Debian 13 has a new-enough HDF5
but GCC 14.

On macOS, build with **Homebrew's LLVM Clang** — it builds the whole project including the embedded C++ console. That toolchain is required *for the console*: Homebrew's libLLVM is libc++-built, so only a matching clang/libc++ build can link the interpreter. A Homebrew **GCC 15** build is a lighter alternative but cannot link the console (see [Build](#build)). For a headless build, no Homebrew LLVM is needed at all — AppleClang suffices:
```bash
brew install llvm sdl2 tbb hdf5
# lighter, no-console alternative (instead of llvm):  brew install gcc@15
```

## Submodules

Use the following git command to update the submodules after an initial clone
```bash
git submodule update --init --recursive
```


## Build

```bash
# Linux
cmake -B build -DCMAKE_BUILD_TYPE=Release

# macOS — Homebrew LLVM Clang. Builds the full project, including the embedded
# C++ console. Confirm `-- Console: clang-repl via LLVM …` at configure time.
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
  -DCMAKE_PREFIX_PATH="$(brew --prefix llvm)"

# Build (either platform)
cmake --build build -j
```

### macOS shortcut: CMake presets

The clang toolchain above is also available as CMake presets (presets need CMake
≥ 3.21; the project itself requires ≥ 3.24), so you don't pass the compiler flags
by hand:

```bash
cmake --preset macos            # -> build/        (clang, full project + console)
cmake --preset macos-tests      # -> build-tests/  (clang, BUILD_TESTING=ON)
cmake --build --preset macos -j
```

The presets pin Homebrew's LLVM Clang (resolved via `$HOMEBREW_PREFIX`, so they
work on both Apple Silicon and Intel) and are gated to macOS — on Linux use the
plain `cmake -B build` flow above. Keeping the macOS build on a single toolchain
matters for the A-tier integration goldens: they are a fingerprint of a long,
chaotic run and diverge ~99% between clang and GCC codegen, so a consistent
compiler keeps locally-generated `macos/` goldens reproducible. The manual
recipes above still work unchanged.

### macOS alternative: GCC 15 (no console)

If you don't need the embedded console, **GCC 15** is a lighter option (no LLVM
dependency). It builds the full GUI + sim; the console is skipped — a harmless
`Console … SKIPPED` warning — because libLLVM is libc++-built and a
GCC/libstdc++ build can't resolve its symbols.

```bash
brew install gcc@15
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$(brew --prefix gcc@15)/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=$(brew --prefix gcc@15)/bin/g++-15
cmake --build build -j
```

Pin the versioned `gcc@15` keg — brew's `gcc` formula rolls forward, so
`$(brew --prefix gcc)/bin/g++-15` stops resolving. (On Linux the console builds
from system `llvm-dev` / `libclang-cpp-dev` with no special compiler flags.)

### GPU backend

GPU support is selected at configure time with the `COAST_GPU_BACKEND` option:

| Value | Effect |
|-------|--------|
| `webgpu` *(default)* | WebGPU compute backend via wgpu-native (Vulkan on Linux, Metal on macOS). GPU configs run on the GPU. |
| `null` | Headless backend with no GPU dependency — exercises the full GPU seam with no-op compute (CI / GPU-free builds). |
| `none` | No GPU layer compiled in; CPU-only. GPU-labeled configs are refused with a clear error. |

```bash
# CPU-only build (no wgpu dependency)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCOAST_GPU_BACKEND=none

# GPU-free build that still links the GPU seam
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCOAST_GPU_BACKEND=null
```

Exactly one backend is compiled per build. The agnostic core talks only to the `gpu_device` interface; each backend owns its own context privately.

### Headless build

`-DCOAST_WITH_GUI=OFF` (default `ON`) drops the entire GUI layer. SDL2 and
OpenGL are never searched for, ImGui/ImPlot/ImNodeFlow are not compiled, and the
embedded console is forced off (it *is* an ImGui panel), so this configures on a
machine with no graphics stack at all — it needs only TBB and HDF5.

```bash
cmake --preset headless          # -> build-headless/  (COAST_WITH_GUI=OFF)
cmake --build build-headless -j

# Leanest possible build: no GUI and no GPU backend
cmake -B build-min -DCMAKE_BUILD_TYPE=Release \
  -DCOAST_WITH_GUI=OFF -DCOAST_GPU_BACKEND=none
```

Measured on one Linux box (Release, gcc): full build 114 s / 15.5 MB / 30 shared
libs; headless 21 s / 4.1 MB / 13 libs; headless + no GPU 20 s / 3.8 MB / 10
libs. A headless binary produces **byte-identical** simulation state to a full
build.

The preset pins no compiler — on macOS it selects AppleClang, which builds it
clean. Combining `--preset headless` with `COAST_GPU_BACKEND=null` gives a build
with no GPU and no graphics dependency whatsoever.

## Usage

A GUI build launched with just a config opens a window; use *System > Open
Configuration* to load a config at runtime instead.

```bash
# From the repository root (configs reference datasets as ./data/...):
./build/coast configs/mnist.json
./build/coast configs/dsprites.json
```

### Headless runs

`-H/--headless` runs without a window, and `-n`/`--unlimited` each imply it — so
a GUI build given a run length runs headless without being told twice. A headless
*binary* is always headless and says so in a one-line notice, unless `-H` was
passed (which then serves only to suppress that notice).

**Every headless run must state how long it lasts** — there is no sentinel:

```bash
./build/coast -n 1000 configs/mnist.json     # finite: 1000 steps (count must be >= 1)
./build/coast --unlimited configs/mnist.json # unbounded: runs until interrupted
./build/coast --headless configs/mnist.json  # ERROR: needs -n or --unlimited
./build/coast -n 0 configs/mnist.json        # ERROR: 0 does not mean "forever"
```

`-n 0` formerly meant "run forever", which turned a mistyped or computed-to-zero
count into an unbounded compute job. It is now rejected, as is a run length left
unspecified.

`--unlimited` also refuses outputs it cannot deliver. Nothing catches the signal
that ends such a run, so `--save-state` and `--emit-run-manifest` would never be
written and `--log-buffers` would leave an unreadable, unclosed HDF5 file:

```bash
./build/coast --unlimited --save-state s.h5 configs/mnist.json  # ERROR
./build/coast --unlimited --checkpoint 5000 configs/mnist.json  # OK — snapshots
                                                                # complete mid-run
```

`--save-on-term` is the exception, and it is what lifts that refusal. It installs
a SIGTERM handler: the run stops, writes a final checkpoint, and exits through the
normal path — closing the buffer log and writing the state file and a manifest
whose `exit_reason` is `user_signal`. This is how `coastd` (see `docs/content/coastd/`)
runs an unbounded job it may later need to stop cleanly.

```bash
./build/coast --unlimited --save-on-term --checkpoint 5000 \
              --save-state s.h5 configs/mnist.json      # OK — SIGTERM writes s.h5
```

SIGINT (Ctrl-C) is still **not** caught, so an interactively-killed run loses
these outputs even with `--save-on-term`. It is a contract with a supervisor that
sends SIGTERM, not a general promise.

## Configuration

COAST supports JSON configuration files for experiment reproducibility.

See `configs/mnist.json` for a complete example. Top-level keys: `datasets`,
one model section (`ngm2`, `ngm25` or `ngm3`), `sim`, and `vis_hints`; some
configs add `observe`, `retina`, `frozen` and `snapshot_dir`.

## Datasets

Download datasets into `data/` — `scripts/download_datasets.sh` fetches the
missing ones, or download them by hand:

### MNIST

```bash
mkdir -p data/mnist && cd data/mnist
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
gunzip *.gz
```

### dSprites

```bash
mkdir -p data/dsprites && cd data/dsprites
curl -LO https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5
```

### Documentation
```bash
cd docs
hugo 
hugo server
```
