# 3.24: $<LINK_LIBRARY:WHOLE_ARCHIVE,...> — the portable spelling of the
# force-link-every-TU trick the coast exe needs (see the exe's link block near
# the end of this file). Previously 3.18, for FetchContent_MakeAvailable on
# content with no CMakeLists (the prebuilt wgpu-native zip) — still required.
cmake_minimum_required(VERSION 3.24)
project(coast)

# Export compile_commands.json so clangd knows include paths and compiler flags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Tests are opt-in. The default build is identical to the pre-test-scaffold one.
option(BUILD_TESTING "Build COAST test suite (requires doctest tree under tests/)" OFF)

# ── GUI vs headless ─────────────────────────────────────────────────────────
# ON  (default): the full app — coast_lib (headless core) + coast_gui (SDL2 /
#                OpenGL / ImGui layer). `coast` can open a window or run headless.
# OFF          : headless-only. coast_gui is never created, SDL2 and OpenGL are
#                never even searched for, and ImGui/ImPlot/ImNodeFlow are not
#                compiled. The resulting `coast` runs headless implicitly.
#
# The split is structural, not conventional: every GUI translation unit lives in
# coast_gui, which links coast_lib — never the reverse.
#
# What that buys, stated precisely, because the short version ("a GUI include in
# a core TU fails to link") is only true in some configurations:
#   * A core TU that includes an ImGui-BEARING header fails to COMPILE in any
#     configuration — the imgui/implot/ImNodeFlow/SDL2 roots are on coast_gui's
#     include path, not coast_lib's. This is the common case and it always bites.
#   * A core TU that includes an ImGui-FREE coast_gui header (gui_plot/figure.h,
#     console_iface.h, …) still compiles: `include/` is a single project root
#     shared by both archives. It then fails to LINK only where coast_gui is
#     absent — a COAST_WITH_GUI=OFF build, or a unit test linking bare coast_lib.
#     In a default GUI build the coast exe links coast_gui in, so it would go
#     unnoticed there.
# The practical consequence: building the `headless` preset is what actually
# enforces the boundary. Run it (and its test suite) before trusting the split —
# a GUI-only build cannot prove the core is GUI-free.
#
# COAST_WITH_GUI is also defined for every TU (see add_compile_definitions below)
# so main.cpp and the CLI usage text can adapt to the build.
option(COAST_WITH_GUI "Build the GUI layer (SDL2/OpenGL/ImGui). OFF = headless-only binary" ON)

# Sanitizer mode. Empty = none. Values: asan, ubsan, tsan. They are mutually
# exclusive: ASan/UBSan can compose in principle, but TSan must stand alone,
# and a single value keeps the option discoverable. Use separate out-of-tree
# build dirs (build-asan, build-ubsan, build-tsan) — flags are added globally
# so every target (coast_lib, coast, tests) gets them.
set(COAST_SANITIZER "" CACHE STRING "Sanitizer: '' (none) | asan | ubsan | tsan")
set_property(CACHE COAST_SANITIZER PROPERTY STRINGS "" asan ubsan tsan)

if(COAST_SANITIZER)
    set(_san_compile_flags -fno-omit-frame-pointer -g)
    set(_san_link_flags)
    if(COAST_SANITIZER STREQUAL "asan")
        list(APPEND _san_compile_flags -fsanitize=address)
        list(APPEND _san_link_flags    -fsanitize=address)
    elseif(COAST_SANITIZER STREQUAL "ubsan")
        list(APPEND _san_compile_flags -fsanitize=undefined)
        list(APPEND _san_link_flags    -fsanitize=undefined)
    elseif(COAST_SANITIZER STREQUAL "tsan")
        list(APPEND _san_compile_flags -fsanitize=thread)
        list(APPEND _san_link_flags    -fsanitize=thread)
    else()
        message(FATAL_ERROR "COAST_SANITIZER='${COAST_SANITIZER}' is not one of '' | asan | ubsan | tsan")
    endif()
    add_compile_options(${_san_compile_flags})
    add_link_options(${_san_link_flags})
    message(STATUS "Sanitizer enabled: ${COAST_SANITIZER} (flags: ${_san_compile_flags})")
endif()

# Dependencies. TBB and HDF5 are core (the simulation and its logging need them
# in every configuration); SDL2 and OpenGL belong to the GUI layer alone and are
# not even searched for in a headless build — so a headless build succeeds on a
# machine where neither is installed.
find_package(TBB REQUIRED)
# 1.12: H5Oget_info3 / H5O_info2_t (mem_state_io); 1.10 fails to compile.
find_package(HDF5 1.12 REQUIRED COMPONENTS C)
if(COAST_WITH_GUI)
    find_package(SDL2 REQUIRED)
    find_package(OpenGL REQUIRED)
    # Visible to every TU (not just coast_gui's) so main.cpp can gate the GUI
    # entry point and cli::print_usage can describe the right flag set.
    add_compile_definitions(COAST_WITH_GUI)
    message(STATUS "GUI: enabled (SDL2 + OpenGL + ImGui -> coast_gui)")
else()
    message(STATUS "GUI: disabled — headless-only build (no SDL2/OpenGL/ImGui)")
endif()

# Build-time git SHA / dirty / build-type capture for the run manifest.
include(${CMAKE_SOURCE_DIR}/cmake/coast_build_info.cmake)

# nlohmann/json - header-only, use submodule directly

# ── GPU backend selection ───────────────────────────────────────────────────
# COAST_GPU_BACKEND selects which (single) GPU backend is compiled in:
#   webgpu - wgpu-native (default); the only backend with real GPU execution.
#   null   - host-array no-op backend; a GPU-free build+run target that proves the
#            agnostic seam is genuinely backend-neutral (CI / portability asset).
#   none   - no GPU support at all (CPU-only build).
# COAST_WITH_GPU is DERIVED (defined for webgpu|null): it gates every backend-
# AGNOSTIC GPU TU — entities, io_buffer variants, gpu_io_entity, the gpu_device
# seam. The webgpu branch additionally defines COAST_WITH_WEBGPU, which gates the
# WebGPU-bearing TUs (the backend itself + the raw micro-benchmark drivers).
# Exactly one backend TU defines coast::gpu::make_gpu_device().
set(COAST_GPU_BACKEND "webgpu" CACHE STRING "GPU backend: none|webgpu|null")
set_property(CACHE COAST_GPU_BACKEND PROPERTY STRINGS none webgpu null)
if(NOT COAST_GPU_BACKEND MATCHES "^(none|webgpu|null)$")
    message(FATAL_ERROR "COAST_GPU_BACKEND must be one of: none|webgpu|null "
                        "(got '${COAST_GPU_BACKEND}')")
endif()
message(STATUS "GPU backend: ${COAST_GPU_BACKEND}")

# ── Toolchain requirements ──────────────────────────────────────────────────
# Checked here so an old compiler fails at configure with a message, not halfway
# through the build. Feature tests rather than version numbers: AppleClang's
# versions do not map onto LLVM's, and what matters is the standard library.
# Measured (fresh-clone builds, 2026-11-08): GCC 13 and 14 (libstdc++) lack
# std::vector::append_range, GCC 14 also lacks #embed; libc++ 19 lacks std::jthread
# (without -fexperimental-library) and floating-point std::from_chars; GCC 15 builds.
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)
include(CheckCXXSourceCompiles)
check_cxx_source_compiles([[
#include <charconv>
#include <thread>
#include <vector>
int main() {
    std::vector<int> v; const int a[] = {1}; v.append_range(a);
    double d = 0; const char s[] = "1.5"; std::from_chars(s, s + 3, d);
    std::jthread t([](std::stop_token) {});
    return v.size() == 1 && d == 1.5 ? 0 : 1;
}
]] COAST_HAVE_CXX23_LIBRARY)
if(NOT COAST_HAVE_CXX23_LIBRARY)
    message(FATAL_ERROR
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} with its standard library "
        "lacks a C++ library feature COAST uses (std::vector::append_range, std::jthread, "
        "floating-point std::from_chars). Use GCC >= 15 (libstdc++ 14 and older lack "
        "append_range); with Clang, a standard library that has all three (libc++ 19 does not).")
endif()
# The GPU TUs embed their WGSL with #embed (C23/C++26, an extension under C++23).
if(NOT COAST_GPU_BACKEND STREQUAL "none")
    check_cxx_source_compiles("
static const unsigned char data[] = {
#embed \"${CMAKE_CURRENT_LIST_FILE}\"
};
int main() { return sizeof(data) > 0 ? 0 : 1; }
" COAST_HAVE_EMBED)
    if(NOT COAST_HAVE_EMBED)
        message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not support #embed, "
            "which the GPU backend (COAST_GPU_BACKEND=${COAST_GPU_BACKEND}) needs. Use "
            "GCC >= 15 or Clang >= 19, or configure with -DCOAST_GPU_BACKEND=none.")
    endif()
endif()

# COAST_GPU_FAST_WAIT (webgpu backend only): realize gpu_context::wait_idle() as a
# non-sleeping OnSubmittedWorkDone + ProcessEvents pump instead of
# wgpuDevicePoll(device, wait=true). Works around the wgpu-native v29 Metal host
# wait being a thread::sleep(1ms) poll loop (event-driven only on wgpu trunk, after
# v29 — see benchmarks/m3_gpu_timing/findings_m3_followup.md): ~3.2-3.7x SPS on
# M3/Metal, ~neutral on RADV/Vulkan (devicePoll(wait) is already cheap there),
# numerically inert. ON by default; retire this switch and bump the wgpu-native pin
# once a release past v29 ships the upstream event-driven wait.
option(COAST_GPU_FAST_WAIT "WebGPU: non-sleeping device wait (wgpu v29 Metal sleep-poll workaround)" ON)

# Embed the clang-repl C++ console in the HDF5 Browser. Adds a system LLVM/Clang
# dependency (libclang-cpp + libLLVM), confined to the coast_console target so
# coast_lib and the small unit tests stay LLVM-free. Turn OFF for an
# LLVM-free build.
option(COAST_WITH_CONSOLE "Embed the clang-repl C++ console (needs system LLVM/Clang dev libs)" ON)

# The console is a GUI feature: console::console_panel IS an ImGui window, and the
# console's plot bridge renders through ImPlot. A headless build therefore cannot
# carry it. Derive the effective switch rather than overwriting the cache entry, so
# the user's COAST_WITH_CONSOLE choice survives flipping COAST_WITH_GUI back ON.
set(_coast_build_console ${COAST_WITH_CONSOLE})
if(COAST_WITH_CONSOLE AND NOT COAST_WITH_GUI)
    set(_coast_build_console OFF)
    message(STATUS "Console: disabled — COAST_WITH_GUI=OFF (the console is an ImGui panel)")
endif()

# ── wgpu-native (prebuilt), fetched only for the webgpu backend ──────────────
# wgpu-native ships per-platform prebuilt release zips (no CMakeLists inside),
# each containing include/webgpu/{webgpu.h,wgpu.h} + lib/libwgpu_native.{so,a}.
# We FetchContent the matching zip (URL_HASH verified), then hand-build an
# IMPORTED SHARED target since the archive carries no CMake config.
if(COAST_GPU_BACKEND STREQUAL "webgpu")
    # Pinned to wgpu-native v29.0.0.0 (latest as of 2026-06-03). SHA256 for
    # linux-x86_64 reproduced locally; the other three are GitHub's signed
    # release digests (hard-fail via URL_HASH on first configure if wrong).
    set(_wgpu_version "v29.0.0.0")
    set(_wgpu_base "https://github.com/gfx-rs/wgpu-native/releases/download/${_wgpu_version}")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(_wgpu_asset "wgpu-linux-x86_64-release.zip")
        set(_wgpu_sha   "cf614af80f23c6364a13f6569e7ae4ca7367ebf6062e3d3d6e80e205264636b4")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_wgpu_asset "wgpu-linux-aarch64-release.zip")
        set(_wgpu_sha   "d763c855a15ab77d43638769514e4cb24b2ab025928160ffbd8b791b1d5a083f")
    elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(_wgpu_asset "wgpu-macos-aarch64-release.zip")
        set(_wgpu_sha   "e3e3f6144f9b197b8199681e956391631cfc6d1cef057a1e8c5a1c6aedef5054")
    elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(_wgpu_asset "wgpu-macos-x86_64-release.zip")
        set(_wgpu_sha   "8201b3b9c05c395b9d3c8bb37d965c0c47c5e53a2e251cf6af04356f01e6c81a")
    else()
        message(FATAL_ERROR "COAST_GPU_BACKEND=webgpu but no prebuilt wgpu-native asset for "
                            "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. "
                            "Use -DCOAST_GPU_BACKEND=null|none or add the asset mapping.")
    endif()

    # Each build tree extracts wgpu into its own <build>/_deps (the FetchContent
    # default). We deliberately do NOT share one FetchContent dir across trees:
    # its populate subbuild is generator- AND toolchain-version specific, so a
    # shared dir is brittle the moment two build trees differ in generator (Make
    # vs Ninja) or cmake/ninja version (e.g. a CLI build vs a CLion build) — they
    # collide on the shared subbuild ("generator does not match" / "failed
    # recompaction"). Only the downloaded zip is shared, through the machine-wide
    # dependency cache (cmake/coast_dep_cache.cmake): a new tree then needs no
    # network, and GitHub is asked only when the cache has no verified copy.
    include(${CMAKE_SOURCE_DIR}/cmake/coast_dep_cache.cmake)
    coast_cached_download(_wgpu_url "${_wgpu_base}/${_wgpu_asset}" "${_wgpu_sha}"
                          "wgpu-native/${_wgpu_version}")
    include(FetchContent)
    FetchContent_Declare(wgpu_native
        URL      "${_wgpu_url}"
        URL_HASH SHA256=${_wgpu_sha}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)  # silence CMP0135; prebuilt asset, mtimes irrelevant
    FetchContent_MakeAvailable(wgpu_native)  # zip has no CMakeLists -> populate+extract only

    # Hand-built imported target. Shared lib per IDD §6.7. CMAKE_SHARED_LIBRARY_SUFFIX
    # resolves .so on Linux / .dylib on macOS (macOS path validated at S13).
    add_library(wgpu_native::wgpu_native SHARED IMPORTED GLOBAL)
    set_target_properties(wgpu_native::wgpu_native PROPERTIES
        IMPORTED_LOCATION             "${wgpu_native_SOURCE_DIR}/lib/libwgpu_native${CMAKE_SHARED_LIBRARY_SUFFIX}"
        # The prebuilt lib carries NO DT_SONAME. By default CMake then links it by
        # full path, so the consumer's DT_NEEDED becomes that (build-relative)
        # path, which the dynamic linker resolves against the CWD and NOT the
        # RUNPATH below — the exe only runs from inside its build dir. Telling
        # CMake the lib has no soname makes it link via -L/-l search instead, so
        # DT_NEEDED is the bare basename (libwgpu_native.so), resolved via RUNPATH
        # from any working directory (e.g. an IDE launching from the repo root).
        IMPORTED_NO_SONAME            TRUE
        INTERFACE_INCLUDE_DIRECTORIES "${wgpu_native_SOURCE_DIR}/include")

    # The .so lives in the build/fetch tree, not a system path. Add its dir to the
    # build RPATH so every executable (coast, smoke, test exes) resolves it at
    # launch with no install step. Set before any target is created so all inherit.
    list(APPEND CMAKE_BUILD_RPATH "${wgpu_native_SOURCE_DIR}/lib")

    message(STATUS "wgpu-native: ${_wgpu_asset} -> ${wgpu_native_SOURCE_DIR}")
endif()
# ────────────────────────────────────────────────────────────────────────────

# Architecture-specific optimizations (applied globally)
if (APPLE)
    # Apple Silicon (M1/M2/M3): -march=native enables ARM NEON SIMD automatically
    # NEON is always available on ARM64 (not optional like AVX on x86)
    # The __ARM_NEON and __aarch64__ macros are defined automatically by the compiler
    add_compile_options(-march=native)
    # GCC on macOS uses libstdc++; a clang build (needed to link Homebrew's
    # libc++-built LLVM for the console) must NOT be forced onto libstdc++.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-stdlib=libstdc++)
    endif()
else()
    # x86-64: Enable AVX512 for SIMD-optimized RNG (falls back to AVX2/scalar if unavailable)
    add_compile_options(-march=native)
endif()

# ── 3rd-party GUI libraries — GUI builds only ───────────────────────────────
# Both sit on top of Dear ImGui (compiled into coast_gui), so neither has any
# meaning in a headless build.
if(COAST_WITH_GUI)
    # ImPlot library (3rd-party, warnings disabled)
    add_library(implot STATIC
        3rd_party/implot/implot.cpp
        3rd_party/implot/implot_items.cpp
    )
    target_include_directories(implot PUBLIC 3rd_party/implot 3rd_party/imgui)
    target_compile_options(implot PRIVATE -w)

    # ImNodeFlow library (3rd-party, warnings disabled)
    add_library(ImNodeFlow STATIC
        3rd_party/ImNodeFlow/src/ImNodeFlow.cpp
    )
    target_include_directories(ImNodeFlow PUBLIC
        3rd_party/ImNodeFlow/include
        3rd_party/ImNodeFlow/src
        3rd_party/imgui
    )
    target_compile_definitions(ImNodeFlow PUBLIC IMGUI_DEFINE_MATH_OPERATORS)
    target_compile_options(ImNodeFlow PRIVATE -w)
endif()

add_library(coast_lib STATIC
        # Diagnostics (header-only: logging facility + run-progress contract)
        include/diag/level.h
        include/diag/diag.h
        include/diag/progress.h

        # CLI parsing
        include/cli/options.h
        include/cli/parse.h
        src/cli/parse.cpp

        # Application entry points (build_env, run_headless). run_gui is the GUI
        # layer's entry point and lives in coast_gui.
        include/app/build_env.h
        src/app/build_env.cpp
        include/app/run_headless.h
        src/app/run_headless.cpp
        include/app/headless_checkpoint.h
        src/app/headless_checkpoint.cpp
        include/app/run_provenance.h
        src/app/run_provenance.cpp
        include/core/local_time.h

        # Application-level location config + per-config output path resolver
        include/core/app_config.h
        src/core/app_config.cpp
        include/core/path_resolver.h
        src/core/path_resolver.cpp
        include/core/run_id.h
        src/core/run_id.cpp

        # Core infrastructure (entity element hierarchy)
        # vis_hints_t is a plain config POD (ints/floats/maps, no GUI types): it is
        # parsed from the experiment config and written into the HDF5 buffer-log
        # topology, both of which happen headless. It lives in core/ — NOT gui_vis/ —
        # so the headless build needs no GUI header at all.
        include/core/vis_hints.h
        include/core/element_type.h
        include/core/entity_element.h
        include/core/entity_element_macros.h
        include/core/structure_registry.h
        include/core/probe_name_registry.h
        # RNG utilities
        include/core/rng/rng_wrapper.h
        include/core/rng/mt19937_simd.h
        include/core/rng/fast_rng.h
        src/core/entity_element.cpp
        src/core/structure_registry.cpp
        src/core/probe_name_registry.cpp

        # Probe system. probe_observer is a pure interface (no GUI types), so the
        # recording side stays here; every concrete observer renders through ImGui
        # and lives in coast_gui alongside the probe browser that constructs them.
        include/probe/spsc_queue.h
        include/probe/probe_observer.h
        include/probe/probe_instance.h
        include/probe/entity_probe_set.h
        src/probe/probe_instance.cpp
        src/probe/entity_probe_set.cpp

        # Observe system
        include/observe/observe_trigger.h
        include/observe/observe_site.h
        include/observe/observe_group.h
        include/observe/observe_registry.h
        include/observe/observe_writer.h
        include/observe/observe_accessor.h
        include/observe/observe_dtype.h
        src/observe/observe_accessor.cpp
        src/observe/observe_registry.cpp
        src/observe/observe_writer.cpp
        src/observe/observe_config.cpp

        # Dataset I/O
        # Shared per-sample index sequencer (sequential + per-epoch shuffle)
        include/io/sample_sequencer.h
        src/io/sample_sequencer.cpp
        # Whether a load restores a loader's sequence (run_id), its identity record (SPEC S.5.14)
        include/io/dataset_run.h
        src/io/dataset_run.cpp
        include/io/mnist_db.h
        src/io/mnist_db.cpp
        # Shared base for IDX-format loaders (mnist, emnist, kmnist, color_mnist)
        include/io/idx_io_base.h
        src/io/idx_io_base.cpp
        src/io/mnist_io.cpp
        include/io/mnist_io.h
        include/io/dsprites_io.h
        src/io/dsprites_io.cpp
        include/io/sequence_params.h
        include/io/mnist_cfg.h
        include/io/dataset_cfg.h
        include/io/dsprites_cfg.h
        # EMNIST / KMNIST configs (their loaders are instantiations in mnist_io)
        include/io/emnist_cfg.h
        include/io/kmnist_cfg.h
        # Shared base for CIFAR binary-batch loaders (cifar10, cifar100)
        include/io/cifar_io_base.h
        src/io/cifar_io_base.cpp
        # CIFAR-10 dataset
        include/io/cifar10_cfg.h
        include/io/cifar10_io.h
        src/io/cifar10_io.cpp
        # CIFAR-100 dataset
        include/io/cifar100_cfg.h
        include/io/cifar100_io.h
        src/io/cifar100_io.cpp
        # 3D Chairs dataset
        include/io/chairs3d_cfg.h
        include/io/chairs3d_io.h
        src/io/chairs3d_io.cpp
        # Color MNIST dataset (MNIST with random R/G/B coloring)
        include/io/color_mnist_cfg.h
        include/io/color_mnist_io.h
        src/io/color_mnist_io.cpp

        # HD-NGM2 neural model
        include/hd_ngm2/hd_ngm2_dendrite.h
        src/hd_ngm2/hd_ngm2_dendrite.cpp
        include/hd_ngm2/hd_ngm2_neuron.h
        src/hd_ngm2/hd_ngm2_neuron.cpp
        include/hd_ngm2/hd_ngm2_tools.h
        include/hd_ngm2/hd_ngm2_neuron_group.h
        src/hd_ngm2/hd_ngm2_neuron_group.cpp
        include/hd_ngm2/hd_ngm2.h
        include/hd_ngm2/hd_ngm2_cfg.h

        # Inhibition (swappable, model-agnostic)
        include/inhibition/inhibition_cfg.h
        include/inhibition/inhibition.h
        src/inhibition/inhibition.cpp
        include/inhibition/random_index.h
        src/inhibition/random_index.cpp
        include/inhibition/max_softmax.h
        src/inhibition/max_softmax.cpp
        include/json_cfg/inhibition_json.h
        include/retina/retina_cfg.h
        include/retina/retina.h
        src/retina/retina.cpp
        include/retina/retina_io.h
        src/retina/retina_io.cpp
        include/json_cfg/retina_json.h
        include/json_cfg/strict_enum.h

        # Simulation core
        include/sim_core/io_entity.h
        src/sim_core/sim_env.cpp
        include/sim_core/sim_env.h
        src/sim_core/io_buffer.cpp
        include/sim_core/io_buffer.h
        src/sim_core/sim_thread_controller.cpp
        include/sim_core/sim_thread_controller.h
        # Simulation instance (RAII container for env+builder+controller+logging_mgr)
        include/sim_core/sim_instance.h
        src/sim_core/sim_instance.cpp
        # Run manifest writer (coast_runtime.json provenance)
        include/sim_core/run_manifest.h
        src/sim_core/run_manifest.cpp

        # JSON configuration
        include/json_cfg/json_loader.h
        src/json_cfg/json_loader.cpp
        include/json_cfg/ngm2_json.h
        include/json_cfg/sim_json.h
        include/json_cfg/dataset_json.h

        # Tools
        include/tools/sim_builder_cfg.h
        include/tools/sim_builder.h
        src/tools/sim_builder.cpp
        include/tools/sim_builder_interface.h
        include/tools/dataset_loader.h
        src/tools/dataset_loader.cpp
        include/tools/metric.h
        include/tools/metric_registry.h
        include/json_cfg/vis_hints_json.h

        # HDF5 executor (single-thread funnel for libhdf5)
        include/hdf5/executor.h
        include/hdf5/handle.h
        include/hdf5/native_type.h
        src/hdf5/executor.cpp

        # Logging system
        include/logging/logging.h
        include/logging/logging_common.h
        include/logging/hdf5_utils.h
        src/logging/hdf5_utils.cpp
        include/logging/buffer_logger.h
        src/logging/buffer_logger.cpp
        include/logging/state_io.h
        include/logging/config_record.h
        src/logging/config_record.cpp
        include/logging/state_checks.h
        src/logging/state_checks.cpp
        include/logging/mem_state_io.h
        src/logging/mem_state_io.cpp
        include/logging/state_logger.h
        src/logging/state_logger.cpp
        include/logging/logging_manager.h
        src/logging/logging_manager.cpp

        # ngm3 version of the neuron model - just a sketch so far
        include/hd_ngm3/hd_ngm3_neuron_group.h
        src/hd_ngm3/hd_ngm3_neuron_group.cpp
        include/hd_ngm3/hd_ngm3_layout.h
        src/hd_ngm3/hd_ngm3_layout.cpp
        include/hd_ngm3/hd_ngm3_cold_init.h
        src/hd_ngm3/hd_ngm3_cold_init.cpp
        include/hd_ngm3/hd_ngm3_host.h
        src/hd_ngm3/hd_ngm3_host.cpp
        include/ngm/bank.h
        include/ngm/common_config_record.h
        include/json_cfg/ngm_bank_json.h

        # ngm2.5 flat-sparse model (design/ngm2_5_flat_sparse.md). Incremental
        # build-out I1-I10; the arena + consistency checker land first (I1).
        include/hd_ngm25/hd_ngm25_cfg.h
        include/hd_ngm25/hd_ngm25_prng.h
        include/hd_ngm25/hd_ngm25_arena.h
        src/hd_ngm25/hd_ngm25_arena.cpp
        include/hd_ngm25/hd_ngm25_response.h
        src/hd_ngm25/hd_ngm25_response.cpp
        include/hd_ngm25/hd_ngm25_adapt.h
        src/hd_ngm25/hd_ngm25_adapt.cpp
        include/hd_ngm25/hd_ngm25_branch.h
        src/hd_ngm25/hd_ngm25_branch.cpp
        # Host code shared by the CPU and GPU group (routing, mirror, checkpoint).
        include/hd_ngm25/hd_ngm25_host.h
        src/hd_ngm25/hd_ngm25_host.cpp
        include/hd_ngm25/hd_ngm25_neuron_group.h
        src/hd_ngm25/hd_ngm25_neuron_group.cpp
        include/json_cfg/ngm25_json.h

        # h5browse — GUI-free HDF5 log-file browser domain module
        include/h5browse/scalar_type.h
        include/h5browse/scan_stats.h
        include/h5browse/base_histogram.h
        src/h5browse/base_histogram.cpp
        include/h5browse/row_filter.h
        src/h5browse/row_filter.cpp
        include/h5browse/stats_kernel.h
        src/h5browse/stats_kernel.cpp
        include/h5browse/h5b_error.h
        src/h5browse/h5b_error.cpp
        include/h5browse/file_model.h
        src/h5browse/file_model.cpp
        include/h5browse/chunk_cache.h
        src/h5browse/chunk_cache.cpp
        include/h5browse/dataset_reader.h
        src/h5browse/dataset_reader.cpp
        include/h5browse/pairwise.h
        src/h5browse/pairwise.cpp
        include/h5browse/scan_engine.h
        src/h5browse/scan_engine.cpp
        include/h5browse/console_api.h
        src/h5browse/console_api.cpp
        # run_config — the editable simulation config (load base JSON, set() by
        # JSON pointer, dump()). The shared core of the "edit then dispatch"
        # console workflow: coast::simc (GUI) runs it, coast::coastd (below)
        # submits it. GUI-free (nlohmann/json only), so it lives in coast_lib
        # where BOTH façades can reach it. See design/run_config_extraction.md.
        include/run_config/run_config.h
        src/run_config/run_config.cpp
        # coastd_console — REPL façade over coastd_client: submit/list/cancel/
        # reorder/machine_info against a coastd daemon. Lean parse boundary (no
        # json/socket headers); wire types re-projected as plain structs. No
        # ImGui, no gui_plot — so it stays in coast_lib and is available to a
        # headless build too.
        include/coastd_console/coastd_console_api.h
        src/coastd_console/coastd_console_api.cpp
        # campaign — the meta-loader over a directory of runs: discovers the swept
        # parameters by DIFFING the run configs against each other, then filters run
        # handles by them. Strictly lazy (the scan reads configs only; HDF5 is opened
        # on demand, through h5browse's console file_console). Lean parse boundary,
        # nlohmann-free header, no ImGui — so it lives in coast_lib and a headless
        # build has it too.
        include/campaign/campaign_api.h
        src/campaign/campaign_api.cpp
)

# 3rd-party includes (marked as SYSTEM to suppress warnings).
# PUBLIC so anything linking coast_lib (the coast exe, tests) inherits them.
# The GUI 3rd-party roots (imgui, implot, ImNodeFlow, imgui-filebrowser, SDL2) are
# deliberately NOT here — they belong to coast_gui. stb stays: src/io/chairs3d_io.cpp
# decodes images with stb_image headlessly.
target_include_directories(coast_lib SYSTEM PUBLIC
        3rd_party/json/include
        3rd_party/stb
        ${HDF5_INCLUDE_DIRS}
)

# Project includes (warnings enabled). PUBLIC for the same reason.
#
# `include` is deliberately the ONLY project header root: every project include
# is written include/-rooted (`#include "core/rng/fast_rng.h"`), so the per-module
# directories that used to sit here (include/core, include/gui_core, …) are not
# just redundant — listing them would let a bare `#include "fast_rng.h"` compile,
# quietly re-admitting the ambiguity the rooted style exists to remove. Keeping
# the list at one entry is what makes the convention a compile error rather than
# a habit. Do not add module directories back.
target_include_directories(coast_lib PUBLIC
        include
        # CMake-generated coast_build_info.h (git SHA / dirty / build type)
        ${CMAKE_BINARY_DIR}/generated
)

# Enable warnings for our own code. GCC's -Wpsabi emits informational ABI
# notes for standard library value types; it is noisy for full local rebuilds.
target_compile_options(coast_lib PRIVATE
        -Wall
        -Wpedantic
        -Wextra
        $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-psabi>
)

# NOTE: do NOT add /opt/homebrew/include/c++/<ver> to the include path here.
# A matching Homebrew GCC already finds its own libstdc++ headers; injecting that
# directory as an explicit -I duplicates it on the search path and breaks the
# #include_next chain in C-library wrapper headers (e.g. <fenv.h> fails with
# "'fenv_t' has not been declared"). An earlier hardcoded gcc-15 line only looked
# harmless because that path was a symlink GCC de-duplicated against its built-in
# path; a real (non-symlinked) Homebrew include dir surfaces it as a build failure.

# Link dependencies as PUBLIC so the coast exe (and any future test exes)
# inherit them transitively via `target_link_libraries(... coast_lib)`.
# SDL2/OpenGL/implot/ImNodeFlow moved to coast_gui — this is what makes a headless
# `coast` (and every unit test that links only coast_lib) genuinely GUI-free.
target_link_libraries(coast_lib PUBLIC
    TBB::tbb
    ${HDF5_C_LIBRARIES}
)

# The coastd console façade lives in coast_lib and speaks the wire protocol through
# the thin, coast_lib-free client. Defined by add_subdirectory(coastd) further down;
# CMake resolves the target at generate time.
#
# PRIVATE, deliberately. exactly ONE coast_lib TU (src/coastd_console/coastd_console_api.cpp)
# includes client/coastd_client.h; the public façade header is a lean parse boundary
# that carries neither json nor sockets, precisely so the clang-repl console can parse
# it without reaching coastd/include. A PUBLIC link contradicted that: it pushed
# coastd's header root (and nlohmann, via coastd_proto) onto the include path of every
# coast_lib consumer — the coast exe, coast_gui, every unit test — so any core TU could
# have `#include "daemon/types.h"` and compiled. That is the same ambiguity the
# single-include-root rule above exists to make a compile error, re-admitted through a
# link line. It is also what widened a two-line daemon defect (an unguarded
# MSG_NOSIGNAL, see coastd/include/daemon/sock_compat.h) into a whole-project macOS
# build blocker. PRIVATE still puts the archive on the final link line for the exe —
# static-library link requirements propagate regardless — it just stops the daemon's
# usage requirements from leaking into code that has no business speaking the protocol.
# Targets that DO speak it (test_coastd_comms, test_coastd_fault_matrix) link
# coastd_client by name, which is the honest declaration of that dependency.
#
# $<LINK_ONLY:> takes it one step further: plain PRIVATE would still apply the
# usage requirements to coast_lib's OWN 66 TUs (PRIVATE means "not propagated to
# consumers", not "not applied here"), leaving every core source able to include
# daemon/*. LINK_ONLY takes the archive and drops the include dirs, and the single
# TU that genuinely needs them gets them per-source below. Net effect: exactly one
# file in coast_lib can see coastd's headers, and `#include "daemon/types.h"`
# anywhere else in the core is a compile error — the property this comment claims.
target_link_libraries(coast_lib PRIVATE $<LINK_ONLY:coastd_client>)
set_source_files_properties(src/coastd_console/coastd_console_api.cpp
    PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/coastd/include")

# GPU support. The AGNOSTIC layer (COAST_WITH_GPU) compiles for any backend ≠ none;
# the chosen backend's private TUs are added inside the per-backend branch below.
# The backend's SDK (headers + library) is NOT a usage requirement of coast_lib:
# see the webgpu branch.
if(NOT COAST_GPU_BACKEND STREQUAL "none")
    target_compile_definitions(coast_lib PUBLIC COAST_WITH_GPU)
    target_sources(coast_lib PRIVATE
        # GPU abstraction seam (agnostic facade — NO SDK types) + shared WGSL source.
        include/core/gpu/gpu_device.h
        include/core/gpu/shared_prng.h
        src/core/gpu/shared_prng.cpp
        # GPU io_buffer variants + GPU entity base (S3)
        include/sim_core/gpu_io_entity.h
        include/sim_core/io_buffer_gpu.h
        src/sim_core/io_buffer_gpu.cpp
        # GPU dataset entity (S5). The .wgsl is #embed'd by the .cpp (GCC tracks
        # the dependency via -MD), so it is not listed as a compiled source.
        include/io/mnist_io_gpu.h
        src/io/mnist_io_gpu.cpp
        # GPU ngm3 neuron group (S6). .wgsl #embed'd by the .cpp like S5.
        include/hd_ngm3/hd_ngm3_gpu_neuron_group.h
        src/hd_ngm3/hd_ngm3_gpu_neuron_group.cpp
        # GPU ngm2.5 neuron group (GI1). CPU-arena-backed; .wgsl #embed'd by the .cpp.
        include/hd_ngm25/hd_ngm25_gpu_neuron_group.h
        src/hd_ngm25/hd_ngm25_gpu_neuron_group.cpp
        # GPU inhibition emitter (kernel injection). WGSL is an inline raw-string
        # literal in the .cpp, so nothing to #embed.
        include/inhibition/gpu_inhibition.h
        src/inhibition/gpu_inhibition.cpp
        include/inhibition/gpu_max_softmax.h
        src/inhibition/gpu_max_softmax.cpp)

    # The three TUs above embed their .wgsl via #embed, which is standardized in
    # C23 / C++26; under our C++23 it is a well-supported GCC/Clang extension that
    # -Wpedantic flags (GCC: -Wc++26-extensions, Clang: -Wc23-extensions). Silence
    # just that one diagnostic, only on these sources, so the rest of coast_lib
    # keeps its full -Wall -Wpedantic -Wextra surface.
    set_source_files_properties(
        src/core/gpu/shared_prng.cpp
        src/io/mnist_io_gpu.cpp
        src/hd_ngm3/hd_ngm3_gpu_neuron_group.cpp
        src/hd_ngm25/hd_ngm25_gpu_neuron_group.cpp
        PROPERTIES COMPILE_OPTIONS
            "$<$<CXX_COMPILER_ID:GNU>:-Wno-c++26-extensions>;$<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-c23-extensions>")

    # ── Backend-private TUs: exactly one backend defines make_gpu_device() ──
    if(COAST_GPU_BACKEND STREQUAL "webgpu")
        # The SDK stays behind gpu_device.h by construction, not by convention:
        # $<LINK_ONLY:> puts the .so on every consumer's link line (a static
        # archive's link requirements always propagate) but applies none of its
        # usage requirements, so no core, GUI or test TU gets <webgpu/webgpu.h>
        # on its include path. Exactly the three backend TUs below get it, per
        # source — the same pattern as coastd_client above. A non-backend
        # `#include <webgpu/webgpu.h>` or `#include "gpu_context.h"` is therefore
        # a compile error. The .so is a separate link item appended outside the
        # coast exe's --whole-archive/--no-whole-archive wrap of coast_lib, so it
        # is unaffected.
        target_link_libraries(coast_lib PRIVATE $<LINK_ONLY:wgpu_native::wgpu_native>)
        target_compile_definitions(coast_lib PUBLIC COAST_WITH_WEBGPU)
        if(COAST_GPU_FAST_WAIT)
            target_compile_definitions(coast_lib PUBLIC COAST_GPU_FAST_WAIT)
            message(STATUS "WebGPU device wait: fast (OnSubmittedWorkDone pump)")
        else()
            message(STATUS "WebGPU device wait: blocking wgpuDevicePoll(wait=true)")
        endif()
        # WebGPU-bearing headers + TUs live ONLY under backends/webgpu/. The
        # backend TUs include their siblings by same-dir relative path; they need
        # only the SDK's own include root.
        set(_coast_webgpu_backend_tus
            src/core/gpu/backends/webgpu/backend_webgpu.cpp
            src/core/gpu/backends/webgpu/gpu_context.cpp
            src/core/gpu/backends/webgpu/wgsl_module.cpp)
        target_sources(coast_lib PRIVATE
            src/core/gpu/backends/webgpu/gpu_context.h
            src/core/gpu/backends/webgpu/wgpu_util.h
            src/core/gpu/backends/webgpu/wgsl_module.h
            ${_coast_webgpu_backend_tus})
        set_source_files_properties(${_coast_webgpu_backend_tus}
            PROPERTIES INCLUDE_DIRECTORIES "${wgpu_native_SOURCE_DIR}/include")

        # The raw micro-bench / smoke drivers (defined further down) are the only
        # other code allowed past the seam: they drive gpu_context directly. They
        # link this to say so — the backend header dir plus the SDK.
        add_library(coast_webgpu_backend_private INTERFACE)
        target_include_directories(coast_webgpu_backend_private
            INTERFACE src/core/gpu/backends/webgpu)
        target_link_libraries(coast_webgpu_backend_private
            INTERFACE wgpu_native::wgpu_native)
    elseif(COAST_GPU_BACKEND STREQUAL "null")
        # Host-array no-op backend: no external deps, no SDK. Proves the seam is
        # backend-neutral (GPU-free build+run target).
        target_compile_definitions(coast_lib PUBLIC COAST_WITH_NULL_GPU)
        target_sources(coast_lib PRIVATE
            src/core/gpu/backends/null/backend_null.cpp)
    endif()
endif()

# ── coast_gui — the GUI layer ────────────────────────────────────────────────
# Everything that includes Dear ImGui, ImPlot, ImNodeFlow, SDL2 or OpenGL lives
# here and nowhere else. coast_gui links coast_lib; coast_lib never links
# coast_gui, so the dependency is one-directional by construction and a GUI
# include that creeps into a core TU shows up as an undefined symbol at link time.
if(COAST_WITH_GUI)
add_library(coast_gui STATIC
        # GUI application entry point (the SDL2/ImGui event loop)
        include/app/run_gui.h
        src/app/run_gui.cpp

        # Dear ImGui core
        3rd_party/imgui/imconfig.h
        3rd_party/imgui/imgui.cpp
        3rd_party/imgui/imgui.h
        3rd_party/imgui/imgui_demo.cpp
        3rd_party/imgui/imgui_draw.cpp
        3rd_party/imgui/imgui_internal.h
        3rd_party/imgui/imgui_tables.cpp
        3rd_party/imgui/imgui_widgets.cpp
        3rd_party/imgui/imstb_rectpack.h
        3rd_party/imgui/imstb_textedit.h
        3rd_party/imgui/imstb_truetype.h

        # Dear ImGui backends for SDL2 + OpenGL3
        3rd_party/imgui/backends/imgui_impl_sdl2.cpp
        3rd_party/imgui/backends/imgui_impl_sdl2.h
        3rd_party/imgui/backends/imgui_impl_opengl3.cpp
        3rd_party/imgui/backends/imgui_impl_opengl3.h
        3rd_party/imgui/backends/imgui_impl_opengl3_loader.h

        # SDL2 application framework
        src/gui_core/sdl_app.cpp
        include/gui_core/sdl_app.h
        # GUI manager (menu bar, dialogs)
        src/gui_core/gui_manager.cpp
        include/gui_core/gui_manager.h
        # Configuration editor
        src/gui_core/config_editor.cpp
        include/gui_core/config_editor.h
        # Network editor (ImNodeFlow-based)
        src/gui_core/network_editor.cpp
        include/gui_core/network_editor.h
        # Browser windows for introspection
        src/gui_core/entity_browser.cpp
        include/gui_core/entity_browser.h
        src/gui_core/buffer_browser.cpp
        include/gui_core/buffer_browser.h
        src/gui_core/dataset_browser.cpp
        include/gui_core/dataset_browser.h
        # Shared dataset enumeration/accessor + reusable thumbnail grid
        src/gui_core/dataset_access.cpp
        include/gui_core/dataset_access.h
        src/gui_core/thumbnail_grid.cpp
        include/gui_core/thumbnail_grid.h
        src/gui_core/logging_window.cpp
        include/gui_core/logging_window.h

        # Probe observers — every one of these renders through ImGui/ImPlot, and
        # only gui_probe's probe browser constructs them. The recording side of the
        # probe system (probe_instance, entity_probe_set) stays in coast_lib.
        include/probe/observers/scalar_observer.h
        include/probe/observers/time_series_observer.h
        include/probe/observers/histogram_observer.h
        include/probe/observers/csv_file_observer.h
        src/probe/observers/scalar_observer.cpp
        src/probe/observers/time_series_observer.cpp
        src/probe/observers/histogram_observer.cpp
        src/probe/observers/csv_file_observer.cpp

        # Probe GUI
        include/gui_probe/probe_browser.h
        include/gui_probe/observer_panel.h
        src/gui_probe/probe_browser.cpp
        src/gui_probe/observer_panel.cpp

        # Observe GUI
        include/gui_observe/observe_panel.h
        src/gui_observe/observe_panel.cpp

        # Visualization core
        src/gui_vis/core/gl_texture.cpp
        include/gui_vis/core/gl_texture.h
        include/gui_vis/core/color_types.h
        include/gui_vis/core/colormap.h
        src/gui_vis/core/colormap.cpp
        include/gui_vis/core/gradient.h

        # Visualization primitives
        src/gui_vis/primitives/vec_vis.cpp
        include/gui_vis/primitives/vec_vis.h
        src/gui_vis/primitives/vec_group_vis.cpp
        include/gui_vis/primitives/vec_group_vis.h

        # IO buffer visualizations
        src/gui_vis/io_buffer/vec_ring_buffer.cpp
        include/gui_vis/io_buffer/vec_ring_buffer.h

        # Sim entity visualizations
        src/gui_vis/sim_entity/neuron_group/ngm_flat_vis.cpp
        include/gui_vis/sim_entity/neuron_group/ngm_flat_vis.h
        src/gui_vis/sim_entity/neuron_group/ngm_tree_vis_v2.cpp
        include/gui_vis/sim_entity/neuron_group/ngm_tree_vis_v2.h
        src/gui_vis/sim_entity/neuron_group/ngm3_flat_vis.cpp
        include/gui_vis/sim_entity/neuron_group/ngm3_flat_vis.h
        src/gui_vis/sim_entity/neuron_group/ngm3_tree_vis.cpp
        include/gui_vis/sim_entity/neuron_group/ngm3_tree_vis.h

        # Visualization registry
        src/gui_vis/vis_registry.cpp
        include/gui_vis/vis_registry.h

        # Force-directed layout engine
        include/gui_layout/layout_graph.h
        src/gui_layout/layout_graph.cpp
        include/gui_layout/force_layout.h
        src/gui_layout/force_layout.cpp
        include/gui_layout/render/node_renderer.h
        src/gui_layout/render/node_renderer.cpp
        include/gui_layout/render/graph_canvas.h
        src/gui_layout/render/graph_canvas.cpp

        # GUI tools (Tools menu — Metric Explorer, HDF5 Browser)
        include/gui_tools/matrix_canvas.h
        src/gui_tools/matrix_canvas.cpp
        include/gui_tools/dataset_source.h
        src/gui_tools/dataset_source.cpp
        include/gui_tools/metric_explorer.h
        src/gui_tools/metric_explorer.cpp
        include/gui_tools/h5_browser_window.h
        src/gui_tools/h5_browser_window.cpp

        # coastd daemon-control panel (Tools menu) — ImGui front over the
        # coast::coastd console façade in coast_lib (same protocol client the
        # REPL uses). ImGui-only, hence coast_gui.
        include/gui_tools/coastd_panel.h
        src/gui_tools/coastd_panel.cpp

        # Campaign browser (Analysis menu) — ImGui front over the coast::campaign
        # meta-loader in coast_lib (same loader the REPL reaches as `camp::`).
        # ImGui-only, hence coast_gui.
        include/gui_tools/campaign_panel.h
        src/gui_tools/campaign_panel.cpp

        # Buffer replay system
        include/gui_replay/buffer_log_reader.h
        src/gui_replay/buffer_log_reader.cpp
        include/gui_replay/replay_node_renderer.h
        src/gui_replay/replay_node_renderer.cpp
        include/gui_replay/buffer_replay_dialog.h
        src/gui_replay/buffer_replay_dialog.cpp

        # sim_console — the `coast::simc` sim-control façade for the embedded C++
        # console (design/sim_console_api.md). Edits a config and spawns headless
        # `coast` subprocesses; its run dashboard and log viewers are ImGui windows,
        # and its only consumers are the console and gui_manager — i.e. it is a
        # console feature, and the console is GUI-only. If a headless consumer ever
        # needs it, split the ImGui windows out and move the façade to coast_lib.
        include/sim_console/sim_console_api.h
        src/sim_console/sim_console_api.cpp

        # gui_plot — ImPlot bridge fed from console macros (GUI-thread rendering).
        # One header/source pair per figure kind; shared base + framework alongside.
        include/gui_plot/plot_data.h        # plain DTOs (grid_result, h5_save_data)
        include/gui_plot/figure.h           # base class + painter seam (ImGui-free)
        include/gui_plot/plot_manager.h     # the singleton object (registry/hook/queues/executor)
        include/gui_plot/plot_detail.h      # module-internal glue (canvas DTO, loader, helpers)
        include/gui_plot/plot_h5.h          # HDF5 (de)serialization decls
        include/gui_plot/plot_tools.h       # render helpers + theme layer
        include/gui_plot/histogram.h
        include/gui_plot/line.h
        include/gui_plot/histogram2d.h
        include/gui_plot/pixels.h
        include/gui_plot/image.h
        include/gui_plot/canvas.h
        include/gui_plot/console_plot.h     # umbrella (includes kinds + free funcs)
        src/gui_plot/figure.cpp             # base: chrome, save, painter, registration
        src/gui_plot/plot_manager.cpp       # singleton: registry, task hook, render loop, load
        src/gui_plot/histogram.cpp
        src/gui_plot/line.cpp
        src/gui_plot/histogram2d.cpp
        src/gui_plot/pixels.cpp
        src/gui_plot/image.cpp
        src/gui_plot/canvas.cpp
        src/gui_plot/plot_load.cpp          # the one kind->type map (make_figure_from_h5)
        src/gui_plot/plot_h5.cpp            # HDF5 read/write
        src/gui_plot/plot_capture.cpp       # off-screen export + save queues + flush
        src/gui_plot/plot_tools.cpp
        src/gui_plot/console_plot.cpp       # registry, task hook, render loop, load
)

# GUI 3rd-party roots, SYSTEM to suppress their warnings. PUBLIC so the coast exe,
# coast_console and any GUI test exe inherit them.
target_include_directories(coast_gui SYSTEM PUBLIC
        3rd_party/imgui
        3rd_party/imgui/backends
        3rd_party/implot
        3rd_party/ImNodeFlow/include
        3rd_party/ImNodeFlow/src
        3rd_party/imgui-filebrowser
        ${SDL2_INCLUDE_DIRS}
)

# Same one-root rule as coast_lib: include/-rooted project headers only.
target_include_directories(coast_gui PUBLIC include)

target_compile_options(coast_gui PRIVATE
        -Wall
        -Wpedantic
        -Wextra
        $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-psabi>
)

target_link_libraries(coast_gui PUBLIC
    coast_lib
    SDL2::SDL2
    OpenGL::GL
    implot
    ImNodeFlow
)
target_compile_definitions(coast_gui PUBLIC IMGUI_DEFINE_MATH_OPERATORS)

# SDL2 needs these on macOS for its Cocoa video driver / IOKit device access.
if (APPLE)
    target_link_libraries(coast_gui PUBLIC "-framework IOKit" "-framework Cocoa")
endif()
endif()

# ── Embedded C++ console (clang-repl) ─────────────────────────────────────────
# The LLVM/Clang dependency lives ONLY in the coast_console target (repl engine).
# coast_lib and the small unit tests never link libLLVM; only coast_console, the
# coast exe, and the console test do. Any executable that runs the interpreter
# must be linked with -rdynamic so the JIT resolves the compiled console_api
# facade symbols against the host process.
# Auto-detecting: the console is built where the system LLVM/Clang development
# libraries are found, and skipped (with a warning) where they are not — so the
# default build never fails just because clang-repl is absent. There is no
# FetchContent fallback: building LLVM from source is impractical (hours + GB).
# Install the dev packages to get the console:
#   Debian/Ubuntu : llvm-NN-dev libclang-cppNN-dev libclang-NN-dev (one LLVM version NN;
#                   Clang's CMake files reference libclang-NN-dev's static libraries)
#   Arch          : clang llvm
#   macOS         : brew install llvm  +  -DCMAKE_PREFIX_PATH="$(brew --prefix llvm)"
# Set -DCOAST_WITH_CONSOLE=OFF to skip it (and silence the warning) explicitly.
# _coast_build_console is COAST_WITH_CONSOLE forced OFF in a headless build.
if(_coast_build_console)
    # Probe first, in a child process (cmake/clang_probe/CMakeLists.txt): a
    # half-installed Clang package hard-errors inside find_package otherwise.
    set(_coast_probe_args "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}" "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
    foreach(_v LLVM_DIR Clang_DIR)
        if(${_v})
            list(APPEND _coast_probe_args "-D${_v}=${${_v}}")
        endif()
    endforeach()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR}/cmake/clang_probe
                -B ${CMAKE_BINARY_DIR}/clang_probe -G ${CMAKE_GENERATOR} ${_coast_probe_args}
        RESULT_VARIABLE _coast_probe_rc
        OUTPUT_VARIABLE _coast_probe_out
        ERROR_VARIABLE  _coast_probe_out)
    if(_coast_probe_rc EQUAL 0)
        find_package(LLVM CONFIG)
        # Only look for Clang once LLVM is present — ClangConfig.cmake internally does
        # find_package(LLVM REQUIRED), which would hard-error if LLVM is unavailable.
        if(LLVM_FOUND)
            find_package(Clang CONFIG)
        endif()
    endif()
    if(NOT _coast_probe_rc EQUAL 0 OR NOT LLVM_FOUND OR NOT Clang_FOUND OR NOT TARGET clang-cpp)
        message(WARNING
            "COAST_WITH_CONSOLE is ON but usable LLVM/Clang development libraries "
            "were not found (see ${CMAKE_BINARY_DIR}/clang_probe for the probe's log). "
            "The embedded C++ console will be SKIPPED; the rest of COAST builds "
            "normally. Install the dev packages (see the comment above this line "
            "in CMakeLists.txt) or set -DCOAST_WITH_CONSOLE=OFF to silence this.")
        file(WRITE ${CMAKE_BINARY_DIR}/clang_probe/probe.log "${_coast_probe_out}")
    elseif(LLVM_PACKAGE_VERSION VERSION_LESS 22)
        # repl.cpp uses clang/Interpreter/IncrementalExecutor.h and
        # IncrementalExecutorBuilder, absent before LLVM 22 (measured: 19 and 21 fail).
        message(WARNING
            "COAST_WITH_CONSOLE is ON but LLVM ${LLVM_PACKAGE_VERSION} is older than "
            "the 22 the console needs. The embedded C++ console will be SKIPPED; the rest "
            "of COAST builds normally. Point CMAKE_PREFIX_PATH at LLVM >= 22 or set "
            "-DCOAST_WITH_CONSOLE=OFF to silence this.")
    else()
    message(STATUS "Console: clang-repl via LLVM ${LLVM_PACKAGE_VERSION}")

    # Clang cannot self-locate its builtin headers when embedded (argv[0] is not
    # the clang binary); bake the resource dir in.
    execute_process(
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang -print-resource-dir
        OUTPUT_VARIABLE COAST_CLANG_RESOURCE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    add_library(coast_console STATIC
        include/console/repl.h
        src/console/repl.cpp
        include/console/session.h
        src/console/session.cpp
        include/console/console_panel.h
        src/console/console_panel.cpp)
    # coast_gui, not coast_lib: console_panel IS an ImGui window and the console's
    # plot bridge is gui_plot/ImPlot. (coast_gui carries coast_lib transitively.)
    target_link_libraries(coast_console PUBLIC coast_gui clang-cpp LLVM)
    target_include_directories(coast_console SYSTEM PUBLIC
        ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
    target_compile_definitions(coast_console
        PUBLIC  COAST_WITH_CONSOLE
        PRIVATE CLANG_RESOURCE_DIR=\"${COAST_CLANG_RESOURCE_DIR}\"
                COAST_INCLUDE_DIR=\"${CMAKE_SOURCE_DIR}/include\")

    # macOS: the embedded interpreter cannot self-locate the SDK sysroot (C
    # headers) or libc++ (the <cstdio> C++ wrappers) the way the clang driver
    # does, so JIT-compiled user code fails with "'cstdio' file not found".
    # Inject the paths this build used, mirroring the CLANG_RESOURCE_DIR idiom.
    if (APPLE)
        set(_coast_sysroot "${CMAKE_OSX_SYSROOT}")
        if (NOT _coast_sysroot)
            execute_process(COMMAND xcrun --show-sdk-path
                OUTPUT_VARIABLE _coast_sysroot OUTPUT_STRIP_TRAILING_WHITESPACE)
        endif()
        target_compile_definitions(coast_console PRIVATE
            COAST_MACOS_SYSROOT=\"${_coast_sysroot}\"
            COAST_LIBCXX_INCLUDE_DIR=\"${LLVM_INSTALL_PREFIX}/include/c++/v1\")
    endif()
    endif()
endif()

# Standalone golden-blob comparison tool. Built by default — it's a real
# user-facing tool, not test-only.
add_subdirectory(tools/coast_compare)

# Thin executable: parses CLI, dispatches to coast::app::run_headless or run_gui.
#
# Several translation units (visualization registrars: ngm_flat_vis,
# ngm_tree_vis_v2, vec_ring_buffer) register themselves with vis_registry via
# static-initializer side effects only. When the archive is static, the linker
# drops .o files whose symbols are unreferenced — silently losing those
# registrations. Force-link the whole archive into the coast exe so every TU's
# static initializers run. (Tests do NOT need this and stay slim.)
#
# The registrars all sit in coast_gui now, but coast_lib keeps the same treatment:
# it is what the pre-split build did, and reducing it would be a separate,
# separately-validated change rather than a side effect of the GUI split.
add_executable(coast src/main.cpp)
set(_coast_whole_archive_libs coast_lib)
if(COAST_WITH_GUI)
    list(INSERT _coast_whole_archive_libs 0 coast_gui)
endif()
# One spelling for both linkers: CMake expands WHOLE_ARCHIVE to
# -Wl,--whole-archive/--no-whole-archive for GNU ld and to -Wl,-force_load per
# archive for ld64. The genex takes a COMMA-separated library list, so the
# CMake ;-list has to be joined first — leaving it a ;-list would split into
# separate link items and the genex would not parse.
#
# This replaces a hand-rolled if(APPLE) split whose ld64 branch named each
# archive twice (once in -force_load, once as an ordinary link item), which is
# what produced `ld: warning: ignoring duplicate libraries` on macOS. The genex
# emits each archive exactly once and carries usage requirements normally.
list(JOIN _coast_whole_archive_libs "," _coast_whole_archive_csv)
target_link_libraries(coast PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_coast_whole_archive_csv}>")
# coast_console PUBLIC-links coast_gui (below), so both archives also reach this
# exe transitively as ordinary link items. CMake treats "same library, two
# different link features" as a hard error unless told which one wins; this says
# WHOLE_ARCHIVE does. Harmless when the console is absent.
set_property(TARGET coast PROPERTY
    LINK_LIBRARY_OVERRIDE "WHOLE_ARCHIVE,${_coast_whole_archive_csv}")

# The console links after (outside) the whole-archive wrap. -rdynamic exports the
# host's dynamic symbols so the embedded JIT resolves the compiled console_api
# facade. COAST_WITH_CONSOLE propagates to main.cpp via coast_console's PUBLIC
# compile definition. Gated on the TARGET existing (not just the option) so a
# missing-LLVM build still links cleanly without the console.
if(TARGET coast_console)
    target_link_libraries(coast PRIVATE coast_console)
    target_link_options(coast PRIVATE -rdynamic)
endif()

# GI1 ngm2.5 GPU isolation probe. Uses only the agnostic env/entity API (no
# webgpu-private headers), so it builds for BOTH the webgpu and null backends —
# the null build exercises the register->build->step->swap->checkpoint plumbing
# with zero GPU, the webgpu build adds the real-adapter forward-path value checks.
if(NOT COAST_GPU_BACKEND STREQUAL "none")
    add_executable(coast_ngm25_gpu_probe src/hd_ngm25/hd_ngm25_gpu_probe_main.cpp)
    target_link_libraries(coast_ngm25_gpu_probe PRIVATE coast_lib)
endif()

# coastd — resident per-machine headless-job daemon. Its own target tree; added
# here (after coast_lib) so it can reuse it. See coastd/CMakeLists.txt.
add_subdirectory(coastd)

# ---- GUI assets: build-time font copy --------------------------------------
# The GUI font is a tracked asset (assets/DejaVuSans.ttf), not a submodule. Copy
# it into the build tree at build time so the running app finds it by an ABSOLUTE
# path regardless of the process CWD (the old "./3rd_party/DejaVuSans.ttf" only
# worked when launched from the repo root). This is new machinery — there is no
# other file(COPY)/add_custom_command asset copy in the project.
#
# COAST_FONT_PATH is the compile-time default the GUI consumes (overridable at
# runtime via the app config's assets.font). Both the compile-def and the copy are
# tied to coast_gui — where sdl_app.cpp (which loads the font) compiles — so EVERY
# GUI consumer (the coast exe today, a GUI smoke test later) inherits the
# build-tree font, not just the coast executable. A headless build neither copies
# the font nor defines the macro: nothing in coast_lib reads either.
if(COAST_WITH_GUI)
    set(COAST_FONT_SRC  "${CMAKE_SOURCE_DIR}/assets/DejaVuSans.ttf")
    set(COAST_FONT_DEST "${CMAKE_BINARY_DIR}/assets/DejaVuSans.ttf")
    add_custom_command(
        OUTPUT  "${COAST_FONT_DEST}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${COAST_FONT_SRC}" "${COAST_FONT_DEST}"
        DEPENDS "${COAST_FONT_SRC}"
        COMMENT "Copying GUI font to build tree")
    add_custom_target(coast_assets DEPENDS "${COAST_FONT_DEST}")
    add_dependencies(coast_gui coast_assets)
    target_compile_definitions(coast_gui PRIVATE COAST_FONT_PATH="${COAST_FONT_DEST}")
endif()

# Minimal wgpu bootstrap smoke test (S1 / checkpoint-alpha A-II1 evidence).
# Links coast_lib normally (no whole-archive) -> pulls gpu_context. The drivers
# that include the WebGPU-private headers (gpu_context.h, <webgpu/webgpu.h>)
# also link coast_webgpu_backend_private, which carries that header dir and the
# SDK; coast_lib deliberately does not. They build ONLY for the webgpu backend.
if(COAST_GPU_BACKEND STREQUAL "webgpu")
    add_executable(coast_gpu_smoke src/core/gpu/smoke_main.cpp)
    target_link_libraries(coast_gpu_smoke PRIVATE coast_lib coast_webgpu_backend_private)

    # Framework spike measurements (unified-memory write path + batched-submission
    # host cost) feeding checkpoint alpha.
    add_executable(coast_gpu_spike src/core/gpu/spike_main.cpp)
    target_link_libraries(coast_gpu_spike PRIVATE coast_lib coast_webgpu_backend_private)

    # S2 WGSL toolchain end-to-end check: add-1 kernel + shared_prng concat +
    # bad-WGSL error-fidelity path. Run unsandboxed (needs /dev/dri).
    add_executable(coast_gpu_min_kernel src/core/gpu/min_kernel_main.cpp)
    target_link_libraries(coast_gpu_min_kernel PRIVATE coast_lib coast_webgpu_backend_private)

    # S4 two-phase encoder dispatcher isolation probe: drives env::process() /
    # env::swap_io() on a hand-built gpu_io_entity + CPU entities, verifying the
    # GPU dispatch round-trips and does not perturb CPU work. Run unsandboxed.
    add_executable(coast_dispatcher_probe src/sim_core/dispatcher_probe_main.cpp)
    target_link_libraries(coast_dispatcher_probe PRIVATE coast_lib)

    # S5 mnist_io_gpu isolation probe: hand-builds an env with mnist_io_gpu + a
    # CPU sink, drives N steps, asserts per-pixel correctness (read-by-index +
    # bounded noise + clamp) and the host schedule. Run unsandboxed
    # from the repo root (loads ./data/mnist).
    add_executable(coast_mnist_io_gpu_probe src/io/mnist_io_gpu_probe_main.cpp)
    target_link_libraries(coast_mnist_io_gpu_probe PRIVATE coast_lib)

    # S5 spike-validate: dataset-upload bandwidth, phase-2 round-trip floor (Cost
    # A), probe-readback latency vs size (Cost B), and live per-step cost. Feeds
    # the checkpoint-beta probe-shape decision. Run unsandboxed from the repo root.
    add_executable(coast_mnist_io_gpu_spike src/io/mnist_io_gpu_spike_main.cpp)
    target_link_libraries(coast_mnist_io_gpu_spike PRIVATE coast_lib coast_webgpu_backend_private)

    # S6 ngm3_gpu isolation probe: hand-builds an env with mnist_io_gpu -> ngm3_gpu
    # -> CPU sink (the sink forces io_buffer_gpu_to_cpu on the ngm3 output so it is
    # CPU-visible), drives N steps, asserts the post-inhibition group_outp is in
    # [0,1] / finite / non-degenerate (the S6 sanity bar). Run unsandboxed from the
    # repo root (loads ./data/mnist).
    add_executable(coast_ngm3_gpu_probe src/hd_ngm3/hd_ngm3_gpu_probe_main.cpp)
    target_link_libraries(coast_ngm3_gpu_probe PRIVATE coast_lib)

    # GI2 ngm2.5 GPU validation: host/WGSL syn_key parity, CPU<->GPU forward
    # agreement on a deterministic config, and GPU run-to-run determinism. Real
    # compute (webgpu only). Run unsandboxed.
    add_executable(coast_ngm25_gpu_validate src/hd_ngm25/hd_ngm25_gpu_validate_main.cpp)
    target_link_libraries(coast_ngm25_gpu_validate PRIVATE coast_lib)

    # GI0 ngm2.5 GPU premise-check micro-benchmark (design/ngm2_5_gpu.md §2):
    # times a realized-sparse vs a dense (ngm3 rep-inflated) response sweep through
    # the real gpu_device facade to measure whether the sparse-arena traffic win
    # materializes on this adapter BEFORE the production kernels are written.
    # Throwaway; inlines its WGSL (no #embed). Run unsandboxed.
    add_executable(coast_ngm25_gpu_premise src/hd_ngm25/hd_ngm25_gpu_premise_main.cpp)
    target_link_libraries(coast_ngm25_gpu_premise PRIVATE coast_lib)
endif()

# Test tree (opt-in via -DBUILD_TESTING=ON).
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)

    # ngm2.5 GPU self-checking binaries as ctest targets (they exit non-zero on
    # any failed gate). They need a real adapter, so they only make sense under the
    # webgpu backend; the `_gpu` in the test name and the "gpu" label let
    # sandboxed / GPU-less CI runs skip them via `ctest -E '_gpu'` or `-LE gpu`,
    # matching the integration._gpu convention.
    if(COAST_GPU_BACKEND STREQUAL "webgpu")
        add_test(NAME ngm25_gpu_validate COMMAND coast_ngm25_gpu_validate
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        add_test(NAME ngm25_gpu_probe COMMAND coast_ngm25_gpu_probe
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        set_tests_properties(ngm25_gpu_validate ngm25_gpu_probe
                             PROPERTIES LABELS "gpu")
    endif()
endif()

# ----------------------------------------------------------------------------
# coast_tidy — on-demand clang-tidy run over first-party sources.
#
# Uses compile_commands.json (already exported via
# CMAKE_EXPORT_COMPILE_COMMANDS) so each TU is checked with its real flags.
# The .clang-tidy config at the repo root determines the rule set.
#
# Invoke:   cmake --build build --target coast_tidy
# Reports; does not gate the build.
# ----------------------------------------------------------------------------
find_program(COAST_CLANG_TIDY clang-tidy)
find_program(COAST_RUN_CLANG_TIDY run-clang-tidy)
if(COAST_RUN_CLANG_TIDY AND COAST_CLANG_TIDY)
    add_custom_target(coast_tidy
        COMMAND ${COAST_RUN_CLANG_TIDY}
                -p ${CMAKE_BINARY_DIR}
                -clang-tidy-binary ${COAST_CLANG_TIDY}
                -quiet
                "^${CMAKE_SOURCE_DIR}/(src|tools|tests)/.*"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
        VERBATIM
        COMMENT "Running clang-tidy over first-party sources"
    )
endif()
