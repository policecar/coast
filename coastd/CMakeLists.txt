# -----------------------------------------------------------------------------
# coastd — resident per-machine headless-job daemon.
#
# Its own target tree (this directory), added from the root via
# add_subdirectory(coastd). coastd orchestrates subprocess `coast` binaries
# (build, schedule, supervise, auto-resume) and NEVER re-embeds the runner. It
# reuses coast_lib for the header-only diag facility, nlohmann/json, the
# diag/progress.h contract, and coast::paths output-location resolution (so
# coastd computes a child's checkpoint directory with the SAME resolver the
# child writes through).
#
# `coastd/include` is coastd's own header root — includes are written
# daemon-rooted (`#include "daemon/types.h"`), parallel to coast_lib's `include`
# root. The inner dir is `daemon/` (not `coastd/`) to avoid confusion with
# coast_lib's own `include/core/`, which coastd consumes (core/app_config.h,
# core/path_resolver.h). coast_lib's headers (diag/, cli/, nlohmann json) come in
# transitively via the PUBLIC link below; coast's `include/` tree carries NO
# coastd headers.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# coastd_proto — the coast_lib-FREE wire contract (header-only).
#
# The single source of truth for the client<->daemon JSONL protocol: command
# tokens, the request/reply envelope, and the job_summary/machine_summary DTOs.
# Carries the include dirs a bare protocol speaker needs — coastd's own header
# root (for daemon/protocol.h) and nlohmann/json — and NOTHING else. Both the
# server (coastd_core) and the client (coastd_client) link it, so the two ends
# can never drift out of a shared, compiler-checked contract. Deliberately does
# NOT depend on coast_lib: a frontend that links coastd_client pays for sockets
# + json, not the whole simulator.
# -----------------------------------------------------------------------------
add_library(coastd_proto INTERFACE)
target_include_directories(coastd_proto INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(coastd_proto SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/3rd_party/json/include)

add_library(coastd_core STATIC
    include/daemon/types.h
    include/daemon/version.h
    include/daemon/protocol.h
    include/daemon/sock_compat.h
    include/daemon/config_load.h
    src/config_load.cpp
    include/daemon/atomic_write.h
    src/atomic_write.cpp
    include/daemon/reconcile.h
    src/reconcile.cpp
    include/daemon/log_sweep.h
    src/log_sweep.cpp
    include/daemon/job_json.h
    src/job_json.cpp
    include/daemon/job_store.h
    src/job_store.cpp

    # M2: core loop, workers, supervision.
    include/daemon/events.h
    include/daemon/event_queue.h
    include/daemon/worker_pool.h
    src/worker_pool.cpp
    include/daemon/line_tailer.h
    include/daemon/child_proc.h
    src/child_proc.cpp
    include/daemon/proc_util.h
    src/proc_util.cpp
    include/daemon/builder.h
    src/builder.cpp
    include/daemon/child_cmd.h
    src/child_cmd.cpp
    include/daemon/supervisor.h
    src/supervisor.cpp
    include/daemon/reality_probe.h
    src/reality_probe.cpp
    include/daemon/core.h
    src/core.cpp

    # M4: comms edges (server side). Connection/discovery threads reach the core
    # only by pushing events — they add no new state and no second writer.
    include/daemon/machine_info.h
    src/machine_info.cpp
    include/daemon/sha256.h
    src/sha256.cpp
    include/daemon/hash_inputs.h
    src/hash_inputs.cpp
    include/daemon/comms_server.h
    src/comms_server.cpp
    include/daemon/discovery_server.h
    src/discovery_server.cpp
)

# PUBLIC so consumers (the coastd exe, the unit tests) resolve `daemon/…` headers.
target_include_directories(coastd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Linked NORMALLY (not --whole-archive): coastd references no static registrars,
# so the linker pulls only the coast_lib objects it uses — the GUI/GPU TUs stay out.
# coastd_proto gives the server the same wire contract the client speaks.
target_link_libraries(coastd_core PUBLIC coast_lib coastd_proto)

target_compile_options(coastd_core PRIVATE
    -Wall
    -Wpedantic
    -Wextra
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-psabi>
)

# -----------------------------------------------------------------------------
# coastd_client — the THIN protocol client library (connect / JSONL round-trip /
# UDP discovery). Frontends (REPL console, GUI) link THIS, never coastd_core:
# it depends only on coastd_proto (header contract + json), so linking it does
# NOT pull in coast_lib, the simulator, or the daemon's state machine.
# -----------------------------------------------------------------------------
add_library(coastd_client STATIC
    include/client/coastd_client.h
    include/daemon/sock_compat.h   # shared with the server: one SIGPIPE contract, both ends
    src/client/coastd_client.cpp
)
target_include_directories(coastd_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(coastd_client PUBLIC coastd_proto)
target_compile_options(coastd_client PRIVATE
    -Wall
    -Wpedantic
    -Wextra
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-psabi>
)

# The daemon EXECUTABLE, and only it, is optional. The libraries above stay
# unconditional deliberately: coastd_proto/coastd_client are cross-platform and are
# what the console façade and the GUI panel link, so gating them would drag
# src/coastd_console/ and gui_tools/coastd_panel.cpp out of coast_lib/coast_gui too —
# a much larger cut than the problem justifies.
#
# What the problem IS: this executable refuses to RUN wherever
# start_token_supported() is false (src/main.cpp — without a recycle-safe start token
# the duplicate-run guard is inoperative, ADR-16; Linux and macOS have one, ADR-17).
# This switch stops paying build time for a binary that cannot run, or is not wanted.
#
# Default ON everywhere, including unsupported platforms: flipping it OFF by platform
# would be a silent behaviour change for anyone who compiles it there deliberately
# (the compile still works and still catches portability breakage — only running is
# refused). Opt out explicitly with -DCOAST_WITH_COASTD_DAEMON=OFF.
#
# The tests that launch a real daemon key off `if(TARGET coastd)`, so turning this off
# drops them from the suite rather than failing the configure.
option(COAST_WITH_COASTD_DAEMON "Build the coastd daemon executable (runs on Linux and macOS)" ON)
if(COAST_WITH_COASTD_DAEMON)
    add_executable(coastd src/main.cpp)
    target_link_libraries(coastd PRIVATE coastd_core)
else()
    message(STATUS "coastd: daemon executable disabled (COAST_WITH_COASTD_DAEMON=OFF); "
                   "the client/console libraries are still built")
endif()
