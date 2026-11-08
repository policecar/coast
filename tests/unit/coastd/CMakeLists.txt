# Load-independent wait helpers (header-only, std-only): tests/support/await_progress.h.
include_directories(${CMAKE_SOURCE_DIR}/tests/support)

# coastd unit tests — the pure spine (job JSON, store, reconcile). These link
# coastd_core only; no subprocesses, no daemon, no sockets.
add_executable(test_coastd_reconcile test_reconcile.cpp)
target_link_libraries(test_coastd_reconcile PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.reconcile COMMAND test_coastd_reconcile)

# Log housekeeping (compress-then-delete by age, plus the append-only size cap).
# Pure filesystem behaviour, no daemon: it runs on a worker thread and touches no
# daemon state, so it is testable in complete isolation.
# The wire-edge anti-crash guard (json_depth_ok / parse_line) and the reply
# envelope. Header-only protocol, so this links nothing but doctest.
add_executable(test_coastd_protocol_guard test_protocol_guard.cpp)
target_link_libraries(test_coastd_protocol_guard PRIVATE doctest::doctest coastd_proto)
add_test(NAME unit.coastd.protocol_guard COMMAND test_coastd_protocol_guard)

add_executable(test_coastd_log_sweep test_log_sweep.cpp)
target_link_libraries(test_coastd_log_sweep PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.log_sweep COMMAND test_coastd_log_sweep)

add_executable(test_coastd_job_json test_job_json.cpp)
target_link_libraries(test_coastd_job_json PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.job_json COMMAND test_coastd_job_json)

# coastd.json loader: the four directory-root keys, the <data-root> token, the
# relaunch-backoff keys, and the tolerant type/range/empty handling.
add_executable(test_coastd_config_load test_config_load.cpp)
target_link_libraries(test_coastd_config_load PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.config_load COMMAND test_coastd_config_load)

# atomic_write: the durability primitive under every job-store write. Covers the
# whole-file replace, the no-leftover-tmp guarantee on both the success and the
# throw path, and the concurrent-writers-to-one-path case that a fixed tmp name
# could not satisfy.
add_executable(test_coastd_atomic_write test_atomic_write.cpp)
target_link_libraries(test_coastd_atomic_write PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.atomic_write COMMAND test_coastd_atomic_write)

add_executable(test_coastd_job_store test_job_store.cpp)
target_link_libraries(test_coastd_job_store PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.job_store COMMAND test_coastd_job_store)

# machine_info: the daemon self-description, incl. data_link_ok (does a config's
# relative "data/…" path resolve here?) — no daemon, no sockets.
add_executable(test_coastd_machine_info test_machine_info.cpp)
target_link_libraries(test_coastd_machine_info PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.machine_info COMMAND test_coastd_machine_info)

# sha256 (FIPS 180-2 vectors, padding boundaries, split invariance) and
# hash_inputs: the dataset_base containment rules and caps behind the
# hash_inputs command — pure filesystem, no daemon, no sockets.
add_executable(test_coastd_sha256 test_sha256.cpp)
target_link_libraries(test_coastd_sha256 PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.sha256 COMMAND test_coastd_sha256)

add_executable(test_coastd_hash_inputs test_hash_inputs.cpp)
target_link_libraries(test_coastd_hash_inputs PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.hash_inputs COMMAND test_coastd_hash_inputs)

# The UDP discovery responder — the daemon's only unauthenticated channel, and the
# only one that can reach a host that never contacted it (forged UDP source). Binds
# a real socket on an OS-assigned port (udp_port = 0, read back via port()).
add_executable(test_coastd_discovery test_discovery.cpp)
target_link_libraries(test_coastd_discovery PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.discovery COMMAND test_coastd_discovery)
# The suite's failure mode is a HANG, not a wrong answer: the listen thread's only
# shutdown wake is a receive timeout, and if that stops taking effect stop()'s join
# blocks forever. Nothing names that hang any more (the idle-stop case was removed:
# it was not load-independent), so this TIMEOUT is its only backstop — 1.2 s
# measured under 2x nproc busy loops.
set_tests_properties(unit.coastd.discovery PROPERTIES TIMEOUT 300)

# rescan's disk probe: the free-space measurement behind reality.disk_pressure,
# incl. the "cannot measure" and "check disabled" arms.
add_executable(test_coastd_reality_probe test_reality_probe.cpp)
target_link_libraries(test_coastd_reality_probe PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.reality_probe COMMAND test_coastd_reality_probe)

# M2 primitives.
add_executable(test_coastd_line_tailer test_line_tailer.cpp)
target_link_libraries(test_coastd_line_tailer PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.line_tailer COMMAND test_coastd_line_tailer)

# plan_launch command derivation (resume budget, the -n 0 bounded-job guard).
add_executable(test_coastd_child_cmd test_child_cmd.cpp)
target_link_libraries(test_coastd_child_cmd PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.child_cmd COMMAND test_coastd_child_cmd)

add_executable(test_coastd_concurrency test_concurrency.cpp)
target_link_libraries(test_coastd_concurrency PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.concurrency COMMAND test_coastd_concurrency)

add_executable(test_coastd_child_proc test_child_proc.cpp)
target_link_libraries(test_coastd_child_proc PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.child_proc COMMAND test_coastd_child_proc)

# run_capture wall-clock deadline: a hung git/cmake must be killed, not block the
# worker thread forever (spawns real short-lived subprocesses).
add_executable(test_coastd_proc_util test_proc_util.cpp)
target_link_libraries(test_coastd_proc_util PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.proc_util COMMAND test_coastd_proc_util)

# supervisor::shutdown SIGTERM-grace-then-SIGKILL escalation (real child procs
# that obey vs. ignore SIGTERM; injected short term_grace).
add_executable(test_coastd_supervisor test_supervisor.cpp)
target_link_libraries(test_coastd_supervisor PRIVATE doctest::doctest coastd_core)
add_test(NAME unit.coastd.supervisor COMMAND test_coastd_supervisor)

add_executable(test_coastd_builder test_builder.cpp)
target_link_libraries(test_coastd_builder PRIVATE doctest::doctest coastd_core)
target_compile_definitions(test_coastd_builder PRIVATE COAST_TEST_REPO="${CMAKE_SOURCE_DIR}")
add_test(NAME unit.coastd.builder COMMAND test_coastd_builder)

# In-process lifecycle test: submit -> resolve -> (pre-placed worktree) -> spawn a
# fake coast -> child_exited -> done. Exercises the whole M2 machine without a real build.
add_executable(test_coastd_core_lifecycle test_core_lifecycle.cpp)
target_link_libraries(test_coastd_core_lifecycle PRIVATE doctest::doctest coastd_core)
target_compile_definitions(test_coastd_core_lifecycle PRIVATE COAST_TEST_REPO="${CMAKE_SOURCE_DIR}")
add_test(NAME unit.coastd.core_lifecycle COMMAND test_coastd_core_lifecycle)

# The real-coast acceptance gates, registered as their OWN ctest test so their
# outcome is visible. The two cases carry doctest::skip(), so the unit test above
# does not run them; this test does, via the driver, and reports one of three
# distinguishable outcomes: passed, FAILED, or Skipped (exit 77 — no dataset, no
# h5diff, no coast binary). What it can never do is contribute a silent green tick
# it did not earn, which is what the old "return early if the env is unset" did.
add_test(NAME gate.coastd.real_coast
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_real_gate.sh
                 ${CMAKE_SOURCE_DIR}
                 $<TARGET_FILE:coast>
                 $<TARGET_FILE:test_coastd_core_lifecycle>)
set_tests_properties(gate.coastd.real_coast PROPERTIES
                     SKIP_RETURN_CODE 77
                     # A real mnist run + a reference run: measured 65 s with 2x
                     # nproc busy loops and 93-119 s with 4x; 900 s left too little
                     # headroom under heavy load, so ctest's default 1500 s.
                     TIMEOUT 1500)
# $<TARGET_FILE:...> inside a test COMMAND is a path, not a build edge — same trap
# documented for unit.coastd.comms below. Without these the gate can run a STALE
# coast or test binary and certify a build that was never made.
add_dependencies(test_coastd_core_lifecycle coast)

# on_child_exited event-ordering folds: the terminal-bucket guard (M2) and the
# cancel guard (L6), driven synchronously via the core_test_access friend seam.
add_executable(test_coastd_child_exit test_child_exit.cpp)
target_link_libraries(test_coastd_child_exit PRIVATE doctest::doctest coastd_core)
target_compile_definitions(test_coastd_child_exit PRIVATE COAST_TEST_REPO="${CMAKE_SOURCE_DIR}")
add_test(NAME unit.coastd.child_exit COMMAND test_coastd_child_exit)

# M3 recovery: boot-time pid duplicate-guard (real orphan process) + crash-loop cap
# (a fake coast that dies with no progress). Both drive a real in-process core.
add_executable(test_coastd_recovery test_recovery.cpp)
target_link_libraries(test_coastd_recovery PRIVATE doctest::doctest coastd_core)
target_compile_definitions(test_coastd_recovery PRIVATE COAST_TEST_REPO="${CMAKE_SOURCE_DIR}")
add_test(NAME unit.coastd.recovery COMMAND test_coastd_recovery)

# ---------------------------------------------------------------------------
# Tests that launch a REAL coastd process. Guarded on the daemon executable
# existing: with -DCOAST_WITH_COASTD_DAEMON=OFF there is no target to point
# COASTD_BIN at, and $<TARGET_FILE:> on a missing target is a CONFIGURE error, not
# a skip. Dropping them from the suite is the honest outcome — the thing they test
# was not built. (Off Linux the daemon is built but refuses to run; that case is
# handled separately, by doctest::skip inside the test files.)
# ---------------------------------------------------------------------------
if(TARGET coastd)

# M4 comms B-tier gate: launches the REAL coastd binary and drives the wire
# protocol through the thin coastd_client (NOT coastd_core) — proving the client
# links WITHOUT the simulator. Subprocess + sockets, like recovery's real procs.
add_executable(test_coastd_comms test_comms.cpp)
target_link_libraries(test_coastd_comms PRIVATE doctest::doctest coastd_client)
target_compile_definitions(test_coastd_comms PRIVATE COASTD_BIN="$<TARGET_FILE:coastd>")
add_test(NAME unit.coastd.comms COMMAND test_coastd_comms)
# COASTD_BIN is a PATH, not a build edge: without this the test happily runs
# against a stale coastd from a previous build and "proves" whatever that binary
# does. This bit us for real — a mutation-test left a modified daemon in the build
# dir, and the reverted source then looked broken.
add_dependencies(test_coastd_comms coastd)

# M3 fault matrix: {kill client, kill daemon} x {building, queued, running}
# against the REAL coastd binary, real child processes and a real cmake build, plus
# the behaviour behind the latency criteria. The fake build and child block on
# release files the test creates (fault_rig.h), so no state is held by a sleep.
# Its own ctest entry — exclude it with -E fault for a quick loop.
add_executable(test_coastd_fault_matrix test_fault_matrix.cpp)
target_link_libraries(test_coastd_fault_matrix PRIVATE doctest::doctest coastd_client)
target_compile_definitions(test_coastd_fault_matrix PRIVATE COASTD_BIN="$<TARGET_FILE:coastd>")
add_test(NAME fault.coastd.matrix COMMAND test_coastd_fault_matrix)
add_dependencies(test_coastd_fault_matrix coastd)   # see the note on unit.coastd.comms
set_tests_properties(fault.coastd.matrix PROPERTIES TIMEOUT 900)

# The latency numbers behind the matrix's criteria (manual benchmark, not a ctest:
# a latency bound would make a verdict depend on the machine's load).
add_executable(bench_coastd_latency bench_coastd_latency.cpp)
target_link_libraries(bench_coastd_latency PRIVATE coastd_client)
target_compile_definitions(bench_coastd_latency PRIVATE COASTD_BIN="$<TARGET_FILE:coastd>")
add_dependencies(bench_coastd_latency coastd)

# Console façade (coast::coastd::session) — lives in coast_lib, so this links
# coast_lib and launches the real coastd. Validates the façade's DTO mapping and
# throw-on-error contract (the clang-repl JIT path needs an LLVM build; not here).
add_executable(test_coastd_console test_console.cpp)
target_link_libraries(test_coastd_console PRIVATE doctest::doctest coast_lib)
target_compile_definitions(test_coastd_console PRIVATE COASTD_BIN="$<TARGET_FILE:coastd>")
add_test(NAME unit.coastd.console COMMAND test_coastd_console)
add_dependencies(test_coastd_console coastd)        # see the note on unit.coastd.comms

else()
    message(STATUS "coastd tests: skipping the real-daemon tests "
                   "(COAST_WITH_COASTD_DAEMON=OFF)")
endif()

# ---------------------------------------------------------------------------
# Skipped-off-Linux entries must report SKIPPED, not PASSED.
#
# doctest exits 0 and prints "Status: SUCCESS!" when every case in a binary is
# skip()-decorated, so ctest — which only sees the exit code — renders it as
# Passed. Off Linux that is an UNEARNED GREEN: three of these binaries have all
# their cases gated, so they would report passing having executed nothing, and a
# macOS suite total would be indistinguishable from one that really ran them.
#
# This is the same hazard run_real_gate.sh already defends against (it asserts an
# exact matched-case count first, because doctest reports SUCCESS on 0 matches),
# and the same outcome gate.coastd.real_coast gets from SKIP_RETURN_CODE 77.
# SKIP_REGULAR_EXPRESSION is the equivalent for a doctest binary we do not own the
# main() of: it matches doctest's summary line ONLY when zero cases ran, so a
# partially-gated binary (child_proc: 1 of 6, comms: 1 of 3) is unaffected, and on
# Linux — where nothing is skipped — it never matches at all.
set_tests_properties(
    unit.coastd.child_proc
    unit.coastd.recovery
    PROPERTIES SKIP_REGULAR_EXPRESSION "test cases: +0 ")
if(TARGET coastd)
    set_tests_properties(
        unit.coastd.comms
        unit.coastd.console
        fault.coastd.matrix
        PROPERTIES SKIP_REGULAR_EXPRESSION "test cases: +0 ")
endif()

# ---------------------------------------------------------------------------
# Serialization. No coastd test depends on timing any more (CLAUDE.md "Testing":
# states are held by construction, waits are on the daemon's heartbeat), so none
# needs RUN_SERIAL to keep it from flapping — and RUN_SERIAL used for that would
# hide the defect, not fix it. The gate keeps it for a different reason: it runs
# two real, TBB-parallel mnist simulations that use every core, and running them
# next to other tests would only slow everything down. (Port collisions were never
# the issue: every real daemon binds OS-assigned ports, test_daemon.h.)
set_tests_properties(gate.coastd.real_coast PROPERTIES RUN_SERIAL TRUE)   # CPU-heavy, not timing
