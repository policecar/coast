# tests/integration_ngm3/CMakeLists.txt — smoke test(s) for the ngm3 module.
#
# Lighter than the A-tier mnist integration tests: no golden machinery,
# just structural assertions after a fixed number of steps. Builds and
# runs sub-second.

set(_target integration_ngm3_synthetic_smoke)
add_executable(${_target} synthetic_smoke_test.cpp)
target_link_libraries(${_target} PRIVATE
    doctest::doctest
    coast_lib
    coast_test_support
    coast_compare_lib
    coast_integration_helpers
)
target_compile_features(${_target} PRIVATE cxx_std_23)
target_compile_definitions(${_target} PRIVATE
    COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/test_ngm3_synthetic.json"
)

add_test(NAME integration.ngm3_synthetic_smoke
         COMMAND ${_target}
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Branch-timing test: pins the schedule's lower bound (no branch before
# branch_interval), which the end-state smoke test cannot observe.
set(_bt_target integration_ngm3_branch_timing)
add_executable(${_bt_target} branch_timing_test.cpp)
target_link_libraries(${_bt_target} PRIVATE
    doctest::doctest
    coast_lib
    coast_test_support
    coast_compare_lib
    coast_integration_helpers
)
target_compile_features(${_bt_target} PRIVATE cxx_std_23)
target_compile_definitions(${_bt_target} PRIVATE
    COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/test_ngm3_synthetic.json"
)

add_test(NAME integration.ngm3_branch_timing
         COMMAND ${_bt_target}
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Read-page restore regression (design §5.3), CPU config. Deterministic and
# RNG-immune: asserts a fresh, never-stepped env's io_buffer read pages equal the
# saved values after a load (exercises io_buffer_cpu::restore). The full
# step-once reproducibility variant is GPU-only (below) — CPU entities carry
# stateful RNGs that save_state does not persist, which would confound a CPU
# step-once. Runs from the repo root for ./data/mnist (the synthetic config drives
# an mnist_input source).
set(_restore_cpu integration_ngm3_state_restore_readpage_cpu)
add_executable(${_restore_cpu} state_restore_readpage_test.cpp)
target_link_libraries(${_restore_cpu} PRIVATE
    doctest::doctest
    coast_lib
    coast_test_support
    coast_compare_lib
    coast_integration_helpers
)
target_compile_features(${_restore_cpu} PRIVATE cxx_std_23)
target_compile_definitions(${_restore_cpu} PRIVATE
    COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/test_ngm3_synthetic.json"
)

add_test(NAME integration.ngm3_state_restore_readpage_cpu
         COMMAND ${_restore_cpu}
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Split-path round-trip (Step 3 gate), CPU config. Exercises the new
# extract_state -> serialize_state -> parse_state -> apply_state pipeline and
# asserts the reproduced io_buffers + entities subtrees match bit-for-bit (no
# stepping → RNG-immune; valid on CPU). The GPU config of the same source is
# registered below. Runs from the repo root for ./data/mnist.
set(_roundtrip_cpu integration_ngm3_state_roundtrip_split_cpu)
add_executable(${_roundtrip_cpu} state_roundtrip_split_test.cpp)
target_link_libraries(${_roundtrip_cpu} PRIVATE
    doctest::doctest
    coast_lib
    coast_test_support
    coast_compare_lib
    coast_integration_helpers
)
target_compile_features(${_roundtrip_cpu} PRIVATE cxx_std_23)
target_compile_definitions(${_roundtrip_cpu} PRIVATE
    COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/test_ngm3_synthetic.json"
)

add_test(NAME integration.ngm3_state_roundtrip_split_cpu
         COMMAND ${_roundtrip_cpu}
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Runtime state-IO glue (Step 4 path), CPU config. End-to-end exercise of
# logging_manager::extract_state + logging_manager::load_state, which run the
# extract/apply + step-counter update INSIDE ctrl.run_quiesced() on the sim
# thread — the production runtime path the GUI and headless callers hit. The
# component layers are covered elsewhere (run_quiesced by the thread-controller
# unit test; buffer restore by state_restore_readpage); this closes the glue gap
# between them. Deterministic, no-step → RNG-immune (valid on CPU). Runs from the
# repo root for ./data/mnist.
set(_load_runtime_cpu integration_ngm3_state_load_runtime_cpu)
add_executable(${_load_runtime_cpu} state_load_runtime_test.cpp)
target_link_libraries(${_load_runtime_cpu} PRIVATE
    doctest::doctest
    coast_lib
    coast_test_support
    coast_compare_lib
    coast_integration_helpers
)
target_compile_features(${_load_runtime_cpu} PRIVATE cxx_std_23)
target_compile_definitions(${_load_runtime_cpu} PRIVATE
    COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/test_ngm3_synthetic.json"
)

add_test(NAME integration.ngm3_state_load_runtime_cpu
         COMMAND ${_load_runtime_cpu}
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Deterministic-resume regression, CPU config. Steps AFTER a load and asserts a
# straight-through N+M run equals checkpoint-at-N + resume + M-steps bit-for-bit.
# Unlike the no-step roundtrip tests, this is the one that fails if any per-step
# carried state (PRNG streams, decaying running maxes, branch counter, dendrite
# thresholds, dataset cursors) is not persisted. Runs from the repo root for
# ./data/mnist (the synthetic config drives an mnist_input source).
set(_resume_step_cpu integration_ngm3_state_resume_stepafter_cpu)
add_executable(${_resume_step_cpu} state_resume_stepafter_test.cpp)
target_link_libraries(${_resume_step_cpu} PRIVATE
    doctest::doctest
    coast_lib
    coast_test_support
    coast_compare_lib
    coast_integration_helpers
)
target_compile_features(${_resume_step_cpu} PRIVATE cxx_std_23)
target_compile_definitions(${_resume_step_cpu} PRIVATE
    COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/test_ngm3_synthetic.json"
)

add_test(NAME integration.ngm3_state_resume_stepafter_cpu
         COMMAND ${_resume_step_cpu}
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# GPU-only state save/load round-trip (B1 regression). Needs a real device and
# the full-GPU ngm3 pipeline (and the forced-rebuild test includes a webgpu-
# private header), so it is gated on the webgpu backend and labeled "gpu"
# (excluded by `ctest -LE gpu`). Runs from the repo root for ./data/mnist.
if(COAST_GPU_BACKEND STREQUAL "webgpu")
    set(_rt_target integration_ngm3_gpu_state_roundtrip)
    add_executable(${_rt_target} gpu_state_roundtrip_test.cpp)
    target_link_libraries(${_rt_target} PRIVATE
        doctest::doctest
        coast_lib
        coast_test_support
        coast_compare_lib
        coast_integration_helpers
    )
    target_compile_features(${_rt_target} PRIVATE cxx_std_23)
    target_compile_definitions(${_rt_target} PRIVATE
        COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist_ngm3_gpu_full.json"
    )

    add_test(NAME integration.ngm3_gpu_state_roundtrip
             COMMAND ${_rt_target}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(integration.ngm3_gpu_state_roundtrip
                         PROPERTIES LABELS "gpu" TIMEOUT 180)

    # GPU config of the step-once reproducibility test (same source as the CPU
    # registration above, full-GPU pipeline). Exercises the GPU read-page restore
    # (cpu_to_gpu upload + io_buffer_gpu upload) being consumed after a load.
    set(_resume_gpu integration_ngm3_state_resume_repro_gpu)
    add_executable(${_resume_gpu} state_resume_repro_test.cpp)
    target_link_libraries(${_resume_gpu} PRIVATE
        doctest::doctest
        coast_lib
        coast_test_support
        coast_compare_lib
        coast_integration_helpers
    )
    target_compile_features(${_resume_gpu} PRIVATE cxx_std_23)
    target_compile_definitions(${_resume_gpu} PRIVATE
        COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist_ngm3_gpu_full.json"
    )

    add_test(NAME integration.ngm3_state_resume_repro_gpu
             COMMAND ${_resume_gpu}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(integration.ngm3_state_resume_repro_gpu
                         PROPERTIES LABELS "gpu" TIMEOUT 180)

    # GPU config of the split-path round-trip (Step 3 gate): same source as the
    # CPU registration above, full-GPU pipeline. Covers extract/serialize/parse/
    # apply over GPU-resident buffers + entities (no stepping).
    set(_roundtrip_gpu integration_ngm3_state_roundtrip_split_gpu)
    add_executable(${_roundtrip_gpu} state_roundtrip_split_test.cpp)
    target_link_libraries(${_roundtrip_gpu} PRIVATE
        doctest::doctest
        coast_lib
        coast_test_support
        coast_compare_lib
        coast_integration_helpers
    )
    target_compile_features(${_roundtrip_gpu} PRIVATE cxx_std_23)
    target_compile_definitions(${_roundtrip_gpu} PRIVATE
        COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist_ngm3_gpu_full.json"
    )

    add_test(NAME integration.ngm3_state_roundtrip_split_gpu
             COMMAND ${_roundtrip_gpu}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(integration.ngm3_state_roundtrip_split_gpu
                         PROPERTIES LABELS "gpu" TIMEOUT 180)

    # GPU-only step-once reproducibility through the load split (parse_state +
    # apply_state) — the runtime path Step 4 moves onto run_quiesced. Exercises
    # the restored read page being consumed after a step.
    set(_resume_split_gpu integration_ngm3_state_resume_split_repro_gpu)
    add_executable(${_resume_split_gpu} state_resume_split_repro_test.cpp)
    target_link_libraries(${_resume_split_gpu} PRIVATE
        doctest::doctest
        coast_lib
        coast_test_support
        coast_compare_lib
        coast_integration_helpers
    )
    target_compile_features(${_resume_split_gpu} PRIVATE cxx_std_23)
    target_compile_definitions(${_resume_split_gpu} PRIVATE
        COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist_ngm3_gpu_full.json"
    )

    add_test(NAME integration.ngm3_state_resume_split_repro_gpu
             COMMAND ${_resume_split_gpu}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(integration.ngm3_state_resume_split_repro_gpu
                         PROPERTIES LABELS "gpu" TIMEOUT 180)

    # GPU config of the deterministic-resume regression (step-after-load equals
    # straight-through, bit-for-bit). Same source as the CPU registration above,
    # full-GPU pipeline. Needs a real device — run outside the sandbox.
    set(_resume_step_gpu integration_ngm3_state_resume_stepafter_gpu)
    add_executable(${_resume_step_gpu} state_resume_stepafter_test.cpp)
    target_link_libraries(${_resume_step_gpu} PRIVATE
        doctest::doctest
        coast_lib
        coast_test_support
        coast_compare_lib
        coast_integration_helpers
    )
    target_compile_features(${_resume_step_gpu} PRIVATE cxx_std_23)
    target_compile_definitions(${_resume_step_gpu} PRIVATE
        COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist_ngm3_gpu_full.json"
    )

    add_test(NAME integration.ngm3_state_resume_stepafter_gpu
             COMMAND ${_resume_step_gpu}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(integration.ngm3_state_resume_stepafter_gpu
                         PROPERTIES LABELS "gpu" TIMEOUT 180)

    # G7 — dirty/rebuild path (B5) proven end-to-end. Self-contained device-level
    # test: a dedicated rebuild-capable demo entity (also the reference shape for
    # a future runtime-plasticity entity) is run with and without a forced
    # epoch-bump rebuild mid-stream; the outputs must be byte-identical. Needs no
    # dataset or config — drives the gpu_device directly. GPU-labeled.
    set(_forced_rebuild integration_ngm3_gpu_forced_rebuild)
    add_executable(${_forced_rebuild} gpu_forced_rebuild_test.cpp)
    target_link_libraries(${_forced_rebuild} PRIVATE
        doctest::doctest
        coast_lib
    )
    target_compile_features(${_forced_rebuild} PRIVATE cxx_std_23)

    add_test(NAME integration.ngm3_gpu_forced_rebuild
             COMMAND ${_forced_rebuild}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(integration.ngm3_gpu_forced_rebuild
                         PROPERTIES LABELS "gpu" TIMEOUT 180)

    # Review #5 — ngm2.5 cross-device checkpoint round-trip. Proves the CPU and
    # GPU save_state/load_state (independent, hand-mirrored) agree by serializing a
    # checkpoint on one device, loading + re-serializing on the other, and asserting
    # the entity subtree is byte-identical. Builds a real GPU sim -> GPU-labeled;
    # runs from the repo root for ./data/mnist.
    set(_xdev integration_ngm25_xdev_checkpoint)
    add_executable(${_xdev} ngm25_xdev_checkpoint_test.cpp)
    target_link_libraries(${_xdev} PRIVATE
        doctest::doctest
        coast_lib
        coast_test_support
        coast_integration_helpers
    )
    target_compile_features(${_xdev} PRIVATE cxx_std_23)
    target_compile_definitions(${_xdev} PRIVATE
        COAST_NGM25_CPU_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist_ngm25.json"
        COAST_NGM25_GPU_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist_ngm25_gpu.json"
    )
    add_test(NAME integration.ngm25_xdev_checkpoint
             COMMAND ${_xdev}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    # TIMEOUT: ctest's default 1500 s, not 180 — measured 44 s with 2x nproc busy
    # loops and 90-107 s with 4x (it is TBB-heavy), so 180 s would trip under a
    # heavier load (CLAUDE.md "Testing": a TIMEOUT must hold under heavy load).
    set_tests_properties(integration.ngm25_xdev_checkpoint
                         PROPERTIES LABELS "gpu" TIMEOUT 1500)

    # ngm3 GPU branch gate after loading a CPU checkpoint (SPEC.md M.7.21): the
    # GPU must branch on the CPU's learning-step schedule, not on its own step
    # counter. Moves a mid-interval CPU group state into a GPU group of the same
    # shape (configs/mnist_ngm3.json vs configs/mnist_ngm3_gpu_full.json).
    set(_bgate integration_ngm3_branch_gate_after_load_gpu)
    add_executable(${_bgate} branch_gate_after_load_gpu_test.cpp)
    target_link_libraries(${_bgate} PRIVATE coast_integration_helpers)
    target_include_directories(${_bgate} PRIVATE ${CMAKE_SOURCE_DIR}/tests/integration/frozen_inert)
    target_compile_features(${_bgate} PRIVATE cxx_std_23)
    add_test(NAME integration.ngm3_branch_gate_after_load_gpu
             COMMAND ${_bgate}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(integration.ngm3_branch_gate_after_load_gpu
                         PROPERTIES LABELS "gpu" TIMEOUT 300)

    # ngm3 GPU load_state refuses a checkpoint that breaks the replica layout
    # (hd_ngm3_layout.h) before uploading anything; edits a branched GPU state.
    set(_layout integration_ngm3_gpu_layout_check)
    add_executable(${_layout} gpu_layout_check_test.cpp)
    target_link_libraries(${_layout} PRIVATE coast_integration_helpers)
    target_include_directories(${_layout} PRIVATE ${CMAKE_SOURCE_DIR}/tests/integration/frozen_inert)
    target_compile_features(${_layout} PRIVATE cxx_std_23)
    add_test(NAME integration.ngm3_gpu_layout_check
             COMMAND ${_layout}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(integration.ngm3_gpu_layout_check
                         PROPERTIES LABELS "gpu" TIMEOUT 300)
endif()
