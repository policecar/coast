# Death-test helper: trips the observe dtype/shape lock on the resolve_site()
# (cache-miss) path so the parent B-tier test can assert it abort()s (an abort
# cannot be caught in-process).
add_executable(observe_dtype_mismatch_probe observe_dtype_mismatch_main.cpp)
target_link_libraries(observe_dtype_mismatch_probe PRIVATE coast_lib)

# Death-test helper: trips the SAME lock on the observe_registry::emit()
# cache-hit path (the guard added in this PR). Needs a real observing sim to make
# a group indexed + active, so it links the full coast_lib and derives its config
# from the repo's mnist.json (repointed at a synthetic dataset).
add_executable(observe_emit_dtype_mismatch_probe observe_emit_dtype_mismatch_main.cpp)
target_link_libraries(observe_emit_dtype_mismatch_probe PRIVATE coast_lib)
target_compile_definitions(observe_emit_dtype_mismatch_probe PRIVATE
    COAST_REPO_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(test_b_tier test_b_tier.cpp)
target_link_libraries(test_b_tier PRIVATE
    doctest::doctest
    coast_test_support
    # coast_gui for the coast::simc sim-control façade end-to-end cases (simc
    # lives in the GUI layer); coast_lib alone in a headless build, where those
    # cases are compiled out.
    $<IF:$<BOOL:${COAST_WITH_GUI}>,coast_gui,coast_lib>
)
target_compile_definitions(test_b_tier PRIVATE
    COAST_BIN="$<TARGET_FILE:coast>"
    COAST_REPO_ROOT="${CMAKE_SOURCE_DIR}"
    OBSERVE_DTYPE_MISMATCH_BIN="$<TARGET_FILE:observe_dtype_mismatch_probe>"
    OBSERVE_EMIT_DTYPE_MISMATCH_BIN="$<TARGET_FILE:observe_emit_dtype_mismatch_probe>"
)
add_test(NAME integration_b COMMAND test_b_tier
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
