# coast_add_integration_test(NAME <name> CONFIG <relpath> STEPS <N>
#                            [SEED_OFFSET <int>] [COMPARE_MODE strict|aggregate]
#                            [TOLERANCES <relpath>] [GPU])
#
# Defines an A-tier in-process integration test executable.
# Expects <NAME>/main.cpp to exist alongside <NAME>/goldens/.
# Registers the test in the global property COAST_INTEGRATION_TESTS so that
# coast_update_goldens can drive a capture pass for it.
#
# COMPARE_MODE  strict (default) | aggregate — passed through as the
#               COAST_TEST_COMPARE_MODE_AGGREGATE define consumed by main.cpp.
# TOLERANCES    optional path (relative to repo root) to a tolerances.json,
#               surfaced as the COAST_TEST_TOLERANCES define.
# GPU           marks the test as requiring a real GPU adapter (its goldens are
#               real-numerics): it is skipped unless COAST_GPU_BACKEND=webgpu
#               (the null backend produces no-op numerics), and labeled "gpu" so a
#               CPU-only / sanitizer ctest run can exclude it with -LE gpu.

function(coast_add_integration_test)
    cmake_parse_arguments(IT "GPU" "NAME;CONFIG;STEPS;SEED_OFFSET;COMPARE_MODE;TOLERANCES" "" ${ARGN})
    if(NOT IT_NAME OR NOT IT_CONFIG OR NOT IT_STEPS)
        message(FATAL_ERROR "coast_add_integration_test: NAME, CONFIG, and STEPS are required")
    endif()
    if(NOT DEFINED IT_SEED_OFFSET)
        set(IT_SEED_OFFSET 0)
    endif()

    # GPU goldens need real GPU numerics; skip unless the webgpu backend is built.
    if(IT_GPU AND NOT COAST_GPU_BACKEND STREQUAL "webgpu")
        message(STATUS "Skipping GPU integration test ${IT_NAME} (COAST_GPU_BACKEND=${COAST_GPU_BACKEND})")
        return()
    endif()

    set(_src_dir   ${CMAKE_CURRENT_SOURCE_DIR}/${IT_NAME})
    set(_blob_dir  ${CMAKE_CURRENT_BINARY_DIR}/${IT_NAME})
    set(_blob_path ${_blob_dir}/${IT_NAME}.json)
    set(_goldens   ${_src_dir}/goldens)

    set(_target integration_${IT_NAME})
    add_executable(${_target} ${_src_dir}/main.cpp)
    target_link_libraries(${_target} PRIVATE
        doctest::doctest
        coast_lib
        coast_test_support
        coast_compare_lib
        coast_integration_helpers
    )
    target_compile_definitions(${_target} PRIVATE
        COAST_TEST_NAME="${IT_NAME}"
        COAST_TEST_CONFIG="${CMAKE_SOURCE_DIR}/${IT_CONFIG}"
        COAST_TEST_STEPS=${IT_STEPS}
        COAST_TEST_SEED_OFFSET=${IT_SEED_OFFSET}
        COAST_TEST_BLOB_PATH="${_blob_path}"
        COAST_TEST_GOLDENS_DIR="${_goldens}"
    )
    if(IT_COMPARE_MODE STREQUAL "aggregate")
        target_compile_definitions(${_target} PRIVATE COAST_TEST_COMPARE_MODE_AGGREGATE=1)
    endif()
    if(IT_TOLERANCES)
        target_compile_definitions(${_target} PRIVATE
            COAST_TEST_TOLERANCES="${CMAKE_SOURCE_DIR}/${IT_TOLERANCES}")
    endif()
    file(MAKE_DIRECTORY ${_blob_dir})

    add_test(NAME integration.${IT_NAME} COMMAND ${_target}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    if(IT_GPU)
        set_tests_properties(integration.${IT_NAME} PROPERTIES LABELS "gpu")
    endif()

    set_property(GLOBAL APPEND PROPERTY COAST_INTEGRATION_TESTS
        "${IT_NAME}|$<TARGET_FILE:${_target}>|${_blob_path}|${_goldens}")
endfunction()
