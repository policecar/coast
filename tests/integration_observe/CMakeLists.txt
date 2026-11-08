# tests/integration_observe/CMakeLists.txt — end-to-end observe capture.
#
# In-process integration: builds a real sim with a small observe capture window,
# runs past it, and reads the flushed HDF5 file back. Complements the observe
# unit tests (config/dtype/trigger/writer) by proving the full pipeline.

set(_target integration_observe_capture)
add_executable(${_target} observe_capture_test.cpp)
target_link_libraries(${_target} PRIVATE
    doctest::doctest
    coast_lib
    coast_test_support
    coast_compare_lib
    coast_integration_helpers
)
target_compile_features(${_target} PRIVATE cxx_std_23)
target_compile_definitions(${_target} PRIVATE
    COAST_BASE_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist.json"
    COAST_NGM3_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist_ngm3.json"
)

# Each test case builds and runs a full sim, and the observe registry is a
# process-wide singleton whose per-call-site caches are not invalidated across a
# registry detach/re-attach (one sim per process is the A-tier convention). So
# register each doctest case as its own ctest entry, filtered by name, to keep
# them in separate processes rather than running both in one binary invocation.
add_test(NAME integration.observe_capture
         COMMAND ${_target} --test-case=*configured*
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME integration.observe_capture_default_base
         COMMAND ${_target} --test-case=*default_base*
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME integration.observe_capture_ngm3
         COMMAND ${_target} --test-case=*ngm3*
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Producer/consumer contract from PR #136, exercised against this worktree's core.
add_executable(integration_observe_contract observe_contract_test.cpp)
target_link_libraries(integration_observe_contract PRIVATE
    doctest::doctest coast_lib coast_test_support coast_compare_lib coast_integration_helpers)
target_compile_features(integration_observe_contract PRIVATE cxx_std_23)
target_compile_definitions(integration_observe_contract PRIVATE
    COAST_OBSERVE_CONTRACT="${CMAKE_CURRENT_SOURCE_DIR}/observe_contract.json"
    COAST_NGM3_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist_ngm3.json"
    COAST_NGM25_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist_ngm25.json")
add_test(NAME integration.observe_contract_checks
    COMMAND integration_observe_contract --test-case-exclude=*real\ run*
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
foreach(model ngm3 ngm25)
    add_test(NAME integration.observe_contract_${model}
        COMMAND integration_observe_contract --test-case=*${model}\ CPU*
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
