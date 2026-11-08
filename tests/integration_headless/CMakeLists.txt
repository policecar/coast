# tests/integration_headless/CMakeLists.txt — run_headless driven in-process.
#
# For behaviour the CLI cannot trigger deterministically (an exception after
# phase=start, #233); the CLI surface itself is B-tier. Each case builds a full
# sim from configs/mnist.json (needs ./data/mnist, hence the source-dir working
# directory) and gets its own ctest entry, one sim per process as in A-tier.

set(_target integration_headless_feed)
add_executable(${_target} run_headless_feed_test.cpp)
target_link_libraries(${_target} PRIVATE doctest::doctest coast_lib coast_test_support)
target_compile_features(${_target} PRIVATE cxx_std_23)
target_compile_definitions(${_target} PRIVATE
    COAST_BASE_CONFIG="${CMAKE_SOURCE_DIR}/configs/mnist.json"
)

add_test(NAME integration.headless_feed_exception
         COMMAND ${_target} --test-case=*std::exception*
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME integration.headless_feed_nonstd_exception
         COMMAND ${_target} --test-case=*non-std*
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
