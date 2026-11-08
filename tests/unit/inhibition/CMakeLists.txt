add_executable(test_inhibition_cfg test_inhibition_cfg.cpp)
target_link_libraries(test_inhibition_cfg PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.inhibition.cfg COMMAND test_inhibition_cfg
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_random_index test_random_index.cpp)
target_link_libraries(test_random_index PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.inhibition.random_index COMMAND test_random_index
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# CPU <-> GPU parity of the injected max_softmax kernel. Needs the webgpu backend
# and a real adapter, so it is labeled "gpu" (exclude with `ctest -LE gpu`).
if(COAST_GPU_BACKEND STREQUAL "webgpu")
    add_executable(test_gpu_max_softmax test_gpu_max_softmax.cpp)
    target_link_libraries(test_gpu_max_softmax PRIVATE doctest::doctest coast_lib)
    add_test(NAME unit.inhibition.max_softmax_gpu COMMAND test_gpu_max_softmax
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(unit.inhibition.max_softmax_gpu PROPERTIES LABELS "gpu")
endif()

add_executable(test_max_softmax test_max_softmax.cpp)
target_link_libraries(test_max_softmax PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.inhibition.max_softmax COMMAND test_max_softmax
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
