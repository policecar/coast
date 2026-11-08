add_executable(test_ngm3_branch_interval_guard test_branch_interval_guard.cpp)
target_link_libraries(test_ngm3_branch_interval_guard PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm3.branch_interval_guard COMMAND test_ngm3_branch_interval_guard
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm3_frozen_inert test_frozen_inert.cpp)
target_link_libraries(test_ngm3_frozen_inert PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm3.frozen_inert COMMAND test_ngm3_frozen_inert
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm3_layout_check test_layout_check.cpp)
target_link_libraries(test_ngm3_layout_check PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm3.layout_check COMMAND test_ngm3_layout_check
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm3_config_validation test_config_validation.cpp)
target_link_libraries(test_ngm3_config_validation PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm3.config_validation COMMAND test_ngm3_config_validation
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm3_cold_init test_cold_init.cpp)
target_link_libraries(test_ngm3_cold_init PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm3.cold_init COMMAND test_ngm3_cold_init
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# #234: GPU readback failures during save / vis refresh. Uses an in-memory
# fault-injection gpu_device, so it needs no adapter (no "gpu" label) — but the
# GPU entity only exists in a GPU build (webgpu or null backend).
if(NOT COAST_GPU_BACKEND STREQUAL "none")
    add_executable(test_ngm3_gpu_readback_faults test_gpu_readback_faults.cpp)
    target_link_libraries(test_ngm3_gpu_readback_faults PRIVATE doctest::doctest coast_lib coast_test_support)
    add_test(NAME unit.hd_ngm3.gpu_readback_faults COMMAND test_ngm3_gpu_readback_faults
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

add_executable(test_ngm3_max_depth test_max_depth.cpp)
target_link_libraries(test_ngm3_max_depth PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm3.max_depth COMMAND test_ngm3_max_depth
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(unit.hd_ngm3.max_depth PROPERTIES TIMEOUT 60)   # a wrapped walk hangs (#237)
