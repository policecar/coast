add_executable(test_ngm25_arena_consistency test_arena_consistency.cpp)
target_link_libraries(test_ngm25_arena_consistency PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.arena_consistency COMMAND test_ngm25_arena_consistency
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm25_repack_splits test_repack_splits.cpp)
target_link_libraries(test_ngm25_repack_splits PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.repack_splits COMMAND test_ngm25_repack_splits
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm25_response test_response.cpp)
target_link_libraries(test_ngm25_response PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.response COMMAND test_ngm25_response
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm25_adapt test_adapt.cpp)
target_link_libraries(test_ngm25_adapt PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.adapt COMMAND test_ngm25_adapt
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm25_branch test_branch.cpp)
target_link_libraries(test_ngm25_branch PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.branch COMMAND test_ngm25_branch
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm25_persistence test_persistence.cpp)
target_link_libraries(test_ngm25_persistence PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.persistence COMMAND test_ngm25_persistence
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm25_vis_mirror test_vis_mirror.cpp)
target_link_libraries(test_ngm25_vis_mirror PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.vis_mirror COMMAND test_ngm25_vis_mirror
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm25_config_validation test_config_validation.cpp)
target_link_libraries(test_ngm25_config_validation PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.config_validation COMMAND test_ngm25_config_validation
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm25_process_guard test_process_guard.cpp)
target_link_libraries(test_ngm25_process_guard PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.process_guard COMMAND test_ngm25_process_guard
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm25_config_record test_config_record.cpp)
target_link_libraries(test_ngm25_config_record PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.config_record COMMAND test_ngm25_config_record
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# #234: GPU readback failures during save / growth / vis refresh. In-memory
# fault-injection gpu_device: no adapter needed, but the GPU entity only exists in
# a GPU build (webgpu or null backend).
if(NOT COAST_GPU_BACKEND STREQUAL "none")
    add_executable(test_ngm25_gpu_readback_faults test_gpu_readback_faults.cpp)
    target_link_libraries(test_ngm25_gpu_readback_faults PRIVATE doctest::doctest coast_lib coast_test_support)
    add_test(NAME unit.hd_ngm25.gpu_readback_faults COMMAND test_ngm25_gpu_readback_faults
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

add_executable(test_ngm25_max_depth test_max_depth.cpp)
target_link_libraries(test_ngm25_max_depth PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm25.max_depth COMMAND test_ngm25_max_depth
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(unit.hd_ngm25.max_depth PROPERTIES TIMEOUT 60)   # a wrapped walk hangs (#237)
