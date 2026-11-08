add_executable(test_run_manifest test_run_manifest.cpp)
target_link_libraries(test_run_manifest PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.sim_core.run_manifest COMMAND test_run_manifest
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_thread_controller_hold test_thread_controller_hold.cpp)
target_link_libraries(test_thread_controller_hold PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.sim_core.thread_controller_hold COMMAND test_thread_controller_hold
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_step_fault test_step_fault.cpp)
target_link_libraries(test_step_fault PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.sim_core.step_fault COMMAND test_step_fault
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_thread_controller_single_step test_thread_controller_single_step.cpp)
target_link_libraries(test_thread_controller_single_step PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.sim_core.thread_controller_single_step COMMAND test_thread_controller_single_step
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_hook_removal test_hook_removal.cpp)
target_link_libraries(test_hook_removal PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.sim_core.hook_removal COMMAND test_hook_removal
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
