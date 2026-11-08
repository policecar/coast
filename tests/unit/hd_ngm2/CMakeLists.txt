add_executable(test_ngm2_pick_winner test_pick_winner.cpp)
target_link_libraries(test_ngm2_pick_winner PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm2.pick_winner COMMAND test_ngm2_pick_winner
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm2_config_record test_config_record.cpp)
target_link_libraries(test_ngm2_config_record PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.hd_ngm2.config_record COMMAND test_ngm2_config_record
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm2_config_validation test_config_validation.cpp)
target_link_libraries(test_ngm2_config_validation PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm2.config_validation COMMAND test_ngm2_config_validation
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm2_state_shape test_state_shape.cpp)
target_link_libraries(test_ngm2_state_shape PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.hd_ngm2.state_shape COMMAND test_ngm2_state_shape
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm2_noise_gate test_noise_gate.cpp)
target_link_libraries(test_ngm2_noise_gate PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm2.noise_gate COMMAND test_ngm2_noise_gate
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm2_window_tracking test_window_tracking.cpp)
target_link_libraries(test_ngm2_window_tracking PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm2.window_tracking COMMAND test_ngm2_window_tracking
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm2_observe_order test_observe_order.cpp)
target_link_libraries(test_ngm2_observe_order PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.hd_ngm2.observe_order COMMAND test_ngm2_observe_order
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm2_max_depth test_max_depth.cpp)
target_link_libraries(test_ngm2_max_depth PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hd_ngm2.max_depth COMMAND test_ngm2_max_depth
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
# A segment-id loop that never ends hangs process() (#237): fail, don't block.
set_tests_properties(unit.hd_ngm2.max_depth PROPERTIES TIMEOUT 60)
