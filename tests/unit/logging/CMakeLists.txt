add_executable(test_mem_state_io test_mem_state_io.cpp)
target_link_libraries(test_mem_state_io PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.logging.mem_state_io COMMAND test_mem_state_io
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_config_record test_config_record.cpp)
target_link_libraries(test_config_record PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.logging.config_record COMMAND test_config_record
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_state_checks test_state_checks.cpp)
target_link_libraries(test_state_checks PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.logging.state_checks COMMAND test_state_checks
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_checkpoint_version test_checkpoint_version.cpp)
target_link_libraries(test_checkpoint_version PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.logging.checkpoint_version COMMAND test_checkpoint_version
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_hdf5_shape_guards test_hdf5_shape_guards.cpp)
target_link_libraries(test_hdf5_shape_guards PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.logging.hdf5_shape_guards COMMAND test_hdf5_shape_guards
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_buffer_logger_lifecycle test_buffer_logger_lifecycle.cpp)
target_link_libraries(test_buffer_logger_lifecycle PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.logging.buffer_logger_lifecycle COMMAND test_buffer_logger_lifecycle
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_rng_state_io test_rng_state_io.cpp)
target_link_libraries(test_rng_state_io PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.logging.rng_state_io COMMAND test_rng_state_io
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_buffer_logger_write_errors test_buffer_logger_write_errors.cpp)
target_link_libraries(test_buffer_logger_write_errors PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.logging.buffer_logger_write_errors COMMAND test_buffer_logger_write_errors)
