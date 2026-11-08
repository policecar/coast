add_executable(test_hdf5_executor test_executor.cpp)
target_link_libraries(test_hdf5_executor PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hdf5.executor COMMAND test_hdf5_executor
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_hdf5_append_rows test_append_rows.cpp)
target_link_libraries(test_hdf5_append_rows PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.hdf5.append_rows COMMAND test_hdf5_append_rows
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
