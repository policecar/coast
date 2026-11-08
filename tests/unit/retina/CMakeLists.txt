add_executable(test_retina test_retina.cpp)
target_link_libraries(test_retina PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.retina COMMAND test_retina
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
