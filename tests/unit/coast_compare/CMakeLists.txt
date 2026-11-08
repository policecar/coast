add_executable(test_compare test_compare.cpp)
target_link_libraries(test_compare PRIVATE doctest::doctest coast_compare_lib)
add_test(NAME unit.compare COMMAND test_compare)
