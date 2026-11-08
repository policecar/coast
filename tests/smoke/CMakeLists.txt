add_executable(test_smoke test_doctest_link.cpp)
target_link_libraries(test_smoke PRIVATE doctest::doctest coast_lib)
add_test(NAME smoke COMMAND test_smoke)
