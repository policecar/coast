add_executable(test_cli_parse test_parse.cpp)
target_link_libraries(test_cli_parse PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.cli.parse COMMAND test_cli_parse)
