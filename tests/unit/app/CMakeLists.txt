add_executable(test_path_resolver test_path_resolver.cpp)
target_link_libraries(test_path_resolver PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.app.path_resolver COMMAND test_path_resolver)

add_executable(test_app_config test_app_config.cpp)
target_link_libraries(test_app_config PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.app.app_config COMMAND test_app_config)
