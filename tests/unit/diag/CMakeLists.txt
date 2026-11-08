add_executable(test_diag_progress test_progress.cpp)
target_link_libraries(test_diag_progress PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.diag.progress COMMAND test_diag_progress)

add_executable(test_diag test_diag.cpp)
target_link_libraries(test_diag PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.diag.core COMMAND test_diag)
