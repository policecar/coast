add_executable(test_blob_writer test_blob_writer.cpp)
target_link_libraries(test_blob_writer PRIVATE doctest::doctest coast_test_support)
add_test(NAME unit.blob_writer COMMAND test_blob_writer)
