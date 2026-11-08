# h5browse unit tests — pure kernels, no HDF5 file I/O in this group.

add_executable(test_h5b_scalar_type test_scalar_type.cpp)
target_link_libraries(test_h5b_scalar_type PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.h5browse.scalar_type COMMAND test_h5b_scalar_type)

add_executable(test_h5b_scan_stats test_scan_stats.cpp)
target_link_libraries(test_h5b_scan_stats PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.h5browse.scan_stats COMMAND test_h5b_scan_stats)

add_executable(test_h5b_base_histogram test_base_histogram.cpp)
target_link_libraries(test_h5b_base_histogram PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.h5browse.base_histogram COMMAND test_h5b_base_histogram)

add_executable(test_h5b_row_filter test_row_filter.cpp)
target_link_libraries(test_h5b_row_filter PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.h5browse.row_filter COMMAND test_h5b_row_filter)

add_executable(test_h5b_chunk_cache test_chunk_cache.cpp)
target_link_libraries(test_h5b_chunk_cache PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.h5browse.chunk_cache COMMAND test_h5b_chunk_cache)

# In-process integration tier (real writers + real executor, temp files).
add_executable(test_h5b_file_model test_file_model.cpp)
target_link_libraries(test_h5b_file_model PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.h5browse.file_model COMMAND test_h5b_file_model)

add_executable(test_h5b_dataset_reader test_dataset_reader.cpp)
target_link_libraries(test_h5b_dataset_reader PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.h5browse.dataset_reader COMMAND test_h5b_dataset_reader)

add_executable(test_h5b_scan_engine test_scan_engine.cpp)
target_link_libraries(test_h5b_scan_engine PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.h5browse.scan_engine COMMAND test_h5b_scan_engine)

add_executable(test_h5b_console_api test_console_api.cpp)
target_link_libraries(test_h5b_console_api PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.h5browse.console_api COMMAND test_h5b_console_api)

add_executable(test_h5b_pairwise test_pairwise.cpp)
target_link_libraries(test_h5b_pairwise PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.h5browse.pairwise COMMAND test_h5b_pairwise)

# G6 zero-crash contract on corrupted real-writer output (T3 fast subset).
add_executable(test_h5b_malformed test_malformed.cpp)
target_link_libraries(test_h5b_malformed PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.h5browse.malformed COMMAND test_h5b_malformed)

# Hot-kernel throughput benchmark (manual; not registered as a ctest — a
# throughput floor would make the verdict depend on the machine and its load).
add_executable(bench_h5b_kernel_throughput bench_kernel_throughput.cpp)
target_link_libraries(bench_h5b_kernel_throughput PRIVATE coast_lib)
