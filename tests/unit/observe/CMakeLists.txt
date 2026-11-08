# Unit (and one in-process integration) tests for the observe system.
# Each is a standalone doctest executable linking coast_lib; HDF5 comes
# transitively (PUBLIC on coast_lib).

add_executable(test_observe_config_parse test_config_parse.cpp)
target_link_libraries(test_observe_config_parse PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.observe.config_parse COMMAND test_observe_config_parse
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_observe_dtype_lock test_dtype_lock.cpp)
target_link_libraries(test_observe_dtype_lock PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.observe.dtype_lock COMMAND test_observe_dtype_lock
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_observe_trigger test_trigger.cpp)
target_link_libraries(test_observe_trigger PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.observe.trigger COMMAND test_observe_trigger
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Round-trips the observe writer through a real hdf5::executor + temp .h5 file.
# In-process integration in spirit; kept here for the lightweight doctest wiring.
add_executable(test_observe_writer_roundtrip test_writer_roundtrip.cpp)
target_link_libraries(test_observe_writer_roundtrip PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.observe.writer_roundtrip COMMAND test_observe_writer_roundtrip
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# N-D (multi-dimensional) observe value capture -> [rows, d0, d1, ...] dataset.
add_executable(test_observe_nd test_nd_observe.cpp)
target_link_libraries(test_observe_nd PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.observe.nd COMMAND test_observe_nd
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Chunk-geometry policy: pick_chunk_rows() math + on-disk chunk readback.
add_executable(test_observe_chunk_geom test_chunk_geom.cpp)
target_link_libraries(test_observe_chunk_geom PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.observe.chunk_geom COMMAND test_observe_chunk_geom
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Writer throughput micro-benchmark (manual; not registered as a ctest).
add_executable(bench_observe_writer bench_writer.cpp)
target_link_libraries(bench_observe_writer PRIVATE coast_lib)
