add_executable(test_sample_sequencer test_sample_sequencer.cpp)
target_link_libraries(test_sample_sequencer PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.io.sample_sequencer COMMAND test_sample_sequencer
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_mnist_io_index_manifest test_mnist_io_index_manifest.cpp)
target_link_libraries(test_mnist_io_index_manifest PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.io.mnist_io_index_manifest COMMAND test_mnist_io_index_manifest
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_dsprites_io test_dsprites_io.cpp)
target_link_libraries(test_dsprites_io PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.io.dsprites_io COMMAND test_dsprites_io
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_loader_input_validation test_loader_input_validation.cpp)
target_link_libraries(test_loader_input_validation PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.io.loader_input_validation COMMAND test_loader_input_validation
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
