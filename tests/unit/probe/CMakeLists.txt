add_executable(test_spsc_queue test_spsc_queue.cpp)
target_link_libraries(test_spsc_queue PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.probe.spsc_queue COMMAND test_spsc_queue)

add_executable(test_probe_instance test_probe_instance.cpp)
target_link_libraries(test_probe_instance PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.probe.probe_instance COMMAND test_probe_instance)

add_executable(test_entity_probe_set test_entity_probe_set.cpp)
target_link_libraries(test_entity_probe_set PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.probe.entity_probe_set COMMAND test_entity_probe_set)
