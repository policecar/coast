add_executable(test_metric test_metric.cpp)
target_link_libraries(test_metric PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.tools.metric COMMAND test_metric)

add_executable(test_distortion test_distortion.cpp)
target_link_libraries(test_distortion PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.tools.distortion COMMAND test_distortion)

add_executable(test_ensemble_code test_ensemble_code.cpp)
target_link_libraries(test_ensemble_code PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.tools.ensemble_code COMMAND test_ensemble_code)

add_executable(test_histogram test_histogram.cpp)
target_link_libraries(test_histogram PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.tools.histogram COMMAND test_histogram)

add_executable(test_metric_registry test_metric_registry.cpp)
target_link_libraries(test_metric_registry PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.tools.metric_registry COMMAND test_metric_registry)

add_executable(test_sim_builder test_sim_builder.cpp)
target_link_libraries(test_sim_builder PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.tools.sim_builder COMMAND test_sim_builder)

add_executable(test_entity_registration test_entity_registration.cpp)
target_link_libraries(test_entity_registration PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.tools.entity_registration COMMAND test_entity_registration
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_entity_types test_entity_types.cpp)
target_link_libraries(test_entity_types PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.tools.entity_types COMMAND test_entity_types)
