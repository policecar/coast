add_executable(test_json_loader test_json_loader.cpp)
target_link_libraries(test_json_loader PRIVATE doctest::doctest coast_lib)
target_compile_definitions(test_json_loader PRIVATE
    COAST_JSON_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    COAST_CONFIGS_DIR="${CMAKE_SOURCE_DIR}/configs"
)
add_test(NAME unit.json_cfg.loader COMMAND test_json_loader
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_ngm3_json test_ngm3_json.cpp)
target_link_libraries(test_ngm3_json PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.json_cfg.ngm3 COMMAND test_ngm3_json
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_dataset_json test_dataset_json.cpp)
target_link_libraries(test_dataset_json PRIVATE doctest::doctest coast_lib)
target_compile_options(test_dataset_json PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>
)
add_test(NAME unit.json_cfg.dataset COMMAND test_dataset_json
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
