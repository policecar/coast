# Standalone golden-blob comparison tool.
# Library form (compare_lib) is unit-testable; main.cpp is the thin CLI shell.

add_library(coast_compare_lib STATIC
    compare.h
    compare.cpp
)
target_include_directories(coast_compare_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/3rd_party/json/include
)
target_compile_features(coast_compare_lib PUBLIC cxx_std_23)
target_compile_options(coast_compare_lib PRIVATE
    -Wall -Wpedantic -Wextra
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-psabi>
)

add_executable(coast_compare main.cpp)
target_link_libraries(coast_compare PRIVATE coast_compare_lib)
target_compile_options(coast_compare PRIVATE
    -Wall -Wpedantic -Wextra
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-psabi>
)
