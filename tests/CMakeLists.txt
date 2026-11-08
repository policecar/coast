# tests/CMakeLists.txt — entry point for the COAST test tree.
# Only included when -DBUILD_TESTING=ON.

add_subdirectory(third_party)
add_subdirectory(support)
add_subdirectory(smoke)
add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(integration_b)
add_subdirectory(integration_ngm3)
add_subdirectory(integration_observe)
add_subdirectory(integration_headless)
add_subdirectory(gpu_smoke)
