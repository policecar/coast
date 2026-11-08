# Test-only static library holding helpers (blob_writer, future integration
# helpers). Never linked into production code.
add_library(coast_test_support STATIC
    await_progress.h
    blob_writer.h
    blob_writer.cpp
    diag_capture.h
    h5_root_links.h
    sim_sync.h
    subprocess.h
    subprocess.cpp
)
target_include_directories(coast_test_support PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/3rd_party/json/include
)
target_compile_features(coast_test_support PUBLIC cxx_std_23)
