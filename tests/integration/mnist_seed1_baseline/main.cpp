#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <filesystem>

#include "integration_helpers.h"

TEST_CASE(COAST_TEST_NAME) {
    coast::test::run_io_buffer_fingerprint_test(
        COAST_TEST_CONFIG,
        COAST_TEST_SEED_OFFSET,
        COAST_TEST_STEPS,
        std::filesystem::path{ COAST_TEST_BLOB_PATH },
        std::filesystem::path{ COAST_TEST_GOLDENS_DIR });
}
