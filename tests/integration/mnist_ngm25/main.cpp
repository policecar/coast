#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <filesystem>

#include "integration_helpers.h"

// Generic ngm2.5 fingerprint test. The compare mode and (optional) tolerances
// path are injected as compile defines by coast_add_integration_test:
//   COAST_TEST_COMPARE_MODE_AGGREGATE  1 -> aggregate, otherwise strict
//   COAST_TEST_TOLERANCES              path to tolerances.json (optional)
#ifndef COAST_TEST_COMPARE_MODE_AGGREGATE
#define COAST_TEST_COMPARE_MODE_AGGREGATE 0
#endif

TEST_CASE(COAST_TEST_NAME) {
    const auto mode = (COAST_TEST_COMPARE_MODE_AGGREGATE != 0)
        ? coast::compare::compare_mode::aggregate
        : coast::compare::compare_mode::strict;
#ifdef COAST_TEST_TOLERANCES
    const std::filesystem::path tolerances{ COAST_TEST_TOLERANCES };
#else
    const std::filesystem::path tolerances{};
#endif
    coast::test::run_io_buffer_fingerprint_test(
        COAST_TEST_CONFIG,
        COAST_TEST_SEED_OFFSET,
        COAST_TEST_STEPS,
        std::filesystem::path{ COAST_TEST_BLOB_PATH },
        std::filesystem::path{ COAST_TEST_GOLDENS_DIR },
        mode,
        tolerances);
}
