// Scaffold smoke test: verifies that doctest is wired up and that the test
// binary can link against coast_lib (the production static library).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "cli/options.h"

TEST_CASE("scaffold: doctest runs and coast_lib is linkable") {
    coast::cli::options opts;
    CHECK(opts.config_path.empty());
    CHECK(opts.max_steps    == 0);
    CHECK(opts.seed_offset  == seed_offset_t{ 0 });
    CHECK_FALSE(opts.headless);
    CHECK_FALSE(opts.auto_start);
}
