// Unit test 3.3 — hdf5::executor error handling and ordering.
//
// The executor is the single-thread funnel every HDF5 user submits work to, so
// its failure semantics matter: a throwing job must not tear down the worker,
// sync jobs must propagate their exception to the caller, async jobs must be
// counted (not fatal), and submission order must be preserved (the observe
// writer relies on value/step rows advancing together). None of this touches
// libhdf5 — the jobs here are plain lambdas — so it is a clean unit test.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include <filesystem>
#include <stdexcept>
#include <vector>

TEST_CASE("hdf5 executor: submit_sync returns the job's result") {
    hdf5::executor exec;
    CHECK(exec.submit_sync([] { return 42; }) == 42);
}

TEST_CASE("hdf5 executor: submit_sync rethrows the job's exception and the worker survives") {
    hdf5::executor exec;
    CHECK_THROWS_AS(
        exec.submit_sync([]() -> int { throw std::runtime_error("boom"); }),
        std::runtime_error);
    // The worker thread must still be alive and able to run more work.
    CHECK(exec.submit_sync([] { return 7; }) == 7);
}

TEST_CASE("hdf5 executor: a throwing async job is counted, not fatal") {
    hdf5::executor exec;
    CHECK(exec.error_count() == 0);

    exec.submit([] { throw std::runtime_error("async boom"); });
    // FIFO: a sync barrier after the async job guarantees it has executed.
    exec.submit_sync([] {});
    CHECK(exec.error_count() == 1);

    // Subsequent work still runs on the same worker.
    CHECK(exec.submit_sync([] { return 1; }) == 1);
}

// hdf5::file::create must materialize missing parent directories: H5Fcreate opens
// the file via the OS and does NOT create intermediate dirs, so a config-relative
// output leaf (e.g. <base>/logs/buffer_log.h5) into a directory no prior run made
// would otherwise fail. This is the choke point for buffer/probe/state logging, so
// covering it here covers every writer. Without the fix, `ok` is false.
TEST_CASE("hdf5 file::create makes missing parent directories") {
    namespace fs = std::filesystem;
    const fs::path base   = fs::temp_directory_path() / "coast_h5_create_test";
    const fs::path target = base / "logs" / "nested" / "buffer_log.h5";
    fs::remove_all(base);  // start clean: the parent must be genuinely absent
    REQUIRE_FALSE(fs::exists(target.parent_path()));

    hdf5::executor exec;  // threading contract: touch libhdf5 only via a submit job
    const bool ok = exec.submit_sync([&]() -> bool {
        auto f = hdf5::file::create(target.string());
        return static_cast<bool>(f);
    });

    CHECK(ok);  // false without the create_directories guard in file::create
    CHECK(fs::exists(target.parent_path()));
    CHECK(fs::exists(target));
    fs::remove_all(base);
}

TEST_CASE("hdf5 executor: jobs execute in FIFO submission order") {
    hdf5::executor exec;
    std::vector<int> order;  // written only by the single worker thread
    for (int i = 0; i < 16; ++i) {
        exec.submit([&order, i] { order.push_back(i); });
    }
    exec.submit_sync([] {});  // barrier: every prior submit() has run; future syncs us
    REQUIRE(order.size() == 16);
    for (int i = 0; i < 16; ++i) {
        CHECK(order[i] == i);
    }
}
