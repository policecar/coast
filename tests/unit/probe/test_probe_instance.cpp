#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstdint>

#include "probe/probe_instance.h"

// These tests focus on the decimation rule in probe_instance::record().
// We construct probes with nullptr element/structure to avoid touching the
// entity_element hierarchy (activate()/deactivate() are explicitly guarded
// against nullptr element).
//
// Decimation rule (see src/probe/probe_instance.cpp):
//   dec == 1: every record() pushes a sample
//   dec >  1: only records where (call_count++) % dec == 0 push
//   call_count is incremented via fetch_add (post-increment),
//   so the FIRST call always pushes (count starts at 0).

namespace {

probe::probe_instance make_probe() {
    return probe::probe_instance(nullptr, nullptr, "test_probe");
}

std::size_t drain_count(probe::probe_instance& p) {
    std::size_t n = 0;
    p.samples.drain([&](const probe::probe_sample&) { ++n; });
    return n;
}

} // namespace

TEST_CASE("probe_instance: decimation=1 records every call") {
    auto p = make_probe();
    p.decimation.store(1);
    for (int i = 0; i < 10; ++i) p.record(static_cast<double>(i));
    CHECK(drain_count(p) == 10u);
}

TEST_CASE("probe_instance: decimation=3 records every 3rd call (first call always)") {
    auto p = make_probe();
    p.decimation.store(3);
    for (int i = 0; i < 10; ++i) p.record(static_cast<double>(i));
    // call_count sequence: 0,1,2,3,4,5,6,7,8,9
    //   keep iff count % 3 == 0  →  0,3,6,9  →  4 samples
    CHECK(drain_count(p) == 4u);
}

TEST_CASE("probe_instance: decimation=5 records 1 of every 5") {
    auto p = make_probe();
    p.decimation.store(5);
    for (int i = 0; i < 20; ++i) p.record(static_cast<double>(i));
    // 0,5,10,15 → 4 samples
    CHECK(drain_count(p) == 4u);
}

TEST_CASE("probe_instance: decimation=1 records sample values verbatim") {
    auto p = make_probe();
    p.decimation.store(1);
    probe::set_current_step(42);
    p.record(3.14);
    p.record(2.71);

    std::vector<probe::probe_sample> got;
    p.samples.drain([&](const probe::probe_sample& s) { got.push_back(s); });
    REQUIRE(got.size() == 2u);
    CHECK(got[0].step == 42u);
    CHECK(got[0].value == doctest::Approx(3.14));
    CHECK(got[1].step == 42u);
    CHECK(got[1].value == doctest::Approx(2.71));
}

TEST_CASE("probe_instance: decimation drops between recorded samples but keeps order") {
    auto p = make_probe();
    p.decimation.store(3);
    probe::set_current_step(0);
    for (int i = 0; i < 9; ++i) {
        probe::set_current_step(static_cast<std::uint64_t>(i));
        p.record(static_cast<double>(i * 10));
    }
    // Calls happen at steps 0..8; kept indices = 0,3,6.
    std::vector<probe::probe_sample> got;
    p.samples.drain([&](const probe::probe_sample& s) { got.push_back(s); });
    REQUIRE(got.size() == 3u);
    CHECK(got[0].step == 0u);  CHECK(got[0].value == doctest::Approx(0.0));
    CHECK(got[1].step == 3u);  CHECK(got[1].value == doctest::Approx(30.0));
    CHECK(got[2].step == 6u);  CHECK(got[2].value == doctest::Approx(60.0));
}
