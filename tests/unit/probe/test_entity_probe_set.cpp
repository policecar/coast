#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <unordered_set>

#include "probe/entity_probe_set.h"

// #249.3: the sim thread calls record(), the GUI thread (or the headless loop)
// calls discover() and traverses with all_instances() / update_observers().
// record() must never insert, and the traversals must not walk the TBB map:
// a lookup may rehash a bucket and move nodes under an unlocked iterator.
//
// The elements are never dereferenced on these paths (record() of an unknown
// probe only hashes the pointer), so fake addresses suffice.

namespace {

const core::entity_element* fake_element(std::uintptr_t i) {
    return reinterpret_cast<const core::entity_element*>(i * 64);
}

} // namespace

TEST_CASE("record() of a probe nobody discovered creates no instance") {
    probe::entity_probe_set ps;
    for (std::uintptr_t e = 1; e <= 100; ++e)
        ps.record(fake_element(e), "synapse_count", 1.0);
    CHECK(ps.all_instances().empty());
}

TEST_CASE("traversal concurrent with record() sees each instance exactly once") {
    // Fails at faeac8f1 (traversal over the TBB map): ~15-25% of rounds see an
    // instance twice or miss one, and TSan reports rehash_bucket vs operator++.
    int bad_rounds = 0;
    for (int round = 0; round < 200; ++round) {
        const std::uintptr_t n = 3 + round % 62;   // map sizes 3..64 grow the table
        probe::entity_probe_set ps;
        std::atomic<bool> stop{false};
        std::thread sim([&] {
            while (!stop.load(std::memory_order_relaxed))
                for (std::uintptr_t e = 1; e <= n; ++e) ps.record(fake_element(e), "p", 1.0);
        });
        bool bad = false;
        for (std::uintptr_t k = 1; k <= n; ++k) {
            ps.discover(fake_element(k), "p");
            for (int t = 0; t < 20; ++t) {
                const auto v = ps.all_instances();
                const std::unordered_set<probe::probe_instance*> unique(v.begin(), v.end());
                bad |= v.size() != k || unique.size() != k;
                ps.update_observers();
            }
        }
        stop = true;
        sim.join();
        bad_rounds += bad;
    }
    CHECK(bad_rounds == 0);
}
