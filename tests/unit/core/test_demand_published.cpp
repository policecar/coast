// tests/unit/core/test_demand_published.cpp
//
// Unit tests for demand_published<T> — the single-slot, demand-driven sim->GUI
// publisher behind the Step 5 race removal (buffer_browser stats, status_str).
//
// Contract under test:
//   - consume() returns the latest published value and requests the next.
//   - produce_if_wanted() recomputes ONLY when the consumer has consumed the last
//     value (the demand-driven throttle): production tracks consumption, not the
//     producer's call rate. When the consumer is idle, the producer early-returns
//     and never overwrites the unconsumed slot.
//   - One producer thread + one consumer thread never tear the value (run this
//     under TSan to prove the race the redesign removes is gone).
//
// The concurrency case mirrors the real wiring: a "sim" thread spins calling
// produce_if_wanted() at a high rate while a "GUI" thread consumes at a slower
// rate; every consumed value must be one a producer actually published (no torn
// reads), and the producer must skip work whenever the slot is still fresh.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "core/demand_published.h"

#include "await_progress.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>


TEST_CASE("consume returns latest publish and drives the demand handshake") {
    coast::demand_published<int> dp;

    // Fresh slot starts empty and "wanted": the first produce runs.
    int produced_calls = 0;
    auto produce = [&](int value) {
        dp.produce_if_wanted([&] { ++produced_calls; return value; });
    };

    produce(7);
    CHECK(produced_calls == 1);

    // Not consumed yet -> further produces are skipped (slot still fresh).
    produce(8);
    produce(9);
    CHECK(produced_calls == 1);

    // Consume returns the published value and requests the next.
    CHECK(dp.consume() == 7);

    // Now a produce runs again.
    produce(42);
    CHECK(produced_calls == 2);
    CHECK(dp.consume() == 42);
}

TEST_CASE("consume without a new publish returns the last value, still requesting more") {
    coast::demand_published<int> dp;
    dp.produce_if_wanted([] { return 5; });

    CHECK(dp.consume() == 5);
    // No produce in between: the slot still holds the last value.
    CHECK(dp.consume() == 5);
}

TEST_CASE("single producer + single consumer never tear the value (TSan target)") {
    // A value where a torn read is detectable: both halves must always match.
    struct pair_t {
        std::uint64_t a = 0;
        std::uint64_t b = 0;
    };

    coast::demand_published<pair_t> dp;
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> torn{0};
    std::atomic<std::uint64_t> consumed{0};
    std::atomic<std::uint64_t> produced{0};

    // Producer: publish strictly increasing, internally-consistent pairs as fast
    // as it can. produce_if_wanted throttles it to the consumer's rate.
    std::thread producer([&] {
        std::uint64_t n = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            dp.produce_if_wanted([&] {
                produced.fetch_add(1, std::memory_order_relaxed);
                return pair_t{n, n};   // a == b by construction
            });
            ++n;
        }
    });

    // Consumer: read repeatedly; a == b must always hold (else the read tore).
    std::thread consumer([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            pair_t v = dp.consume();
            if (v.a != v.b) torn.fetch_add(1, std::memory_order_relaxed);
            consumed.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // Run until both sides have done a fixed amount of work — not for a fixed
    // number of consumer reads, which a producer thread scheduled late could
    // sleep through entirely. The heartbeat is `produced`: each produce needs a
    // consume before it, so production only advances while BOTH sides work —
    // and a throttle that never lets the producer through stops it (the consumer
    // alone spins forever and would keep any count of its own ticking).
    constexpr std::uint64_t k_consumed = 200'000;
    constexpr std::uint64_t k_produced = 1'000;
    const bool ran = coast_test::await_progress(
        [&] { return consumed.load() >= k_consumed && produced.load() >= k_produced; },
        [&] { return produced.load(); });
    stop.store(true, std::memory_order_relaxed);
    consumer.join();
    producer.join();

    REQUIRE(ran);
    CHECK(torn.load() == 0);
    // Invariant of the throttle (each produce needs a consume in between). Not
    // its detector: whether a broken throttle overshoots here depends on the two
    // threads' relative speed. The single-threaded case above pins it exactly.
    CHECK(produced.load() <= consumed.load() + 1);
}
