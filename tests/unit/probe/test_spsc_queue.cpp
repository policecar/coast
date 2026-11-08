#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstdint>
#include <vector>

#include "probe/spsc_queue.h"

using probe::spsc_queue;

TEST_CASE("spsc_queue: starts empty") {
    spsc_queue<int> q(8);
    CHECK(q.empty());
    CHECK(q.size_approx() == 0u);
    int x = 0;
    CHECK_FALSE(q.try_pop(x));
}

TEST_CASE("spsc_queue: FIFO order with simple push/pop") {
    spsc_queue<int> q(8);
    CHECK(q.try_push(1));
    CHECK(q.try_push(2));
    CHECK(q.try_push(3));
    CHECK(q.size_approx() == 3u);
    int x = 0;
    CHECK(q.try_pop(x)); CHECK(x == 1);
    CHECK(q.try_pop(x)); CHECK(x == 2);
    CHECK(q.try_pop(x)); CHECK(x == 3);
    CHECK(q.empty());
}

TEST_CASE("spsc_queue: full → try_push returns false; one slot wasted by ring") {
    // capacity 4 means 3 usable slots (head==tail represents empty,
    // and one slot is sacrificed to distinguish full from empty).
    spsc_queue<int> q(4);
    CHECK(q.try_push(1));
    CHECK(q.try_push(2));
    CHECK(q.try_push(3));
    CHECK_FALSE(q.try_push(4));     // full
    CHECK(q.size_approx() == 3u);
    int x = 0;
    CHECK(q.try_pop(x));            // free one
    CHECK(q.try_push(4));           // now fits
    CHECK_FALSE(q.try_push(5));     // full again
}

TEST_CASE("spsc_queue: wrap-around preserves values") {
    spsc_queue<int> q(4);
    // Drive head/tail around the ring twice.
    int x = 0;
    for (int round = 0; round < 5; ++round) {
        CHECK(q.try_push(round * 10 + 1));
        CHECK(q.try_push(round * 10 + 2));
        CHECK(q.try_push(round * 10 + 3));
        CHECK(q.try_pop(x)); CHECK(x == round * 10 + 1);
        CHECK(q.try_pop(x)); CHECK(x == round * 10 + 2);
        CHECK(q.try_pop(x)); CHECK(x == round * 10 + 3);
        CHECK(q.empty());
    }
}

TEST_CASE("spsc_queue: drain visits every value in order") {
    spsc_queue<int> q(8);
    for (int v : {7, 8, 9, 10, 11}) CHECK(q.try_push(v));

    std::vector<int> seen;
    q.drain([&](int v) { seen.push_back(v); });
    CHECK(seen == std::vector<int>{7, 8, 9, 10, 11});
    CHECK(q.empty());
}

TEST_CASE("spsc_queue: clear discards pending without popping") {
    spsc_queue<int> q(8);
    CHECK(q.try_push(1));
    CHECK(q.try_push(2));
    q.clear();
    CHECK(q.empty());
    int x = 0;
    CHECK_FALSE(q.try_pop(x));
}

TEST_CASE("spsc_queue: works with non-trivial payload type") {
    struct payload { int a; double b; };
    spsc_queue<payload> q(8);
    CHECK(q.try_push({3, 1.5}));
    CHECK(q.try_push({-1, 2.25}));
    payload p{};
    CHECK(q.try_pop(p)); CHECK(p.a == 3);
    CHECK(q.try_pop(p)); CHECK(p.a == -1);
    CHECK(p.b == doctest::Approx(2.25));
}
