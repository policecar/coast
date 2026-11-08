#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <chrono>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <variant>

#include "daemon/event_queue.h"
#include "daemon/worker_pool.h"

// Waits on worker threads running trivial tasks: no progress counter, so the
// stall limit is the only bound (CLAUDE.md "Testing"), never a performance budget.
#include "await_progress.h"

using namespace coastd;
using namespace std::chrono_literals;

// If the core thread dies with a client_command still queued, close() must DROP
// it so the captured promise breaks and the connection thread's future::get()
// unblocks (throws) instead of wedging shutdown forever.
TEST_CASE("event_queue: close() drops a pending client_command and breaks its promise") {
    event_queue q;
    auto prom = std::make_shared<std::promise<int>>();
    auto fut  = prom->get_future();
    // A client_command carrying the promise, as core::submit/etc. build it. The
    // QUEUED LAMBDA must be the promise's sole owner (submit's local shared_ptr
    // goes out of scope right after push), so dropping the command destroys it —
    // reset our reference to reproduce that.
    q.push(client_command{[prom](core&) { prom->set_value(1); }, "test"});
    prom.reset();

    q.close();  // core-thread-died path

    // The command was dropped, so the promise is destroyed unfulfilled -> the
    // future is broken. get() must THROW rather than block.
    CHECK_THROWS_AS(fut.get(), std::future_error);

    // And the queue is drained: pop_until returns false immediately.
    event e;
    CHECK_FALSE(q.pop_until(std::chrono::steady_clock::now() + 20ms, e));
}

TEST_CASE("event_queue: pop_until times out with no event") {
    event_queue q;
    event e;
    const auto t0 = std::chrono::steady_clock::now();
    const bool got = q.pop_until(t0 + 20ms, e);
    CHECK_FALSE(got);
}

TEST_CASE("event_queue: a pushed event is popped") {
    event_queue q;
    q.push(child_exited{"j1", 0, 100, "ck.h5"});
    event e;
    REQUIRE(q.pop_until(std::chrono::steady_clock::now() + coast_test::default_stall, e));
    REQUIRE(std::holds_alternative<child_exited>(e));
    CHECK(std::get<child_exited>(e).id == "j1");
    CHECK(std::get<child_exited>(e).steps_done == 100);
}

TEST_CASE("event_queue: close wakes a waiter and drops ALL pending events") {
    event_queue q;
    q.push(worker_result{task_kind::gc, "", "sha_x", true, ""});  // pre-close
    q.close();
    q.push(worker_result{task_kind::gc, "", "late", true, ""});   // post-close, dropped

    // close() now drops pending events too (not just later pushes): a dead core
    // must not leave a queued client_command's promise alive and wedge a client.
    // The queue is empty immediately.
    event e;
    CHECK_FALSE(q.pop_until(std::chrono::steady_clock::now() + 20ms, e));
    CHECK(q.is_closed());
}

TEST_CASE("worker_pool: every submitted task posts exactly one result") {
    event_queue q;
    constexpr int N = 50;
    {
        worker_pool pool(4, q);
        for (int i = 0; i < N; ++i)
            pool.submit({task_kind::resolve, "j" + std::to_string(i), {}}, [i] {
                return worker_result{task_kind::resolve, "j" + std::to_string(i),
                                     "sha" + std::to_string(i), true, ""};
            });
        // Drain N results (pool still alive so nothing is dropped).
        std::set<std::string> ids;
        for (int i = 0; i < N; ++i) {
            event e;
            REQUIRE(q.pop_until(std::chrono::steady_clock::now() + coast_test::default_stall, e));
            REQUIRE(std::holds_alternative<worker_result>(e));
            ids.insert(std::get<worker_result>(e).id);
        }
        CHECK(ids.size() == N);  // all distinct tasks ran
    }
}

TEST_CASE("worker_pool: a throwing task cannot take down a worker") {
    event_queue q;
    worker_pool pool(2, q);
    pool.submit({task_kind::build, {}, "boomsha"},
                []() -> worker_result { throw std::runtime_error("boom"); });
    pool.submit({task_kind::build, {}, "ok"},
                [] { return worker_result{task_kind::build, "", "ok", true, ""}; });

    // The good task still completes (the thrower was swallowed).
    event e;
    bool saw_ok = false;
    for (int i = 0; i < 2 && !saw_ok; ++i)
        if (q.pop_until(std::chrono::steady_clock::now() + coast_test::default_stall, e) &&
            std::holds_alternative<worker_result>(e) &&
            std::get<worker_result>(e).sha == "ok")
            saw_ok = true;
    CHECK(saw_ok);
}

// A throwing task must not merely fail to kill its worker — it must still POST a
// result. The core erases its in-flight marker for a (kind, key) only when the
// matching worker_result arrives, so a swallowed throw leaked that key forever:
// with build_concurrency == 1 no build ever started again for the life of the
// daemon, behind a single log line. The key travels with the task precisely so the
// backstop can synthesize this.
TEST_CASE("worker_pool: a throwing task still reports a keyed failure") {
    event_queue q;
    worker_pool pool(1, q);
    pool.submit({task_kind::build, {}, "sha_thrower"},
                []() -> worker_result { throw std::runtime_error("boom"); });

    event e;
    REQUIRE(q.pop_until(std::chrono::steady_clock::now() + coast_test::default_stall, e));
    REQUIRE(std::holds_alternative<worker_result>(e));
    const auto& wr = std::get<worker_result>(e);
    CHECK(wr.kind == task_kind::build);
    CHECK(wr.sha == "sha_thrower");   // keyed, so the core can clear the right marker
    CHECK_FALSE(wr.ok);
    CHECK(wr.message.find("boom") != std::string::npos);
}

TEST_CASE("worker_pool: a non-exception throw is reported too") {
    event_queue q;
    worker_pool pool(1, q);
    pool.submit({task_kind::gc, {}, "sha_odd"},
                []() -> worker_result { throw 42; });

    event e;
    REQUIRE(q.pop_until(std::chrono::steady_clock::now() + coast_test::default_stall, e));
    REQUIRE(std::holds_alternative<worker_result>(e));
    const auto& wr = std::get<worker_result>(e);
    CHECK(wr.kind == task_kind::gc);
    CHECK(wr.sha == "sha_odd");
    CHECK_FALSE(wr.ok);
}
