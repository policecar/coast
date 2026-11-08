#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "diag/diag.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <new>
#include <string>
#include <string_view>

using namespace coast::diag;

namespace {
// Restore global diag state (sink + level) at scope exit. Every test mutates the
// process-global sink/level; a mid-case REQUIRE failure unwinds the case before
// any trailing restore statement runs, which would leak a custom sink/level into
// later cases and turn one failure into a confusing cascade. RAII avoids that.
struct diag_state_guard {
    diag_state_guard() = default;
    diag_state_guard(const diag_state_guard&) = delete;
    diag_state_guard& operator=(const diag_state_guard&) = delete;
    ~diag_state_guard() { set_sink({}); set_level(level::info); }
};
}  // namespace

// dispatch() holds a non-recursive mutex across the sink call. These lock in the
// no-deadlock guarantees: a sink that re-enters the logger (by logging or by
// reconfiguring the sink) must not hang the process. If a guard regresses, the
// COAST_* call below never returns and the test times out rather than failing —
// which is itself a clear signal. (The first re-entrant log also emits a
// one-shot "sink re-entered the logger" notice to stderr; that's expected.)

TEST_CASE("diag: log_nothrow swallows a throwing emit action") {
    diag_state_guard guard;
    bool ran = false;
    // A throwing emit (std::format bad_alloc on a thread entry / in a graceful-
    // degradation catch) must not escape log_nothrow.
    log_nothrow([&] { ran = true; throw std::bad_alloc{}; });
    CHECK(ran);   // the action ran, the exception was swallowed, we reached here
}

TEST_CASE("diag: a throwing sink never escapes a bare COAST_* call") {
    diag_state_guard guard;
    set_level(level::info);
    // The point of routing every COAST_* macro through detail::emit_guarded: a
    // sink that throws (stand-in for a std::format bad_alloc under OOM) must not
    // unwind the caller, so COAST_* is safe in a noexcept context / thread entry
    // with no per-call wrapping.
    int calls = 0;
    set_sink([&](level, std::string_view, std::string_view) {
        ++calls;
        throw std::bad_alloc{};
    });
    CHECK_NOTHROW(COAST_ERROR("test", "boom {}", 1));  // swallowed, does not escape
    CHECK(calls == 1);                                 // the sink did run
    // dispatch's in_dispatch flag must be cleared even though the sink threw
    // through it — otherwise this second record would be dropped as "reentrant".
    CHECK_NOTHROW(COAST_ERROR("test", "again {}", 2));
    CHECK(calls == 2);
}

TEST_CASE("diag: a sink that logs drops the reentrant record (no deadlock)") {
    diag_state_guard guard;
    set_level(level::info);
    int calls = 0;
    set_sink([&](level, std::string_view, std::string_view) {
        ++calls;
        COAST_ERROR("reenter", "must be dropped, not delivered");  // re-enters dispatch
    });
    COAST_INFO("t", "outer");
    CHECK(calls == 1);   // the reentrant record was dropped, never reached the sink
}

TEST_CASE("diag: reconfiguring the sink from within a sink is a safe no-op (no deadlock)") {
    diag_state_guard guard;
    set_level(level::info);
    int calls = 0;
    set_sink([&](level, std::string_view, std::string_view) {
        ++calls;
        // Would relock the non-recursive sink_mtx and destroy this running
        // lambda — refused as a no-op instead of deadlocking.
        set_sink([](level, std::string_view, std::string_view) {});
    });
    COAST_ERROR("t", "x");
    CHECK(calls == 1);
    // The inner set_sink was refused, so the counting sink is still installed:
    COAST_ERROR("t", "y");
    CHECK(calls == 2);
}

TEST_CASE("diag: set_log_file from within a sink is refused (no deadlock)") {
    diag_state_guard guard;
    set_level(level::info);
    bool attempted = false;
    set_sink([&](level, std::string_view, std::string_view) {
        attempted = true;
        CHECK_FALSE(set_log_file("/tmp/coast_diag_should_not_open.log"));  // refused
    });
    COAST_ERROR("t", "x");
    CHECK(attempted);
}

// --- runtime behavior -------------------------------------------------------

TEST_CASE("diag: sink receives level, category, and formatted message") {
    diag_state_guard guard;
    level ll{};
    std::string cc, mm;
    set_sink([&](level l, std::string_view c, std::string_view m) {
        ll = l; cc = std::string(c); mm = std::string(m);
    });
    set_level(level::trace);
    COAST_ERROR("gpu", "adapter {} idx {}", "RADV", 3);
    CHECK(ll == level::error);
    CHECK(cc == "gpu");
    CHECK(mm == "adapter RADV idx 3");
}

TEST_CASE("diag: runtime level gates records below the threshold") {
    diag_state_guard guard;
    std::string cap;
    set_sink([&](level l, std::string_view, std::string_view m) {
        cap += std::string(level_name(l)) + "|" + std::string(m) + "\n";
    });
    set_level(level::warn);
    COAST_INFO("t", "info-suppressed");
    COAST_WARN("t", "warn-shown");
    COAST_ERROR("t", "error-shown");
    CHECK(cap.find("info-suppressed") == std::string::npos);
    CHECK(cap.find("warn-shown")      != std::string::npos);
    CHECK(cap.find("error-shown")     != std::string::npos);
}

TEST_CASE("diag: set_log_file appends formatted records to the file") {
    diag_state_guard guard;
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() / "coast_diag_test.log";
    std::error_code ec;
    fs::remove(path, ec);

    REQUIRE(set_log_file(path.string().c_str()));
    set_level(level::info);
    COAST_ERROR("io", "boom {}", 7);

    std::ifstream in(path);
    const std::string content((std::istreambuf_iterator<char>(in)), {});
    CHECK(content.find("[ERROR][io] boom 7") != std::string::npos);

    fs::remove(path, ec);
}

TEST_CASE("diag: a second set_log_file retargets to the new file (closes the previous)") {
    diag_state_guard guard;
    namespace fs = std::filesystem;
    const auto a = fs::temp_directory_path() / "coast_diag_a.log";
    const auto b = fs::temp_directory_path() / "coast_diag_b.log";
    std::error_code ec; fs::remove(a, ec); fs::remove(b, ec);
    set_level(level::info);

    REQUIRE(set_log_file(a.string().c_str()));
    COAST_ERROR("t", "to-a");
    REQUIRE(set_log_file(b.string().c_str()));   // closes a's handle, retargets to b
    COAST_ERROR("t", "to-b");

    std::ifstream ina(a), inb(b);
    const std::string ca((std::istreambuf_iterator<char>(ina)), {});
    const std::string cb((std::istreambuf_iterator<char>(inb)), {});
    CHECK(ca.find("to-a") != std::string::npos);
    CHECK(ca.find("to-b") == std::string::npos);   // second record did not go to a
    CHECK(cb.find("to-b") != std::string::npos);

    fs::remove(a, ec); fs::remove(b, ec);
}
