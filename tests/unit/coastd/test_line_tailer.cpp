#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <string>
#include <vector>

#include "daemon/line_tailer.h"
#include "diag/progress.h"

using coastd::line_tailer;
using coast::diag::parse_progress;
using coast::diag::run_phase;

// Mirrors Demonstrate spike 04 PART A: a byte stream chunked at hostile
// boundaries — a complete progress line, an interleaved human line, a progress
// line TORN across two feeds, and a trailing partial with no newline.
TEST_CASE("line_tailer: partial-line + human-line robustness (spike 04 PART A)") {
    std::vector<std::string> chunks = {
        "@coast/progress v1 phase=start step=0 start=0 target=20000 sps=0.000 logged=0\n"
        "Checkpoint @ step 2000: mnist_x.h5\n"
        "@coast/progress v1 phase=che",                       // torn mid-line
        "ckpoint step=2000 start=0 target=20000 sps=3950.000 logged=0\n"
        "Simulation completed. Total steps: 20000\n"
        "@coast/progress v1 phase=done step=20000 start=0 target=20000 sps=3994.000 logged=0\n"
        "@coast/progress v1 phase=step step=21000 start=0 targ",  // trailing partial
    };

    line_tailer t;
    int parsed = 0, ignored = 0;
    std::vector<coast::diag::progress> got;
    for (auto& c : chunks) {
        t.feed(c, [&](std::string_view line) {
            if (auto p = parse_progress(line)) { ++parsed; got.push_back(*p); }
            else                               { ++ignored; }
        });
    }

    // 3 progress lines recovered (start@0, checkpoint@2000, done@20000); the torn
    // line is recovered exactly once with the right phase & step.
    REQUIRE(got.size() == 3);
    CHECK(got[1].phase == run_phase::checkpoint);
    CHECK(got[1].step  == 2000);
    CHECK(got[2].phase == run_phase::done);
    CHECK(got[2].step  == 20000);
    CHECK(parsed  == 3);
    CHECK(ignored == 2);  // the two human lines

    // The trailing partial must still be buffered, unparsed.
    CHECK(t.pending().find("phase=step") != std::string_view::npos);
}

TEST_CASE("line_tailer: byte-at-a-time feeding recovers the same lines") {
    const std::string s = "alpha\nbeta\ngamma";  // no trailing newline
    line_tailer t;
    std::vector<std::string> lines;
    for (char c : s)
        t.feed(std::string_view(&c, 1), [&](std::string_view l) { lines.emplace_back(l); });

    REQUIRE(lines.size() == 2);
    CHECK(lines[0] == "alpha");
    CHECK(lines[1] == "beta");
    CHECK(t.pending() == "gamma");  // last, unterminated
}

// ---- regression: bounded partial line + clean resync (review 2026-07-21) -------
// A producer that streams bytes with no newline must not grow the buffer without
// bound; past the cap the over-long line is dropped and parsing resynchronizes at
// the next newline so a subsequent well-formed line is still delivered intact.
TEST_CASE("line_tailer: over-long unterminated line is bounded, then resyncs") {
    line_tailer t;
    std::vector<std::string> lines;
    auto sink = [&](std::string_view l) { lines.emplace_back(l); };

    // Feed well over the cap with no '\n'.
    const std::string huge(line_tailer::max_line_bytes + 4096, 'x');
    t.feed(huge, sink);
    CHECK(lines.empty());                                   // nothing emitted yet
    CHECK(t.pending().size() <= line_tailer::max_line_bytes);  // buffer stayed bounded

    // The tail of the dropped line ends, then a clean line arrives: it must parse.
    t.feed("more-garbage\ngood-line\n", sink);
    REQUIRE(lines.size() == 1);
    CHECK(lines[0] == "good-line");                         // realigned at the boundary
}

TEST_CASE("line_tailer: many small lines in one feed all delivered (no O(n^2) drop)") {
    line_tailer t;
    std::string blob;
    for (int i = 0; i < 5000; ++i) blob += "line" + std::to_string(i) + "\n";
    int n = 0;
    std::string last;
    t.feed(blob, [&](std::string_view l) { ++n; last = l; });
    CHECK(n == 5000);
    CHECK(last == "line4999");
    CHECK(t.pending().empty());
}
