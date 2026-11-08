#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "diag/progress.h"

using coast::diag::format_progress;
using coast::diag::parse_progress;
using coast::diag::progress;
using coast::diag::run_phase;

// The whole point of the shared-header contract is that format_progress (emitter,
// in coast) and parse_progress (consumer, in consim) cannot silently drift. The
// compiler enforces field *names*; this locks in the *wire format*.

TEST_CASE("diag::progress: format→parse round-trips every phase") {
    for (auto ph : {run_phase::start, run_phase::step, run_phase::checkpoint,
                    run_phase::done, run_phase::failed}) {
        progress p{ph, 30000, 5000, 500000, 12345.678, 42};
        auto got = parse_progress(format_progress(p));
        REQUIRE(got);
        CHECK(got->phase         == p.phase);
        CHECK(got->step          == 30000);
        CHECK(got->start         == 5000);
        CHECK(got->target        == 500000);
        CHECK(got->logged_frames == 42);
        CHECK(got->sps == doctest::Approx(12345.678));
    }
}

TEST_CASE("diag::progress: tolerates leading whitespace, CRLF, and unknown trailing keys") {
    auto r = parse_progress(
        "  @coast/progress v1 phase=step step=7 start=0 target=100 sps=9.5 logged=0 foo=bar\r\n");
    REQUIRE(r);
    CHECK(r->phase  == run_phase::step);
    CHECK(r->step   == 7);
    CHECK(r->target == 100);
}

TEST_CASE("diag::progress: rejects non-sentinel and phase-less lines") {
    CHECK_FALSE(parse_progress("Steps: 30000 | SPS: 12345"));     // legacy human line
    CHECK_FALSE(parse_progress("@coast/progress v1 step=5"));     // no phase
    CHECK_FALSE(parse_progress(""));                              // empty
}

TEST_CASE("diag::progress: missing numeric fields default to zero") {
    auto r = parse_progress("@coast/progress v1 phase=done");
    REQUIRE(r);
    CHECK(r->step   == 0);
    CHECK(r->target == 0);
    CHECK(r->sps    == 0.0);
}

TEST_CASE("diag::progress: an out-of-range integer field does not throw") {
    // from_chars leaves the field untouched on overflow; parsing must stay
    // non-throwing (the line still carries a valid phase).
    auto r = parse_progress(
        "@coast/progress v1 phase=step step=999999999999999999999999999999");
    REQUIRE(r);
    CHECK(r->phase == run_phase::step);
}
