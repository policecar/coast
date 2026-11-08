// coastd latency benchmark (manual; NOT a ctest).
//
// The numbers Discover §5 named as acceptance criteria — submit -> visible,
// child death -> detected, cancel -> acknowledged / settled — measured against
// the REAL coastd with the fault-matrix rig (fault_rig.h). These used to be
// pass/fail gates inside fault.coastd.matrix; a latency bound is a verdict that
// depends on the machine and its load (CLAUDE.md "Testing"), so they are reported
// here instead, and the matrix asserts the behaviour the numbers are about.
//
// Usage: bench_coastd_latency [reps]     (default 4)
// Prints median and max per measurement, in ms. Exit code 1 only if the rig or a
// step of the measurement itself failed.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "daemon/child_proc.h"   // start_token_supported
#include "fault_rig.h"

using namespace coastd_fault;

namespace {

void report(const char* what, std::vector<double> v) {
    if (v.empty()) { std::printf("%-44s  (no samples)\n", what); return; }
    std::sort(v.begin(), v.end());
    std::printf("%-44s  median %9.2f ms   max %9.2f ms   (n=%zu)\n", what, v[v.size() / 2],
                v.back(), v.size());
}

// Poll at 10 ms and time until pred(job) holds; -1 if the job froze first (its
// fingerprint, everything observable about it, stood still for the stall limit).
template <class Pred>
double time_until(const rig& r, const std::string& id, Pred pred) {
    const auto t0 = clk::now();
    const bool ok = coast_test::await_progress(
        [&] {
            const auto j = r.job(id);
            return j && pred(*j);
        },
        [&] { return r.fingerprint(id); }, coast_test::default_stall, std::chrono::milliseconds{10});
    return ok ? ms_since(t0) : -1.0;
}

[[noreturn]] void fail(const char* step, const rig& r) {
    std::fprintf(stderr, "bench_coastd_latency: %s failed\n%s", step, r.log_tail().c_str());
    std::exit(1);
}

} // namespace

int main(int argc, char** argv) {
    if (!coastd::start_token_supported()) {
        std::puts("bench_coastd_latency: coastd does not run on this platform; skipped");
        return 0;
    }
    const int reps = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 4;

    // ---- submit -> visible, child SIGKILL -> no longer running ----------------
    {
        rig r;
        if (!r.ok) fail("rig", r);
        std::vector<double> visible, death;
        for (int i = 0; i < reps; ++i) {
            const auto t0 = clk::now();
            const auto id = r.submit("lat" + std::to_string(i));
            if (!id) fail("submit", r);
            if (time_until(r, *id, [](const auto&) { return true; }) < 0) fail("visible", r);
            visible.push_back(ms_since(t0));

            if (time_until(r, *id, is_running) < 0) fail("running", r);
            const auto t1 = clk::now();
            (void)shell_ok("pkill -9 -P " + std::to_string(r.daemon) + " -f coast || true");
            if (time_until(r, *id, [](const auto& j) { return j.status != "running"; }) < 0)
                fail("death", r);
            death.push_back(ms_since(t1));

            // Let the relaunched run finish, then hold the next one again.
            r.release_run();
            if (time_until(r, *id, is_settled) < 0) fail("done", r);
            std::error_code ec;
            fs::remove(r.run_release, ec);
        }
        report("submit -> visible in list", visible);
        report("child SIGKILL -> no longer running", death);
    }

    // ---- cancel of an obedient child ------------------------------------------
    {
        rig r;
        if (!r.ok) fail("rig", r);
        std::vector<double> settled;
        for (int i = 0; i < reps; ++i) {
            const auto id = r.submit("cq" + std::to_string(i));
            if (!id || time_until(r, *id, is_running) < 0) fail("running", r);
            auto c = r.open_conn();
            std::string err;
            const auto t0 = clk::now();
            if (!c || !c->cancel(*id, err)) fail("cancel", r);
            if (time_until(r, *id, [](const auto& j) { return j.bucket == "cancelled"; }) < 0)
                fail("settle", r);
            settled.push_back(ms_since(t0));
        }
        report("cancel -> settled (child exits on SIGTERM)", settled);
    }

    // ---- cancel of a child that ignores SIGTERM (term_grace 3 s) ---------------
    {
        rig r({.child_ignores_term = true, .term_grace_seconds = 3});
        if (!r.ok) fail("rig", r);
        std::vector<double> rpc, ack, settled;
        for (int i = 0; i < std::min(reps, 2); ++i) {
            const auto id = r.submit("cs" + std::to_string(i));
            if (!id || time_until(r, *id, is_running) < 0) fail("running", r);
            auto c = r.open_conn();
            std::string err;
            const auto t0 = clk::now();
            if (!c || !c->cancel(*id, err)) fail("cancel", r);
            rpc.push_back(ms_since(t0));
            if (time_until(r, *id, [](const auto& j) { return j.cancel_requested; }) < 0)
                fail("ack", r);
            ack.push_back(ms_since(t0));
            if (time_until(r, *id, [](const auto& j) { return j.bucket == "cancelled"; }) < 0)
                fail("settle", r);
            settled.push_back(ms_since(t0));
        }
        report("cancel RPC round-trip", rpc);
        report("cancel -> acknowledged on the wire", ack);
        report("cancel -> settled (child ignores SIGTERM)", settled);
    }
    return 0;
}
