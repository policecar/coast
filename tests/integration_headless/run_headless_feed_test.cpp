// tests/integration_headless/run_headless_feed_test.cpp
//
// A headless run that fails by an EXCEPTION after phase=start must end its
// @coast/progress feed on exactly one phase=failed line, with nothing after it
// (#233). The return and fault paths are covered from the CLI in B-tier
// (b_tier.failed_run_terminal_progress); no CLI input throws deterministically
// after phase=start, so this drives run_headless in-process and injects the
// throw through the one main-thread seam the run loop calls every poll: a probe
// observer. The loop's update_observers() pushes each sample into it, and it
// throws.
//
// --checkpoint is on so the sim thread is emitting phase=checkpoint lines while
// the main thread fails: the feed must still end on failed, which needs the sim
// thread stopped before that line is written. The test checks that directly (the
// step count must not move after run_headless returns), and captures stdout
// (fd 1) across the whole sim lifetime, teardown included, so a late line shows.
//
// Each case builds a full sim (one sim per process, as in A-tier), so CMake
// registers each as its own ctest entry.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "app/build_env.h"
#include "app/run_headless.h"
#include "cli/options.h"
#include "cli/parse.h"
#include "core/app_config.h"
#include "core/structure_registry.h"
#include "hdf5/executor.h"
#include "probe/probe_instance.h"
#include "probe/probe_observer.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_thread_controller.h"

#include "sim_sync.h"

namespace fs = std::filesystem;

namespace {

// Calls `throw_fn` (which throws) from push() once a sample of step >= k_throw_step
// arrives, i.e. from run_headless's poll loop while the sim is stepping and has
// already written checkpoints.
constexpr std::uint64_t k_throw_step = 2000;
struct throwing_observer final : probe::probe_observer {
    std::function<void()> throw_fn;
    explicit throwing_observer(std::function<void()> r) : throw_fn(std::move(r)) {}
    void push(std::uint64_t step, double) override { if (step >= k_throw_step) throw_fn(); }
    void render() override {}
    [[nodiscard]] std::string_view name() const override { return "throwing"; }
};

// Redirects fd 1 to `path` for its lifetime. Restored on unwind too, so a throw
// escaping run_headless still lets doctest report to the real stdout.
struct stdout_redirect {
    int saved = -1;
    explicit stdout_redirect(const fs::path& path) {
        std::fflush(stdout);
        saved = ::dup(1);
        REQUIRE(saved >= 0);
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        REQUIRE(fd >= 0);
        REQUIRE(::dup2(fd, 1) >= 0);
        ::close(fd);
    }
    ~stdout_redirect() {
        std::fflush(stdout);
        ::dup2(saved, 1);
        ::close(saved);
    }
    stdout_redirect(const stdout_redirect&) = delete;
    stdout_redirect& operator=(const stdout_redirect&) = delete;
};

// Run `fn` with fd 1 redirected to `path`; return what was written.
std::string capture_stdout(const fs::path& path, const std::function<void()>& fn) {
    {
        stdout_redirect r(path);
        fn();
    }
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> progress_lines(const std::string& out) {
    std::vector<std::string> v;
    std::istringstream in(out);
    for (std::string line; std::getline(in, line);)
        if (line.starts_with("@coast/progress ")) v.push_back(line);
    return v;
}

bool has(const std::string& s, const std::string& needle) {
    return s.find(needle) != std::string::npos;
}

// Activate the "activity" probe on the first child element of the first entity
// that has any (an ngm2 neuron in configs/mnist.json) and attach `obs` to it.
void attach_observer(coast::sim_instance& sim, std::unique_ptr<probe::probe_observer> obs) {
    for (auto& e : sim.get_sim_env().iterate_entities()) {
        auto* root = e.structure().root();
        if (!root || root->children.empty()) continue;
        auto* inst = e.probes().activate(root->children.front()->element, "activity");
        REQUIRE(inst);
        inst->add_observer(std::move(obs));
        return;
    }
    FAIL("no entity with child elements to probe");
}

// Run configs/mnist.json headless for 20000 steps with --checkpoint 500, with an
// observer that throws via `throw_fn`. Returns {exit code, captured stdout}.
std::pair<int, std::string> run_with_throw(const std::string& tag,
                                           std::function<void()> throw_fn) {
    const fs::path scratch = fs::temp_directory_path() / ("coast_headless_feed_" + tag);
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch);
    // A copy of the config in scratch, so the default config_base puts the
    // checkpoints under scratch/cfg/snapshots, not into the source tree.
    const fs::path cfg = scratch / "cfg.json";
    fs::copy_file(COAST_BASE_CONFIG, cfg);

    const std::string cfg_s = cfg.string();
    std::vector<std::string> args{"coast", "-n", "20000", "--checkpoint", "500", cfg_s};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    const auto opts = coast::cli::parse(static_cast<int>(argv.size()), argv.data());
    const coast::paths::app_config app_cfg;

    int rc = -1;
    std::uint64_t steps_at_return = 0, steps_after_wait = 0;
    bool          paused_at_return = false;
    hdf5::executor h5;
    const std::string out = capture_stdout(scratch / "stdout.txt", [&] {
        auto sim = coast::app::build_env(opts, app_cfg, h5);
        REQUIRE(sim);
        attach_observer(*sim, std::make_unique<throwing_observer>(std::move(throw_fn)));
        rc = coast::app::run_headless(*sim, opts, app_cfg);
        // The sim thread must be stopped by the time run_headless returns: an
        // unstopped one keeps stepping, and its checkpoint hook then writes
        // lines after phase=failed. "Stopped" is paused, and a full sim-loop
        // iteration — the dispatch where a step would be taken — takes none
        // (tests/support/sim_sync.h), instead of watching the count for 1.5 s.
        auto* ctrl = sim->get_thread_controller();
        REQUIRE(ctrl);
        paused_at_return = ctrl->get_mode() == sim::sim_mode::paused;
        steps_at_return  = ctrl->get_step_count();
        coast_test::sim_iterations(*ctrl);
        steps_after_wait = ctrl->get_step_count();
        sim.reset();   // teardown still captured: nothing may follow phase=failed
    });
    fs::remove_all(scratch, ec);
    // Checked here, not in the lambda: doctest reports there would go to the file.
    CHECK(paused_at_return);
    CHECK(steps_after_wait == steps_at_return);
    return {rc, out};
}

void check_feed_ends_failed(int rc, const std::string& out) {
    CHECK(rc == 1);
    const auto lines = progress_lines(out);
    REQUIRE(!lines.empty());
    CHECK(has(lines.front(), "phase=start "));
    // The throw came mid-run, after checkpoints: the sim thread was emitting.
    CHECK(has(out, "phase=checkpoint "));
    CAPTURE(lines.back());
    CHECK(has(lines.back(), "phase=failed "));
    int terminal = 0;
    for (const auto& l : lines)
        if (has(l, "phase=failed ") || has(l, "phase=done ")) ++terminal;
    CHECK(terminal == 1);
}

} // namespace

TEST_CASE("headless feed: a std::exception after phase=start ends the feed on phase=failed") {
    const auto [rc, out] = run_with_throw("std", [] {
        throw std::runtime_error("injected observer failure");
    });
    check_feed_ends_failed(rc, out);
}

TEST_CASE("headless feed: a non-std exception after phase=start ends the feed on phase=failed") {
    // run_headless is documented not to throw; a non-std exception used to
    // escape it (std::terminate in main).
    const auto [rc, out] = run_with_throw("nonstd", [] { throw 42; });
    check_feed_ends_failed(rc, out);
}
