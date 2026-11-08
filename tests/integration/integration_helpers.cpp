#include "integration_helpers.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <unistd.h>   // ::getpid() for scratch-dir uniqueness

#include <doctest.h>

#include "compare.h"
#include "cli/options.h"
#include "app/build_env.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_thread_controller.h"
#include "sim_core/io_buffer.h"
#include "hdf5/executor.h"

namespace coast::test {

namespace {

[[nodiscard]] bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, "1") == 0;
}


} // namespace

scratch_dir_t::scratch_dir_t(std::string_view label) {
    static std::atomic<std::uint64_t> counter{0};
    path_ = std::filesystem::temp_directory_path() /
            (std::string(label) + "_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
    std::filesystem::create_directories(path_);
}

scratch_dir_t::~scratch_dir_t() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
}

void run_for(sim_instance& sim, std::uint64_t steps) {
    auto* controller = sim.get_thread_controller();
    if (!controller) {
        throw std::runtime_error("run_for: sim has no thread controller");
    }
    const std::uint64_t start  = controller->get_step_count();
    const std::uint64_t target = start + steps;

    controller->set_step_limit(target);
    controller->set_mode(sim::sim_mode::unlimited);

    while (controller->get_step_count() < target &&
           controller->get_mode() != sim::sim_mode::paused) {
        const auto now       = controller->get_step_count();
        const auto remaining = (target > now) ? (target - now) : 0;
        auto sleep_ms = std::chrono::milliseconds{100};
        if (remaining < 1000)  sleep_ms = std::chrono::milliseconds{1};
        else if (remaining < 10000) sleep_ms = std::chrono::milliseconds{10};
        std::this_thread::sleep_for(sleep_ms);
    }

    controller->set_mode(sim::sim_mode::paused);
    controller->set_step_limit(0);
}

blob_writer open_blob(const std::filesystem::path& path) {
    return blob_writer{ path };
}

void fingerprint_io_buffers(blob_writer& w, sim_instance& sim) {
    auto& buffers = sim.get_sim_env().get_io_buffers();

    // Deterministic order: sort by buffer ID.
    std::vector<std::size_t> ids;
    ids.reserve(buffers.size());
    for (const auto& [id, _] : buffers) ids.push_back(id);
    std::sort(ids.begin(), ids.end());

    for (std::size_t id : ids) {
        auto& buf = buffers.at(id);
        std::span<const float> values = buf->cur_read_buffer();
        const std::string base = "buffer_" + std::to_string(id);

#ifdef COAST_WITH_GPU
        // GPU-backed buffer with no standing CPU mirror (pure io_buffer_gpu): empty
        // CPU span but a live device read page and a non-zero logical size. Read it
        // back via the agnostic device seam so it fingerprints like any other
        // variant. read_back_sync() routes to gpu_device::readback() of the READ
        // side once env has registered the page (gpu-abstraction-layer F). `staged`
        // owns the data for the rest of this iteration.
        std::vector<float> staged;
        if (values.empty() && buf->size() > 0 && buf->gpu_backed()) {
            staged.resize(buf->size());
            if (buf->read_back_sync(staged)) values = staged;
        }
#endif

        // size() reflects the buffer's logical element count; for a pure-GPU
        // buffer this matches the read-back span (values), for CPU variants the
        // span itself. Use the span we will actually summarize below.
        w.scalar(base + "/size", static_cast<std::int64_t>(values.size()));

        if (values.empty()) {
            w.scalar(base + "/sum",  0.0);
            w.scalar(base + "/mean", 0.0);
            w.scalar(base + "/min",  0.0);
            w.scalar(base + "/max",  0.0);
            continue;
        }

        double sum =  0.0;
        double mn  =  std::numeric_limits<double>::infinity();
        double mx  = -std::numeric_limits<double>::infinity();
        for (float v : values) {
            const double d = static_cast<double>(v);
            sum += d;
            if (d < mn) mn = d;
            if (d > mx) mx = d;
        }
        w.scalar(base + "/sum",  sum);
        w.scalar(base + "/mean", sum / static_cast<double>(values.size()));
        w.scalar(base + "/min",  mn);
        w.scalar(base + "/max",  mx);

        // Promote raw values to f64 (golden format uses doubles uniformly).
        std::vector<double> as_f64(values.begin(), values.end());
        if (as_f64.size() <= 64) {
            w.array(base + "/values", std::span<const double>{ as_f64 });
        } else {
            w.array_binary(base + "/values",
                           std::span<const double>{ as_f64 },
                           base + ".bin");
        }
    }
}

int compare(const std::filesystem::path& blob,
            const std::filesystem::path& goldens_dir,
            const std::filesystem::path& tolerances_path,
            coast::compare::compare_mode mode) {
    if (env_flag("COAST_TEST_CAPTURE_ONLY")) {
        std::cout << "[COAST_TEST_CAPTURE_ONLY=1] skipping compare; blob at "
                  << blob << "\n";
        return 0;
    }
    coast::compare::tolerance_set tols;
    if (!tolerances_path.empty() && std::filesystem::exists(tolerances_path)) {
        tols = coast::compare::tolerance_set::load(tolerances_path);
    }
    auto outcome = coast::compare::compare_against_goldens(blob, goldens_dir, tols, mode);
    coast::compare::render_report(std::cout, outcome, /*verbose=*/false);
    return coast::compare::exit_code_of(outcome);
}

void run_io_buffer_fingerprint_test(
    const char*                  config_path,
    std::uint32_t                seed_offset,
    std::uint64_t                steps,
    const std::filesystem::path& blob_path,
    const std::filesystem::path& goldens_dir,
    coast::compare::compare_mode mode,
    const std::filesystem::path& tolerances_path)
{
    coast::cli::options opts;
    opts.config_path = config_path;
    opts.seed_offset = seed_offset_t{ seed_offset };

    hdf5::executor h5;
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);

    run_for(*sim, steps);

    auto blob = open_blob(blob_path);
    blob.scalar("meta/step_count",
        static_cast<std::int64_t>(sim->get_thread_controller()->get_step_count()));
    fingerprint_io_buffers(blob, *sim);
    blob.close();

    const int code = compare(blob_path, goldens_dir, tolerances_path, mode);
    CHECK(code == 0);
}

} // namespace coast::test
