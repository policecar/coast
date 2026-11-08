// ngm3_noise_floor — Shape-A aggregate noise-floor probe (s12.tolerance-pin).
//
// Runs a CPU config at several RNG seed offsets, and for every io_buffer reports
// the final-step (mean, max, population-std) — the same three aggregates S11's
// coast_compare aggregate mode reduces to. The spread of these aggregates across
// seeds is the intrinsic run-to-run variation of the dynamics: the "noise floor"
// against which a CPU-vs-GPU aggregate gap is judged for the Shape-A parity claim
// (MH3) at checkpoint gamma.
//
// CPU-only config assumed (every io_buffer is io_buffer_cpu, CPU-visible) so no
// GPU readback is needed here — aggregates are computed straight from
// cur_read_buffer().
//
// Usage: ngm3_noise_floor <config.json> <steps> <seed0> [<seed1> ...]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <chrono>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "cli/options.h"
#include "app/build_env.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_thread_controller.h"
#include "sim_core/io_buffer.h"
#include "hdf5/executor.h"

namespace {

// Drive the sim `steps` further, synchronously. Inlined from
// integration_helpers::run_for to avoid pulling doctest into this tool.
void run_for(coast::sim_instance& sim, std::uint64_t steps) {
    auto* c = sim.get_thread_controller();
    const std::uint64_t target = c->get_step_count() + steps;
    c->set_step_limit(target);
    c->set_mode(sim::sim_mode::unlimited);
    while (c->get_step_count() < target && c->get_mode() != sim::sim_mode::paused) {
        std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }
    c->set_mode(sim::sim_mode::paused);
    c->set_step_limit(0);
}

struct agg { double mean, max, std; std::size_t size; };

agg reduce(std::span<const float> v) {
    if (v.empty()) return {0.0, 0.0, 0.0, 0};
    double sum = 0.0, sq = 0.0, mx = -std::numeric_limits<double>::infinity();
    for (float f : v) { double d = f; sum += d; sq += d * d; if (d > mx) mx = d; }
    const double n    = static_cast<double>(v.size());
    const double mean = sum / n;
    const double var  = sq / n - mean * mean;          // population variance
    return { mean, mx, std::sqrt(var < 0.0 ? 0.0 : var), v.size() };
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <config.json> <steps> <seed0> [<seed1> ...]\n";
        return 2;
    }
    const std::string config = argv[1];
    const std::uint64_t steps = std::strtoull(argv[2], nullptr, 10);
    std::vector<int> seeds;
    for (int i = 3; i < argc; ++i) seeds.push_back(std::atoi(argv[i]));

    // buffer id -> per-seed aggregates
    std::map<std::size_t, std::vector<agg>> by_buf;

    for (int seed : seeds) {
        coast::cli::options opts;
        opts.config_path = config;
        opts.seed_offset = seed_offset_t{ static_cast<std::uint32_t>(seed) };
        hdf5::executor h5;
        auto sim = coast::app::build_env(opts, h5);
        if (!sim) { std::cerr << "build_env failed for seed " << seed << "\n"; return 1; }
        run_for(*sim, steps);

        auto& bufs = sim->get_sim_env().get_io_buffers();
        std::vector<std::size_t> ids;
        for (auto& [id, _] : bufs) ids.push_back(id);
        std::sort(ids.begin(), ids.end());
        for (std::size_t id : ids) by_buf[id].push_back(reduce(bufs.at(id)->cur_read_buffer()));
        std::cout << "seed " << seed << " done (" << steps << " steps)\n";
    }

    std::cout << "\n=== aggregate noise floor over seeds";
    for (int s : seeds) std::cout << ' ' << s;
    std::cout << " (config " << config << ", " << steps << " steps) ===\n";
    std::cout << "buffer  size    mean[min..max]            spread%   "
                 "max[min..max]             spread%   std[min..max]             spread%\n";
    for (auto& [id, v] : by_buf) {
        auto span_pct = [](auto sel, const std::vector<agg>& a) {
            double lo = 1e300, hi = -1e300, sum = 0;
            for (auto& x : a) { double d = sel(x); lo = std::min(lo, d); hi = std::max(hi, d); sum += d; }
            double mean = sum / static_cast<double>(a.size());
            double pct  = (std::abs(mean) > 1e-12) ? 100.0 * (hi - lo) / std::abs(mean) : 0.0;
            return std::tuple{lo, hi, pct};
        };
        auto [m_lo, m_hi, m_pct] = span_pct([](const agg& x){ return x.mean; }, v);
        auto [x_lo, x_hi, x_pct] = span_pct([](const agg& x){ return x.max;  }, v);
        auto [s_lo, s_hi, s_pct] = span_pct([](const agg& x){ return x.std;  }, v);
        std::printf("buf_%-3zu %4zu  [%.6g .. %.6g] %7.3f%%  [%.6g .. %.6g] %7.3f%%  [%.6g .. %.6g] %7.3f%%\n",
                    id, v.front().size, m_lo, m_hi, m_pct, x_lo, x_hi, x_pct, s_lo, s_hi, s_pct);
    }
    return 0;
}
