// ngm_parity_probe — aggregate-parity measurement for any neuron group.
//
// SPEC.md M.0.6: ngm2, ngm25 and ngm3 (CPU and GPU) must compute the same model;
// for the stochastic parts that is judged statistically (M.0.6 b; for ngm3
// CPU vs GPU, M.7.20). ADR-5 once accepted a ~16-25 % lower ngm3-GPU mean on
// the strength of one GPU seed against a CPU seed band. This tool produces the
// per-seed numbers for a proper two-sample comparison: run a config at several
// seed offsets and print, per seed, (a) the group output buffer's final-step
// mean/max/std plus its mean over the last kAvgSteps steps and the input it saw,
// and (b) summary statistics of the group's learned state, read through
// save_state (so every implementation is read the same way; GPU groups read
// back).
//
// The config must hold exactly one neuron group, of any of the five types. The
// learned-state statistics are taken over LOGICAL synapses — (neuron, dendrite,
// segment id, input index) — so the implementations' layouts compare directly:
//   ngm2  per-dendrite SoA; input index advanced by input_inc's low bit;
//   ngm25 CSR arena; explicit input_idx per slot;
//   ngm3  dense [n][d][rep][K] with replica rows; consecutive equal segment ids
//         along rep are one logical synapse.
// (integration.impl_equivalence builds the same view for its fixed shapes; this
// tool reads every dimension from the state tree instead.)
//
// Usage: ngm_parity_probe <config.json> <steps> <perm_threshold> <seed0> [<seed1> ...]
// Output: CSV lines "seed,metric,value" on stdout.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "app/build_env.h"
#include "cli/options.h"
#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#ifdef COAST_WITH_GPU
#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"
#endif
#include "hdf5/executor.h"
#include "logging/mem_state_io.h"
#include "sim_core/io_buffer.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_thread_controller.h"

namespace {

constexpr std::uint64_t kAvgSteps = 200;

// Drive the sim `steps` further, synchronously (as ngm3_noise_floor does).
void run_for(coast::sim_instance& sim, std::uint64_t steps)
{
    auto* c = sim.get_thread_controller();
    const std::uint64_t target = c->get_step_count() + steps;
    c->set_step_limit(target);
    c->set_mode(sim::sim_mode::unlimited);
    while (c->get_step_count() < target && c->get_mode() != sim::sim_mode::paused)
        std::this_thread::sleep_for(std::chrono::milliseconds{2});
    c->set_mode(sim::sim_mode::paused);
    c->set_step_limit(0);
}

std::vector<float> read_buffer(sim::io_buffer& buf)
{
    auto v = buf.cur_read_buffer();
    std::vector<float> out(v.begin(), v.end());
#ifdef COAST_WITH_GPU
    if (out.empty() && buf.size() > 0 && buf.gpu_backed()) {
        out.resize(buf.size());
        if (!buf.read_back_sync(out)) out.clear();
    }
#endif
    return out;
}

struct moments { double mean = 0, max = 0, std = 0; };

template <class T>
moments reduce(const std::vector<T>& v)
{
    moments m;
    if (v.empty()) return m;
    double sum = 0, sq = 0, mx = -std::numeric_limits<double>::infinity();
    for (T x : v) { const double d = static_cast<double>(x); sum += d; sq += d * d; mx = std::max(mx, d); }
    m.mean = sum / static_cast<double>(v.size());
    m.max  = mx;
    m.std  = std::sqrt(std::max(0.0, sq / static_cast<double>(v.size()) - m.mean * m.mean));
    return m;
}

// Typed access to the state tree; a missing field or a changed type throws
// (std::out_of_range / std::bad_variant_access), never reads a default.
template <class T>
const std::vector<T>& field(const logging::mem_group& g, const std::string& name)
{
    return std::get<std::vector<T>>(g.fields.at(name));
}

template <class T>
T scalar(const logging::mem_group& g, const std::string& name)
{
    return std::get<T>(g.fields.at(name));
}

void emit(int seed, const std::string& metric, double value)
{
    std::printf("%d,%s,%.9g\n", seed, metric.c_str(), value);
}

// ---- the group, whatever its type -------------------------------------------

enum class kind_t { ngm2, ngm25, ngm3 };   // the state layout (GPU = its CPU twin's)

struct found_group { kind_t kind; logging::mem_group tree; };

template <class G>
std::optional<logging::mem_group> state_of_single(sim::env& env)
{
    auto v = env.get_entities<G>();
    if (!v.has_value() || (*v)->size() != 1) return std::nullopt;
    logging::mem_state_writer w;
    (**v)[0].save_state(w);
    return w.take_root();
}

std::optional<found_group> group_state(coast::sim_instance& sim)
{
    auto& env = sim.get_sim_env();
    std::vector<found_group> found;
    if (auto t = state_of_single<ngm2::neuron_group_t>(env))  found.push_back({ kind_t::ngm2,  std::move(*t) });
    if (auto t = state_of_single<ngm25::neuron_group_t>(env)) found.push_back({ kind_t::ngm25, std::move(*t) });
    if (auto t = state_of_single<ngm3::neuron_group_t>(env))  found.push_back({ kind_t::ngm3,  std::move(*t) });
#ifdef COAST_WITH_GPU
    if (auto t = state_of_single<ngm25::neuron_group_gpu_t>(env)) found.push_back({ kind_t::ngm25, std::move(*t) });
    if (auto t = state_of_single<ngm3::neuron_group_gpu_t>(env))  found.push_back({ kind_t::ngm3,  std::move(*t) });
#endif
    if (found.size() != 1) return std::nullopt;
    return std::move(found[0]);
}

// ---- the logical view ----------------------------------------------------------

struct logical_syn { std::size_t dendrite; std::uint32_t seg; float p, mm, ah; };

struct logical_view {
    std::vector<logical_syn>   syn;        // every logical synapse; `dendrite` is global
    std::size_t                dendrites = 0;
    std::vector<float>         mx;         // per neuron: running activity max
    std::vector<std::uint64_t> cnt;        // per neuron: learning-step (branch) counter
};

logical_view view_ngm2(const logging::mem_group& root)
{
    logical_view v;
    const auto& neurons = root.groups.at("neurons");
    const auto n_cnt = scalar<std::uint64_t>(root, "neuron_count");
    for (std::uint64_t n = 0; n < n_cnt; ++n) {
        const auto& ng = neurons.groups.at(std::to_string(n));
        v.mx.push_back(scalar<float>(ng, "max_activity"));
        v.cnt.push_back(scalar<std::uint64_t>(ng, "input_count"));
        const auto d_cnt = scalar<std::uint64_t>(ng, "dendrite_count");
        const auto& dends = ng.groups.at("dendrites");
        for (std::uint64_t d = 0; d < d_cnt; ++d, ++v.dendrites) {
            const auto& dg  = dends.groups.at(std::to_string(d));
            const auto& p   = field<float>(dg, "permanence");
            const auto& mm  = field<float>(dg, "mismatch");
            const auto& ah  = field<float>(dg, "adapt_history");
            const auto& seg = field<std::uint16_t>(dg, "segment_idx");
            for (std::size_t k = 0; k < p.size(); ++k)
                v.syn.push_back({ v.dendrites, seg[k], p[k], mm[k], ah[k] });
        }
    }
    return v;
}

logical_view view_ngm25(const logging::mem_group& root)
{
    logical_view v;
    const auto& p   = field<float>(root, "perm");
    const auto& mm  = field<float>(root, "mismatch");
    const auto& ah  = field<float>(root, "adapt_hist");
    const auto& seg = field<std::uint8_t>(root, "seg");
    const auto& off = field<std::uint32_t>(root, "dend_offset");
    v.dendrites = off.size() - 1;
    for (std::size_t d = 0; d < v.dendrites; ++d)
        for (std::size_t s = off[d]; s < off[d + 1]; ++s)
            v.syn.push_back({ d, seg[s], p[s], mm[s], ah[s] });
    v.mx  = field<float>(root, "neuron_mx_act");
    v.cnt = field<std::uint64_t>(root, "input_count");
    return v;
}

logical_view view_ngm3(const logging::mem_group& root)
{
    logical_view v;
    const auto n_cnt = scalar<std::uint64_t>(root, "neuron_cnt");
    const std::size_t rep = std::size_t{1} << scalar<std::uint64_t>(root, "max_branch_level");
    // Banks in the order a neuron's dendrites are counted: apical, then proximal.
    for (const char* bank : { "apical", "proximal" }) {
        const auto it = root.groups.find(bank);
        if (it == root.groups.end()) continue;
        const std::uint64_t d_cnt = scalar<std::uint64_t>(root, std::string(bank) + "_cnt");
        const std::uint64_t k     = scalar<std::uint64_t>(root, std::string(bank) + "_input_size");
        const auto& p   = field<float>(it->second, "permanences");
        const auto& mm  = field<float>(it->second, "mismatch");
        const auto& ah  = field<float>(it->second, "adapt_history");
        const auto& seg = field<std::uint8_t>(it->second, "segment_ids");
        for (std::uint64_t n = 0; n < n_cnt; ++n)
            for (std::uint64_t d = 0; d < d_cnt; ++d) {
                const std::size_t gd = v.dendrites++;
                for (std::uint64_t i = 0; i < k; ++i)
                    for (std::size_t r = 0; r < rep; ++r) {
                        const std::size_t x = ((n * d_cnt + d) * rep + r) * k + i;
                        // a run of equal segment ids along rep is ONE logical synapse
                        if (r > 0 && seg[x] == seg[x - k]) continue;
                        v.syn.push_back({ gd, seg[x], p[x], mm[x], ah[x] });
                    }
            }
    }
    v.mx = field<float>(root, "neuron_mx_act");
    v.cnt.assign(n_cnt, scalar<std::uint64_t>(root, "input_cnt"));
    return v;
}

void emit_learned_state(int seed, const logical_view& v, float tau)
{
    std::vector<float> p, mm, ah;
    std::size_t connected = 0, split = 0;
    std::vector<char> branched(v.dendrites, 0);
    for (const auto& s : v.syn) {
        p.push_back(s.p); mm.push_back(s.mm); ah.push_back(s.ah);
        connected += (s.p > tau);
        if (s.seg != 1) { ++split; branched[s.dendrite] = 1; }
    }
    const double n_syn = static_cast<double>(v.syn.size());
    const auto pm = reduce(p), mmm = reduce(mm);
    emit(seed, "perm_mean",       pm.mean);
    emit(seed, "perm_std",        pm.std);
    emit(seed, "frac_connected",  static_cast<double>(connected) / n_syn);
    emit(seed, "syn_per_dendrite", n_syn / static_cast<double>(v.dendrites));
    emit(seed, "frac_split_syn",  static_cast<double>(split) / n_syn);
    emit(seed, "frac_branched_dendrites",
         static_cast<double>(std::count(branched.begin(), branched.end(), 1)) / static_cast<double>(v.dendrites));
    emit(seed, "adapt_hist_mean", reduce(ah).mean);
    emit(seed, "mismatch_mean",   mmm.mean);
    emit(seed, "mismatch_std",    mmm.std);
    emit(seed, "mx_act_mean",     reduce(v.mx).mean);
    emit(seed, "learn_cnt_mean",  reduce(v.cnt).mean);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <config.json> <steps> <perm_threshold> <seed0> [<seed1> ...]\n", argv[0]);
        return 2;
    }
    const std::string   config = argv[1];
    const std::uint64_t steps  = std::strtoull(argv[2], nullptr, 10);
    const float         tau    = std::strtof(argv[3], nullptr);
    if (steps <= kAvgSteps) { std::fprintf(stderr, "steps must exceed %llu\n", (unsigned long long)kAvgSteps); return 2; }

    std::printf("seed,metric,value\n");
    for (int a = 4; a < argc; ++a) {
        const int seed = std::atoi(argv[a]);
        hdf5::executor h5;
        coast::cli::options opts;
        opts.config_path = config;
        opts.seed_offset = seed_offset_t{ static_cast<std::uint32_t>(seed) };
        auto sim = coast::app::build_env(opts, h5);
        if (!sim) { std::fprintf(stderr, "build_env failed\n"); return 1; }

        // The group's output is the highest-id buffer in these 2-node configs
        // (node 0 = dataset input, node 1 = the neuron group).
        auto& buffers = sim->get_sim_env().get_io_buffers();
        std::size_t group_buf = 0, input_buf = static_cast<std::size_t>(-1);
        for (const auto& [id, b] : buffers) { group_buf = std::max(group_buf, id); input_buf = std::min(input_buf, id); }

        run_for(*sim, steps - kAvgSteps);
        double out_mean_acc = 0;
        for (std::uint64_t s = 0; s < kAvgSteps; ++s) {
            run_for(*sim, 1);
            out_mean_acc += reduce(read_buffer(*buffers.at(group_buf))).mean;
        }
        const auto out = reduce(read_buffer(*buffers.at(group_buf)));
        emit(seed, "out_final_mean", out.mean);
        emit(seed, "out_final_max",  out.max);
        emit(seed, "out_final_std",  out.std);
        emit(seed, "out_mean_last200", out_mean_acc / static_cast<double>(kAvgSteps));
        // The input the group saw (the dataset node's output buffer).
        const auto in = reduce(read_buffer(*buffers.at(input_buf)));
        emit(seed, "in_final_mean", in.mean);
        emit(seed, "in_final_max",  in.max);

        const auto g = group_state(*sim);
        if (!g) { std::fprintf(stderr, "expected exactly one neuron group (ngm2, ngm25 or ngm3)\n"); return 1; }
        switch (g->kind) {
            case kind_t::ngm2:  emit_learned_state(seed, view_ngm2(g->tree),  tau); break;
            case kind_t::ngm25: emit_learned_state(seed, view_ngm25(g->tree), tau); break;
            case kind_t::ngm3:  emit_learned_state(seed, view_ngm3(g->tree),  tau); break;
        }
        std::fflush(stdout);
    }
    return 0;
}
