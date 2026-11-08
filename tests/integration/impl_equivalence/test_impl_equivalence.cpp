// tests/integration/impl_equivalence/test_impl_equivalence.cpp
//
// Differential equivalence test for SPEC.md M.0.6: ngm2, ngm25 and ngm3 (CPU)
// MUST compute the same model — they may compute it differently (layout,
// parallelisation, PRNG, FP evaluation order), but given the same state, input
// and random draws they must produce the same results up to FP evaluation-order
// error.
//
// How the three are made comparable without touching production code:
//   * Same initial state: each group is built, its native state is saved to an
//     in-memory tree (logging::mem_state_writer), the permanences are
//     overwritten from ONE canonical [n][d][i] array, and the tree is loaded
//     back (load_state validates it). Fresh groups are root-only, so the only
//     differing initial field is the permanence draw.
//   * No randomness that matters:
//       - stochastic_win_thres = 1  -> win_act ~ U(mx, mx) == mx exactly, so the
//         winner is the neuron with out + eps >= mx in all three — one neuron,
//         as long as no two outputs tie within eps of the max (a tie would be
//         broken by each implementation's own draw, M.4.4);
//       - neuron noise range [c, c] -> exactly c (c != 0: zero noise changes
//         model semantics, and keeps the group output sum away from 0);
//       - max_softmax inhibition, the default (deterministic);
//       - every input is either exactly 0 or in [max/2, max]. The low-input
//         penalty fires iff U(0, max/2) > input: always at 0 (to float
//         resolution), never at >= max/2, and its amount does not depend on the
//         draw. So the penalty and the low-input learning branch are both
//         exercised, deterministically.
//   * Constants are float-exact (tau = 0.3125, noise = 1/32): ngm2 keeps several
//     parameters as double, the others as float.
//
// State is compared under a common logical key (neuron, dendrite, segment id,
// input index), reconstructed from each implementation's native layout:
//   ngm2  per-dendrite SoA, input index advanced by input_inc's low bit;
//   ngm25 CSR arena, explicit input_idx per synapse;
//   ngm3  dense [n][d][rep][K] with replica rows; consecutive equal segment ids
//         along rep are one logical synapse.
// plus, per neuron, the decaying running activity max the adapt pass's
// learning window reads (ngm2 max_activity, ngm25/ngm3 neuron_mx_act) and the
// branch counter (ngm2/ngm25 per neuron; ngm3 keeps one per group, replicated).
//
// GPU entities: this file is also built as integration.impl_equivalence_gpu
// (COAST_EQUIV_GPU, webgpu builds, label "gpu"), which adds ngm25-GPU and
// ngm3-GPU rigs to every case they can run: each GPU group runs in its own
// GPU-enabled sim::env, fed by a no-op source whose CPU page the rig writes;
// swap_io() before process() uploads that page so the group consumes it on the
// same call (the env's cross-device copy has a one-step lag otherwise). Their
// state trees match the CPU groups', so injection and readout are shared.
// random_index inhibition is CPU-only (M.7.14), so cases that use it run CPU
// rigs only. The stochastic parts (M.0.6 b: draw distributions and
// granularity) need a separate distributional test.
//
// The same rigs carry the M.7.8 checkpoint cases at the end (configuration wins
// on load, the checkpoint's random state is kept, changes are reported), and the
// load_state refusal cases after them (a checkpoint of another shape, or a
// corrupt one, throws in every implementation): those compare each
// implementation with itself, not with the others.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstdint>
#include <cstdio>
#include <format>
#include <functional>
#include <type_traits>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "diag/diag.h"
#include "hd_ngm2/hd_ngm2_cfg.h"
#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "ngm/window_tracking.h"
#include "hd_ngm25/hd_ngm25_cfg.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "hd_ngm3/hd_ngm3_cfg.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "logging/config_record.h"
#include "logging/state_checks.h"
#include "logging/mem_state_io.h"
#include "sim_core/io_buffer.h"
#include "sim_core/io_entity.h"
#include "sim_core/sim_env.h"
#ifdef COAST_EQUIV_GPU
#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"
#endif

namespace {

// ---------------------------------------------------------------------------
// Shared logical configuration
// ---------------------------------------------------------------------------

// The group shape: neuron count n and input width k. Each case states it in its
// knobs (edge cases need n < 2 or k = 1), and every helper receives it from
// there; everything else is fixed.
struct shape_t {
    std::size_t n = 12;   // neurons
    std::size_t k = 48;   // input width
};

constexpr std::size_t D   = 3;         // proximal dendrites per neuron
constexpr std::uint8_t L  = 3;         // max branch depth
constexpr float TAU       = 0.3125f;   // float-exact, off the 0.01 Poisson grid
constexpr float NOISE     = 0.03125f;  // float-exact constant neuron noise

constexpr float TOL       = 1e-5f;     // FP evaluation-order tolerance on [0,1] values

constexpr sim::output_id_t IN_ID = 0;

struct knobs_t {
    shape_t     shape;
    bool        frozen          = false;
    std::size_t branch_interval = 1'000'000'000;   // effectively never
    float       theta_thres     = 2.0f;
    float       common_lr       = 0.0001f;
    float       noise           = NOISE;
    bool        full_inhibition = false;   // random_index, reach 1, strength 1 (see below)
    bool        cpu_only        = false;   // leave the GPU rigs out (see make_rigs)
    seed_offset_t seed_offset   = {};      // the --seed-offset every group is built with
    float       tau             = TAU;     // permanence threshold
    // Randomness the other cases neutralise (defaults): a winner threshold below
    // 1 draws the winner, a noise spread above 0 draws the neuron noise from
    // [noise, noise + spread]. The checkpoint case needs both, so a seed matters.
    float       win_thres       = 1.0f;
    float       noise_spread    = 0.0f;
    float       inhibition_strength = 5.0f;   // max_softmax beta (local_inhibition_strength)
    // Input channels routed into the proximal bank, in declared order; empty
    // means one channel of width shape.k. Widths must sum to shape.k.
    std::vector<std::size_t> channel_widths;
    // Declare those channels in reverse order (the proximal routes; in ngm2 the
    // dendrite's input_channels). Their names, ids and widths stay as they are.
    bool        reverse_declared = false;
};

std::vector<std::size_t> widths(const knobs_t& k)
{
    return k.channel_widths.empty() ? std::vector<std::size_t>{ k.shape.k } : k.channel_widths;
}

// init_io map for the knobs' channels: "ga" for a single channel (ngm2's wiring
// uses that name), "c0", "c1", ... otherwise; ids IN_ID, IN_ID + 1, ...
std::unordered_map<std::string, sim::output_channel_t> channel_map(const knobs_t& k)
{
    const auto w = widths(k);
    std::unordered_map<std::string, sim::output_channel_t> m;
    for (std::size_t c = 0; c < w.size(); ++c)
        m[w.size() == 1 ? std::string("ga") : "c" + std::to_string(c)] =
            sim::output_channel_t{ IN_ID + c, w[c] };
    return m;
}

template <class Route, class Bank>
std::vector<Route> proximal_routes(const knobs_t& k, Bank proximal)
{
    const auto w = widths(k);
    std::vector<Route> r;
    for (std::size_t c = 0; c < w.size(); ++c)
        r.push_back({ w.size() == 1 ? std::string("ga") : "c" + std::to_string(c), proximal });
    if (k.reverse_declared) std::ranges::reverse(r);
    return r;
}

// random_index with reach 1 connects every pair (U[0,1) < 1 always, so the
// matrix values do not matter) and strength 1 gives
//   out_i = clamp(raw_i - sum_{j != i} raw_j, 0, 1),
// which is 0 for every neuron unless one neuron outweighs all the others
// combined — a deterministic way to make the group output sum to 0 while the
// neuron activities stay positive.
inhibition::inhibition_params_t full_inhibition_params()
{
    inhibition::inhibition_params_t ip;
    ip.type        = inhibition::inhibition_type_t::random_index;
    ip.ri_reach    = 1.0;
    ip.ri_strength = 1.0;
    return ip;
}

// ---------------------------------------------------------------------------
// Logical state
// ---------------------------------------------------------------------------

struct syn_key {
    std::size_t   n, d;
    std::uint32_t seg;
    std::size_t   i;
    auto operator<=>(const syn_key&) const = default;
};
struct syn_val { float p, mm, ah; };
struct logical_state {
    std::map<syn_key, syn_val>  syn;   // per logical synapse
    std::vector<float>          mx;    // per neuron: decaying running activity max
    std::vector<std::uint64_t>  cnt;   // per neuron: branch counter (learning steps)
    std::vector<float>          edges; // activity window edges: winner lo/hi, other lo/hi
};

bool has_branched(const logical_state& s)
{
    return std::any_of(s.syn.begin(), s.syn.end(), [](const auto& kv) { return kv.first.seg != 1; });
}

// ---------------------------------------------------------------------------
// mem tree helpers
// ---------------------------------------------------------------------------

template <class T>
std::vector<T>& field(logging::mem_group& g, const std::string& name)
{
    auto it = g.fields.find(name);
    REQUIRE_MESSAGE(it != g.fields.end(), "state field missing: " << name);
    auto* v = std::get_if<std::vector<T>>(&it->second);
    REQUIRE_MESSAGE(v != nullptr, "state field has unexpected type: " << name);
    return *v;
}

template <class T>
T scalar(const logging::mem_group& g, const std::string& name)
{
    auto it = g.fields.find(name);
    REQUIRE_MESSAGE(it != g.fields.end(), "state field missing: " << name);
    const auto* v = std::get_if<T>(&it->second);
    REQUIRE_MESSAGE(v != nullptr, "state field has unexpected type: " << name);
    return *v;
}

logging::mem_group& sub(logging::mem_group& g, const std::string& name)
{
    auto it = g.groups.find(name);
    REQUIRE_MESSAGE(it != g.groups.end(), "state group missing: " << name);
    return it->second;
}

// The four activity window edges, the same subgroup in every implementation.
std::vector<float> edges_of(logging::mem_group& root)
{
    auto& w = sub(root, "activity_window_tracking");
    return { scalar<float>(w, "winner_lower"), scalar<float>(w, "winner_upper"),
             scalar<float>(w, "other_lower"),  scalar<float>(w, "other_upper") };
}

template <class G>
logging::mem_group save_tree(const G& grp)
{
    logging::mem_state_writer w;
    grp.save_state(w);
    return w.take_root();
}

// Loads as state_logger does: a group without a configuration record is noted
// on the reader and reported once the load is done.
template <class G>
void load_tree(G& grp, const logging::mem_group& root)
{
    logging::mem_state_reader r(root, logging::k_format_version);
    grp.load_state(r);
    logging::report_unrecorded_configs(r);
}

// ---------------------------------------------------------------------------
// One rig per implementation, behind a common interface
// ---------------------------------------------------------------------------

using perm_init_t = std::vector<float>;   // canonical [n][d][i], size sh.n * D * sh.k

std::size_t pidx(const shape_t& sh, std::size_t n, std::size_t d, std::size_t i) { return (n * D + d) * sh.k + i; }

// A branched logical state, the same in every implementation: per (n, d), every
// input i % 3 == 0 is split into segments 2 and 3 (permanences p2, p3), every
// other input stays on the root segment 1 (permanence p1). mismatch and adapt
// history are 0. In ngm3 segment 2 covers the lower half of the replica rows,
// segment 3 the upper half; ngm2 and ngm25 store the lower clone first.
struct branched_state {
    std::vector<float> p1, p2, p3;   // each [n][d][i] (pidx)
    static bool split(std::size_t i) { return i % 3 == 0; }
};

struct rig_base {
    std::string                          name;
    shape_t                              shape;   // the shape the group was built with
    std::unique_ptr<sim::io_buffer_cpu>  out;
    // Rigs with the same non-empty family draw the SAME random numbers by design
    // (ngm25 CPU and GPU share one structural-key hash, M.7.22), so their
    // stochastic outputs must agree step by step, not just in distribution.
    std::string                          draw_family;

    rig_base(std::string nm, const shape_t& sh) : name(std::move(nm)), shape(sh) {}
    virtual ~rig_base() = default;

    virtual sim::io_entity& entity() = 0;
    virtual void            inject(const perm_init_t& p) = 0;
    virtual void            inject_branched(const branched_state& b) = 0;
    virtual logical_state   state() = 0;

    using inputs_t = std::vector<std::unique_ptr<sim::io_buffer_cpu>>;

    // CPU rigs: the group reads the trio's input pages and writes `out`.
    virtual void wire(const inputs_t& ins)
    {
        out = std::make_unique<sim::io_buffer_cpu>(entity().get_outp_size());
        for (std::size_t c = 0; c < ins.size(); ++c)
            entity().set_input_buffer(IN_ID + c, *ins[c]);
        entity().set_output_buffer(*out);
    }

    virtual std::vector<float> step()
    {
        auto r = entity().process();
        REQUIRE_MESSAGE(r.has_value(), name << ": process() reported a fault");
        auto o = out->cur_write_buffer();
        return { o.begin(), o.end() };
    }
};

// ---- where a rig's group lives ----------------------------------------------

template <class G> inline constexpr bool is_gpu_group_v = false;

#ifdef COAST_EQUIV_GPU
template <> inline constexpr bool is_gpu_group_v<ngm25::neuron_group_gpu_t> = true;
template <> inline constexpr bool is_gpu_group_v<ngm3::neuron_group_gpu_t>  = true;

// Input channel of a GPU rig's env. Its CPU write page is filled by the rig
// before each step; process() is a no-op.
struct feed_source final : sim::io_entity {
    sim::output_id_t id;
    std::size_t      width;
    feed_source(sim::output_id_t i, std::size_t w) : id(i), width(w) {}
    void set_output_buffer(sim::io_buffer&) override {}
    void set_input_buffer(sim::output_id_t, sim::io_buffer&) override {}
    sim::step_result process() override { return {}; }
    [[nodiscard]] sim::output_id_t get_outp_id() const override { return id; }
    [[nodiscard]] std::size_t get_outp_size() const override { return width; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

// The group's output id inside a GPU rig's env: clear of the input channel ids
// IN_ID, IN_ID + 1, ... (the CPU rigs need no env, so they keep id 1).
constexpr sim::output_id_t GPU_GROUP_ID = 100;

// Consumer of the group's output, so the env wires it as for a real run.
struct out_sink final : sim::io_entity {
    std::array<std::size_t, 1> in{ GPU_GROUP_ID };
    void set_output_buffer(sim::io_buffer&) override {}
    void set_input_buffer(sim::output_id_t, sim::io_buffer&) override {}
    sim::step_result process() override { return {}; }
    [[nodiscard]] sim::output_id_t get_outp_id() const override { return 1000; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 1; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return in; }
};

// One step of a GPU rig: copy this step's input into the env's source pages,
// swap_io() (uploads them; the group reads them on the next process()), then
// process(). Returns this step's post-inhibition output.
template <class G>
std::vector<float> gpu_step(sim::env& env, G& g, const rig_base::inputs_t& ins, const std::string& name)
{
    for (std::size_t c = 0; c < ins.size(); ++c) {
        const auto src = ins[c]->cur_read_buffer();
        auto buf = env.get_io_buffer(IN_ID + c);
        REQUIRE_MESSAGE(buf.has_value(), name << ": no env buffer for input channel " << c);
        auto dst = (*buf)->cur_write_buffer();
        REQUIRE(dst.size() == src.size());
        std::copy(src.begin(), src.end(), dst.begin());
    }
    REQUIRE_MESSAGE(env.swap_io().has_value(), name << ": swap_io() reported a fault");
    REQUIRE_MESSAGE(env.process().has_value(), name << ": process() reported a fault");
    return g.readback_group_outp();
}
#endif

// Build the group with the rig's knobs: owned directly (CPU), or inside a
// GPU-enabled env with one feed_source per input channel (GPU).
template <class G, class CS>
G* make_group(const CS& cs, const knobs_t& k, std::unique_ptr<G>& owned, std::unique_ptr<sim::env>& env)
{
    if constexpr (is_gpu_group_v<G>) {
#ifdef COAST_EQUIV_GPU
        env = std::make_unique<sim::env>();
        const auto w = widths(k);
        for (std::size_t c = 0; c < w.size(); ++c)
            env->emplace_back<feed_source>(IN_ID + c, w[c]);
        env->emplace_back<G>(cs, "g", GPU_GROUP_ID, k.seed_offset);
        env->emplace_back<out_sink>();
        env->enable_gpu();
        auto ents = env->get_entities<G>();
        REQUIRE(ents.has_value());
        G* g = &(*ents.value())[0];
        g->init_io(channel_map(k));
        env->init_io_buffers();
        return g;
#endif
    } else {
        owned = std::make_unique<G>(cs, "g", /*output_id=*/1, k.seed_offset);
        owned->init_io(channel_map(k));
        return owned.get();
    }
}

// ---- ngm2 -----------------------------------------------------------------

struct rig_ngm2 final : rig_base {
    ngm2::cfg_store                        cs;
    std::unique_ptr<ngm2::neuron_group_t>  owned;
    std::unique_ptr<sim::env>              env;   // unused: ngm2 has no GPU group
    ngm2::neuron_group_t*                  g = nullptr;
    // Wiring that must not matter (M.7.5): a dendrite's channel order is its
    // declared input_channels, which this flag leaves alone. It lists the
    // channel mappings in reverse, gives channel c the id IN_ID + 50 - c
    // instead of IN_ID + c, and hands the buffers over last channel first.
    bool                                   scrambled = false;
    std::size_t                            n_chan    = 1;

    sim::output_id_t chan_id(std::size_t c) const { return scrambled ? IN_ID + 50 - c : IN_ID + c; }

    // k.reverse_declared reverses only the dendrite's OWN declared order
    // (dp.input_channels), independently of scrambled: every channel name this
    // rig ever declares is already ascending ("da0", "da1", ...), so a mutation
    // that sorted the declared names before walking them would go undetected
    // without a case whose true declared order is not already sorted.
    //
    // swap_group_targets / swap_neuron_targets (n_chan == 2 only): swap the
    // to_channel of the group's or the neuron's own two channel_mappings
    // entries, e.g. {c0->na0, c1->na1} becomes {c0->na1, c1->na0}. Every
    // declared name at every level, and every width, stays exactly as the
    // unswapped rig's -- only which external channel reaches which dendrite
    // position changes. A fix that records only the dendrite's own, innermost
    // channel key (not the group-level name it resolves to, SPEC.md M.7.5)
    // cannot tell this apart from the unswapped rig.
    explicit rig_ngm2(const knobs_t& k, bool scramble = false, std::string nm = "ngm2",
                      bool swap_group_targets = false, bool swap_neuron_targets = false)
        : rig_base(std::move(nm), k.shape), scrambled(scramble), n_chan(widths(k).size())
    {
        // One declared channel per input channel, in the order the knobs give
        // them: the group's channel names are channel_map's ("ga", or "c0",
        // "c1", ...), the neuron's and dendrite's "na"/"da" or "na0"/"da0", ...
        auto name = [&](const char* level, std::size_t c) {
            return n_chan == 1 ? std::string(level) : level + std::to_string(c);
        };
        ngm2::dendrite_params_t dp;
        ngm2::neuron_params_t   np;
        ngm2::neuron_group_params_t gp;
        for (std::size_t c = 0; c < n_chan; ++c) {
            const std::string gc = n_chan == 1 ? std::string("ga") : "c" + std::to_string(c);
            dp.input_channels.push_back(name("da", c));
            np.input_channels.push_back(name("na", c));
            gp.input_channels.push_back(gc);
            np.channel_mappings.push_back({ name("na", c), "p", name("da", c) });
            gp.channel_mappings.push_back({ gc, "pyr", name("na", c) });
        }
        if (k.reverse_declared) std::ranges::reverse(dp.input_channels);
        if (scrambled) {
            std::ranges::reverse(np.channel_mappings);
            std::ranges::reverse(gp.channel_mappings);
        }
        if (swap_group_targets) {
            REQUIRE(n_chan == 2);
            std::swap(gp.channel_mappings[0].to_channel, gp.channel_mappings[1].to_channel);
        }
        if (swap_neuron_targets) {
            REQUIRE(n_chan == 2);
            std::swap(np.channel_mappings[0].to_channel, np.channel_mappings[1].to_channel);
        }

        dp.type                            = ngm2::dendrite_type_t::proximal;
        // As a config gives it: ngm2 parses the written decimal into a double,
        // the others into a float (0.3 -> 0.3 and 0.3f, which differ).
        dp.permanence_threshold            = std::stod(std::format("{}", k.tau));
        dp.max_branch_level                = L;
        dp.default_accumulated_theta_thres = k.theta_thres;
        cs.dendrites["prox"] = dp;

        np.dendrites                  = { { "p", "prox", D } };
        np.default_branch_interval    = k.branch_interval;
        np.default_neuron_noise_range = { k.noise, k.noise + k.noise_spread };
        cs.neurons["n"] = np;

        gp.neurons                      = { { "pyr", "n", k.shape.n } };
        gp.default_stochastic_win_thres = k.win_thres;
        gp.default_local_inhibition_strength = k.inhibition_strength;
        gp.default_common_learning_rate = k.common_lr;
        cs.frozen                       = k.frozen;   // the run's
        if (k.full_inhibition) { cs.inhibitions["ri"] = full_inhibition_params(); gp.inhibition_cfg = "ri"; }
        cs.neuron_groups["g"] = gp;

        owned = std::make_unique<ngm2::neuron_group_t>(cs, "g", /*output_id=*/1, k.seed_offset);
        auto m = channel_map(k);
        for (auto& oc : m | std::views::values) oc.first = chan_id(oc.first - IN_ID);
        owned->init_io(m);
        g = owned.get();
    }

    void wire(const inputs_t& ins) override
    {
        REQUIRE(ins.size() == n_chan);
        out = std::make_unique<sim::io_buffer_cpu>(g->get_outp_size());
        for (std::size_t j = 0; j < ins.size(); ++j) {
            const std::size_t c = scrambled ? ins.size() - 1 - j : j;
            g->set_input_buffer(chan_id(c), *ins[c]);
        }
        g->set_output_buffer(*out);
    }

    sim::io_entity& entity() override { return *g; }

    void inject(const perm_init_t& p) override
    {
        auto root = save_tree(*g);
        auto& neurons = sub(root, "neurons");
        for (std::size_t n = 0; n < shape.n; ++n) {
            auto& dends = sub(sub(neurons, std::to_string(n)), "dendrites");
            for (std::size_t d = 0; d < D; ++d) {
                auto& perm = field<float>(sub(dends, std::to_string(d)), "permanence");
                REQUIRE_MESSAGE(perm.size() == shape.k, "ngm2: fresh dendrite is not root-only");
                for (std::size_t i = 0; i < shape.k; ++i) perm[i] = p[pidx(shape, n, d, i)];
            }
        }
        load_tree(*g, root);
    }

    void inject_branched(const branched_state& b) override
    {
        auto root = save_tree(*g);
        auto& neurons = sub(root, "neurons");
        for (std::size_t n = 0; n < shape.n; ++n) {
            auto& dends = sub(sub(neurons, std::to_string(n)), "dendrites");
            for (std::size_t d = 0; d < D; ++d) {
                auto& dg = sub(dends, std::to_string(d));
                std::vector<float> perm, mm, ah;
                std::vector<std::uint16_t> seg;
                std::vector<std::uint8_t> inc;
                for (std::size_t i = 0; i < shape.k; ++i) {
                    if (branched_state::split(i)) {   // lower clone first, advance on the last
                        perm.push_back(b.p2[pidx(shape, n, d, i)]); seg.push_back(2); inc.push_back(0);
                        perm.push_back(b.p3[pidx(shape, n, d, i)]); seg.push_back(3); inc.push_back(1);
                    } else {
                        perm.push_back(b.p1[pidx(shape, n, d, i)]); seg.push_back(1); inc.push_back(1);
                    }
                }
                mm.assign(perm.size(), 0.0f); ah.assign(perm.size(), 0.0f);
                dg.fields["permanence"] = perm;   dg.fields["mismatch"] = mm;
                dg.fields["adapt_history"] = ah;  dg.fields["segment_idx"] = seg;
                dg.fields["input_inc"] = inc;
            }
        }
        load_tree(*g, root);
    }

    logical_state state() override
    {
        logical_state s;
        auto root = save_tree(*g);
        auto& neurons = sub(root, "neurons");
        for (std::size_t n = 0; n < shape.n; ++n) {
            auto& ng    = sub(neurons, std::to_string(n));
            s.mx.push_back(scalar<float>(ng, "max_activity"));
            s.cnt.push_back(scalar<std::uint64_t>(ng, "input_count"));
            auto& dends = sub(ng, "dendrites");
            for (std::size_t d = 0; d < D; ++d) {
                auto& dg   = sub(dends, std::to_string(d));
                auto& perm = field<float>(dg, "permanence");
                auto& mm   = field<float>(dg, "mismatch");
                auto& ah   = field<float>(dg, "adapt_history");
                auto& seg  = field<std::uint16_t>(dg, "segment_idx");
                auto& inc  = field<std::uint8_t>(dg, "input_inc");
                std::size_t i = 0;
                for (std::size_t k = 0; k < perm.size(); ++k) {
                    s.syn[{ n, d, seg[k], i }] = { perm[k], mm[k], ah[k] };
                    i += inc[k] & 1u;
                }
                REQUIRE_MESSAGE(i == shape.k, "ngm2: input_inc does not walk all K inputs");
            }
        }
        s.edges = edges_of(root);
        return s;
    }
};

// ---- ngm25 ----------------------------------------------------------------

template <class G>
struct rig_ngm25_t final : rig_base {
    ngm25::cfg_store                        cs;
    std::unique_ptr<G>                      owned;
    std::unique_ptr<sim::env>               env;
    G*                                      g = nullptr;
    const inputs_t*                         gpu_ins = nullptr;

    explicit rig_ngm25_t(const knobs_t& k, std::string nm = "ngm25") : rig_base(std::move(nm), k.shape)
    {
        draw_family = "ngm25";
        ngm25::neuron_group_params_t p;
        p.neuron_cnt              = k.shape.n;
        p.apical_cnt              = 0;
        p.proximal_cnt            = D;
        p.max_branch_level        = L;
        p.permanence_threshold    = k.tau;
        p.stochastic_win_thres    = k.win_thres;
        p.local_inhibition_strength = k.inhibition_strength;
        p.common_learning_rate    = k.common_lr;
        p.neuron_noise_range      = { k.noise, k.noise + k.noise_spread };
        p.branch_interval         = k.branch_interval;
        p.accumulated_theta_thres = k.theta_thres;
        cs.frozen                 = k.frozen;   // the run's
        if (k.full_inhibition) { cs.inhibitions["ri"] = full_inhibition_params(); p.inhibition_cfg = "ri"; }
        p.inputs                  = proximal_routes<ngm25::input_route_t>(k, ngm25::bank_t::proximal);
        cs.neuron_groups["g"] = p;

        g = make_group(cs, k, owned, env);
    }

    sim::io_entity& entity() override { return *g; }

    void wire(const inputs_t& ins) override
    {
        if constexpr (is_gpu_group_v<G>) gpu_ins = &ins; else rig_base::wire(ins);
    }
    std::vector<float> step() override
    {
#ifdef COAST_EQUIV_GPU
        if constexpr (is_gpu_group_v<G>) return gpu_step(*env, *g, *gpu_ins, name);
#endif
        return rig_base::step();
    }

    void inject(const perm_init_t& p) override
    {
        auto root = save_tree(*g);
        auto& perm = field<float>(root, "perm");
        auto& idx  = field<std::uint16_t>(root, "input_idx");
        auto& seg  = field<std::uint8_t>(root, "seg");
        auto& off  = field<std::uint32_t>(root, "dend_offset");
        for (std::size_t n = 0; n < shape.n; ++n)
            for (std::size_t d = 0; d < D; ++d) {
                const std::size_t gd = n * D + d;
                REQUIRE_MESSAGE(off[gd + 1] - off[gd] == shape.k, "ngm25: fresh dendrite is not root-only");
                for (std::size_t i = 0; i < shape.k; ++i) {
                    const std::size_t slot = off[gd] + i;
                    REQUIRE(idx[slot] == i);
                    REQUIRE(seg[slot] == 1);
                    perm[slot] = p[pidx(shape, n, d, i)];
                }
            }
        load_tree(*g, root);
    }

    void inject_branched(const branched_state& b) override
    {
        auto root = save_tree(*g);
        std::vector<float> perm, mm, ah;
        std::vector<std::uint16_t> idx;
        std::vector<std::uint8_t> seg;
        std::vector<std::uint32_t> off{ 0 };
        for (std::size_t n = 0; n < shape.n; ++n)
            for (std::size_t d = 0; d < D; ++d) {
                for (std::size_t i = 0; i < shape.k; ++i) {   // input-major, clones adjacent, lower first
                    if (branched_state::split(i)) {
                        perm.push_back(b.p2[pidx(shape, n, d, i)]); idx.push_back(static_cast<std::uint16_t>(i)); seg.push_back(2);
                        perm.push_back(b.p3[pidx(shape, n, d, i)]); idx.push_back(static_cast<std::uint16_t>(i)); seg.push_back(3);
                    } else {
                        perm.push_back(b.p1[pidx(shape, n, d, i)]); idx.push_back(static_cast<std::uint16_t>(i)); seg.push_back(1);
                    }
                }
                off.push_back(static_cast<std::uint32_t>(perm.size()));
            }
        mm.assign(perm.size(), 0.0f); ah.assign(perm.size(), 0.0f);
        root.fields["perm"] = perm;        root.fields["input_idx"] = idx;
        root.fields["seg"] = seg;          root.fields["adapt_hist"] = ah;
        root.fields["mismatch"] = mm;      root.fields["dend_offset"] = off;
        load_tree(*g, root);
    }

    logical_state state() override
    {
        logical_state s;
        auto root = save_tree(*g);
        auto& perm = field<float>(root, "perm");
        auto& mm   = field<float>(root, "mismatch");
        auto& ah   = field<float>(root, "adapt_hist");
        auto& idx  = field<std::uint16_t>(root, "input_idx");
        auto& seg  = field<std::uint8_t>(root, "seg");
        auto& off  = field<std::uint32_t>(root, "dend_offset");
        for (std::size_t n = 0; n < shape.n; ++n)
            for (std::size_t d = 0; d < D; ++d) {
                const std::size_t gd = n * D + d;
                for (std::size_t slot = off[gd]; slot < off[gd + 1]; ++slot)
                    s.syn[{ n, d, seg[slot], idx[slot] }] = { perm[slot], mm[slot], ah[slot] };
            }
        s.mx = field<float>(root, "neuron_mx_act");
        s.cnt = field<std::uint64_t>(root, "input_count");
        s.edges = edges_of(root);
        return s;
    }
};

// ---- ngm3 (CPU) -----------------------------------------------------------

template <class G>
struct rig_ngm3_t final : rig_base {
    ngm3::cfg_store                        cs;
    std::unique_ptr<G>                     owned;
    std::unique_ptr<sim::env>              env;
    G*                                     g = nullptr;
    const inputs_t*                        gpu_ins = nullptr;
    static constexpr std::size_t REP = std::size_t{1} << L;

    explicit rig_ngm3_t(const knobs_t& k, std::string nm = "ngm3") : rig_base(std::move(nm), k.shape)
    {
        ngm3::neuron_group_params_t p;
        p.neuron_cnt              = k.shape.n;
        p.apical_cnt              = 0;
        p.proximal_cnt            = D;
        p.max_branch_level   = L;
        p.permanence_threshold    = k.tau;
        p.stochastic_win_thres    = k.win_thres;
        p.local_inhibition_strength = k.inhibition_strength;
        p.common_learning_rate    = k.common_lr;
        p.neuron_noise_range      = { k.noise, k.noise + k.noise_spread };
        p.branch_interval         = k.branch_interval;
        p.accumulated_theta_thres = k.theta_thres;
        cs.frozen                 = k.frozen;   // the run's
        if (k.full_inhibition) { cs.inhibitions["ri"] = full_inhibition_params(); p.inhibition_cfg = "ri"; }
        p.inputs                  = proximal_routes<ngm3::input_route_t>(k, ngm3::bank_t::proximal);
        cs.neuron_groups["g"] = p;

        g = make_group(cs, k, owned, env);
    }

    void wire(const inputs_t& ins) override
    {
        if constexpr (is_gpu_group_v<G>) gpu_ins = &ins; else rig_base::wire(ins);
    }
    std::vector<float> step() override
    {
#ifdef COAST_EQUIV_GPU
        if constexpr (is_gpu_group_v<G>) return gpu_step(*env, *g, *gpu_ins, name);
#endif
        return rig_base::step();
    }

    std::size_t didx(std::size_t n, std::size_t d, std::size_t r, std::size_t i) const
    {
        return ((n * D + d) * REP + r) * shape.k + i;
    }

    sim::io_entity& entity() override { return *g; }

    void inject(const perm_init_t& p) override
    {
        auto root = save_tree(*g);
        auto& bank = sub(root, "proximal");
        auto& perm = field<float>(bank, "permanences");
        auto& seg  = field<std::uint8_t>(bank, "segment_ids");
        REQUIRE(perm.size() == shape.n * D * REP * shape.k);
        for (std::size_t n = 0; n < shape.n; ++n)
            for (std::size_t d = 0; d < D; ++d)
                for (std::size_t r = 0; r < REP; ++r)
                    for (std::size_t i = 0; i < shape.k; ++i) {
                        REQUIRE(seg[didx(n, d, r, i)] == 1);
                        perm[didx(n, d, r, i)] = p[pidx(shape, n, d, i)];
                    }
        load_tree(*g, root);
    }

    void inject_branched(const branched_state& b) override
    {
        auto root = save_tree(*g);
        auto& bank = sub(root, "proximal");
        auto& perm = field<float>(bank, "permanences");
        auto& seg  = field<std::uint8_t>(bank, "segment_ids");
        for (std::size_t n = 0; n < shape.n; ++n)
            for (std::size_t d = 0; d < D; ++d)
                for (std::size_t i = 0; i < shape.k; ++i)
                    for (std::size_t r = 0; r < REP; ++r) {
                        const std::size_t x = didx(n, d, r, i);
                        if (branched_state::split(i)) {
                            const bool upper = r >= REP / 2;   // seg 2: lower half of rows, seg 3: upper half
                            seg[x]  = upper ? 3 : 2;
                            perm[x] = upper ? b.p3[pidx(shape, n, d, i)] : b.p2[pidx(shape, n, d, i)];
                        } else {
                            seg[x]  = 1;
                            perm[x] = b.p1[pidx(shape, n, d, i)];
                        }
                    }
        load_tree(*g, root);
    }

    logical_state state() override
    {
        logical_state s;
        auto root = save_tree(*g);
        auto& bank = sub(root, "proximal");
        auto& perm = field<float>(bank, "permanences");
        auto& mm   = field<float>(bank, "mismatch");
        auto& ah   = field<float>(bank, "adapt_history");
        auto& seg  = field<std::uint8_t>(bank, "segment_ids");
        for (std::size_t n = 0; n < shape.n; ++n)
            for (std::size_t d = 0; d < D; ++d)
                for (std::size_t i = 0; i < shape.k; ++i)
                    for (std::size_t r = 0; r < REP; ++r) {
                        // a run of equal segment ids along rep is ONE logical synapse
                        if (r > 0 && seg[didx(n, d, r, i)] == seg[didx(n, d, r - 1, i)]) continue;
                        const std::size_t x = didx(n, d, r, i);
                        s.syn[{ n, d, seg[x], i }] = { perm[x], mm[x], ah[x] };
                    }
        s.mx = field<float>(root, "neuron_mx_act");
        s.cnt.assign(shape.n, scalar<std::uint64_t>(root, "input_cnt"));
        s.edges = edges_of(root);
        return s;
    }
};

using rig_ngm25 = rig_ngm25_t<ngm25::neuron_group_t>;
using rig_ngm3  = rig_ngm3_t<ngm3::neuron_group_t>;
#ifdef COAST_EQUIV_GPU
using rig_ngm25_gpu = rig_ngm25_t<ngm25::neuron_group_gpu_t>;
using rig_ngm3_gpu  = rig_ngm3_t<ngm3::neuron_group_gpu_t>;
#endif

// ---------------------------------------------------------------------------
// Inputs and initial state
// ---------------------------------------------------------------------------

// Every value is exactly 0 or in [0.5, 1.0] (>= max/2 for any max <= 1), so the
// low-input penalty outcome is deterministic. Six prototypes over the shape's
// k inputs, with value jitter, so different neurons win.
class input_source {
public:
    input_source(const shape_t& sh, std::uint32_t seed) : k_(sh.k), rng_(seed)
    {
        std::bernoulli_distribution on(0.35);
        for (auto& proto : protos_) {
            proto.assign(k_, false);
            std::size_t active = 0;
            for (std::size_t i = 0; i < k_; ++i) active += (proto[i] = on(rng_));
            if (active == 0) proto[0] = true;   // never an all-zero input (M.3.10)
        }
    }

    std::vector<float> next()
    {
        std::uniform_int_distribution<std::size_t> pick(0, protos_.size() - 1);
        std::uniform_real_distribution<float>      val(0.5f, 1.0f);
        const auto& proto = protos_[pick(rng_)];
        std::vector<float> v(k_, 0.0f);
        for (std::size_t i = 0; i < k_; ++i)
            if (proto[i]) v[i] = val(rng_);
        return v;
    }

private:
    std::size_t                       k_;
    std::mt19937                      rng_;
    std::vector<std::vector<bool>>    protos_ = std::vector<std::vector<bool>>(6);
};

perm_init_t canonical_permanences(const shape_t& sh, std::uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.05f, 0.60f);
    perm_init_t p(sh.n * D * sh.k);
    for (auto& x : p) {
        do { x = u(rng); } while (std::fabs(x - TAU) < 1e-4f);   // keep clear of p == tau
    }
    return p;
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

// Every implementation the knobs allow, unwired. The GPU rigs (GPU build only) are left out
// where random_index inhibition is used, as it is CPU-only (M.7.14), and where
// a case asks for cpu_only.
std::vector<std::unique_ptr<rig_base>> make_rigs(const knobs_t& k)
{
    std::vector<std::unique_ptr<rig_base>> rigs;
    rigs.push_back(std::make_unique<rig_ngm2>(k));
    rigs.push_back(std::make_unique<rig_ngm25>(k));
    rigs.push_back(std::make_unique<rig_ngm3>(k));
#ifdef COAST_EQUIV_GPU
    if (!k.full_inhibition && !k.cpu_only) {
        rigs.push_back(std::make_unique<rig_ngm25_gpu>(k, "ngm25-gpu"));
        rigs.push_back(std::make_unique<rig_ngm3_gpu>(k, "ngm3-gpu"));
    }
#endif
    return rigs;
}

struct trio {
    shape_t                                          shape;  // every rig's shape
    std::vector<std::unique_ptr<sim::io_buffer_cpu>> ins;    // one per input channel
    std::vector<std::size_t>                         chan;   // channel widths
    std::vector<std::unique_ptr<rig_base>>           rigs;

    trio(const knobs_t& k, const perm_init_t& p, std::vector<std::unique_ptr<rig_base>> rs)
        : shape(k.shape), chan(widths(k)), rigs(std::move(rs))
    {
        for (std::size_t w : chan) ins.push_back(std::make_unique<sim::io_buffer_cpu>(w));
        for (auto& r : rigs) { r->wire(ins); r->inject(p); }
    }
    trio(const knobs_t& k, const perm_init_t& p) : trio(k, p, make_rigs(k)) {}
    trio(const knobs_t& k, std::uint32_t perm_seed) : trio(k, canonical_permanences(k.shape, perm_seed)) {}

    std::vector<std::vector<float>> step(const std::vector<float>& x)
    {
        std::size_t off = 0;
        for (std::size_t c = 0; c < ins.size(); ++c) {
            auto w = ins[c]->cur_write_buffer();
            std::copy(x.begin() + off, x.begin() + off + chan[c], w.begin());
            ins[c]->swap_buffer();
            off += chan[c];
        }
        std::vector<std::vector<float>> outs;
        for (auto& r : rigs) outs.push_back(r->step());
        return outs;
    }
};

// |a - b|, where a NaN on either side counts as an infinite difference (two
// NaNs as none). std::max(m, NaN) would silently drop it otherwise.
float absdiff(float a, float b)
{
    const bool na = std::isnan(a), nb = std::isnan(b);
    if (na || nb) return (na && nb) ? 0.0f : INFINITY;
    return std::fabs(a - b);
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    REQUIRE(a.size() == b.size());
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, absdiff(a[i], b[i]));
    return m;
}

bool all_finite(const std::vector<float>& v)
{
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}
bool all_finite(const logical_state& s)
{
    return all_finite(s.mx) && std::all_of(s.syn.begin(), s.syn.end(), [](const auto& kv) {
        return std::isfinite(kv.second.p) && std::isfinite(kv.second.mm) && std::isfinite(kv.second.ah); });
}

struct state_diff {
    bool        same_keys = true;
    std::size_t only_a = 0, only_b = 0;
    float       dp = 0, dmm = 0, dah = 0;
    float       dmx = 0;   // per-neuron running activity max
    float       dcnt = 0;  // per-neuron branch counter
    float       dedge = 0; // activity window edges
    float worst() const { return std::max({ dp, dmm, dah, dmx, dcnt, dedge }); }
};

state_diff compare(const logical_state& a, const logical_state& b)
{
    state_diff r;
    for (const auto& [k, va] : a.syn) {
        auto it = b.syn.find(k);
        if (it == b.syn.end()) { r.same_keys = false; ++r.only_a; continue; }
        r.dp  = std::max(r.dp,  absdiff(va.p,  it->second.p));
        r.dmm = std::max(r.dmm, absdiff(va.mm, it->second.mm));
        r.dah = std::max(r.dah, absdiff(va.ah, it->second.ah));
    }
    for (const auto& kv : b.syn)
        if (!a.syn.contains(kv.first)) { r.same_keys = false; ++r.only_b; }
    r.dmx = max_abs_diff(a.mx, b.mx);   // REQUIREs one entry per neuron on both sides
    r.dedge = max_abs_diff(a.edges, b.edges);
    REQUIRE(a.cnt.size() == b.cnt.size());
    for (std::size_t n = 0; n < a.cnt.size(); ++n)
        r.dcnt = std::max(r.dcnt, static_cast<float>(a.cnt[n] > b.cnt[n] ? a.cnt[n] - b.cnt[n]
                                                                         : b.cnt[n] - a.cnt[n]));
    return r;
}

// Per-pair record of where two implementations first stop agreeing.
struct pair_report {
    std::string a, b;
    long        first_out_step   = -1;   // first step whose output differs by > TOL
    long        first_state_step = -1;   // first step whose logical state differs by > TOL
    float       max_out = 0;
    state_diff  first_state{};           // the state difference at first_state_step
    state_diff  last_state{};

    bool ok() const { return first_out_step < 0 && first_state_step < 0; }

    std::string str() const
    {
        std::ostringstream o;
        o << a << " vs " << b << ": ";
        if (ok()) { o << "equivalent (max output diff " << max_out << ")"; return o.str(); }
        if (first_out_step >= 0)   o << "outputs diverge at step " << first_out_step << " (max " << max_out << "); ";
        if (first_state_step >= 0) {
            const auto& s = first_state;
            o << "state diverges at step " << first_state_step << " [";
            if (!s.same_keys) o << "segment structure differs: " << s.only_a << " synapses only in " << a
                                << ", " << s.only_b << " only in " << b << "; ";
            o << "|dp| " << s.dp << ", |dmm| " << s.dmm << ", |dah| " << s.dah << ", |dmx| " << s.dmx
              << ", |dcnt| " << s.dcnt << ", |dedge| " << s.dedge << "]";
            o << "; at end |dp| " << last_state.dp << ", |dmm| " << last_state.dmm << ", |dah| " << last_state.dah
              << ", |dmx| " << last_state.dmx << ", |dcnt| " << last_state.dcnt << ", |dedge| " << last_state.dedge;
        }
        return o.str();
    }
};

// Run `steps` steps on the inputs `next_input` returns, comparing every pair's
// output (and, if `track_state`, its logical state) after every step.
std::vector<pair_report> run_differential(trio& t, std::size_t steps, bool track_state,
                                          const std::function<std::vector<float>()>& next_input)
{
    std::vector<pair_report> reps;
    for (std::size_t a = 0; a < t.rigs.size(); ++a)
        for (std::size_t b = a + 1; b < t.rigs.size(); ++b)
            reps.push_back({ t.rigs[a]->name, t.rigs[b]->name });

    if (track_state) {   // the injected initial state must already agree
        std::vector<logical_state> st;
        for (auto& r : t.rigs) st.push_back(r->state());
        std::size_t k = 0;
        for (std::size_t a = 0; a < st.size(); ++a)
            for (std::size_t b = a + 1; b < st.size(); ++b, ++k) {
                auto d = compare(st[a], st[b]);
                REQUIRE_MESSAGE((d.same_keys && d.worst() == 0.0f),
                                "injected initial states differ: " << reps[k].a << " vs " << reps[k].b);
            }
    }

    for (std::size_t s = 0; s < steps; ++s) {
        const auto outs = t.step(next_input());
        std::vector<logical_state> st;
        if (track_state) for (auto& r : t.rigs) st.push_back(r->state());

        std::size_t k = 0;
        for (std::size_t a = 0; a < t.rigs.size(); ++a)
            for (std::size_t b = a + 1; b < t.rigs.size(); ++b, ++k) {
                auto& rp = reps[k];
                const float od = max_abs_diff(outs[a], outs[b]);
                rp.max_out = std::max(rp.max_out, od);
                if (od > TOL && rp.first_out_step < 0) rp.first_out_step = static_cast<long>(s);
                if (track_state) {
                    auto d = compare(st[a], st[b]);
                    rp.last_state = d;
                    if ((!d.same_keys || d.worst() > TOL) && rp.first_state_step < 0) {
                        rp.first_state_step = static_cast<long>(s);
                        rp.first_state = d;
                    }
                }
            }
    }
    return reps;
}

// The same, on input_source's inputs.
std::vector<pair_report> run_differential(trio& t, std::size_t steps, bool track_state,
                                          std::uint32_t input_seed)
{
    input_source src(t.shape, input_seed);
    return run_differential(t, steps, track_state, [&] { return src.next(); });
}

void check_all(const std::vector<pair_report>& reps)
{
    for (const auto& r : reps) {
        MESSAGE(r.str());
        CHECK_MESSAGE(r.ok(), r.str());
    }
}

} // namespace

// ---------------------------------------------------------------------------

TEST_CASE("M.0.6 response: identical state and input give identical output (frozen)")
{
    // Frozen: response + combine + inhibition only, no learning. Isolates the
    // response function (M.3) — synapse contribution, penalty, segment/path
    // sums, attenuation, dendrite/neuron max, noise, max_softmax.
    knobs_t k; k.frozen = true;
    trio t(k, /*perm_seed=*/11);
    check_all(run_differential(t, /*steps=*/200, /*track_state=*/false, /*input_seed=*/101));
}

TEST_CASE("M.0.6 learning: identical state, input and draws give identical learning, step by step")
{
    // Full learning (winner + common pass, mismatch, adapt history), branching
    // disabled. Compares outputs and the logical state after every step.
    knobs_t k;
    trio t(k, /*perm_seed=*/12);
    check_all(run_differential(t, /*steps=*/300, /*track_state=*/true, /*input_seed=*/102));
}

TEST_CASE("M.0.6 learning, winner only (common_learning_rate = 0)")
{
    // Localises winner-pass vs common-pass differences: with a zero common rate
    // the non-winners' permanence update weight is 0, so any remaining
    // difference comes from the winner's adaptation or from updates that do not
    // scale with the learning weight (mismatch).
    knobs_t k; k.common_lr = 0.0f;
    trio t(k, /*perm_seed=*/13);
    check_all(run_differential(t, /*steps=*/300, /*track_state=*/true, /*input_seed=*/103));
}

TEST_CASE("M.0.6 branching: implementations branch at the same step with the same synapses")
{
    // Branching enabled with a short interval and a low adapt-history gate so
    // events happen inside the window. Clone permanences get a random jitter
    // (M.6.7), so values cannot be compared after an event — this case compares
    // WHEN each implementation first branches and WHICH synapses it moves.
    knobs_t k; k.branch_interval = 20; k.theta_thres = 0.05f;
    trio t(k, /*perm_seed=*/14);

    input_source src(k.shape, 104);
    std::vector<long> first_branch(t.rigs.size(), -1);
    std::vector<logical_state> at_branch(t.rigs.size());
    for (std::size_t s = 0; s < 400; ++s) {
        t.step(src.next());
        for (std::size_t r = 0; r < t.rigs.size(); ++r) {
            if (first_branch[r] >= 0) continue;
            auto st = t.rigs[r]->state();
            if (has_branched(st)) { first_branch[r] = static_cast<long>(s); at_branch[r] = std::move(st); }
        }
        if (std::all_of(first_branch.begin(), first_branch.end(), [](long x) { return x >= 0; })) break;
    }

    for (std::size_t r = 0; r < t.rigs.size(); ++r) {
        MESSAGE(t.rigs[r]->name << ": first branch event at step " << first_branch[r]);
        REQUIRE_MESSAGE(first_branch[r] >= 0, t.rigs[r]->name << " never branched in 400 steps");
    }
    for (std::size_t a = 0; a < t.rigs.size(); ++a)
        for (std::size_t b = a + 1; b < t.rigs.size(); ++b) {
            const auto& na = t.rigs[a]->name;
            const auto& nb = t.rigs[b]->name;
            CHECK_MESSAGE(first_branch[a] == first_branch[b],
                          na << " first branches at step " << first_branch[a] << ", "
                             << nb << " at step " << first_branch[b]);
            if (first_branch[a] == first_branch[b]) {
                const auto d = compare(at_branch[a], at_branch[b]);
                CHECK_MESSAGE(d.same_keys, na << " vs " << nb << ": different synapses moved ("
                                               << d.only_a << " only in " << na << ", "
                                               << d.only_b << " only in " << nb << ")");
            }
        }
}

TEST_CASE("M.4.2a window edges move by at most one step per learning step, across branching and GPU rebuilds")
{
    // An edge moves by at most rate * max(tau, 1 - tau) <= rate per learning step
    // (M.4.2a). Branching grows the synapse arena; on the GPU, growing past the
    // initial capacity rebuilds the step plan (ngm25 grow_if_branched). A rebuild
    // that re-seeded the device edges from a stale host copy would show as a jump
    // larger than one step. Guard against a vacuous pass: by the time an arena
    // first grows, the edges must already be measurably away from their start.
    namespace wt = ngm::window_tracking;
    knobs_t k; k.branch_interval = 20; k.theta_thres = 0.05f;
    trio t(k, /*perm_seed=*/15);
    input_source src(t.shape, 105);

    const std::vector<float> initial = { wt::initial_lower, wt::initial_upper, wt::initial_lower, wt::initial_upper };
    const float max_step = wt::rate * 1.0001f;
    const std::size_t R = t.rigs.size();
    std::vector<std::vector<float>> prev(R, initial);
    std::vector<std::size_t> syn0(R);
    std::vector<long>  grown_at(R, -1);
    std::vector<float> drift_at_growth(R, 0.0f);
    for (std::size_t r = 0; r < R; ++r) syn0[r] = t.rigs[r]->state().syn.size();

    long all_grown_at = -1;
    for (long s = 0; s < 800 && (all_grown_at < 0 || s < all_grown_at + 100); ++s) {
        t.step(src.next());
        for (std::size_t r = 0; r < R; ++r) {
            const auto st = t.rigs[r]->state();
            REQUIRE(st.edges.size() == 4);
            for (std::size_t e = 0; e < 4; ++e)
                CHECK_MESSAGE(std::fabs(st.edges[e] - prev[r][e]) <= max_step,
                              t.rigs[r]->name << ": edge " << e << " jumped from " << prev[r][e]
                                              << " to " << st.edges[e] << " at step " << s);
            if (grown_at[r] < 0 && st.syn.size() > syn0[r]) {
                grown_at[r] = s;
                for (std::size_t e = 0; e < 4; ++e)
                    drift_at_growth[r] = std::max(drift_at_growth[r], std::fabs(prev[r][e] - initial[e]));
            }
            prev[r] = st.edges;
        }
        if (all_grown_at < 0 && std::all_of(grown_at.begin(), grown_at.end(), [](long g) { return g >= 0; }))
            all_grown_at = s;
    }
    for (std::size_t r = 0; r < R; ++r) {
        MESSAGE(t.rigs[r]->name << ": arena first grew at step " << grown_at[r]
                                << ", edges then " << drift_at_growth[r] << " from their start");
        REQUIRE_MESSAGE(grown_at[r] >= 0, t.rigs[r]->name << ": the synapse arena never grew in 800 steps");
        REQUIRE_MESSAGE(drift_at_growth[r] > 10.0f * wt::rate,
                        t.rigs[r]->name << ": edges too close to their start when the arena grew for a reset to show");
    }
}

TEST_CASE("M.0.6 connectivity: a permanence exactly at the threshold is unconnected in every implementation")
{
    // M.1.9: a synapse is connected iff p > tau. Every 7th synapse is placed
    // exactly at tau (float-exact 0.3125), the rest as usual. Frozen, so only
    // the response is compared.
    knobs_t k; k.frozen = true;
    auto p = canonical_permanences(k.shape, 15);
    for (std::size_t x = 0; x < p.size(); x += 7) p[x] = TAU;
    trio t(k, p);
    check_all(run_differential(t, /*steps=*/200, /*track_state=*/false, /*input_seed=*/105));
}

TEST_CASE("M.0.6 connectivity at the default threshold: tau = 0.3, which float cannot hold exactly")
{
    // The case above uses a float-exact tau. The model default 0.3 is not one:
    // 0.3f = 0.30000001..., and initial permanences are Pois(30)/100, so ~7 % of
    // synapses start at exactly 0.3f. An implementation that compares a float
    // permanence with a double threshold (0.3f > 0.3 is true) connects those
    // synapses — and takes the other mismatch branch — where the others do not.
    //
    // Two runs. Frozen, every implementation including the GPU ones: the
    // permanences stay at exactly 0.3f, so the response comparison p > tau is
    // exact everywhere, and a double threshold shows at step 0. Learning on, the
    // CPU implementations only, which also exercises the mismatch branch. The GPU
    // rigs are left out of that run on purpose: the branch tests the UPDATED
    // permanence, pnew = clamp(p * (1 - theta) + add, 0, 1), which for a synapse
    // at tau lands within one ulp of tau, so whether pnew > tau holds depends on
    // whether the multiply-add is fused (one rounding) or not (two). The CPU
    // compiler and the GPU shader compiler need not contract alike (GCC + RADV do
    // not; AppleClang + Metal do), and WGSL gives no per-driver control. Those
    // synapses can then legitimately take different mismatch branches on CPU and
    // GPU after their first update — the same class of divergence that makes the
    // GPU goldens compare in aggregate. A branch flip, not rounding: no tolerance
    // would cover it.
    auto p = canonical_permanences(shape_t{}, 16);   // the default shape, as both runs
    for (std::size_t x = 0; x < p.size(); x += 7) p[x] = 0.3f;
    {
        knobs_t k; k.tau = 0.3f; k.frozen = true;
        trio t(k, p);
        check_all(run_differential(t, /*steps=*/200, /*track_state=*/true, /*input_seed=*/109));
    }
    {
        knobs_t k; k.tau = 0.3f; k.cpu_only = true;
        trio t(k, p);
        check_all(run_differential(t, /*steps=*/200, /*track_state=*/true, /*input_seed=*/109));
    }
}

TEST_CASE("M.0.6 (b) initial permanences: the same Poisson mean at thresholds float rounds differently")
{
    // M.2.1: initial permanences are Pois(lambda)/100 with lambda = int(100 * tau).
    // At some thresholds that integer depends on how the product is evaluated
    // (0.29: 28 in double, 29 in float), so an implementation computing it
    // differently draws from another distribution. Compares the mean freshly
    // initialised (not injected) permanence: an off-by-one lambda shifts it by
    // 0.01, ~11 standard errors over the n * D * k = 13824 draws of this shape.
    for (const float tau : { 0.29f, 0.53f, 0.57f }) {
        CAPTURE(tau);
        knobs_t k; k.tau = tau; k.shape = { .n = 48, .k = 96 };
        std::vector<std::pair<std::string, double>> means;
        for (auto& r : make_rigs(k)) {
            const auto st = r->state();
            double sum = 0.0;
            for (const auto& [key, v] : st.syn) sum += v.p;
            means.emplace_back(r->name, sum / st.syn.size());
        }
        const double se = std::sqrt(std::floor(100.0 * tau)) / 100.0 / std::sqrt(static_cast<double>(k.shape.n * D * k.shape.k));
        for (std::size_t a = 0; a < means.size(); ++a)
            for (std::size_t b = a + 1; b < means.size(); ++b)
                CHECK_MESSAGE(std::fabs(means[a].second - means[b].second) < 4.0 * std::sqrt(2.0) * se,
                              means[a].first << " mean initial permanence " << means[a].second << ", "
                              << means[b].first << " " << means[b].second);
    }
}

TEST_CASE("M.0.6 zero group output: the step is not a learning step in any implementation")
{
    // M.4.15: full random_index inhibition (see
    // full_inhibition_params) drives every output to 0 while the neuron
    // activities stay positive. Such a step must write its (all-zero) output and
    // change nothing in the learning state: permanences, mismatch, adapt
    // history, segment structure and the running max stay bit-identical. Each
    // output page is pre-filled with NaN, so a skip path that forgets to write
    // the output fails the finiteness check. ngm25's stateless step key must
    // still advance (it keys the next step's draws). Precondition checked
    // below: the outputs really do sum to 0, otherwise the edge is not exercised.
    knobs_t k; k.full_inhibition = true;
    trio t(k, /*perm_seed=*/16);

    auto ngm25_step = [&]() -> std::uint32_t {
        for (auto& r : t.rigs)
            if (auto* g = dynamic_cast<rig_ngm25*>(r.get()))
                return scalar<std::uint32_t>(save_tree(*g->g), "step");
        FAIL("no ngm25 rig");
        return 0;
    };

    input_source src(k.shape, 106);
    std::size_t zero_steps = 0;
    for (std::size_t s = 0; s < 5; ++s) {
        std::vector<logical_state> before;
        for (auto& r : t.rigs) {
            before.push_back(r->state());
            if (!r->out) continue;   // GPU rigs: no CPU output page (and not built here)
            auto w = r->out->cur_write_buffer();
            std::fill(w.begin(), w.end(), std::numeric_limits<float>::quiet_NaN());
        }
        const std::uint32_t step_before = ngm25_step();

        const auto outs = t.step(src.next());

        bool zero = false;
        for (std::size_t r = 0; r < t.rigs.size(); ++r) {
            const auto& name = t.rigs[r]->name;
            REQUIRE_MESSAGE(all_finite(outs[r]), name << ": output not written (or non-finite) at step " << s);
            float sum = 0.0f;
            for (float v : outs[r]) sum += v;
            if (r == 0) zero = (sum == 0.0f);
            CHECK_MESSAGE((sum == 0.0f) == zero, name << " disagrees on whether step " << s << " has zero output");
            if (sum != 0.0f) continue;

            const auto d = compare(before[r], t.rigs[r]->state());
            CHECK_MESSAGE((d.same_keys && d.worst() == 0.0f),
                          name << ": zero-output step " << s << " changed the learning state: |dp| " << d.dp
                               << ", |dmm| " << d.dmm << ", |dah| " << d.dah << ", |dmx| " << d.dmx
                               << ", |dcnt| " << d.dcnt << ", |dedge| " << d.dedge);
        }
        if (zero) ++zero_steps;
        CHECK_MESSAGE(ngm25_step() == step_before + 1, "ngm25: step key did not advance at step " << s);
    }
    REQUIRE_MESSAGE(zero_steps > 0, "precondition: no step produced an all-zero group output");
}

TEST_CASE("M.0.6 zero group output from silent input: not a learning step on CPU or GPU")
{
    // M.4.15 without random_index inhibition, which is CPU-only (M.7.14), so the
    // GPU groups are covered too: with neuron baseline noise [0,0] an all-zero
    // input makes every activity 0 (M.3.10), max_softmax leaves the output at 0, and
    // the output sums to 0. Twenty learning steps first raise the running max
    // and the branch counters, so a silent step that still decayed the one or
    // advanced the other would show. Noise [0,0] changes the model's resting
    // state (M.7.3); that is fine for an edge-case check.
    knobs_t k; k.noise = 0.0f;
    trio t(k, /*perm_seed=*/18);
    check_all(run_differential(t, /*steps=*/20, /*track_state=*/true, /*input_seed=*/108));

    const auto warm = t.rigs[0]->state();
    REQUIRE_MESSAGE(std::any_of(warm.mx.begin(), warm.mx.end(), [](float m) { return m > 0.0f; }),
                    "precondition: no running max was raised by the learning steps");
    REQUIRE_MESSAGE(std::any_of(warm.cnt.begin(), warm.cnt.end(), [](std::uint64_t c) { return c > 0; }),
                    "precondition: no branch counter advanced during the learning steps");

    const std::vector<float> silent(k.shape.k, 0.0f);
    for (std::size_t s = 0; s < 5; ++s) {
        std::vector<logical_state> before;
        for (auto& r : t.rigs) before.push_back(r->state());
        const auto outs = t.step(silent);
        for (std::size_t r = 0; r < t.rigs.size(); ++r) {
            const auto& name = t.rigs[r]->name;
            REQUIRE_MESSAGE(std::all_of(outs[r].begin(), outs[r].end(), [](float v) { return v == 0.0f; }),
                            name << ": silent step " << s << " did not produce an all-zero output");
            const auto d = compare(before[r], t.rigs[r]->state());
            CHECK_MESSAGE((d.same_keys && d.worst() == 0.0f),
                          name << ": silent step " << s << " changed the learning state: |dp| " << d.dp
                               << ", |dmm| " << d.dmm << ", |dah| " << d.dah << ", |dmx| " << d.dmx
                               << ", |dcnt| " << d.dcnt << ", |dedge| " << d.dedge);
        }
    }
}

namespace {

// Edge-case check used by the width-1 and single-neuron cases:
//  (a) frozen: outputs finite in every implementation and identical across them;
//  (b) learning: outputs and learned state stay finite in every implementation
//      (M.3.10/M.3.11 — a per-implementation property, so the known winner-
//      adaptation divergence between implementations does not confound it).
// Equality alone would not do: two NaNs compare as no difference.
void check_edge_shape(const shape_t& sh, std::uint32_t seed)
{
    {
        knobs_t k; k.shape = sh; k.frozen = true;
        trio t(k, seed);
        input_source src(sh, seed + 1000);
        for (std::size_t s = 0; s < 50; ++s) {
            const auto outs = t.step(src.next());
            for (std::size_t r = 0; r < t.rigs.size(); ++r)
                REQUIRE_MESSAGE(all_finite(outs[r]), t.rigs[r]->name << ": non-finite output (frozen) at step " << s);
            for (std::size_t a = 0; a < outs.size(); ++a)
                for (std::size_t b = a + 1; b < outs.size(); ++b)
                    REQUIRE_MESSAGE(max_abs_diff(outs[a], outs[b]) <= TOL,
                                    t.rigs[a]->name << " vs " << t.rigs[b]->name << ": outputs differ (frozen) at step " << s);
        }
    }
    {
        knobs_t k; k.shape = sh;
        trio t(k, seed);
        input_source src(sh, seed + 2000);
        for (std::size_t s = 0; s < 50; ++s) {
            const auto outs = t.step(src.next());
            for (std::size_t r = 0; r < t.rigs.size(); ++r) {
                REQUIRE_MESSAGE(all_finite(outs[r]), t.rigs[r]->name << ": non-finite output (learning) at step " << s);
                REQUIRE_MESSAGE(all_finite(t.rigs[r]->state()),
                                t.rigs[r]->name << ": non-finite permanence/mismatch/adapt-history/running max at step " << s);
            }
        }
    }
}

} // namespace

TEST_CASE("M.0.6 width-1 input: every implementation stays finite and they agree")
{
    // A 1-wide input makes the normalized input entropy 0/log2(1). The GPU
    // shaders define it as 0; every CPU implementation must too (M.3.10).
    check_edge_shape({ .n = 12, .k = 1 }, /*seed=*/17);
}


namespace {

// The error message thrown while constructing one implementation's rig, or ""
// if construction succeeded.
template <class Rig>
std::string construction_error(const knobs_t& k = {})
{
    try {
        Rig r(k);
        (void)r;
        return "";
    } catch (const std::exception& e) {
        return e.what();
    }
}

} // namespace

TEST_CASE("M.0.6 configuration domain: all implementations reject fewer than two neurons and accept two")
{
    // M.0.6 (c): a configuration is either computed the same way everywhere or
    // rejected everywhere. A neuron group needs at least two neurons to compete
    // (SPEC.md M.1.1), so 0 and 1 are errors in every implementation and 2 runs.
    for (std::size_t n : { std::size_t{0}, std::size_t{1} }) {
        CAPTURE(n);
        knobs_t k; k.shape = { .n = n, .k = 48 };
        const std::string e2  = construction_error<rig_ngm2>(k);
        const std::string e25 = construction_error<rig_ngm25>(k);
        const std::string e3  = construction_error<rig_ngm3>(k);
        CHECK_MESSAGE(e2.find("must be >= 2")  != std::string::npos, "ngm2 accepted or gave: " << e2);
        CHECK_MESSAGE(e25.find("must be >= 2") != std::string::npos, "ngm25 accepted or gave: " << e25);
        CHECK_MESSAGE(e3.find("must be >= 2")  != std::string::npos, "ngm3 accepted or gave: " << e3);
    }
    {
        knobs_t k; k.shape = { .n = 2, .k = 48 }; k.frozen = true;
        trio t(k, /*perm_seed=*/19);
        input_source src(k.shape, 107);
        for (std::size_t s = 0; s < 10; ++s) {
            const auto outs = t.step(src.next());
            for (std::size_t a = 0; a < outs.size(); ++a)
                for (std::size_t b = a + 1; b < outs.size(); ++b)
                    CHECK_MESSAGE(max_abs_diff(outs[a], outs[b]) <= TOL,
                                  t.rigs[a]->name << " vs " << t.rigs[b]->name << ": outputs differ with 2 neurons at step " << s);
        }
    }
}

TEST_CASE("M.0.6 seed semantics: a different seed offset draws fresh streams in every implementation")
{
    // M.0.6 (b) / M.2.3: changing the seed offset must give new random streams,
    // not the old ones shifted to a neighbouring neuron or dendrite. Compares
    // the freshly initialised (not injected) permanences at offsets 0 and 1:
    // no dendrite's initial permanence vector at offset 1 may equal any
    // dendrite's at offset 0. (Initial permanences are Poisson draws on a 0.01
    // grid over K = 48 inputs, so an accidental whole-vector match is
    // practically impossible; a shifted stream matches exactly.)
    auto vectors = [](const logical_state& s) {
        std::map<std::pair<std::size_t, std::size_t>, std::vector<float>> v;
        for (const auto& [key, val] : s.syn) v[{ key.n, key.d }].push_back(val.p);
        return v;
    };
    auto check = [&](auto make, const char* name) {
        knobs_t k0; k0.seed_offset = seed_offset_t{ 0 };
        knobs_t k1; k1.seed_offset = seed_offset_t{ 1 };
        const auto v0 = vectors(make(k0)->state());
        const auto v1 = vectors(make(k1)->state());
        std::size_t aliased = 0;
        for (const auto& [nd1, p1] : v1)
            for (const auto& [nd0, p0] : v0)
                if (p1 == p0) { ++aliased; break; }
        CHECK_MESSAGE(aliased == 0, std::string(name) << ": " << aliased << " of " << v1.size()
                      << " dendrites at seed offset 1 reuse a stream from offset 0");
    };
    check([](const knobs_t& k) { return std::make_unique<rig_ngm2>(k); },  "ngm2");
    check([](const knobs_t& k) { return std::make_unique<rig_ngm25>(k); }, "ngm25");
    check([](const knobs_t& k) { return std::make_unique<rig_ngm3>(k); },  "ngm3");
}

namespace {

template <class G, class CfgStore, class Params>
std::string group_init_error(Params p, std::size_t apical_width, std::size_t proximal_width)
{
    try {
        CfgStore cs;
        cs.neuron_groups["g"] = p;
        G g(cs, "g", /*output_id=*/1);
        std::unordered_map<std::string, sim::output_channel_t> m;
        if (apical_width)   m["ap"] = sim::output_channel_t{ 1, apical_width };
        if (proximal_width) m["px"] = sim::output_channel_t{ 2, proximal_width };
        g.init_io(m);
        return "";
    } catch (const std::exception& e) {
        return e.what();
    }
}

} // namespace

TEST_CASE("M.0.6 configuration domain: parameter ranges and bank routing are rejected, never silently changed")
{
    // M.0.6 (c): an implementation may reject a configuration the others
    // accept, but must not compute something different without saying so.
    // Each config below used to run silently differently (or crash) in at
    // least one implementation; now every implementation rejects it.
    auto all_reject = [](const knobs_t& k, const char* what) {
        const std::string e2  = construction_error<rig_ngm2>(k);
        const std::string e25 = construction_error<rig_ngm25>(k);
        const std::string e3  = construction_error<rig_ngm3>(k);
        CHECK_MESSAGE(!e2.empty(),  "ngm2 accepted "  << what);
        CHECK_MESSAGE(!e25.empty(), "ngm25 accepted " << what);
        CHECK_MESSAGE(!e3.empty(),  "ngm3 accepted "  << what);
    };
    { knobs_t k; k.tau = 0.005f;      all_reject(k, "permanence_threshold 0.005 (Poisson mean 0)"); }
    { knobs_t k; k.tau = 1.0f;        all_reject(k, "permanence_threshold 1.0 (nothing can connect)"); }
    { knobs_t k; k.branch_interval = 0; all_reject(k, "branch_interval 0 (modulo by zero)"); }

    // ngm2 depth is capped at 4 (#237); depth 16 used to be capped at 15 silently.
    {
        ngm2::cfg_store cs;
        ngm2::dendrite_params_t dp;
        dp.max_branch_level = 5;
        cs.dendrites["deep"] = dp;
        CHECK_THROWS_AS(ngm2::dendrite_t(cs, "deep"), std::runtime_error);
    }

    // Bank routing (ngm3 and ngm25; ngm2 wires dendrites by channel mapping
    // instead of banks). Apical dendrites without an apical input: ngm3 CPU
    // used to segfault and its GPU output noise only. An input routed to the
    // apical bank of a group with no apical dendrites: ngm25 used to drop it.
    auto params3 = [](std::size_t apical_cnt, bool route_apical) {
        ngm3::neuron_group_params_t p;
        p.neuron_cnt = 4; p.apical_cnt = apical_cnt; p.proximal_cnt = 2; p.max_branch_level = 2;
        p.inputs = { { "px", ngm3::bank_t::proximal } };
        if (route_apical) p.inputs.push_back({ "ap", ngm3::bank_t::apical });
        return p;
    };
    auto params25 = [](std::size_t apical_cnt, bool route_apical) {
        ngm25::neuron_group_params_t p;
        p.neuron_cnt = 4; p.apical_cnt = apical_cnt; p.proximal_cnt = 2; p.max_branch_level = 2;
        p.inputs = { { "px", ngm25::bank_t::proximal } };
        if (route_apical) p.inputs.push_back({ "ap", ngm25::bank_t::apical });
        return p;
    };
    // apical dendrites, no apical input
    CHECK_FALSE(group_init_error<ngm3::neuron_group_t, ngm3::cfg_store>(params3(1, false), 0, 16).empty());
    CHECK_FALSE(group_init_error<ngm25::neuron_group_t, ngm25::cfg_store>(params25(1, false), 0, 16).empty());
    // an apical input, no apical dendrites
    CHECK_FALSE(group_init_error<ngm3::neuron_group_t, ngm3::cfg_store>(params3(0, true), 16, 16).empty());
    CHECK_FALSE(group_init_error<ngm25::neuron_group_t, ngm25::cfg_store>(params25(0, true), 16, 16).empty());
    // and the consistent configs still initialise
    CHECK(group_init_error<ngm3::neuron_group_t, ngm3::cfg_store>(params3(1, true), 16, 16).empty());
    CHECK(group_init_error<ngm25::neuron_group_t, ngm25::cfg_store>(params25(1, true), 16, 16).empty());
}

TEST_CASE("M.0.6 branched response: each segment's sum is clamped at 0 on its own (ngm2's rule)")
{
    // M.7.31: after a connected synapse's low-input penalty, the
    // activity is clamped at 0 per SEGMENT, and a leaf's activity is the sum of
    // its path's segment sums. Inject one branched state into all three
    // (every third input split into segments 2 and 3), freeze learning, and
    // drive inputs that are exactly 0 or >= max/2, so the penalty fires
    // deterministically (always at 0, never at >= max/2).
    knobs_t k; k.frozen = true;
    branched_state b;
    b.p1 = canonical_permanences(k.shape, 21);
    b.p2 = canonical_permanences(k.shape, 22);
    b.p3 = canonical_permanences(k.shape, 23);
    trio t(k, /*perm_seed=*/21);
    for (auto& r : t.rigs) r->inject_branched(b);

    std::vector<logical_state> st;
    for (auto& r : t.rigs) st.push_back(r->state());
    for (std::size_t a = 0; a < st.size(); ++a)
        for (std::size_t c = a + 1; c < st.size(); ++c) {
            const auto d = compare(st[a], st[c]);
            REQUIRE_MESSAGE((d.same_keys && d.worst() == 0.0f), "injected branched states differ: "
                            << t.rigs[a]->name << " vs " << t.rigs[c]->name);
        }
    check_all(run_differential(t, /*steps=*/200, /*track_state=*/false, /*input_seed=*/121));
}

namespace {

// Continuous inputs: every value in (0, 1), so U(0, max/2) > input is a real
// coin flip for the low inputs and the low-input penalty is stochastic.
std::vector<float> continuous_input(const shape_t& sh, std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> v(sh.k);
    for (auto& x : v) x = u(rng);
    return v;
}

} // namespace

TEST_CASE("M.0.6 stochastic penalty: one draw per synapse per step (no extra draws per replica row)")
{
    // M.0.6 (b): the low-input penalty is drawn once per connected
    // synapse per step. ngm3 stores each dendrite as 2^L replica rows; drawing
    // per row made an unbranched dendrite's activity the max of 2^L noisy
    // evaluations, biased upward by ~1 %. The other cases neutralise the draws,
    // so they cannot see such a bias; here the penalty fires at random
    // (continuous input), with the same state and inputs, frozen.
    //
    // Two checks, by how a pair draws:
    //  - Same draw family (ngm25 CPU and GPU, M.7.22): the draws are identical,
    //    so the outputs must agree STEP BY STEP within TOL. A statistic would be
    //    the wrong tool: their differences are pure rounding, so any consistent
    //    rounding asymmetry gives a large z for a negligible difference.
    //  - Independent draws: the per-step difference of the group's mean output
    //    must be shown NEGLIGIBLE — an equivalence test (TOST), not a test of
    //    "exactly zero": its 90 % confidence interval must lie within +-DELTA,
    //    DELTA = 0.25 % of the mean output (a quarter of item #4's bias). It
    //    fails on a real shift >= DELTA, ignores negligible ones, and fails
    //    rather than passes when the data are too noisy to decide.
    knobs_t k; k.frozen = true;
    trio t(k, /*perm_seed=*/31);
    std::mt19937 rng(131);
    constexpr std::size_t S       = 3000;
    constexpr double      DELTA   = 0.0025;   // relative equivalence margin
    constexpr double      Z90     = 1.6448536269514722;   // two one-sided 5 % tests
    const auto shared = [&](std::size_t a, std::size_t c) {
        return !t.rigs[a]->draw_family.empty() && t.rigs[a]->draw_family == t.rigs[c]->draw_family;
    };

    std::vector<std::vector<double>> means(t.rigs.size());
    std::map<std::pair<std::size_t, std::size_t>, float> worst_step;   // shared-draw pairs
    for (std::size_t s = 0; s < S; ++s) {
        const auto outs = t.step(continuous_input(k.shape, rng));
        for (std::size_t r = 0; r < outs.size(); ++r) {
            double m = 0;
            for (float v : outs[r]) m += v;
            means[r].push_back(m / static_cast<double>(outs[r].size()));
        }
        for (std::size_t a = 0; a < outs.size(); ++a)
            for (std::size_t c = a + 1; c < outs.size(); ++c)
                if (shared(a, c)) {
                    auto& w = worst_step[{ a, c }];
                    w = std::max(w, max_abs_diff(outs[a], outs[c]));
                }
    }
    for (std::size_t a = 0; a < t.rigs.size(); ++a)
        for (std::size_t c = a + 1; c < t.rigs.size(); ++c) {
            const auto& na = t.rigs[a]->name;
            const auto& nc = t.rigs[c]->name;
            if (shared(a, c)) {
                const float w = worst_step[{ a, c }];
                MESSAGE(nc << " vs " << na << " (same draws): max per-step output diff " << w);
                CHECK_MESSAGE(w <= TOL, nc << " vs " << na << ": same draws, but outputs differ by " << w);
                continue;
            }
            double sum = 0, sq = 0;
            for (std::size_t s = 0; s < S; ++s) { const double x = means[c][s] - means[a][s]; sum += x; sq += x * x; }
            const double mean = sum / S;
            const double sd   = std::sqrt(std::max(0.0, sq / S - mean * mean));
            const double half = Z90 * sd / std::sqrt(static_cast<double>(S));   // 90 % CI half-width
            double base = 0; for (double x : means[a]) base += x; base /= S;
            const double margin = DELTA * base;
            MESSAGE(nc << " - " << na << ": mean output difference " << 100.0 * mean / base << " % +- "
                    << 100.0 * half / base << " % (90 % CI), margin +-" << 100.0 * DELTA << " %");
            CHECK_MESSAGE(std::fabs(mean) + half < margin,
                          nc << " vs " << na << ": difference " << 100.0 * mean / base << " % +- "
                             << 100.0 * half / base << " % not shown to lie within +-" << 100.0 * DELTA << " %");
        }
}

namespace {

// Two channels routed into the proximal bank, with different scales: channel A
// (the first 20 inputs) peaks at 0.6, channel B (the other 28) at 1.0. Every
// value is 0 or >= half its own channel's max, so with per-channel draw ranges
// (U(0, max_c / 2)) the low-input penalty is deterministic; with ranges pooled
// over the bank (max 1.0), channel A's values below 0.5 would become coin flips.
// The cases using them run at the default shape, so its k must be their sum.
constexpr std::size_t KA = 20, KB = 28;
static_assert(KA + KB == shape_t{}.k, "the two channels must fill the default input width");

std::vector<float> two_scale_input(const shape_t& sh, std::mt19937& rng)
{
    std::bernoulli_distribution on(0.4);
    std::uniform_real_distribution<float> hi(0.5f, 1.0f);
    std::vector<float> v(sh.k, 0.0f);
    for (std::size_t i = 0; i < sh.k; ++i)
        if (on(rng)) v[i] = hi(rng) * (i < KA ? 0.6f : 1.0f);
    v[0] = 0.6f;     // pin each channel's max
    v[KA] = 1.0f;
    return v;
}

} // namespace

TEST_CASE("M.0.6 multi-channel bank: input statistics are computed per channel (ngm2's rule)")
{
    // M.3.2: the penalty's 1/sum and draw range, the adapt-side
    // attenuation and high/low threshold are per input channel; the response
    // attenuation uses the minimum entropy and average over the channels, and
    // the leaf is normalised by the total sum. ngm3 pooled them over the
    // concatenated bank, on the CPU and the GPU. Every pair is compared: ngm2,
    // ngm25 and ngm3, and in a GPU build also ngm25-gpu and ngm3-gpu. The
    // permanences are not uniform, so the learned state only agrees if each
    // implementation also ties every synapse to the same channel (M.7.5).
    {
        knobs_t k; k.frozen = true; k.channel_widths = { KA, KB };
        trio t(k, /*perm_seed=*/41);
        std::mt19937 rng(141);
        check_all(run_differential(t, /*steps=*/200, /*track_state=*/false, [&] { return two_scale_input(k.shape, rng); }));
    }
    {
        // Winner-only learning: the whole learned state must agree.
        knobs_t k; k.common_lr = 0.0f; k.channel_widths = { KA, KB };
        trio t(k, /*perm_seed=*/42);
        std::mt19937 rng(142);
        check_all(run_differential(t, /*steps=*/100, /*track_state=*/true, [&] { return two_scale_input(k.shape, rng); }));
    }
}

TEST_CASE("M.0.6 1-wide channel: it stays out of the response gate's min in every implementation")
{
    // M.7.32: a channel of one entry has 1 - Hoyer = 0 (no shape) and is left
    // out of the response gate's min. Its sibling is 40 % active in [0.5, 1]
    // (max pinned at 1), so its own 1 - Hoyer gates the response; the 1-wide
    // channel's 0 would open the gate fully. Every input is 0 or >= max/2 of
    // its channel, so the penalty is deterministic. ngm25 and ngm3 (and in a
    // GPU build their GPU groups) must agree step by step.
    const shape_t shape;   // the default: 48 inputs
    std::mt19937 rng(161);
    auto input = [&] {
        std::bernoulli_distribution on(0.4);
        std::uniform_real_distribution<float> hi(0.5f, 1.0f);
        std::vector<float> v(shape.k, 0.0f);
        v[0] = hi(rng);                        // the 1-wide channel
        for (std::size_t i = 1; i < shape.k; ++i)
            if (on(rng)) v[i] = hi(rng);
        v[1] = 1.0f;                           // pin the sibling's max
        return v;
    };
    {
        knobs_t k; k.frozen = true; k.channel_widths = { 1, shape.k - 1 };
        trio t(k, /*perm_seed=*/61);
        check_all(run_differential(t, /*steps=*/100, /*track_state=*/false, input));
    }
    {
        knobs_t k; k.common_lr = 0.0f; k.channel_widths = { 1, shape.k - 1 };
        trio t(k, /*perm_seed=*/62);
        check_all(run_differential(t, /*steps=*/100, /*track_state=*/true, input));
    }
}

TEST_CASE("M.7.5 multi-channel response: ngm2 walks a dendrite's channels in declared order, as the others do")
{
    // Every permanence is the same, so only the order of the arithmetic can
    // differ, not which weight meets which input. It does differ: a segment's
    // sum is clamped at 0 after every penalised synapse (ngm2's rule, see the
    // branched-response case), so a dense channel walked before a sparse one
    // keeps its sum, and walked after it is partly clamped away. ngm2 walked
    // its channels in unordered_map order, the others in declared order; before
    // that was fixed, ngm2 gave 1 where the others gave 0.4945, and exchanging
    // the channels' contents exchanged the results. Each input is 0
    // or the channel's max, so every penalty is decided (U(0, max/2) > input).
    //
    // A fourth rig declares the SAME two channels in the opposite order
    // ("da1" before "da0" — every other rig in this file only ever declares
    // channels in ascending order, so a mutation that sorted the declared
    // names before walking them would still pass if this were the only case).
    // Whichever channel is declared first keeps its sum; the other is partly
    // clamped. So this rig's response for "dense channel 0" must equal the
    // forward rig's response for "dense channel 1", and vice versa.
    knobs_t k; k.shape = { .n = 3, .k = 12 }; k.frozen = true; k.channel_widths = { 6, 6 };
    std::vector<std::unique_ptr<rig_base>> rs;
    rs.push_back(std::make_unique<rig_ngm2>(k));
    rs.push_back(std::make_unique<rig_ngm25>(k));
    rs.push_back(std::make_unique<rig_ngm3>(k));
    knobs_t kr = k; kr.reverse_declared = true;
    rs.push_back(std::make_unique<rig_ngm2>(kr, /*scramble=*/false, "ngm2-reversed-declare"));
    trio t(k, perm_init_t(k.shape.n * D * k.shape.k, 0.7f), std::move(rs));
    std::array<float, 2> fwd{}, rev{};
    for (const std::size_t dense : { 0u, 1u }) {
        std::vector<float> x(k.shape.k, 0.0f);
        for (std::size_t c = 0; c < 2; ++c) {
            const std::size_t off = c * 6;
            if (c == dense) std::fill(x.begin() + off, x.begin() + off + 6, 0.75f);
            else            x[off + 5] = 0.75f;
        }
        const auto outs = t.step(x);
        fwd[dense] = outs[0][0];
        rev[dense] = outs[3][0];
        for (std::size_t r = 1; r < 3; ++r)
            CHECK_MESSAGE(max_abs_diff(outs[0], outs[r]) <= TOL,
                          "dense channel c" << dense << ": " << t.rigs[0]->name << " gives " << outs[0][0]
                          << ", " << t.rigs[r]->name << " gives " << outs[r][0]);
    }
    CHECK_MESSAGE(std::fabs(rev[0] - fwd[1]) <= TOL,
                  "reversed declaration, dense channel 0: " << rev[0]
                  << " should equal the forward declaration's dense channel 1: " << fwd[1]);
    CHECK_MESSAGE(std::fabs(rev[1] - fwd[0]) <= TOL,
                  "reversed declaration, dense channel 1: " << rev[1]
                  << " should equal the forward declaration's dense channel 0: " << fwd[0]);
}

TEST_CASE("M.7.5 multi-channel wiring: ngm2's channel order is the declared one, through wiring and resume")
{
    // Channel ids, the order of the channel mappings and the order in which
    // the buffers are handed over are wiring; only the dendrite's
    // input_channels declare the order. A scrambled ngm2 must learn exactly as
    // the plain one and ngm3 do. Halfway, the scrambled one is replaced by a
    // fresh scrambled group that resumes the plain one's checkpoint, so the
    // learned permanences must come back on the same channels. The two
    // channels have different scales, so a channel mix-up changes the output.
    knobs_t k; k.common_lr = 0.0f; k.channel_widths = { KA, KB };
    std::vector<std::unique_ptr<rig_base>> rs;
    rs.push_back(std::make_unique<rig_ngm2>(k));
    rs.push_back(std::make_unique<rig_ngm2>(k, /*scramble=*/true, "ngm2-scrambled"));
    rs.push_back(std::make_unique<rig_ngm3>(k));
    trio t(k, canonical_permanences(k.shape, /*seed=*/43), std::move(rs));
    std::mt19937 rng(143);
    check_all(run_differential(t, /*steps=*/60, /*track_state=*/true, [&] { return two_scale_input(k.shape, rng); }));

    auto resumed = std::make_unique<rig_ngm2>(k, /*scramble=*/true, "ngm2-scrambled-resumed");
    resumed->wire(t.ins);
    load_tree(*resumed->g, save_tree(*dynamic_cast<rig_ngm2&>(*t.rigs[0]).g));
    t.rigs[1] = std::move(resumed);
    // Outputs step by step (run_differential's state tracking would require the
    // states to agree bit for bit at the start, and ngm3's agrees only to TOL),
    // then the learned state at the end.
    check_all(run_differential(t, /*steps=*/60, /*track_state=*/false, [&] { return two_scale_input(k.shape, rng); }));
    const auto plain = t.rigs[0]->state();
    for (std::size_t r = 1; r < t.rigs.size(); ++r) {
        const auto d = compare(plain, t.rigs[r]->state());
        CHECK_MESSAGE((d.same_keys && d.worst() <= TOL),
                      t.rigs[0]->name << " vs " << t.rigs[r]->name << ": learned state differs after resume, |dp| "
                      << d.dp << ", |dmm| " << d.dmm << ", |dah| " << d.dah << ", |dmx| " << d.dmx);
    }
}

TEST_CASE("M.7.5 shared source: one source feeding two channels of a dendrite fills both")
{
    // A dendrite holds one slot per declared channel, so a source that feeds
    // two of its channels fills both. ngm2 used to keep a dendrite's sources
    // as a set of ids, which held this source once while the input width
    // counted it twice, and process() crashed. Here the group's channels c0
    // and c1 come from one source; it must respond and learn exactly as a
    // group fed by two sources that carry the same data.
    knobs_t k; k.shape = { .n = 3, .k = 12 }; k.channel_widths = { 6, 6 };
    rig_ngm2 two(k);   // c0 from IN_ID, c1 from IN_ID + 1
    auto m = channel_map(k);
    m.at("c1").first = IN_ID;
    ngm2::neuron_group_t one(two.cs, "g", /*output_id=*/1, k.seed_offset);
    one.init_io(m);

    sim::io_buffer_cpu a(6), b(6), out_two(two.g->get_outp_size()), out_one(one.get_outp_size());
    two.g->set_input_buffer(IN_ID, a);
    two.g->set_input_buffer(IN_ID + 1, b);
    two.g->set_output_buffer(out_two);
    one.set_input_buffer(IN_ID, a);
    one.set_output_buffer(out_one);

    std::mt19937 rng(181);
    std::bernoulli_distribution on(0.4);
    std::uniform_real_distribution<float> val(0.1f, 1.0f);
    bool responded = false;
    for (int s = 0; s < 60; ++s) {
        std::vector<float> x(6);
        for (auto& v : x) v = on(rng) ? val(rng) : 0.0f;
        for (auto* buf : { &a, &b }) {
            auto w = buf->cur_write_buffer();
            std::ranges::copy(x, w.begin());
            buf->swap_buffer();
        }
        REQUIRE(two.g->process().has_value());
        REQUIRE(one.process().has_value());
        const auto o2 = out_two.cur_write_buffer();
        const auto o1 = out_one.cur_write_buffer();
        REQUIRE_MESSAGE(std::ranges::equal(o1, o2), "step " << s << ": one source gives " << o1[0]
                        << ", two sources with the same data give " << o2[0]);
        responded = responded || std::ranges::any_of(o1, [](float v) { return v != 0.0f; });
    }
    CHECK_MESSAGE(responded, "no neuron ever responded, so the comparison shows nothing");
}

TEST_CASE("M.7.5 legacy checkpoint: ngm2 refuses a format-1 or format-2 checkpoint of a multi-channel dendrite, and only that")
{
    // Before format version 3, ngm2 walked a dendrite's channels in an order
    // that depended on the build and the wiring, and the checkpoint does not
    // record it, so a multi-channel dendrite's learned synapses cannot be tied
    // back to their channels. Such a checkpoint is refused before anything is
    // assigned; a single channel has no order, so its checkpoint still loads.
    // The source and target groups hold different permanences, so a load that
    // happened (fully or partly) shows in the target's state.
    auto load_from = [](const knobs_t& k, std::uint32_t version) {
        rig_ngm2 src(k), dst(k);
        src.inject(canonical_permanences(k.shape, /*seed=*/61));
        dst.inject(canonical_permanences(k.shape, /*seed=*/62));
        const auto before = dst.state();
        const auto tree = save_tree(*src.g);   // the reader keeps a reference to it
        logging::mem_state_reader r(tree, version);
        std::string err;
        try { dst.g->load_state(r); } catch (const std::exception& e) { err = e.what(); }
        const auto after = dst.state();
        struct result { std::string err; bool unchanged, loaded; };
        auto same = [](const logical_state& a, const logical_state& b) {
            const auto d = compare(a, b); return d.same_keys && d.worst() == 0.0f; };
        return result{ err, same(after, before), same(after, src.state()) };
    };

    knobs_t multi; multi.channel_widths = { KA, KB };
    for (const std::uint32_t version : { 1u, 2u }) {
        CAPTURE(version);
        const auto r = load_from(multi, version);
        CHECK_MESSAGE(r.err.find("dendrite 0") != std::string::npos, "refusal does not name the dendrite: " << r.err);
        CHECK_MESSAGE(r.err.find("2 input channels") != std::string::npos, "refusal does not say why: " << r.err);
        CHECK_MESSAGE(r.unchanged, "a refused checkpoint changed the group");
    }
    {
        const auto r = load_from(multi, logging::k_format_version);
        CHECK_MESSAGE(r.err.empty(), "current-format multi-channel checkpoint refused: " << r.err);
        CHECK(r.loaded);
    }
    for (const std::uint32_t version : { 1u, 2u }) {
        CAPTURE(version);
        const auto r = load_from(knobs_t{}, version);
        CHECK_MESSAGE(r.err.empty(), "format-" << version << " single-channel checkpoint refused: " << r.err);
        CHECK(r.loaded);
    }
}

TEST_CASE("M.7.5 configuration domain: ngm2 rejects a dendrite whose declared channels are not its mapped ones")
{
    // The declared input_channels set the dendrite's input layout, so they
    // must name exactly the channels mapped to it: a mapped channel left
    // undeclared has no place in the layout, and a declared one left unmapped
    // has no input (ngm3 and ngm25 reject a route without a source the same way).
    // Several channels must be declared in full. Each case below checks a
    // substring specific to the ONE validation it means to exercise, not just
    // that some error was thrown — "da0", "da0" also fails on da1 being
    // unmapped, so on its own it does not prove the duplicate-declaration
    // check ran; "da0", "da1", "da0" declares both mapped channels (one of
    // them twice) so the only way it can still be refused is that check.
    knobs_t k; k.shape = { .n = 4, .k = 12 }; k.channel_widths = { 6, 6 };
    auto init_error = [&](auto edit) -> std::string {
        rig_ngm2 r(k);
        edit(r.cs.dendrites["prox"].input_channels);
        try { ngm2::neuron_group_t g(r.cs, "g", 1); g.init_io(channel_map(k)); }
        catch (const std::exception& e) { return e.what(); }
        return {};
    };
    CHECK(init_error([](auto&) {}).empty());
    {
        const auto err = init_error([](auto& v) { v = { "da0" }; });   // da1 mapped, not declared
        CHECK_MESSAGE(err.find("does not declare 'da1'") != std::string::npos, "unexpected message: " << err);
    }
    {
        const auto err = init_error([](auto& v) { v = { "da0", "da1", "da2" }; });   // da2 declared, not mapped
        CHECK_MESSAGE(err.find("declares 'da2', but no channel mapping") != std::string::npos,
                      "unexpected message: " << err);
    }
    CHECK_FALSE(init_error([](auto& v) { v = { "da0", "da0" }; }).empty());   // da0 twice, da1 also missing
    {
        // da0 and da1 both declared (da0 twice); the missing-channel check
        // above cannot fire here, so this isolates the duplicate check.
        const auto err = init_error([](auto& v) { v = { "da0", "da1", "da0" }; });
        CHECK_MESSAGE(err.find("declares 'da0' more than once") != std::string::npos,
                      "unexpected message: " << err);
    }
    {
        const auto err = init_error([](auto& v) { v = {}; });   // nothing declared
        CHECK_MESSAGE(err.find("does not declare") != std::string::npos, "unexpected message: " << err);
    }

    // One channel has no order to declare, so ngm2 may leave input_channels
    // empty there (configs written while the field was ignored do); a wrong
    // declaration is still refused.
    k.channel_widths = {};
    CHECK(init_error([](auto& v) { v = {}; }).empty());
    CHECK_FALSE(init_error([](auto& v) { v = { "other" }; }).empty());
}

TEST_CASE("M.7.5 configuration domain: two channel mappings onto the same dendrite channel are rejected")
{
    // neuron_t::init_input used to build its to_channel -> source map with
    // operator[], so a second mapping targeting a to_channel already claimed
    // by another silently replaced it instead of being refused — the dropped
    // mapping's input would then be missing from the dendrite without any
    // error. The group->neuron step has the same check; its own case follows.
    knobs_t k; k.channel_widths = { KA, KB };   // na0->da0, na1->da1
    rig_ngm2 base(k);
    for (auto& m : base.cs.neurons["n"].channel_mappings)
        if (m.from_channel == "na1") m.to_channel = "da0";   // now clashes with na0's mapping
    std::string err;
    try { ngm2::neuron_group_t g(base.cs, "g", 1); g.init_io(channel_map(k)); }
    catch (const std::exception& e) { err = e.what(); }
    CHECK_MESSAGE((err.find("da0") != std::string::npos && err.find("silently replace") != std::string::npos),
                  "unexpected message: " << err);
}

TEST_CASE("M.7.5 configuration domain: two group channel mappings onto the same neuron channel are rejected")
{
    // The group->neuron step of the case above: neuron_group_t::init_io used
    // operator[] for its to_channel -> source map too, so a second group
    // mapping onto a neuron channel replaced the first. This is the level a
    // retina variant hit: it ran with exit 0 and half its inputs dropped. If
    // the group level drops c0 the neuron still fails, but on na1 being
    // unmapped, so the message is matched to the group's own refusal.
    knobs_t k; k.channel_widths = { KA, KB };   // c0->na0, c1->na1
    rig_ngm2 base(k);
    for (auto& m : base.cs.neuron_groups["g"].channel_mappings)
        if (m.from_channel == "c1") m.to_channel = "na0";   // now clashes with c0's mapping
    std::string err;
    try { ngm2::neuron_group_t g(base.cs, "g", 1); g.init_io(channel_map(k)); }
    catch (const std::exception& e) { err = e.what(); }
    CHECK_MESSAGE((err.find("neuron set 'pyr'") != std::string::npos && err.find("'na0'") != std::string::npos
                   && err.find("silently replace") != std::string::npos),
                  "unexpected message: " << err);
}

TEST_CASE("M.7.5 configuration domain: a dendrite with nothing declared and nothing mapped is rejected")
{
    // With nothing declared and nothing mapped, both validation loops in
    // dendrite_t::init_input walk empty collections and neither fires, so
    // init_io used to succeed with an empty input_buf; the first
    // get_response()/adapt_synapses() call then dereferences it unconditionally
    // (hd_ngm2_dendrite.cpp) instead of failing cleanly at configuration time.
    // rig_ngm2 always declares its single channel explicitly, so it must be
    // cleared too, alongside the mapping that would otherwise feed it.
    knobs_t k;
    rig_ngm2 base(k);
    base.cs.dendrites["prox"].input_channels.clear();
    base.cs.neurons["n"].channel_mappings.clear();   // nothing maps to dendrite set "p" anymore
    std::string err;
    try { ngm2::neuron_group_t g(base.cs, "g", 1); g.init_io(channel_map(k)); }
    catch (const std::exception& e) { err = e.what(); }
    CHECK_MESSAGE(err.find("has no input channels") != std::string::npos, "unexpected message: " << err);
}

TEST_CASE("M.0.6 zero-sum channel: every implementation stays finite and they agree")
{
    // M.3.10: a channel whose inputs are all 0 while another channel is
    // active. Its 1/sum is 1/0 in ngm2 (guarded to 0 in ngm25 and ngm3), but
    // the penalty never fires there: the draw range is U(0, max/2) = U(0, 0),
    // so "draw > input" is never true and the 0 * inf term is never used.
    // This case guards that all three stay finite on such input, and that
    // ngm25 and ngm3 (and their GPU groups) leave the silent channel out of
    // the response gate's min alike (M.7.32): the other channel is a mix of
    // zeros and high values, so its 1 - Hoyer gates the response, while the
    // silent channel's 0 would open the gate fully.
    const shape_t shape;   // the default: KA + KB inputs
    std::mt19937 rng(151);
    auto input = [&] {
        auto v = two_scale_input(shape, rng);
        std::fill(v.begin(), v.begin() + KA, 0.0f);   // channel A all zero
        return v;
    };

    // ngm2, two channels, built directly (only finiteness is checked, so its
    // channel order does not matter).
    {
        ngm2::cfg_store cs;
        ngm2::dendrite_params_t dp;
        dp.input_channels = { "da", "db" };
        dp.permanence_threshold = TAU;
        cs.dendrites["prox"] = dp;
        ngm2::neuron_params_t np;
        np.input_channels = { "na", "nb" };
        np.dendrites = { { "p", "prox", D } };
        np.channel_mappings = { { "na", "p", "da" }, { "nb", "p", "db" } };
        np.default_neuron_noise_range = { NOISE, NOISE };
        cs.neurons["n"] = np;
        ngm2::neuron_group_params_t gp;
        gp.input_channels = { "ca", "cb" };
        gp.neurons = { { "pyr", "n", shape.n } };
        gp.channel_mappings = { { "ca", "pyr", "na" }, { "cb", "pyr", "nb" } };
        cs.neuron_groups["g"] = gp;
        ngm2::neuron_group_t g(cs, "g", /*output_id=*/1);
        g.init_io({ { "ca", sim::output_channel_t{ 0, KA } }, { "cb", sim::output_channel_t{ 1, KB } } });
        sim::io_buffer_cpu a(KA), b(KB), out(g.get_outp_size());
        g.set_input_buffer(0, a); g.set_input_buffer(1, b); g.set_output_buffer(out);
        for (std::size_t s = 0; s < 20; ++s) {
            const auto x = input();
            std::copy(x.begin(), x.begin() + KA, a.cur_write_buffer().begin()); a.swap_buffer();
            std::copy(x.begin() + KA, x.end(),   b.cur_write_buffer().begin()); b.swap_buffer();
            REQUIRE(g.process().has_value());
            auto o = out.cur_write_buffer();
            REQUIRE_MESSAGE(all_finite(std::vector<float>(o.begin(), o.end())), "ngm2: non-finite output at step " << s);
            logging::mem_state_writer w; g.save_state(w);
            const auto root = w.take_root();
            bool finite = true;
            for (const auto& [nn, ng] : root.groups.at("neurons").groups)
                for (const auto& [dn, dg] : ng.groups.at("dendrites").groups)
                    for (const char* f : { "permanence", "mismatch", "adapt_history" })
                        for (float x2 : std::get<std::vector<float>>(dg.fields.at(f))) finite = finite && std::isfinite(x2);
            REQUIRE_MESSAGE(finite, "ngm2: non-finite permanence/mismatch/adapt-history/running max at step " << s);
        }
    }
    // ngm25 and ngm3 through the rigs.
    {
        knobs_t k; k.shape = shape; k.channel_widths = { KA, KB };
        trio t(k, /*perm_seed=*/51);
        for (std::size_t s = 0; s < 20; ++s) {
            const auto outs = t.step(input());
            for (std::size_t r = 0; r < t.rigs.size(); ++r) {
                REQUIRE_MESSAGE(all_finite(outs[r]), t.rigs[r]->name << ": non-finite output at step " << s);
                REQUIRE_MESSAGE(all_finite(t.rigs[r]->state()), t.rigs[r]->name << ": non-finite state at step " << s);
            }
        }
    }
    // ...and compute the same thing: frozen (response) and winner-only learning.
    {
        knobs_t k; k.frozen = true; k.channel_widths = { KA, KB };
        trio t(k, /*perm_seed=*/52);
        check_all(run_differential(t, /*steps=*/100, /*track_state=*/false, input));
    }
    {
        knobs_t k; k.common_lr = 0.0f; k.channel_widths = { KA, KB };
        trio t(k, /*perm_seed=*/53);
        check_all(run_differential(t, /*steps=*/100, /*track_state=*/true, input));
    }
}

TEST_CASE("M.4.4 winner: neurons tied at the maximum win equally often in every implementation")
{
    // M.4.4 / M.0.6 (b): the winner is uniform among the neurons that clear the
    // winner threshold, whatever their index. Every neuron gets the same
    // permanences and the same constant noise, so all N tie at the maximum on
    // every step; with common_lr = 0 only the winner's permanences move, which
    // identifies it. inject() resets the permanences before each step but keeps
    // the RNG state and step counter, so every step is an independent draw. The
    // former rule (lowest qualifying index) gave neuron 0 every win.
    knobs_t k;
    k.common_lr = 0.0f;
    const perm_init_t one = canonical_permanences(k.shape, 5);
    perm_init_t p(k.shape.n * D * k.shape.k);
    for (std::size_t n = 0; n < k.shape.n; ++n)
        for (std::size_t d = 0; d < D; ++d)
            for (std::size_t i = 0; i < k.shape.k; ++i) p[pidx(k.shape, n, d, i)] = one[pidx(k.shape, 0, d, i)];

    trio t(k, p);
    input_source src(k.shape, 3);
    constexpr int steps = 360;                 // 30 expected wins per neuron at N = 12
    std::vector<std::vector<int>> wins(t.rigs.size(), std::vector<int>(k.shape.n, 0));
    std::vector<int> unclear(t.rigs.size(), 0);
    for (int s = 0; s < steps; ++s) {
        for (auto& r : t.rigs) r->inject(p);
        t.step(src.next());
        for (std::size_t r = 0; r < t.rigs.size(); ++r) {
            const auto st = t.rigs[r]->state();
            std::vector<std::size_t> moved;
            for (std::size_t n = 0; n < k.shape.n; ++n) {
                bool any = false;
                for (std::size_t d = 0; d < D && !any; ++d)
                    for (std::size_t i = 0; i < k.shape.k && !any; ++i)
                        any = st.syn.at({ n, d, 1, i }).p != p[pidx(k.shape, n, d, i)];
                if (any) moved.push_back(n);
            }
            if (moved.size() == 1) ++wins[r][moved[0]]; else ++unclear[r];
        }
    }
    // chi-square against uniform, N - 1 = 11 degrees of freedom: 31.26 is the
    // p = 0.001 critical value. All draws are seeded, so the result is fixed.
    REQUIRE(k.shape.n == 12);
    const double expected = static_cast<double>(steps) / static_cast<double>(k.shape.n);
    for (std::size_t r = 0; r < t.rigs.size(); ++r) {
        const std::string& name = t.rigs[r]->name;
        CHECK_MESSAGE(unclear[r] == 0, name << ": " << unclear[r] << " steps without exactly one learning neuron");
        double chi2 = 0.0;
        std::ostringstream counts;
        for (std::size_t n = 0; n < k.shape.n; ++n) {
            chi2 += (wins[r][n] - expected) * (wins[r][n] - expected) / expected;
            counts << ' ' << wins[r][n];
        }
        MESSAGE(name << ": wins per neuron" << counts.str() << " (chi2 " << chi2 << ")");
        CHECK_MESSAGE(chi2 < 31.26, name << ": winners are not uniform over tied neurons:" << counts.str());
    }
}

namespace {

// Two-sample Kolmogorov-Smirnov statistic D = sup |F_a - F_b|.
double ks_statistic(std::vector<double> a, std::vector<double> b)
{
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::size_t i = 0, j = 0;
    double d = 0.0;
    while (i < a.size() && j < b.size()) {
        const double x = std::min(a[i], b[j]);
        while (i < a.size() && a[i] <= x) ++i;
        while (j < b.size() && b[j] <= x) ++j;
        d = std::max(d, std::fabs(static_cast<double>(i) / a.size() - static_cast<double>(j) / b.size()));
    }
    return d;
}

// The KS critical value at significance alpha (asymptotic, two-sided).
double ks_critical(std::size_t n, std::size_t m, double alpha)
{
    return std::sqrt(-std::log(alpha / 2.0) / 2.0) * std::sqrt(static_cast<double>(n + m) / (n * m));
}

// Upper critical value of chi-square with df degrees of freedom at p = 0.001
// (Wilson-Hilferty; within ~1 % of the exact value for df >= 2).
double chi2_critical_001(std::size_t df)
{
    const double k = static_cast<double>(df), z = 3.0902;
    const double c = 1.0 - 2.0 / (9.0 * k) + z * std::sqrt(2.0 / (9.0 * k));
    return k * c * c * c;
}

// P(neuron j wins) for one step's post-inhibition output, by M.4.4 and M.7.34:
// win_act ~ U(lo, mx) with mx = max(out) and lo = mx * win_thres (the same float
// product as the implementations), then uniform among the neurons with
// out + FLT_EPSILON >= win_act. Exact: the qualifying set only changes at the
// points out_j + eps, so the integral over win_act is a sum over intervals.
std::vector<double> winner_probabilities(const std::vector<float>& out, float win_thres)
{
    const float eps = std::numeric_limits<float>::epsilon();
    const float mx  = *std::max_element(out.begin(), out.end());
    const double lo = static_cast<double>(mx * win_thres), hi = static_cast<double>(mx);
    std::vector<double> p(out.size(), 0.0), cut;
    auto c = [&](std::size_t j) { return static_cast<double>(out[j] + eps); };
    auto add = [&](double w, double mass) {   // uniform among the neurons qualifying at w
        std::size_t k = 0;
        for (std::size_t j = 0; j < out.size(); ++j) k += c(j) >= w;
        for (std::size_t j = 0; j < out.size(); ++j) if (c(j) >= w) p[j] += mass / k;
    };
    if (hi <= lo) { add(lo, 1.0); return p; }
    cut = { lo, hi };
    for (std::size_t j = 0; j < out.size(); ++j) if (c(j) > lo && c(j) < hi) cut.push_back(c(j));
    std::sort(cut.begin(), cut.end());
    for (std::size_t q = 0; q + 1 < cut.size(); ++q)
        if (cut[q + 1] > cut[q]) add((cut[q] + cut[q + 1]) / 2.0, (cut[q + 1] - cut[q]) / (hi - lo));
    return p;
}

} // namespace

TEST_CASE("M.0.6 (b) neuron noise: every implementation draws it from the same distribution")
{
    // One noise draw per neuron per step, U[lo, hi] (M.0.6 b; matched by
    // inspection so far). Frozen, with silent input, each neuron's activity is
    // exactly its noise draw (apical 1 x proximal 0 + noise), so the output is
    // one fixed function — max_softmax, the same in every implementation — of the
    // step's noise vector. Then the per-step output mean and max have the same
    // distribution in two implementations iff their noise does; with no
    // learning they are independent across steps, so a two-sample KS test
    // applies to every pair, GPU included. Pairs that share draws (ngm25 CPU and
    // GPU, M.7.22) must agree step by step instead.
    knobs_t k;
    k.frozen       = true;
    k.noise        = 0.01f;   // the model default range [0.01, 0.05]
    k.noise_spread = 0.04f;
    trio t(k, /*perm_seed=*/19);
    const std::vector<float> silent(k.shape.k, 0.0f);

    constexpr std::size_t steps = 500;
    std::vector<std::vector<double>> step_mean(t.rigs.size()), step_max(t.rigs.size());
    std::vector<float> shared_dev(t.rigs.size(), 0.0f);
    for (std::size_t s = 0; s < steps; ++s) {
        const auto outs = t.step(silent);
        for (std::size_t r = 0; r < t.rigs.size(); ++r) {
            REQUIRE_MESSAGE(all_finite(outs[r]), t.rigs[r]->name << ": non-finite output at step " << s);
            double sum = 0.0;
            for (float v : outs[r]) sum += v;
            step_mean[r].push_back(sum / outs[r].size());
            step_max[r].push_back(*std::max_element(outs[r].begin(), outs[r].end()));
            for (std::size_t q = 0; q < t.rigs.size(); ++q)
                if (q != r && !t.rigs[r]->draw_family.empty() && t.rigs[q]->draw_family == t.rigs[r]->draw_family)
                    shared_dev[r] = std::max(shared_dev[r], max_abs_diff(outs[r], outs[q]));
        }
    }
    // The draws vary the output (otherwise the comparison proves nothing).
    REQUIRE(std::fabs(*std::max_element(step_mean[0].begin(), step_mean[0].end())
                      - *std::min_element(step_mean[0].begin(), step_mean[0].end())) > 1e-3);

    const double crit = ks_critical(steps, steps, 0.001);
    for (std::size_t a = 0; a < t.rigs.size(); ++a)
        for (std::size_t b = a + 1; b < t.rigs.size(); ++b) {
            const auto& na = t.rigs[a]->name;
            const auto& nb = t.rigs[b]->name;
            if (!t.rigs[a]->draw_family.empty() && t.rigs[a]->draw_family == t.rigs[b]->draw_family) {
                CHECK_MESSAGE(shared_dev[a] <= TOL, na << " vs " << nb << " share draws but differ by " << shared_dev[a]);
                continue;
            }
            const double d_mean = ks_statistic(step_mean[a], step_mean[b]);
            const double d_max  = ks_statistic(step_max[a],  step_max[b]);
            MESSAGE(na << " vs " << nb << ": KS D (step mean) " << d_mean << ", (step max) " << d_max
                    << ", critical " << crit);
            CHECK_MESSAGE(d_mean < crit, na << " vs " << nb << ": per-step output mean differs in distribution");
            CHECK_MESSAGE(d_max  < crit, na << " vs " << nb << ": per-step output max differs in distribution");
        }
}

TEST_CASE("M.0.6 (b) winner draw: who wins follows the threshold distribution and the uniform choice")
{
    // M.4.4 / M.7.34: win_act ~ U(mx * win_thres, mx), then a uniform choice
    // among the neurons clearing it. For each step the output the winner is
    // drawn on gives every neuron's exact win probability (winner_probabilities),
    // so the expected win counts over many steps are known and the observed ones
    // are compared with them (chi-square). As in the M.4.4 case, common_lr = 0
    // makes the winner the only neuron whose permanences move, and re-injecting
    // the permanences before each step (RNG state and step counter kept) makes
    // the steps independent. The permanences differ per neuron and the input
    // varies, so the outputs are spread and several neurons share the band.
    knobs_t k;
    k.common_lr           = 0.0f;
    k.win_thres           = 0.5f;
    k.inhibition_strength = 2.0f;
    const perm_init_t p = canonical_permanences(k.shape, 6);
    trio t(k, p);
    input_source src(k.shape, 5);

    constexpr int steps = 600;
    std::vector<std::vector<int>>    obs(t.rigs.size(), std::vector<int>(k.shape.n, 0));
    std::vector<std::vector<double>> expct(t.rigs.size(), std::vector<double>(k.shape.n, 0.0));
    std::vector<int> unclear(t.rigs.size(), 0);
    std::vector<double> contested(t.rigs.size(), 0.0);   // expected wins not by the top neuron
    for (int s = 0; s < steps; ++s) {
        for (auto& r : t.rigs) r->inject(p);
        const auto outs = t.step(src.next());
        for (std::size_t r = 0; r < t.rigs.size(); ++r) {
            const auto pr = winner_probabilities(outs[r], k.win_thres);
            const auto top = std::max_element(outs[r].begin(), outs[r].end()) - outs[r].begin();
            for (std::size_t n = 0; n < k.shape.n; ++n) expct[r][n] += pr[n];
            contested[r] += 1.0 - pr[top];
            const auto st = t.rigs[r]->state();
            std::vector<std::size_t> moved;
            for (std::size_t n = 0; n < k.shape.n; ++n) {
                bool any = false;
                for (std::size_t d = 0; d < D && !any; ++d)
                    for (std::size_t i = 0; i < k.shape.k && !any; ++i)
                        any = st.syn.at({ n, d, 1, i }).p != p[pidx(k.shape, n, d, i)];
                if (any) moved.push_back(n);
            }
            if (moved.size() == 1) ++obs[r][moved[0]]; else ++unclear[r];
        }
    }
    for (std::size_t r = 0; r < t.rigs.size(); ++r) {
        const std::string& name = t.rigs[r]->name;
        CHECK_MESSAGE(unclear[r] == 0, name << ": " << unclear[r] << " steps without exactly one learning neuron");
        // Power: enough wins must be up for grabs beyond the top neuron.
        REQUIRE_MESSAGE(contested[r] >= 100.0, name << ": only " << contested[r] << " contested wins expected");
        // Bins: neurons expected to win at least 5 times; the rest pooled.
        double chi2 = 0.0, rest_e = 0.0;
        int rest_o = 0;
        std::size_t bins = 0;
        std::ostringstream detail;
        for (std::size_t n = 0; n < k.shape.n; ++n) {
            if (expct[r][n] >= 5.0) {
                chi2 += (obs[r][n] - expct[r][n]) * (obs[r][n] - expct[r][n]) / expct[r][n];
                ++bins;
                detail << ' ' << obs[r][n] << '/' << std::lround(expct[r][n]);
            } else { rest_e += expct[r][n]; rest_o += obs[r][n]; }
        }
        if (rest_e >= 5.0) { chi2 += (rest_o - rest_e) * (rest_o - rest_e) / rest_e; ++bins; }
        else REQUIRE_MESSAGE(rest_o <= 5, name << ": " << rest_o << " wins by neurons expected to win " << rest_e << " times");
        REQUIRE(bins >= 3);
        const double crit = chi2_critical_001(bins - 1);
        MESSAGE(name << ": observed/expected wins" << detail.str() << " (chi2 " << chi2 << ", critical " << crit << ")");
        CHECK_MESSAGE(chi2 < crit, name << ": winners do not follow win_act ~ U(mx*t, mx) + uniform choice");
    }
}

// ---------------------------------------------------------------------------
// M.7.8: checkpoint versus configuration
// ---------------------------------------------------------------------------

namespace {

struct diag_line {
    coast::diag::level lvl;
    std::string        msg;
};

// Collects the "checkpoint" diag records of its lifetime.
struct checkpoint_log {
    std::vector<diag_line> got;
    checkpoint_log()
    {
        coast::diag::set_level(coast::diag::level::info);
        coast::diag::set_sink([this](coast::diag::level l, std::string_view c, std::string_view m) {
            if (c == "checkpoint") got.push_back({ l, std::string(m) });
        });
    }
    ~checkpoint_log() { coast::diag::set_sink({}); coast::diag::set_level(coast::diag::level::info); }
    checkpoint_log(const checkpoint_log&) = delete;
    checkpoint_log& operator=(const checkpoint_log&) = delete;
};

// A run whose random draws matter: stochastic winner and neuron noise.
knobs_t stochastic_knobs()
{
    knobs_t k;
    k.win_thres    = 0.8f;
    k.noise_spread = 0.03125f;
    k.common_lr    = 0.001f;
    return k;
}

// The settings the checkpoint cases change, and how each implementation's
// record names them: (setting, recorded -> current).
knobs_t changed_knobs()
{
    knobs_t k = stochastic_knobs();
    k.common_lr           = 0.003f;
    k.inhibition_strength = 3.0f;
    k.seed_offset         = seed_offset_t{ 1 };
    return k;
}
const std::vector<std::string> k_changes = {
    "common_learning_rate 0.001 -> 0.003",
    "local_inhibition_strength 5 -> 3",
    "inhibition.strength 5 -> 3",
    "effective_seed 0 -> 1 (not applied: the random state comes from the checkpoint)",
};

} // namespace

TEST_CASE("M.7.8 checkpoint record: a changed configuration is reported once per group, setting by setting")
{
    // Each group records the settings it was built with in its checkpoint and, on
    // load, reports every difference from its current settings in one warning;
    // a seed is reported as not applied (the random state is the checkpoint's).
    // An unchanged configuration reports nothing; a checkpoint without a record
    // gives a notice that none was found.
    const knobs_t a = stochastic_knobs();
    const knobs_t b = changed_knobs();
    auto ra = make_rigs(a);
    auto rb = make_rigs(b);
    auto rs = make_rigs(a);
    REQUIRE(ra.size() == rb.size());
    for (std::size_t i = 0; i < ra.size(); ++i) {
        const std::string name = ra[i]->name;
        const auto tree = save_tree(ra[i]->entity());
        REQUIRE_MESSAGE(tree.groups.contains("config"), name << ": the checkpoint has no configuration record");
        {
            checkpoint_log log;
            load_tree(rb[i]->entity(), tree);
            REQUIRE_MESSAGE(log.got.size() == 1, name << ": " << log.got.size() << " checkpoint messages, expected 1");
            CHECK(log.got[0].lvl == coast::diag::level::warn);
            for (const auto& c : k_changes)
                CHECK_MESSAGE(log.got[0].msg.find(c) != std::string::npos,
                              name << ": the warning does not name '" << c << "': " << log.got[0].msg);
            // Nothing else differs, so nothing else is named.
            CHECK_MESSAGE(std::count(log.got[0].msg.begin(), log.got[0].msg.end(), '>') == 4,
                          name << ": " << log.got[0].msg);
        }
        {
            checkpoint_log log;
            load_tree(rs[i]->entity(), tree);
            CHECK_MESSAGE(log.got.empty(), name << ": an unchanged configuration reported: "
                          << (log.got.empty() ? std::string() : log.got[0].msg));
        }
        {
            auto old = tree;
            old.groups.erase("config");   // a checkpoint written before records existed
            checkpoint_log log;
            load_tree(rs[i]->entity(), old);
            REQUIRE_MESSAGE(log.got.size() == 1, name << ": " << log.got.size() << " messages for a record-less checkpoint");
            CHECK(log.got[0].lvl == coast::diag::level::info);
            CHECK(log.got[0].msg.find("no configuration record") != std::string::npos);
        }
    }
}

namespace {

// The runtime tunables older checkpoints carried (and loads restored), at
// whatever depth an implementation wrote them; the configuration record is
// left alone.
void strip_legacy_tunables(logging::mem_group& g)
{
    static const std::vector<std::string> names = {
        // group (ngm2, ngm3 CPU and GPU)
        "local_inhibition_strength", "common_learning_rate", "stochastic_win_thres",
        "weight_filter_steepness", "weight_filter_transition_point",
        // ngm2 dendrite
        "primary_learning_rate", "secondary_learning_rate", "mismatch_smoothing",
        "accumulated_theta_thres", "min_mismatch_deviation", "min_mismatch_percentage",
        "mismatch_act_thres",
    };
    for (const auto& n : names) g.fields.erase(n);
    for (auto& [name, sub_g] : g.groups)
        if (name != "config") strip_legacy_tunables(sub_g);
}

} // namespace

TEST_CASE("M.7.8 checkpoint vs configuration: the configuration's settings apply, the checkpoint's random state is kept")
{
    // Train under configuration A, checkpoint, and continue under B, which changes
    // the learning rate, the inhibition strength and the seed. M.7.8: B's settings
    // apply, while the random state stays the checkpoint's so the resume is exact
    // (M.7.17). The reference for "B's settings, A's random state" is a group
    // built under B with A's seed, loaded with the same checkpoint minus any
    // checkpointed tunables: the two must continue identically, step by step.
    // That they also differ from A's own continuation shows B's settings changed
    // the run (otherwise the comparison would prove nothing).
    const knobs_t a = stochastic_knobs();
    const knobs_t b = changed_knobs();
    knobs_t b_seed_a = b;
    b_seed_a.seed_offset = a.seed_offset;

    trio ta(a, 11), tb(b, 11), tr(b_seed_a, 11);
    std::mt19937 rng(23);
    for (int s = 0; s < 20; ++s) ta.step(continuous_input(a.shape, rng));

    REQUIRE(ta.rigs.size() == tb.rigs.size());
    for (std::size_t i = 0; i < ta.rigs.size(); ++i) {
        const auto tree = save_tree(ta.rigs[i]->entity());
        auto stripped = tree;
        strip_legacy_tunables(stripped);
        checkpoint_log quiet;   // the reports are checked by the case above
        load_tree(tb.rigs[i]->entity(), tree);
        load_tree(tr.rigs[i]->entity(), stripped);
    }

    std::vector<int> first_diff(ta.rigs.size(), -1), changed(ta.rigs.size(), 0);
    for (int s = 0; s < 40; ++s) {
        const auto x  = continuous_input(a.shape, rng);
        const auto oa = ta.step(x);
        const auto ob = tb.step(x);
        const auto orf = tr.step(x);
        for (std::size_t i = 0; i < ta.rigs.size(); ++i) {
            if (first_diff[i] < 0 && ob[i] != orf[i]) first_diff[i] = s;
            if (oa[i] != ob[i]) changed[i] = 1;
        }
    }
    for (std::size_t i = 0; i < ta.rigs.size(); ++i) {
        const std::string& name = ta.rigs[i]->name;
        CHECK_MESSAGE(first_diff[i] < 0, name << ": the continuation under B departs from B's settings with "
                      "the checkpoint's random state at step " << first_diff[i]);
        const auto d = compare(tb.rigs[i]->state(), tr.rigs[i]->state());
        CHECK_MESSAGE((d.same_keys && d.worst() == 0.0f), name << ": the states differ by " << d.worst());
        CHECK_MESSAGE(changed[i], name << ": B's settings did not change the run");
    }
}

TEST_CASE("M.2.4 seed provenance: after a load under another seed, every implementation reports and re-saves the checkpoint's seed")
{
    // The random state is the checkpoint's (M.7.8), so the effective seed the
    // run manifest reads (effective_seed()) and the next checkpoint records must
    // be the checkpoint's too, not the loading configuration's (#239). Restoring
    // it must not touch the stream: a group loaded under another seed continues
    // exactly as one loaded under the checkpoint's own.
    const knobs_t a = stochastic_knobs();
    knobs_t b = a;
    b.seed_offset = seed_offset_t{ 5 };

    trio ta(a, 11), tb(b, 11), tr(a, 11);
    std::mt19937 rng(29);
    for (int s = 0; s < 20; ++s) ta.step(continuous_input(a.shape, rng));

    REQUIRE(ta.rigs.size() == tb.rigs.size());
    for (std::size_t i = 0; i < ta.rigs.size(); ++i) {
        const std::string& name = ta.rigs[i]->name;
        const auto saved_seed = ta.rigs[i]->entity().effective_seed();
        REQUIRE_MESSAGE(saved_seed.has_value(), name);
        REQUIRE_MESSAGE(tb.rigs[i]->entity().effective_seed() != saved_seed, name << ": the seeds do not differ");
        const auto tree = save_tree(ta.rigs[i]->entity());
        {
            checkpoint_log quiet;   // the seed is reported as not applied (case above)
            load_tree(tb.rigs[i]->entity(), tree);
            load_tree(tr.rigs[i]->entity(), tree);
        }
        CHECK_MESSAGE(tb.rigs[i]->entity().effective_seed() == saved_seed,
                      name << ": effective_seed() after the load is "
                           << std::to_underlying(tb.rigs[i]->entity().effective_seed().value_or(seed_t{}))
                           << ", the checkpoint's is " << std::to_underlying(*saved_seed));
        auto resaved = save_tree(tb.rigs[i]->entity());
        CHECK_MESSAGE(std::get<std::uint64_t>(sub(resaved, "config").fields.at("effective_seed"))
                          == std::to_underlying(*saved_seed),
                      name << ": the re-saved configuration record names another seed");
    }

    std::vector<int> first_diff(ta.rigs.size(), -1);
    for (int s = 0; s < 40; ++s) {
        const auto x  = continuous_input(a.shape, rng);
        const auto ob = tb.step(x);
        const auto orf = tr.step(x);
        for (std::size_t i = 0; i < ta.rigs.size(); ++i)
            if (first_diff[i] < 0 && ob[i] != orf[i]) first_diff[i] = s;
    }
    for (std::size_t i = 0; i < ta.rigs.size(); ++i) {
        const std::string& name = ta.rigs[i]->name;
        CHECK_MESSAGE(first_diff[i] < 0, name << ": the continuation under another configured seed departs at step "
                                              << first_diff[i]);
        const auto d = compare(tb.rigs[i]->state(), tr.rigs[i]->state());
        CHECK_MESSAGE((d.same_keys && d.worst() == 0.0f), name << ": the states differ by " << d.worst());
    }
}

// ---------------------------------------------------------------------------
// load_state refusals: a checkpoint that does not fit, or a corrupt one, throws
// ---------------------------------------------------------------------------

namespace {

// The structural fields each implementation checks, with the value the rig's
// shape gives them.
struct shape_case { std::string field; std::uint64_t built; };

std::vector<shape_case> shape_fields(const std::string& rig, const shape_t& sh)
{
    if (rig == "ngm2") return { { "neuron_count", sh.n } };
    if (rig.starts_with("ngm25"))
        return { { "neuron_cnt", sh.n }, { "dend_count", sh.n * D }, { "max_seg_id", (1u << (L + 1)) - 1 },
                 { "apical_cnt", 0 }, { "proximal_cnt", D } };
    return { { "neuron_cnt", sh.n }, { "apical_cnt", 0 }, { "proximal_cnt", D },
             { "max_branch_level", L }, { "apical_input_size", 0 }, { "proximal_input_size", sh.k } };
}

// Adds `by` to an integer scalar field, keeping its stored type.
void bump(logging::mem_group& g, const std::string& name, std::uint64_t by)
{
    auto it = g.fields.find(name);
    REQUIRE_MESSAGE(it != g.fields.end(), "state field missing: " << name);
    std::visit([&](auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) v = static_cast<T>(v + by);
        else FAIL("not an integer scalar: " << name);
    }, it->second);
}

// Drops the last element of a vector field.
void truncate(logging::mem_group& g, const std::string& name)
{
    auto it = g.fields.find(name);
    REQUIRE_MESSAGE(it != g.fields.end(), "state field missing: " << name);
    std::visit([&](auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (requires { v.pop_back(); }) { REQUIRE(!v.empty()); v.pop_back(); }
        else FAIL("not a vector field: " << name);
    }, it->second);
}

// Loads `tree` into the rig, which must throw; returns the message. `log`, if
// given, receives what the load reported on the "checkpoint" channel.
std::string refusal(rig_base& r, const logging::mem_group& tree, std::vector<diag_line>* log = nullptr)
{
    checkpoint_log cap;
    std::string what;
    try {
        load_tree(r.entity(), tree);
        FAIL_CHECK(r.name << ": load_state accepted the checkpoint");
    } catch (const std::runtime_error& e) {
        what = e.what();
    }
    if (log) *log = cap.got;
    return what;
}

// Changes the first float or double setting in the tree's configuration record,
// so a load that compares the record reports a difference.
void perturb_record(logging::mem_group& tree)
{
    for (auto& [name, v] : sub(tree, "config").fields) {
        if (auto* f = std::get_if<float>(&v))  { *f += 0.5f; return; }
        if (auto* d = std::get_if<double>(&v)) { *d += 0.5;  return; }
    }
    FAIL("no float or double setting in the configuration record");
}

bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

} // namespace

TEST_CASE("load_state: every implementation refuses a checkpoint of another shape, naming the field")
{
    // Each structural field in turn is given another value (the built one + 2, so
    // a swapped pair of numbers in the message cannot pass), then removed. The
    // shape is checked before anything is read, so every group is unchanged, and
    // before the configuration record: a checkpoint that does not fit reports no
    // settings, although its record differs too.
    trio t(knobs_t{}, 11);
    std::mt19937 rng(23);
    for (int s = 0; s < 5; ++s) t.step(continuous_input(t.shape, rng));

    for (auto& r : t.rigs) {
        const auto tree = save_tree(r->entity());
        auto recorded = tree;
        perturb_record(recorded);
        for (const auto& f : shape_fields(r->name, r->shape)) {
            INFO(r->name, ": ", f.field);
            auto edited = recorded;
            bump(edited, f.field, 2);
            std::vector<diag_line> log;
            auto what = refusal(*r, edited, &log);
            INFO("message: ", what);
            CHECK(contains(what, "checkpoint does not fit this group"));
            CHECK(contains(what, std::format("{} {} (checkpoint) vs {} (configured)", f.field, f.built + 2, f.built)));
            CHECK_MESSAGE(log.empty(), "the record was reported before the shape: " << log.front().msg);
            CHECK(save_tree(r->entity()) == tree);

            auto removed = tree;
            removed.fields.erase(f.field);
            what = refusal(*r, removed);
            CHECK(contains(what, f.field + " missing"));
            CHECK(save_tree(r->entity()) == tree);
        }

        // The perturbed record alone is reported, so its silence above means the
        // shape was checked first.
        checkpoint_log cap;
        load_tree(r->entity(), recorded);
        CHECK_MESSAGE(!cap.got.empty(), r->name << ": the perturbed record was not reported");
    }
}

TEST_CASE("load_state: ngm2 names the neuron whose dendrite count does not fit")
{
    trio t(knobs_t{}, 11);
    auto& r = *t.rigs.at(0);
    REQUIRE(r.name == "ngm2");
    auto tree = save_tree(r.entity());
    bump(sub(sub(tree, "neurons"), "5"), "dendrite_count", 2);
    const auto what = refusal(r, tree);
    INFO("message: ", what);
    CHECK(contains(what, std::format("ngm2 neuron group 1: neuron 5: checkpoint does not fit this group — "
                                     "dendrite_count {} (checkpoint) vs {} (configured)", D + 2, D)));
}

TEST_CASE("load_state: ngm2 refuses a missing or mis-sized piece, naming where it is")
{
    trio t(knobs_t{}, 11);
    auto& r = *t.rigs.at(0);
    REQUIRE(r.name == "ngm2");
    const auto tree = save_tree(r.entity());
    const std::string at = "ngm2 neuron group 1: neuron 5: dendrite 2: corrupt checkpoint — ";
    auto dendrite = [](logging::mem_group& g) -> logging::mem_group& {
        return sub(sub(sub(sub(g, "neurons"), "5"), "dendrites"), "2");
    };

    for (const char* name : { "mismatch", "adapt_history", "segment_idx", "input_inc" }) {
        INFO(name);
        auto edited = tree;
        truncate(dendrite(edited), name);
        const auto what = refusal(r, edited);
        INFO("message: ", what);
        CHECK(contains(what, at + "mismatch/adapt_history/segment_idx/input_inc hold"));
        CHECK(contains(what, std::format("permanence {}", r.shape.k)));
    }
    {
        auto edited = tree;
        dendrite(edited).fields.erase("permanence");
        CHECK(contains(refusal(r, edited), at + "permanence missing"));
    }
    {
        auto edited = tree;
        sub(sub(sub(edited, "neurons"), "5"), "dendrites").groups.erase("2");
        CHECK(contains(refusal(r, edited), "ngm2 neuron group 1: neuron 5: corrupt checkpoint — dendrite 2 missing"));
    }
    {
        auto edited = tree;
        sub(sub(edited, "neurons"), "5").groups.erase("dendrites");
        CHECK(contains(refusal(r, edited), "ngm2 neuron group 1: neuron 5: corrupt checkpoint — dendrites missing"));
    }
    {
        auto edited = tree;
        sub(edited, "neurons").groups.erase("7");
        CHECK(contains(refusal(r, edited), "ngm2 neuron group 1: corrupt checkpoint — neuron 7 missing"));
    }
    {
        auto edited = tree;
        edited.groups.erase("neurons");
        CHECK(contains(refusal(r, edited), "ngm2 neuron group 1: corrupt checkpoint — neurons missing"));
    }
}

TEST_CASE("load_state: ngm3 and ngm25 refuse a missing or mis-sized payload and keep their state")
{
    // Both stage the payload before adopting it, so a refused load leaves the
    // group as it was (ngm3 for every piece; ngm25 for a missing perm, which it
    // checks before anything is assigned).
    trio t(knobs_t{}, 11);
    std::mt19937 rng(23);
    for (int s = 0; s < 5; ++s) t.step(continuous_input(t.shape, rng));

    for (auto& r : t.rigs) {
        if (r->name == "ngm2") continue;
        INFO(r->name);
        const auto tree = save_tree(r->entity());
        auto expect = [&](const logging::mem_group& edited, const std::string& part) {
            const auto what = refusal(*r, edited);
            INFO("message: ", what);
            CHECK(contains(what, "corrupt checkpoint — " + part));
            CHECK(save_tree(r->entity()) == tree);
        };
        if (r->name.starts_with("ngm25")) {
            auto edited = tree;
            edited.fields.erase("perm");
            expect(edited, "perm missing or empty");
            continue;
        }
        for (const char* name : { "permanences", "segment_ids", "adapt_history", "mismatch" }) {
            INFO(name);
            auto edited = tree;
            truncate(sub(edited, "proximal"), name);
            expect(edited, "proximal bank: array sizes");
        }
        {
            auto edited = tree;
            edited.groups.erase("proximal");
            expect(edited, "proximal bank missing");
        }
        {
            auto edited = tree;
            truncate(edited, "neuron_mx_act");
            expect(edited, std::format("neuron_mx_act holds {} values, expected {}", r->shape.n - 1, r->shape.n));
        }
    }
}

TEST_CASE("load_state: an ngm3 checkpoint without neuron_mx_act (pre-field) keeps the group's running max")
{
    // Checkpoints written before the format gained neuron_mx_act have none. They
    // load, and the group keeps the running max it has: here one that ten steps
    // moved away from both a fresh group's and the checkpoint's, so neither a
    // reset nor a stale read can pass. Everything else comes from the checkpoint.
    trio fresh(knobs_t{}, 11);
    trio t(knobs_t{}, 11);
    std::mt19937 rng(23);
    for (int s = 0; s < 5; ++s) t.step(continuous_input(t.shape, rng));

    for (std::size_t i = 0; i < t.rigs.size(); ++i) {
        auto& r = *t.rigs[i];
        if (!r.name.starts_with("ngm3")) continue;
        INFO(r.name);
        auto tree = save_tree(r.entity());
        const auto saved = r.state();
        tree.fields.erase("neuron_mx_act");
        std::mt19937 more(29);
        for (int s = 0; s < 5; ++s) t.step(continuous_input(t.shape, more));
        const auto kept = r.state().mx;
        REQUIRE(kept != fresh.rigs[i]->state().mx);
        REQUIRE(kept != saved.mx);

        {
            checkpoint_log quiet;
            CHECK_NOTHROW(load_tree(r.entity(), tree));
        }
        const auto after = r.state();
        CHECK(after.mx == kept);
        const auto d = compare(saved, after);
        CHECK(d.same_keys);
        CHECK(d.dp == 0.0f);
        CHECK(d.dmm == 0.0f);
        CHECK(d.dah == 0.0f);
        CHECK(d.dcnt == 0.0f);
    }
}

// ---------------------------------------------------------------------------
// M.7.5 channel record: which channel each synapse position belongs to
// ---------------------------------------------------------------------------

namespace {

// What loading one rig's checkpoint into another did: the refusal (empty if the
// load went through), whether the target kept its own state or took the
// source's, and the "checkpoint" lines the load reported.
struct cross_load_t {
    std::string            err;
    bool                   unchanged = false;
    bool                   loaded    = false;
    std::vector<diag_line> log;
};

// Saves `src`'s group and loads it into `dst` through a reader of format
// `version`, then reports the reader's notices as state_logger does after its
// entity loop.
// `edit` changes the saved tree first. The two hold different permanences, so a
// load that happened, fully or partly, shows in the target's state.
cross_load_t cross_load(rig_base& src, rig_base& dst, std::uint32_t version,
                        const std::function<void(logging::mem_group&)>& edit = {})
{
    src.inject(canonical_permanences(src.shape, /*seed=*/61));
    dst.inject(canonical_permanences(dst.shape, /*seed=*/62));
    const auto before = dst.state();
    auto tree = save_tree(src.entity());
    if (edit) edit(tree);

    cross_load_t res;
    {
        checkpoint_log cap;
        logging::mem_state_reader r(tree, version);   // keeps a reference to tree
        try { dst.entity().load_state(r); } catch (const std::runtime_error& e) { res.err = e.what(); }
        logging::report_unrecorded_configs(r);
        logging::report_unrecorded_channels(r);
        logging::report_unchecked_dendrite_types(r);
        res.log = cap.got;
    }
    auto same = [](const logical_state& a, const logical_state& b) {
        const auto d = compare(a, b); return d.same_keys && d.worst() == 0.0f; };
    const auto after = dst.state();
    res.unchanged = same(after, before);
    res.loaded    = same(after, src.state());
    return res;
}

// The channel names every rig's synapses are laid out by are the group's own,
// external channel names (channel_map's "c0", "c1", ...; "ga" for a single
// channel): ngm2's record carries the group-level name a channel resolves to
// through its own channel_mappings (SPEC.md M.7.5, hd_ngm2_tools.h
// channel_source_t), not its dendrite's innermost, local channel key.
std::string channel_prefix(const rig_base&) { return "c"; }

// Removes every channel record from a checkpoint tree, as a checkpoint written
// before format version 4 has none.
void strip_channel_records(logging::mem_group& g)
{
    for (const auto* name : { "input_channels", "apical_channels", "proximal_channels" })
        g.groups.erase(name);
    for (auto& child : g.groups | std::views::values) strip_channel_records(child);
}

} // namespace

TEST_CASE("M.7.5 channel record: every implementation refuses a checkpoint whose channels are declared in another order")
{
    // Both keep the total width, which is all the shape checks compare: two
    // equal-width channels declared in reverse, and two widths swapped under the
    // same names. Either is refused before anything is assigned or reported.
    struct case_t { std::vector<std::size_t> saved, loaded; bool reverse; std::string from, to; };
    const std::vector<case_t> cases = {
        { { 24, 24 }, { 24, 24 }, true,  "[{0}0 (24), {0}1 (24)]", "[{0}1 (24), {0}0 (24)]" },
        { { KA, KB }, { KB, KA }, false, "[{0}0 (20), {0}1 (28)]", "[{0}0 (28), {0}1 (20)]" },
    };
    for (const auto& c : cases) {
        knobs_t saved;  saved.channel_widths  = c.saved;
        knobs_t loaded; loaded.channel_widths = c.loaded; loaded.reverse_declared = c.reverse;
        auto src = make_rigs(saved);
        auto dst = make_rigs(loaded);
        REQUIRE(src.size() == dst.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            INFO(src[i]->name, ": ", c.from, " -> ", c.to);
            const auto p = channel_prefix(*src[i]);
            const auto r = cross_load(*src[i], *dst[i], logging::k_format_version);
            INFO("message: ", r.err);
            CHECK(contains(r.err, "checkpoint does not fit this group"));
            CHECK(contains(r.err, std::vformat(c.from + " (checkpoint) vs " + c.to + " (configured)",
                                               std::make_format_args(p))));
            if (src[i]->name == "ngm2") CHECK(contains(r.err, "neuron 0: dendrite 0"));
            CHECK_MESSAGE(r.unchanged, "a refused checkpoint changed the group");
            CHECK_MESSAGE(r.log.empty(), "the load reported something before refusing: " << r.log.front().msg);
        }
    }
}

TEST_CASE("M.7.5 channel record: ngm2 catches a reorder made through channel_mappings alone")
{
    // Two width-24 inputs saved with group mappings {c0->na0, c1->na1} and
    // loaded with {c0->na1, c1->na0} keep the same declared names, in the same
    // order, at every level (the dendrite's own "da0"/"da1" included) -- only
    // the swap itself moves c1's data to where c0's used to be. A record of
    // the dendrite's own, innermost channel key is identical in both and the
    // load goes through; a record of the group-level name each channel
    // resolves to differs and is refused. The neuron's own channel_mappings
    // can make the identical swap one hop later (SUBCASE two).
    knobs_t k; k.channel_widths = { 24, 24 };
    rig_ngm2 base(k);
    SUBCASE("through the group's own channel_mappings") {
        rig_ngm2 swapped(k, /*scramble=*/false, "ngm2-group-swapped", /*swap_group_targets=*/true);
        const auto r = cross_load(base, swapped, logging::k_format_version);
        INFO("message: ", r.err);
        CHECK(contains(r.err, "checkpoint does not fit this group"));
        CHECK(contains(r.err, "c0"));
        CHECK(contains(r.err, "c1"));
        CHECK(contains(r.err, "neuron 0: dendrite 0"));
        CHECK_MESSAGE(r.unchanged, "a refused checkpoint changed the group");
    }
    SUBCASE("through the neuron's own channel_mappings") {
        rig_ngm2 swapped(k, /*scramble=*/false, "ngm2-neuron-swapped",
                         /*swap_group_targets=*/false, /*swap_neuron_targets=*/true);
        const auto r = cross_load(base, swapped, logging::k_format_version);
        INFO("message: ", r.err);
        CHECK(contains(r.err, "checkpoint does not fit this group"));
        CHECK(contains(r.err, "c0"));
        CHECK(contains(r.err, "c1"));
        CHECK(contains(r.err, "neuron 0: dendrite 0"));
        CHECK_MESSAGE(r.unchanged, "a refused checkpoint changed the group");
    }
}

TEST_CASE("M.7.5 channel record: ngm2 checks every dendrite set, not only the first")
{
    // Every other ngm2 case in this file gives each neuron ONE dendrite set
    // (all its dendrites share one dendrite_params_t), so a mutation that
    // restricted the structural pre-pass to neuron 0's dendrite 0 would still
    // pass them all. Here a neuron has two dendrite sets, "p1" (one channel,
    // unchanged between the two configurations) and "p2" (two channels,
    // declared in opposite order between them) -- so only this neuron's
    // SECOND dendrite (index 1, from set "p2") differs, and a check that stops
    // after the first would miss it.
    auto make = [](bool reversed_p2) {
        ngm2::cfg_store cs;
        cs.dendrites["prox1"] = ngm2::dendrite_params_t{};   // single channel: no declaration needed
        ngm2::dendrite_params_t dp2;
        dp2.input_channels = reversed_p2 ? std::vector<std::string>{ "dz1", "dz0" }
                                         : std::vector<std::string>{ "dz0", "dz1" };
        cs.dendrites["prox2"] = dp2;

        ngm2::neuron_params_t np;
        np.dendrites = { { "p1", "prox1", 1 }, { "p2", "prox2", 1 } };
        np.channel_mappings = {
            { "na1",  "p1", "da1" },
            { "na2a", "p2", "dz0" },
            { "na2b", "p2", "dz1" },
        };
        cs.neurons["n"] = np;

        ngm2::neuron_group_params_t gp;
        gp.neurons = { { "pyr", "n", 2 } };
        gp.input_channels = { "c1", "c2a", "c2b" };   // a group with several inputs declares them
        gp.channel_mappings = {
            { "c1",  "pyr", "na1"  },
            { "c2a", "pyr", "na2a" },
            { "c2b", "pyr", "na2b" },
        };
        cs.neuron_groups["g"] = gp;

        auto g = std::make_unique<ngm2::neuron_group_t>(cs, "g", /*output_id=*/1);
        g->init_io({ { "c1",  sim::output_channel_t{ 0, 6 } },
                     { "c2a", sim::output_channel_t{ 1, 4 } },
                     { "c2b", sim::output_channel_t{ 2, 4 } } });
        return g;
    };

    auto src = make(/*reversed_p2=*/false);
    auto dst = make(/*reversed_p2=*/true);
    const auto tree = save_tree(*src);
    logging::mem_state_reader r(tree, logging::k_format_version);
    std::string err;
    try { dst->load_state(r); } catch (const std::exception& e) { err = e.what(); }
    INFO("message: ", err);
    CHECK(contains(err, "checkpoint does not fit this group"));
    CHECK(contains(err, "neuron 0: dendrite 1"));
    CHECK(contains(err, "c2a"));
    CHECK(contains(err, "c2b"));
}

TEST_CASE("M.7.5 channel record: the checkpoint names each channel and its width, and an unchanged configuration loads silently")
{
    knobs_t k; k.channel_widths = { KA, KB };
    auto src = make_rigs(k);
    auto dst = make_rigs(k);
    REQUIRE(src.size() == dst.size());

    for (std::size_t i = 0; i < src.size(); ++i) {
        INFO(src[i]->name);
        const auto p = channel_prefix(*src[i]);

        // The record, in declared order. ngm2 keeps one per dendrite, the others
        // one per bank; an empty bank records no channels.
        auto tree = save_tree(src[i]->entity());
        std::vector<logging::mem_group*> records;
        if (src[i]->name == "ngm2") {
            for (auto& neuron : sub(tree, "neurons").groups | std::views::values)
                for (auto& dend : sub(neuron, "dendrites").groups | std::views::values)
                    records.push_back(&sub(dend, "input_channels"));
        } else {
            CHECK(scalar<std::uint64_t>(sub(tree, "apical_channels"), "count") == 0);
            records.push_back(&sub(tree, "proximal_channels"));
        }
        for (auto* rec : records) {
            CHECK(scalar<std::uint64_t>(*rec, "count") == 2);
            CHECK(field<std::uint64_t>(*rec, "widths") == std::vector<std::uint64_t>{ KA, KB });
            CHECK(rec->attrs == std::map<std::string, std::string, std::less<>>{ { "0", p + "0" }, { "1", p + "1" } });
        }

        const auto r = cross_load(*src[i], *dst[i], logging::k_format_version);
        CHECK_MESSAGE(r.err.empty(), "an unchanged configuration refused its own checkpoint: " << r.err);
        CHECK(r.loaded);
        CHECK_MESSAGE(r.log.empty(), "an unchanged configuration reported: " << r.log.front().msg);
    }

    // A CPU checkpoint resumes on GPU and back: both record the same channels.
    for (auto& d : dst) {
        if (!d->name.ends_with("-gpu")) continue;
        const auto cpu = d->name.substr(0, d->name.size() - 4);
        for (auto& s : src) {
            if (s->name != cpu) continue;
            INFO(s->name, " -> ", d->name);
            const auto r = cross_load(*s, *d, logging::k_format_version);
            CHECK_MESSAGE(r.err.empty(), r.err);
            CHECK(r.log.empty());
        }
    }
    // And the other direction: a GPU checkpoint resumes on CPU.
    for (auto& s : src) {
        if (!s->name.ends_with("-gpu")) continue;
        const auto cpu = s->name.substr(0, s->name.size() - 4);
        for (auto& d : dst) {
            if (d->name != cpu) continue;
            INFO(s->name, " -> ", d->name);
            const auto r = cross_load(*s, *d, logging::k_format_version);
            CHECK_MESSAGE(r.err.empty(), r.err);
            CHECK(r.log.empty());
        }
    }
}

TEST_CASE("M.7.5 channel record: a checkpoint without one loads unchecked, with one notice per checkpoint")
{
    // One notice per checkpoint, not per group or per dendrite.
    knobs_t k; k.channel_widths = { 24, 24 };
    auto src = make_rigs(k);
    auto dst = make_rigs(k);
    REQUIRE(src.size() == dst.size());
    // Format 3 is the newest without the record; ngm2 refuses multi-channel
    // checkpoints older than that (its channel order was not declared then).
    for (std::size_t i = 0; i < src.size(); ++i) {
        INFO(src[i]->name);
        const auto r = cross_load(*src[i], *dst[i], 3, strip_channel_records);
        CHECK_MESSAGE(r.err.empty(), "a format-3 checkpoint was refused: " << r.err);
        CHECK(r.loaded);
        // ngm2 adds its own notice: a checkpoint before format 5 records no
        // dendrite types (#236). Each is one notice per checkpoint, too.
        const std::size_t expected = src[i]->name == "ngm2" ? 2 : 1;
        REQUIRE_MESSAGE(r.log.size() == expected, "expected " << expected << " notices, got " << r.log.size());
        for (const auto& l : r.log) CHECK(l.lvl == coast::diag::level::info);
        CHECK(contains(r.log[0].msg, "channel order"));
        CHECK(contains(r.log[0].msg, "neuron group"));
        if (expected == 2) CHECK(contains(r.log[1].msg, "dendrite types (apical/proximal) (ngm2 neuron group"));
    }
}

TEST_CASE("M.7.5 channel record: strip_channel_records removes every record it covers")
{
    // The "loads unchecked" case above passes whether or not the strip
    // actually strips anything (a format-3 reader never looks for a record
    // either way), so it alone would not catch strip_channel_records silently
    // doing nothing. This confirms the helper itself does what its name says.
    auto has_channel_record = [](const logging::mem_group& g) {
        auto walk = [](const logging::mem_group& g, auto&& self) -> bool {
            for (const char* name : { "input_channels", "apical_channels", "proximal_channels" })
                if (g.groups.contains(name)) return true;
            for (const auto& child : g.groups | std::views::values)
                if (self(child, self)) return true;
            return false;
        };
        return walk(g, walk);
    };

    knobs_t k; k.channel_widths = { 24, 24 };
    for (auto& r : make_rigs(k)) {
        INFO(r->name);
        auto tree = save_tree(r->entity());
        REQUIRE_MESSAGE(has_channel_record(tree), "fixture has no channel record to strip");
        strip_channel_records(tree);
        CHECK_MESSAGE(!has_channel_record(tree), "strip_channel_records left a record behind");
    }
}

TEST_CASE("M.7.5 channel record: a format-4 checkpoint without one is corrupt, not merely unrecorded")
{
    // Gating the check on "does a record exist" instead of
    // state_reader::format_version() >= 4 would read a stripped record on a
    // current-format checkpoint as if it predated the record (one quiet
    // notice, as the case above gets at format 3) instead of refusing a
    // corrupt checkpoint: a format-4 checkpoint without a record is never
    // loaded through a group's load_state otherwise.
    knobs_t k; k.channel_widths = { 24, 24 };
    auto src = make_rigs(k);
    auto dst = make_rigs(k);
    REQUIRE(src.size() == dst.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        INFO(src[i]->name);
        const auto r = cross_load(*src[i], *dst[i], logging::k_format_version, strip_channel_records);
        INFO("message: ", r.err);
        CHECK(contains(r.err, "corrupt checkpoint"));
        CHECK(contains(r.err, src[i]->name == "ngm2" ? "input_channels missing" : "apical_channels missing"));
        CHECK_MESSAGE(r.unchanged, "a refused checkpoint changed the group");
    }
}

namespace {

// A routed apical channel, on top of the knobs' usual proximal-only routing:
// every other M.7.5 rig in this file has apical_cnt == 0, so its channel
// record case never checks the apical bank at all -- dropping a model's apical
// check_channel_record call would leave every other test in this file passing.
// ngm25 and ngm3's neuron_group_params_t and input_route_t are shaped alike
// but are distinct types (the models stay decoupled), so each gets its own,
// otherwise identical, helper.
ngm25::neuron_group_t* make_apical_group(const knobs_t& k, std::unique_ptr<ngm25::neuron_group_t>& owned)
{
    ngm25::cfg_store cs;
    ngm25::neuron_group_params_t p;
    p.neuron_cnt           = k.shape.n;
    p.apical_cnt           = 1;
    p.proximal_cnt         = D;
    p.max_branch_level     = L;
    p.permanence_threshold = k.tau;
    p.inputs               = { { "ap", ngm25::bank_t::apical }, { "ga", ngm25::bank_t::proximal } };
    cs.neuron_groups["g"]  = p;
    owned = std::make_unique<ngm25::neuron_group_t>(cs, "g", /*output_id=*/1, k.seed_offset);
    owned->init_io({ { "ap", sim::output_channel_t{ IN_ID,     8 } },
                     { "ga", sim::output_channel_t{ IN_ID + 1, k.shape.k } } });
    return owned.get();
}

ngm3::neuron_group_t* make_apical_group(const knobs_t& k, std::unique_ptr<ngm3::neuron_group_t>& owned)
{
    ngm3::cfg_store cs;
    ngm3::neuron_group_params_t p;
    p.neuron_cnt              = k.shape.n;
    p.apical_cnt              = 1;
    p.proximal_cnt            = D;
    p.max_branch_level   = L;
    p.permanence_threshold    = k.tau;
    p.inputs                  = { { "ap", ngm3::bank_t::apical }, { "ga", ngm3::bank_t::proximal } };
    cs.neuron_groups["g"]     = p;
    owned = std::make_unique<ngm3::neuron_group_t>(cs, "g", /*output_id=*/1, k.seed_offset);
    owned->init_io({ { "ap", sim::output_channel_t{ IN_ID,     8 } },
                     { "ga", sim::output_channel_t{ IN_ID + 1, k.shape.k } } });
    return owned.get();
}

// Renames the apical bank's first channel in a saved tree and checks that
// loading it back is refused as a channel-record mismatch, naming the bank.
template <class G>
void check_apical_record_enforced(G& src, G& dst)
{
    auto tree = save_tree(src);
    REQUIRE(scalar<std::uint64_t>(sub(tree, "apical_channels"), "count") == 1);
    sub(tree, "apical_channels").attrs["0"] = "renamed";
    logging::mem_state_reader r(tree, logging::k_format_version);
    std::string err;
    try { dst.load_state(r); } catch (const std::exception& e) { err = e.what(); }
    INFO("message: ", err);
    CHECK(contains(err, "checkpoint does not fit this group"));
    CHECK(contains(err, "apical_channels"));
}

} // namespace

TEST_CASE("M.7.5 channel record: a routed apical channel is checked too, in every CPU implementation")
{
    // make_apical_group bypasses rig_base (apical routing is not one of the
    // knobs the shared rigs take), so this compares each implementation with
    // itself, as the other load_state refusal cases in this file do. GPU
    // apical routing is not covered here (left for a follow-up).
    knobs_t k; k.shape = { .n = 3, .k = 12 };

    SUBCASE("ngm25") {
        std::unique_ptr<ngm25::neuron_group_t> src_owned, dst_owned;
        auto* src = make_apical_group(k, src_owned);
        auto* dst = make_apical_group(k, dst_owned);
        check_apical_record_enforced(*src, *dst);
    }
    SUBCASE("ngm3") {
        std::unique_ptr<ngm3::neuron_group_t> src_owned, dst_owned;
        auto* src = make_apical_group(k, src_owned);
        auto* dst = make_apical_group(k, dst_owned);
        check_apical_record_enforced(*src, *dst);
    }
}
