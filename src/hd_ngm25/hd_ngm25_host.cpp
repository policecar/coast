#include "hd_ngm25/hd_ngm25_host.h"

#include <format>
#include <stdexcept>
#include <string>
#include <utility>

#include "logging/state_checks.h"
#include "logging/state_io.h"

namespace ngm25 {

// ------------------------------- configuration -----------------------------

void validate_params(const neuron_group_params_t &p, const group_caps_t &caps)
{
    // Config contracts — JSON-deserialized, so a malformed config is a
    // Release-time event: validate with throws (asserts are NDEBUG no-ops).
    const std::string m(caps.model);
    if (p.proximal_cnt < 1)
        throw std::runtime_error(m + ": proximal_cnt must be >= 1");
    if (p.max_branch_level < 1 || p.max_branch_level > caps.max_branch_level)
        throw std::runtime_error(std::format("{}: max_branch_level must be in [1,{}]{}",
                                             m, caps.max_branch_level, caps.depth_note));
    if (p.branch_interval == 0)
        throw std::runtime_error(m + ": branch_interval must be >= 1");
    // permanence_threshold feeds two math paths that both break outside a bounded
    // range, and it is never re-validated downstream:
    //   * response attenuation  inv_perm_range = 1/(1-thr)  -> +inf/NaN at thr==1,
    //     silently negative above 1, poisoning every segment activity in the group
    //     (on GPU: gpu_params_t::inv_perm_range);
    //   * Poisson cold-init      mean = int(100*thr)        -> mean<=0 is UB per
    //     [rand.dist.pois], and thr<0.01 truncates that mean to 0.
    // Written as a range test so NaN/inf are rejected too (both fail both compares).
    if (!(p.permanence_threshold >= 0.01f && p.permanence_threshold < 1.0f))
        throw std::runtime_error(m + ": permanence_threshold must be in [0.01, 1.0); got "
                                 + std::to_string(p.permanence_threshold));
    if (p.neuron_cnt < 2)
        throw std::runtime_error(m + ": neuron_cnt must be >= 2 (a neuron group needs at least two "
                                 "neurons to compete); got " + std::to_string(p.neuron_cnt));
    // dend_neuron (u16, arena directory) stores neuron ids 0..neuron_cnt-1, so the
    // largest id (neuron_cnt-1) must fit u16 => neuron_cnt <= 65536. (Testing
    // neuron_cnt-1 is safe only because < 2 is rejected just above.)
    if (p.neuron_cnt - 1 > 0xFFFFu)
        throw std::runtime_error(m + ": neuron_cnt " + std::to_string(p.neuron_cnt)
                                 + " exceeds the u16 dend_neuron id space (max 65536)");
    // Only values that cause UB, out-of-bounds access or a silently dead group are
    // refused here. Learning rates, mismatch thresholds, stochastic_win_thres and
    // the noise-range order degrade gracefully: ngm25 draws by hash + lerp, which
    // is defined for a reversed range (ngm2/ngm3 refuse it: UB in their std
    // distributions, #199).

    // ---- backend limits ----
    // The GPU WGSL implements only the stochastic penalty (response_main hardcodes
    // ngm2's D1 draw); refuse the others rather than silently run the wrong model.
    if (caps.stochastic_penalty_only && p.penalty_policy != penalty_policy_t::stochastic)
        throw std::runtime_error(m + ": only penalty_policy=stochastic is "
                                 "implemented on GPU; use the CPU model for "
                                 "expected_value / none");
    // dend_count is fixed by config and invariant across growth, so this bound
    // holds for the whole run; checked here so an over-large config fails with a
    // clear message rather than an opaque backend validation error later.
    const std::size_t dend_cnt = p.neuron_cnt * (p.apical_cnt + p.proximal_cnt);
    if (caps.max_dend_count != 0 && dend_cnt > caps.max_dend_count)
        throw std::runtime_error(m + ": dend_count " + std::to_string(dend_cnt)
                                 + " exceeds the GPU dispatch limit "
                                   "(maxComputeWorkgroupsPerDimension = "
                                 + std::to_string(caps.max_dend_count)
                                 + "); reduce neuron_cnt/dendrites or use the CPU model");
    if (caps.branch_interval_u32 && p.branch_interval >= (std::uint64_t{1} << 32))
        throw std::runtime_error(m + ": branch_interval must be < 2^32; got "
                                 + std::to_string(p.branch_interval));
}

group_shape_t base_shape(const neuron_group_params_t &p)
{
    group_shape_t s;
    s.neuron_cnt   = p.neuron_cnt;
    s.apical_cnt   = p.apical_cnt;
    s.proximal_cnt = p.proximal_cnt;
    s.max_depth    = p.max_branch_level;
    s.max_seg_id   = static_cast<std::uint8_t>((1u << (p.max_branch_level + 1)) - 1);
    return s;
}

adapt_params_t make_adapt_params(const neuron_group_params_t &p)
{
    adapt_params_t a;
    a.perm_threshold          = p.permanence_threshold;
    a.primary_learning_rate   = p.primary_learning_rate;
    a.secondary_learning_rate = p.secondary_learning_rate;
    a.common_learning_rate    = p.common_learning_rate;
    a.stochastic_win_thres    = p.stochastic_win_thres;
    a.weight_filter           = p.weight_filter;
    a.mismatch_act_thres      = p.mismatch_act_thres;
    a.mismatch_smoothing      = p.mismatch_smoothing;
    return a;
}

branch_params_t make_branch_params(const neuron_group_params_t &p)
{
    branch_params_t b;
    b.min_mismatch_deviation  = p.min_mismatch_deviation;
    b.accumulated_theta_thres = p.accumulated_theta_thres;
    b.min_mismatch_percentage = p.min_mismatch_percentage;
    b.branch_interval         = p.branch_interval;
    return b;
}

// --------------------------------- routing ---------------------------------

input_routing_t route_inputs(
    const neuron_group_params_t                                  &p,
    const std::unordered_map<std::string, sim::output_channel_t> &input_map,
    group_shape_t                                                &shape,
    const group_caps_t                                           &caps,
    sim::output_id_t                                              group_id)
{
    const std::string m(caps.model);
    input_routing_t out;

    std::uint32_t pos = 0;
    auto route_bank = [&](bank_t bank) {
        const std::uint32_t lo = pos;
        for (const auto &r : p.inputs) {
            if (r.bank != bank) continue;
            // The channel record (M.7.5) cannot store an empty name, so every save
            // would fail: refuse it at configuration time.
            if (r.channel_name.empty())
                throw std::runtime_error(std::format(
                    "{} neuron group {}: init_io: a {} input route has an empty channel name",
                    m, group_id, bank == bank_t::apical ? "apical" : "proximal"));
            const auto it = input_map.find(r.channel_name);
            if (it == input_map.end())
                throw std::runtime_error(m + ": input channel '" + r.channel_name
                                         + "' not found in routing map");
            const sim::output_id_t chan_id   = it->second.first;
            const auto             chan_size = static_cast<std::uint32_t>(it->second.second);
            out.routes.push_back({ chan_id, bank, pos, pos + chan_size, nullptr, r.channel_name });
            out.inp_ids.push_back(chan_id);
            pos += chan_size;
        }
        const auto ti = static_cast<std::size_t>(bank);
        shape.input_lo[ti]   = lo;
        shape.input_hi[ti]   = pos;
        shape.input_size[ti] = pos - lo;
    };
    route_bank(bank_t::apical);
    route_bank(bank_t::proximal);

    const std::size_t apical_w   = shape.input_size[static_cast<std::size_t>(bank_t::apical)];
    const std::size_t proximal_w = shape.input_size[static_cast<std::size_t>(bank_t::proximal)];
    if (proximal_w < 1)
        throw std::runtime_error(m + ": no proximal input channel routed");
    // Symmetric guard for the apical bank. apical_cnt (a structural dendrite count)
    // and the per-route bank tag are independent config fields; if apical_cnt > 0
    // but no channel is routed to the apical bank, build_arena produces apical
    // dendrites with ZERO synapses. They respond 0, but because apical dendrites
    // EXIST the "no apical bank" sentinel (neuron_type_max apical -> 1.0) never
    // fires, so combine multiplies every neuron by apical=0 and the whole group is
    // silently, permanently dead (no throw, no NaN, consistency-checker-clean).
    if (p.apical_cnt > 0 && apical_w < 1)
        throw std::runtime_error(m + ": apical_cnt > 0 but no apical input channel "
                                 "routed (apical dendrites would be dead-empty)");
    // The converse: a channel routed to the apical bank of a group with no apical
    // dendrites has nowhere to go and would be silently dropped (ngm3 rejects it
    // too; SPEC.md M.0.6 c).
    if (p.apical_cnt == 0 && apical_w > 0)
        throw std::runtime_error(m + ": an input channel is routed to the apical bank "
                                 "but apical_cnt == 0");
    if (pos > 0xFFFFu)                 // input_idx is u16; positions span [0, pos)
        throw std::runtime_error(m + ": total input width " + std::to_string(pos)
                                 + " exceeds u16 input_idx capacity (65535)");
    out.width = pos;
    return out;
}

logging::channel_layout channel_layout(std::span<const channel_route_t> routes, bank_t bank)
{
    logging::channel_layout layout;
    for (const auto &r : routes)
        if (r.bank == bank) layout.push_back({ r.name, r.pos_hi - r.pos_lo });
    return layout;
}

// ------------------------------- vis mirror --------------------------------

void dense_mirror_t::refresh(const synapse_arena_t &arena, const group_shape_t &shape)
{
    const std::size_t rep_cnt    = std::size_t{1} << shape.max_depth;
    const std::size_t per_neuron = shape.apical_cnt + shape.proximal_cnt;

    for (const bank_t type : { bank_t::apical, bank_t::proximal }) {
        const auto ti = static_cast<std::size_t>(type);
        bank_mirror_t &b = banks_[ti];
        b.d_cnt      = (type == bank_t::apical) ? shape.apical_cnt : shape.proximal_cnt;
        b.rep_cnt    = rep_cnt;
        b.input_size = shape.input_size[ti];
        if (b.d_cnt == 0 || b.input_size == 0) { b.perm.clear(); b.seg.clear(); continue; }

        const std::size_t total = shape.neuron_cnt * b.d_cnt * rep_cnt * b.input_size;
        b.perm.assign(total, 0.0f);
        b.seg .assign(total, 0);

        // Each synapse's seg covers a dyadic leaf interval [lo,hi) of the rep axis
        // (detail::seg_interval), i.e. exactly the rep rows r in [lo,hi) for that
        // input position — so one pass over the synapses fills every rep slot
        // exactly once (antichain-cut property, invariant 4).
        const std::size_t type_base = (type == bank_t::apical) ? 0 : shape.apical_cnt;
        const auto        lo        = static_cast<std::uint16_t>(shape.input_lo[ti]);
        for (std::size_t n = 0; n < shape.neuron_cnt; ++n)
            for (std::size_t ld = 0; ld < b.d_cnt; ++ld) {
                const std::size_t row = b.offset(n, ld, 0);
                const auto rng = arena.dendrite(n * per_neuron + type_base + ld);
                for (std::uint32_t s = rng.begin; s < rng.end; ++s) {
                    const auto         i    = static_cast<std::size_t>(arena.input_idx[s] - lo);
                    const std::uint8_t seg  = arena.seg[s];
                    const auto         span = detail::seg_interval(seg, shape.max_depth);
                    for (int r = span.lo; r < span.hi; ++r) {
                        const std::size_t off = row + static_cast<std::size_t>(r) * b.input_size + i;
                        b.perm[off] = arena.perm[s];
                        b.seg [off] = seg;
                    }
                }
            }
    }
}

std::span<const float>
dense_mirror_t::perm(bank_t bank, std::size_t n, std::size_t d, std::size_t r) const
{
    const bank_mirror_t &b = banks_[static_cast<std::size_t>(bank)];
    const std::size_t off = b.offset(n, d, r);
    if (b.d_cnt == 0 || off + b.input_size > b.perm.size()) return {};
    return { b.perm.data() + off, b.input_size };
}

std::span<const std::uint8_t>
dense_mirror_t::seg(bank_t bank, std::size_t n, std::size_t d, std::size_t r) const
{
    const bank_mirror_t &b = banks_[static_cast<std::size_t>(bank)];
    const std::size_t off = b.offset(n, d, r);
    if (b.d_cnt == 0 || off + b.input_size > b.seg.size()) return {};
    return { b.seg.data() + off, b.input_size };
}

// ------------------------------- checkpoint --------------------------------

void write_checkpoint(logging::state_writer          &w,
                      const group_identity_t         &g,
                      std::uint32_t                   step,
                      std::uint32_t                   branch_epoch,
                      seed_t                          effective_seed,
                      const synapse_arena_t          &arena,
                      std::span<const std::uint64_t>  input_count,
                      std::span<const float>          neuron_mx_act,
                      const ngm::window_tracking::edges_t &window_edges)
{
    // Structural sizes — validated on load (refuse a mismatched checkpoint).
    w.write("neuron_cnt",   static_cast<std::uint64_t>(g.params.neuron_cnt));
    w.write("dend_count",   static_cast<std::uint64_t>(arena.dend_count()));
    w.write("max_seg_id",   static_cast<std::uint32_t>(g.shape.max_seg_id));
    w.write("apical_cnt",   static_cast<std::uint64_t>(g.params.apical_cnt));
    w.write("proximal_cnt", static_cast<std::uint64_t>(g.params.proximal_cnt));

    logging::write_channel_record(w, "apical_channels",   channel_layout(g.routes, bank_t::apical));
    logging::write_channel_record(w, "proximal_channels", channel_layout(g.routes, bank_t::proximal));

    // Hash-key state.
    w.write("step",           step);
    w.write("branch_epoch",   branch_epoch);
    w.write("effective_seed", std::to_underlying(effective_seed));

    // Arena SoA + directory.
    w.write("perm",        std::span<const float>(arena.perm));
    w.write("input_idx",   std::span<const std::uint16_t>(arena.input_idx));
    w.write("seg",         std::span<const std::uint8_t>(arena.seg));
    w.write("adapt_hist",  std::span<const float>(arena.adapt_hist));
    w.write("mismatch",    std::span<const float>(arena.mismatch));
    w.write("dend_offset", std::span<const std::uint32_t>(arena.dend_offset));
    {
        std::vector<std::uint8_t> dtype(arena.dend_type.size());
        for (std::size_t i = 0; i < dtype.size(); ++i)
            dtype[i] = static_cast<std::uint8_t>(arena.dend_type[i]);
        w.write("dend_type", std::span<const std::uint8_t>(dtype));
    }
    w.write("dend_neuron", std::span<const std::uint16_t>(arena.dend_neuron));

    // Per-neuron carried state.
    w.write("input_count",   input_count);
    w.write("neuron_mx_act", neuron_mx_act);
    ngm::window_tracking::save(w, window_edges);

    logging::write_config_record(w, g.config);
}

checkpoint_state_t stage_checkpoint(logging::state_reader  &r,
                                    const group_identity_t &g,
                                    std::uint32_t           step,
                                    std::uint32_t           branch_epoch,
                                    seed_t                  effective_seed)
{
    const std::string who(g.who);
    auto corrupt = [&](std::string_view what) {
        return std::runtime_error(std::format("{}: corrupt checkpoint — {}", who, what));
    };

    // A checkpoint from a differently shaped group cannot be loaded into this one.
    // apical_cnt/proximal_cnt are recorded from format version 4 on, added to
    // the same call as the rest rather than a second check_shape: two calls
    // would stop at the first mismatch and never report a second (M.7.18).
    const std::size_t dend_count = g.params.neuron_cnt * (g.params.apical_cnt + g.params.proximal_cnt);
    std::vector<logging::shape_field> shape_fields{
        { "neuron_cnt", g.params.neuron_cnt },
        { "dend_count", dend_count          },
        { "max_seg_id", g.shape.max_seg_id  },
    };
    if (r.format_version() >= 4) {
        shape_fields.push_back({ "apical_cnt",   g.params.apical_cnt   });
        shape_fields.push_back({ "proximal_cnt", g.params.proximal_cnt });
    }
    logging::check_shape(r, who, shape_fields);

    if (r.format_version() >= 4) {
        logging::check_channel_record(r, "apical_channels",   channel_layout(g.routes, bank_t::apical),   who);
        logging::check_channel_record(r, "proximal_channels", channel_layout(g.routes, bank_t::proximal), who);
    } else {
        r.note_unrecorded_channels(who);
    }

    logging::check_config_record(r, g.config, who);

    checkpoint_state_t s;

    // A truncated checkpoint whose structural header matches but whose `perm`
    // dataset is absent/empty is refused by name (check_arena_consistency would
    // refuse it too, with a less telling message).
    s.arena.perm = r.read_float_vec("perm");
    if (s.arena.perm.empty())
        throw corrupt("perm missing or empty");

    // A format-1 checkpoint has no window edges: the trackers start over, see read().
    s.window_edges = ngm::window_tracking::read(r, who);

    s.step         = r.read_uint32("step",         step);
    s.branch_epoch = r.read_uint32("branch_epoch", branch_epoch);
    // The hash-RNG seed is authoritative for the checkpointed run (every D1..D5
    // draw is keyed on it). Resuming under a config whose random_seed/seed_offset
    // differs would otherwise silently diverge from the checkpointed stream. The
    // default (the group's own) keeps a pre-field checkpoint's old behaviour.
    s.effective_seed = seed_t{ r.read_uint32("effective_seed", std::to_underlying(effective_seed)) };

    s.arena.input_idx   = r.read_uint16_vec("input_idx");
    s.arena.seg         = r.read_uint8_vec("seg");
    s.arena.adapt_hist  = r.read_float_vec("adapt_hist");
    s.arena.mismatch    = r.read_float_vec("mismatch");
    s.arena.dend_offset = r.read_uint32_vec("dend_offset");
    {
        const auto dtype = r.read_uint8_vec("dend_type");
        s.arena.dend_type.resize(dtype.size());
        for (std::size_t i = 0; i < dtype.size(); ++i)
            s.arena.dend_type[i] = static_cast<bank_t>(dtype[i]);
    }
    s.arena.dend_neuron = r.read_uint16_vec("dend_neuron");

    s.input_count   = r.read_uint64_vec("input_count");
    s.neuron_mx_act = r.read_float_vec("neuron_mx_act");

    // check_arena_consistency (below) does not cover these two per-neuron vectors,
    // and read_*_vec returns an EMPTY vector on a missing/mistyped field (it never
    // throws). An absent or truncated one would OOB on the next process()
    // (neuron_mx_act_[n], n in [0,neuron_cnt)). The structural check above pinned
    // neuron_cnt.
    if (s.input_count.size() != g.params.neuron_cnt || s.neuron_mx_act.size() != g.params.neuron_cnt)
        throw corrupt(std::format("per-neuron vectors (input_count/neuron_mx_act) length "
                                  "mismatch vs neuron_cnt={}", g.params.neuron_cnt));

    // A matching structural header does NOT guarantee a self-consistent payload: a
    // truncated or corrupt checkpoint — or one from another apical/proximal split or
    // input width that predates those records — can leave the SoA vectors, CSR
    // offsets and directory with mismatched lengths or out-of-range values. The
    // invariant net used after every repack decides, on the staged arena.
    std::string why;
    if (!check_arena_consistency(s.arena, g.shape, &why))
        throw corrupt("arena inconsistent: " + why);

    return s;
}

} // namespace ngm25
