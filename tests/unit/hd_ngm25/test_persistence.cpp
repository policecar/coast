// tests/unit/hd_ngm25/test_persistence.cpp
//
// ngm2.5 increment I7: persistence + exact resume (map R9). The hash-RNG payoff
// test, in-process and dataset-free: build a group, drive it k steps against a
// fixed input, checkpoint (save_state -> mem tree -> load_state into a fresh
// group), then drive both the original and the resumed group n more steps and
// assert their per-step outputs are BIT-IDENTICAL. A small branch_interval makes
// the window cross branch events, so arena growth + branch_epoch persistence are
// exercised too. If step_ (the hash-key input) weren't persisted, the resumed
// draws would diverge and this test would fail.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <variant>

#include "hd_ngm25/hd_ngm25_cfg.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#ifdef COAST_WITH_GPU
#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#endif
#include "sim_core/io_buffer.h"
#include "sim_core/io_entity.h"
#include "logging/mem_state_io.h"

using namespace ngm25;

namespace {

neuron_group_params_t small_group()
{
    neuron_group_params_t p;
    p.neuron_cnt      = 8;
    p.apical_cnt      = 0;
    p.proximal_cnt    = 2;
    p.max_branch_level = 3;
    // Tuned to force branch events inside the test window: a short interval and
    // a low accumulated-theta gate so qualifying outliers appear quickly. (The
    // full-scale schedule is validated on real mnist via the CLI resume run.)
    p.branch_interval          = 15;
    p.accumulated_theta_thres  = 0.05f;
    p.min_mismatch_percentage  = 0.002f;
    p.random_seed              = seed_t{ 7 };
    p.inputs                   = { { "in", bank_t::proximal } };
    return p;
}

// A group wired to a fixed-content input buffer and a private output buffer.
struct rig {
    cfg_store                            cs;
    std::unique_ptr<neuron_group_t>      grp;
    std::unique_ptr<sim::io_buffer_cpu>  in_buf;
    std::unique_ptr<sim::io_buffer_cpu>  out_buf;
    static constexpr std::size_t W = 16;
    static constexpr sim::output_id_t in_id = 0;

    explicit rig(const std::vector<float> &input, std::uint32_t seed_offset = 0, bool frozen = false)
    {
        neuron_group_params_t p = small_group();
        cs.neuron_groups["g"] = p;
        cs.frozen = frozen;
        grp = std::make_unique<neuron_group_t>(cs, "g", /*output_id=*/1, seed_offset_t{ seed_offset });
        grp->init_io({{"in", sim::output_channel_t{in_id, W}}});

        in_buf  = std::make_unique<sim::io_buffer_cpu>(W);
        out_buf = std::make_unique<sim::io_buffer_cpu>(grp->get_outp_size());
        auto w = in_buf->cur_write_buffer();
        std::copy(input.begin(), input.end(), w.begin());
        in_buf->swap_buffer();                       // fixed read page
        grp->set_input_buffer(in_id, *in_buf);
        grp->set_output_buffer(*out_buf);
    }

    // One step; returns the freshly written output (copied).
    std::vector<float> step()
    {
        REQUIRE(grp->process().has_value());
        auto o = out_buf->cur_write_buffer();
        return { o.begin(), o.end() };
    }
};

} // namespace

TEST_CASE("I7 exact resume: checkpoint/restore reproduces subsequent steps bit-for-bit") {
    std::vector<float> input(rig::W);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.2f + 0.7f * static_cast<float>((i * 3 + 1) % 7) / 7.0f;

    // The checkpoint goes 30 steps before the fixture's first growth, so the
    // post-resume window crosses a branch event. That step is found at runtime:
    // the run is chaotic, so it depends on floating-point codegen (179 with
    // AppleClang on arm64, later with GCC, where a hardcoded window missed it).
    int first_growth = -1;
    {
        rig probe(input);
        const std::size_t syn0 = probe.grp->get_synapse_count();
        for (int s = 0; s < 5000 && first_growth < 0; ++s) {
            probe.step();
            if (probe.grp->get_synapse_count() > syn0) first_growth = s;
        }
    }
    MESSAGE("first growth at step " << first_growth);
    REQUIRE(first_growth >= 0);
    const int k = std::max(1, first_growth - 30);   // steps before checkpoint
    constexpr int n = 60;                            // steps after

    rig a(input);
    for (int s = 0; s < k; ++s) a.step();

    // Checkpoint a's state and restore into a fresh group b.
    logging::mem_state_writer wtr;
    a.grp->save_state(wtr);
    const logging::mem_group tree = wtr.take_root();

    rig b(input);
    { logging::mem_state_reader rdr(tree, logging::k_format_version); b.grp->load_state(rdr); }

    // Structural agreement right after load.
    CHECK(b.grp->get_synapse_count() == a.grp->get_synapse_count());

    // Drive both n more steps; every step's output must match bit-for-bit.
    bool all_equal = true;
    bool saw_growth = false;
    const std::size_t syn_at_resume = a.grp->get_synapse_count();
    for (int s = 0; s < n; ++s) {
        const std::vector<float> oa = a.step();
        const std::vector<float> ob = b.step();
        if (oa != ob) { all_equal = false; break; }
        if (a.grp->get_synapse_count() > syn_at_resume) saw_growth = true;
    }
    CHECK(all_equal);
    CHECK(a.grp->get_synapse_count() == b.grp->get_synapse_count());
    CHECK(saw_growth);   // the window actually crossed a branch event
}

TEST_CASE("I7 exact resume: the checkpoint's seed is authoritative over a divergent "
          "resuming config") {
    // effective_seed_ is derived from (random_seed + seed_offset) at construction
    // and keys EVERY hash draw (D1 response penalty, D3 winner, D4 split, D5 init).
    // A checkpoint must reproduce the run that produced it even if the resuming
    // group was built with a different seed — otherwise "exact resume" is a lie.
    // Here b is built with a DIFFERENT seed_offset than a; load_state overwrites
    // b's arena with a's AND must restore a's effective_seed. Without the restore,
    // b keeps its own seed and its stochastic D1 draws diverge from a's on the very
    // first post-resume step. This test fails if effective_seed is not read back.
    std::vector<float> input(rig::W);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.2f + 0.7f * static_cast<float>((i * 3 + 1) % 7) / 7.0f;

    constexpr int k = 40;
    constexpr int n = 40;

    rig a(input, /*seed_offset=*/0);      // effective_seed = 7
    for (int s = 0; s < k; ++s) a.step();

    logging::mem_state_writer wtr;
    a.grp->save_state(wtr);
    const logging::mem_group tree = wtr.take_root();

    rig b(input, /*seed_offset=*/1000);   // effective_seed = 1007 (distinct stream)
    { logging::mem_state_reader rdr(tree, logging::k_format_version); b.grp->load_state(rdr); }

    CHECK(b.grp->get_synapse_count() == a.grp->get_synapse_count());

    bool all_equal = true;
    for (int s = 0; s < n; ++s) {
        const std::vector<float> oa = a.step();
        const std::vector<float> ob = b.step();
        if (oa != ob) { all_equal = false; break; }
    }
    CHECK(all_equal);
}

TEST_CASE("I7 load_state: a truncated checkpoint (no synapse payload) throws, "
          "leaving the group in its constructed state") {
    std::vector<float> input(rig::W);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.2f + 0.7f * static_cast<float>((i * 3 + 1) % 7) / 7.0f;

    // Forge a checkpoint whose structural header MATCHES a fresh group but which
    // carries no `perm` dataset (a truncated write). load_state must throw, and
    // before adopting anything: the group keeps its constructed state (step_ == 0)
    // rather than a stored step on a freshly-built arena, which would resume at a
    // bogus step with mismatched hash-key state. Structural fields are derived from
    // a fresh group so the structural guard passes and we exercise the empty-payload
    // path specifically.
    rig fresh(input);
    const auto ncnt = static_cast<std::uint64_t>(fresh.grp->get_neuron_count());
    const auto dcnt = static_cast<std::uint64_t>(fresh.grp->get_neuron_count()
                                                 * fresh.grp->get_proximal_count());  // apical_cnt==0
    const auto msid = static_cast<std::uint32_t>(fresh.grp->get_max_segment_id());

    logging::mem_state_writer wtr;
    wtr.write("neuron_cnt", ncnt);
    wtr.write("dend_count", dcnt);
    wtr.write("max_seg_id", msid);
    wtr.write("apical_cnt",   static_cast<std::uint64_t>(fresh.grp->get_apical_count()));
    wtr.write("proximal_cnt", static_cast<std::uint64_t>(fresh.grp->get_proximal_count()));
    wtr.write("step", static_cast<std::uint32_t>(999));   // would-be resumed step
    // deliberately NO "perm" (nor any other arena payload) — a truncated checkpoint
    logging::mem_group tree = wtr.take_root();
    // The channel records are structural too (format version 4): take the fresh
    // group's, so they pass as well.
    {
        logging::mem_state_writer fw;
        fresh.grp->save_state(fw);
        auto full = fw.take_root();
        for (const auto* name : { "apical_channels", "proximal_channels" })
            tree.groups[name] = full.groups.at(name);
    }

    rig loaded(input);
    std::string what;
    try {
        logging::mem_state_reader rdr(tree, logging::k_format_version);
        loaded.grp->load_state(rdr);
    } catch (const std::runtime_error& e) {
        what = e.what();
    }
    INFO("message: ", what);
    CHECK(what.find("corrupt checkpoint — perm missing or empty") != std::string::npos);

    // The refused group must be indistinguishable from a brand-new one: identical
    // first-step output. A stale step_=999 would key the D1 penalty draw differently
    // and diverge here.
    rig control(input);
    CHECK(loaded.step() == control.step());
}

TEST_CASE("I7 load_state: a header-matching but length-inconsistent checkpoint is rejected") {
    std::vector<float> input(rig::W);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.2f + 0.7f * static_cast<float>((i * 3 + 1) % 7) / 7.0f;

    // A well-formed checkpoint from a fresh group...
    rig src(input);
    logging::mem_state_writer wtr;
    src.grp->save_state(wtr);
    logging::mem_group tree = wtr.take_root();

    // ...then corrupt it: overwrite adapt_hist with a wrong-length vector so it no
    // longer matches perm/seg/input_idx (arena consistency invariant 1). The
    // structural header (neuron_cnt/dend_count/max_seg_id) still matches, so a
    // load_state that skips the payload consistency check would accept this and
    // then read out of bounds on the next process().
    tree.fields["adapt_hist"] = std::vector<float>(3, 0.0f);

    rig dst(input);
    logging::mem_state_reader rdr(tree, logging::k_format_version);
    CHECK_THROWS_AS(dst.grp->load_state(rdr), std::runtime_error);
}

TEST_CASE("I7 load_state: a checkpoint with a short per-neuron vector is rejected") {
    std::vector<float> input(rig::W);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.2f + 0.7f * static_cast<float>((i * 3 + 1) % 7) / 7.0f;

    // A well-formed checkpoint from a fresh group...
    rig src(input);
    logging::mem_state_writer wtr;
    src.grp->save_state(wtr);
    logging::mem_group tree = wtr.take_root();

    // ...then corrupt a PER-NEURON vector, not an arena vector: shorten
    // neuron_mx_act to length 3. Unlike adapt_hist (previous test), neuron_mx_act is
    // NOT covered by check_arena_consistency — it is a group member, not an arena
    // field. read_float_vec returns it verbatim (no throw), so the ONLY guard that
    // can catch this is the explicit per-neuron length check in load_state. Without
    // it, neuron_mx_act_ ends up size 3 and the next process() writes index [0,8) out
    // of bounds. The arena itself is untouched, so the arena consistency check passes
    // — this test therefore isolates the per-neuron guard specifically.
    tree.fields["neuron_mx_act"] = std::vector<float>(3, 0.0f);

    rig dst(input);
    logging::mem_state_reader rdr(tree, logging::k_format_version);
    CHECK_THROWS_AS(dst.grp->load_state(rdr), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Frozen mode: a pure forward pass — output is produced, but no learning state
// changes (no adapt, no branching), gated by params.frozen in process().
// ---------------------------------------------------------------------------
namespace {
// Flatten the proximal permanence mirror into one vector (learning writes perms).
std::vector<float> snapshot_perm(neuron_group_t &g)
{
    g.refresh_vis_mirror();
    std::vector<float> v;
    for (std::size_t n = 0; n < g.get_neuron_count(); ++n)
        for (std::size_t d = 0; d < g.get_proximal_count(); ++d)
            for (std::size_t r = 0; r < g.get_rep_count(); ++r) {
                const auto s = g.get_proximal_representation(n, d, r);
                v.insert(v.end(), s.begin(), s.end());
            }
    return v;
}
} // namespace

TEST_CASE("frozen group: forward pass runs but no learning state changes") {
    std::vector<float> input(rig::W);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.2f + 0.7f * static_cast<float>((i * 3 + 1) % 7) / 7.0f;

    constexpr int k = 40;   // spans several branch_interval boundaries (interval 15)

    rig frozen(input, /*seed_offset=*/0, /*frozen=*/true);
    const std::size_t     syn0        = frozen.grp->get_synapse_count();
    const std::vector<float> perm_before = snapshot_perm(*frozen.grp);

    std::vector<float> last;
    for (int s = 0; s < k; ++s) last = frozen.step();

    // Forward pass produced finite, in-range output.
    REQUIRE(last.size() == frozen.grp->get_neuron_count());
    for (float o : last) { CHECK(std::isfinite(o)); CHECK(o >= 0.0f); CHECK(o <= 1.0f); }

    // No learning: no branching, and adapt (which writes perms + mismatch) never ran.
    CHECK(frozen.grp->get_synapse_count() == syn0);
    CHECK(snapshot_perm(*frozen.grp) == perm_before);
    CHECK(frozen.grp->get_max_mismatch() == 0.0f);
    CHECK(frozen.grp->get_avg_mismatch() == 0.0f);

    // Non-vacuity: a NON-frozen group on the same input/steps DOES change — so the
    // frozen assertions above are meaningful, not trivially true of any group.
    rig live(input);
    for (int s = 0; s < k; ++s) live.step();
    CHECK((live.grp->get_synapse_count() > syn0 || live.grp->get_max_mismatch() > 0.0f));
}

// ---------------------------------------------------------------------------
// Apical/proximal split (checkpoint format version 4 on). dend_count alone
// does not tell apart an N-apical/M-proximal group from an M-apical/N-proximal
// one of the same total: a checkpoint from the wrong split is otherwise
// refused only by check_arena_consistency, after the arena is assigned. A
// pre-4 checkpoint has neither apical_cnt nor proximal_cnt and still loads;
// check_arena_consistency is what refuses one from another split then.
//
// The CPU and the GPU group share one checkpoint schema; these run without an
// adapter (no device plan), as the width tests above do.
// ---------------------------------------------------------------------------
namespace {

// A group with `apical_cnt` apical and `proximal_cnt` proximal dendrites per
// neuron, every bank wired to a same-width channel. Save and load need
// neither stepping nor buffers.
template <class G>
struct split_rig {
    cfg_store          cs;
    std::unique_ptr<G> grp;
    static constexpr std::size_t w = 12;

    // `inhibition`: a cfg_store inhibition entry to use; none = the default.
    split_rig(std::size_t apical_cnt, std::size_t proximal_cnt,
              const inhibition::inhibition_params_t* inhibition = nullptr)
    {
        neuron_group_params_t p = small_group();
        if (inhibition) {
            cs.inhibitions["inh"] = *inhibition;
            p.inhibition_cfg      = "inh";
        }
        p.apical_cnt   = apical_cnt;
        p.proximal_cnt = proximal_cnt;
        p.inputs       = { { "in", bank_t::proximal } };
        std::unordered_map<std::string, sim::output_channel_t> routes{
            { "in", sim::output_channel_t{ 0, w } }
        };
        if (apical_cnt > 0) {
            p.inputs.push_back({ "ctx", bank_t::apical });
            routes.emplace("ctx", sim::output_channel_t{ 1, w });
        }
        cs.neuron_groups["g"] = p;
        grp = std::make_unique<G>(cs, "g", /*output_id=*/1);
        grp->init_io(routes);
    }

    [[nodiscard]] logging::mem_group saved() const
    {
        logging::mem_state_writer w_;
        grp->save_state(w_);
        return w_.take_root();
    }

    // The message of the runtime_error a load of `tree`, as a checkpoint of
    // format `version`, throws; empty if the load succeeds.
    std::string load_error(const logging::mem_group& tree,
                           std::uint32_t version = logging::k_format_version)
    {
        try {
            logging::mem_state_reader r(tree, version);
            grp->load_state(r);
        } catch (const std::runtime_error& e) {
            return e.what();
        }
        return {};
    }
};

// Turns `tree` into what a pre-4 checkpoint holds: no split. Returns how many
// fields it removed.
std::size_t erase_split(logging::mem_group& tree)
{
    return tree.fields.erase("apical_cnt") + tree.fields.erase("proximal_cnt");
}

void advance_step(logging::mem_group& tree) { tree.fields.at("step") = std::uint32_t{ 40 }; }

bool contains(std::string_view s, std::string_view part) { return s.find(part) != std::string_view::npos; }

} // namespace

#ifdef COAST_WITH_GPU
#define NGM25_SPLIT_GROUPS neuron_group_t, neuron_group_gpu_t
#else
#define NGM25_SPLIT_GROUPS neuron_group_t
#endif

TEST_CASE_TEMPLATE("apical/proximal split: a checkpoint records both and loads back unchanged",
                   G, NGM25_SPLIT_GROUPS) {
    auto tree = split_rig<G>(1, 2).saved();
    CHECK(tree.fields.at("apical_cnt")   == logging::mem_value{ std::uint64_t{ 1 } });
    CHECK(tree.fields.at("proximal_cnt") == logging::mem_value{ std::uint64_t{ 2 } });
    advance_step(tree);

    split_rig<G> dst(1, 2);
    REQUIRE(dst.saved() != tree);
    CHECK(dst.load_error(tree).empty());
    CHECK(dst.saved() == tree);
}

TEST_CASE_TEMPLATE("apical/proximal split: a checkpoint from the swapped split is refused "
                   "before anything is assigned", G, NGM25_SPLIT_GROUPS) {
    // Same total dendrites per neuron (3) and the same channel widths, so
    // dend_count and max_seg_id both match: only apical_cnt/proximal_cnt tell
    // the two apart.
    auto tree = split_rig<G>(1, 2).saved();
    advance_step(tree);

    split_rig<G> dst(2, 1);
    const auto before = dst.saved();
    const auto err = dst.load_error(tree);
    INFO("message: ", err);
    CHECK(contains(err, "checkpoint does not fit this group"));
    CHECK(contains(err, "apical_cnt 1 (checkpoint) vs 2 (configured)"));
    CHECK(contains(err, "proximal_cnt 2 (checkpoint) vs 1 (configured)"));
    CHECK(dst.saved() == before);
}

TEST_CASE_TEMPLATE("apical/proximal split: a pre-4 checkpoint without it loads", G, NGM25_SPLIT_GROUPS) {
    auto tree = split_rig<G>(1, 2).saved();
    REQUIRE(erase_split(tree) == 2);
    advance_step(tree);

    split_rig<G> dst(1, 2);
    CHECK(dst.load_error(tree, 3).empty());
    auto resaved = dst.saved();
    REQUIRE(erase_split(resaved) == 2);
    CHECK(resaved == tree);
}

TEST_CASE_TEMPLATE("apical/proximal split: a pre-4 checkpoint from the swapped split is still "
                   "refused by the arena check, before anything is assigned", G, NGM25_SPLIT_GROUPS) {
    auto tree = split_rig<G>(1, 2).saved();
    REQUIRE(erase_split(tree) == 2);
    advance_step(tree);   // a partial apply would show as the stored step

    split_rig<G> dst(2, 1);
    const auto before = dst.saved();
    const auto err = dst.load_error(tree, 3);
    INFO("message: ", err);
    CHECK(contains(err, "corrupt checkpoint — arena inconsistent: inv2"));
    CHECK(dst.saved() == before);
}

// ---------------------------------------------------------------------------
// Any payload the header does not describe is checked before the group is
// touched (#189, M.7.18): load_state stages and validates the whole
// checkpoint, then commits. Each case corrupts one part of a well-formed
// checkpoint whose step is advanced, so a partial apply (the old order: hash-key
// state and arena assigned, then checked) shows in the re-saved state.
// ---------------------------------------------------------------------------
namespace {
template <class G>
void check_refused_unchanged(void (*corrupt)(logging::mem_group&), std::string_view needle)
{
    auto tree = split_rig<G>(1, 2).saved();
    advance_step(tree);
    tree.fields.at("branch_epoch")   = std::uint32_t{ 3 };
    tree.fields.at("effective_seed") = std::uint32_t{ 4242 };
    corrupt(tree);

    split_rig<G> dst(1, 2);
    const auto before = dst.saved();
    const auto err = dst.load_error(tree);
    INFO("message: ", err);
    CHECK(contains(err, needle));
    CHECK(dst.saved() == before);
}
} // namespace

TEST_CASE_TEMPLATE("staged load: a corrupt payload is refused before anything is assigned",
                   G, NGM25_SPLIT_GROUPS) {
    SUBCASE("a wrong-length arena vector") {
        check_refused_unchanged<G>([](logging::mem_group& t) {
            t.fields["adapt_hist"] = std::vector<float>(3, 0.0f);
        }, "corrupt checkpoint — arena inconsistent: inv1");
    }
    SUBCASE("an out-of-range permanence") {
        check_refused_unchanged<G>([](logging::mem_group& t) {
            std::get<std::vector<float>>(t.fields.at("perm")).at(0) = 2.0f;
        }, "corrupt checkpoint — arena inconsistent");
    }
    SUBCASE("a short per-neuron vector") {
        check_refused_unchanged<G>([](logging::mem_group& t) {
            t.fields["neuron_mx_act"] = std::vector<float>(3, 0.0f);
        }, "corrupt checkpoint — per-neuron vectors");
    }
    SUBCASE("a dendrite directory of another split (same lengths, apical/proximal flipped)") {
        check_refused_unchanged<G>([](logging::mem_group& t) {
            auto& dt = std::get<std::vector<std::uint8_t>>(t.fields.at("dend_type"));
            dt.at(0) = dt.at(0) == 0 ? 1 : 0;
        }, "corrupt checkpoint — arena inconsistent: inv2");
    }
}

// The inhibitor is loaded before the group commits and is all or nothing
// itself, so a misfit inhibition part leaves the whole group unchanged.
// random_index is the only stateful kind, and CPU only.
TEST_CASE("staged load: a misfit inhibition matrix is refused before anything is assigned") {
    inhibition::inhibition_params_t ri;
    ri.type        = inhibition::inhibition_type_t::random_index;
    ri.ri_reach    = 0.3;
    ri.ri_strength = 0.5;
    ri.ri_seed     = seed_t{ 5 };

    auto tree = split_rig<neuron_group_t>(1, 2, &ri).saved();
    advance_step(tree);
    auto& inh = tree.groups.at("inhibition");
    const std::size_t wrong = small_group().neuron_cnt + 1;
    inh.fields["initialized"] = true;
    inh.fields["n"]           = static_cast<std::uint64_t>(wrong);
    inh.fields["matrix"]      = logging::mem_array2d{ std::vector<float>(wrong * wrong, 0.5f),
                                                      wrong, wrong };

    split_rig<neuron_group_t> dst(1, 2, &ri);
    const auto before = dst.saved();
    const auto err = dst.load_error(tree);
    INFO("message: ", err);
    CHECK(contains(err, "the group has 8 neurons"));
    CHECK(dst.saved() == before);
}
