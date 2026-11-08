// tests/unit/hd_ngm2/test_state_shape.cpp
//
// ngm2 refuses a checkpoint whose synapses do not fit the dendrite it loads into.
// A dendrite's synapse arrays take the checkpoint's length (they grow by
// branching), so their length alone says nothing about the input. Without these
// checks a checkpoint from a wider input loaded silently and process() then read
// past the end of the input page, and a segment id beyond this dendrite's tree
// wrote past segment_activity. No dataset or IO buffer is needed: a load does not
// touch the input, and save/load run through the in-memory state IO.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "diag_capture.h"
#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "logging/mem_state_io.h"
#include "logging/state_checks.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using logging::mem_group;
using logging::mem_state_reader;
using logging::mem_state_writer;

namespace {

constexpr sim::output_id_t k_id    = 0;
constexpr sim::output_id_t k_in_id = 1;

ngm2::cfg_store make_store()
{
    ngm2::cfg_store cs;
    ngm2::dendrite_params_t dp;
    dp.input_channels   = { "da" };
    dp.max_branch_level = 3;          // segment ids 1 .. 15
    cs.dendrites["prox"] = dp;

    ngm2::neuron_params_t np;
    np.input_channels   = { "na" };
    np.dendrites        = { { "p", "prox", 2 } };
    np.channel_mappings = { { "na", "p", "da" } };
    cs.neurons["n"] = np;

    ngm2::neuron_group_params_t gp;
    gp.input_channels   = { "ga" };
    gp.neurons          = { { "pyr", "n", 3 } };
    gp.channel_mappings = { { "ga", "pyr", "na" } };
    cs.neuron_groups["g"] = gp;
    return cs;
}

// A group wired to one input channel of `width` values.
std::unique_ptr<ngm2::neuron_group_t> make_group(std::size_t width)
{
    auto g = std::make_unique<ngm2::neuron_group_t>(make_store(), "g", k_id);
    g->init_io({ { "ga", sim::output_channel_t{ k_in_id, width } } });
    return g;
}

mem_group saved(const ngm2::neuron_group_t& g)
{
    mem_state_writer w;
    g.save_state(w);
    return w.take_root();
}

// The message of the runtime_error a load of `tree` into `g`, read as format
// `version`, throws; empty if the load succeeds.
std::string load_error(ngm2::neuron_group_t& g, const mem_group& tree,
                       std::uint32_t version = logging::k_format_version)
{
    try {
        mem_state_reader r(tree, version);
        g.load_state(r);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

bool contains(std::string_view s, std::string_view part) { return s.find(part) != std::string_view::npos; }

} // namespace

// From format version 4 the dendrite's channel record (M.7.5) names each channel
// with its width, and refuses first. Before it, the dendrite's own check on its
// synapses' input width is what refuses.
TEST_CASE("a checkpoint from a wider input is refused")
{
    const auto tree = saved(*make_group(16));
    auto g = make_group(12);
    const auto err = load_error(*g, tree);
    CHECK(contains(err, "does not fit"));
    CHECK(contains(err, "input_channels [ga (16)] (checkpoint) vs [ga (12)] (configured)"));
    const auto old = load_error(*g, tree, 3);
    CHECK(contains(old, "does not fit"));
    CHECK(contains(old, "input width 16 (checkpoint) vs 12 (configured)"));
}

TEST_CASE("a checkpoint from a narrower input is refused")
{
    const auto tree = saved(*make_group(12));
    auto g = make_group(16);
    CHECK(contains(load_error(*g, tree), "input_channels [ga (12)] (checkpoint) vs [ga (16)] (configured)"));
    CHECK(contains(load_error(*g, tree, 3), "input width 12 (checkpoint) vs 16 (configured)"));
}

TEST_CASE("a checkpoint with a segment id beyond this dendrite's tree is refused")
{
    auto tree = saved(*make_group(12));
    auto& seg = std::get<std::vector<std::uint16_t>>(
        tree.groups.at("neurons").groups.at("1").groups.at("dendrites").groups.at("0").fields.at("segment_idx"));
    seg.back() = 16;                  // one past max_segment_idx at max_branch_level 3

    auto g = make_group(12);
    CHECK(contains(load_error(*g, tree), "segment id 16 (checkpoint) vs at most 15 (configured)"));
}

TEST_CASE("a checkpoint with a segment id beyond this dendrite's tree is refused "
          "before anything is assigned")
{
    // The bad segment id sits on neuron 1's dendrite; the group's own rng and
    // neuron 0 come earlier in load_state's loop. If the depth check ran only
    // where it used to -- inside dendrite_t::load_state, reached after the
    // group's rng and neuron 0 have already loaded -- this checkpoint's
    // different rng and input_count would end up adopted despite the throw.
    auto g = make_group(12);
    const auto before = saved(*g);

    auto tree = saved(*make_group(12));
    std::get<std::uint64_t>(tree.groups.at("rng").fields.at("mti")) += 1;
    std::get<std::uint64_t>(tree.groups.at("neurons").groups.at("0").fields.at("input_count")) += 1;
    auto& seg = std::get<std::vector<std::uint16_t>>(
        tree.groups.at("neurons").groups.at("1").groups.at("dendrites").groups.at("0").fields.at("segment_idx"));
    seg.back() = 16;

    CHECK(contains(load_error(*g, tree), "segment id 16 (checkpoint) vs at most 15 (configured)"));
    CHECK(saved(*g) == before);
}

TEST_CASE("a checkpoint of the same shape loads")
{
    const auto tree = saved(*make_group(12));
    auto g = make_group(12);
    CHECK(load_error(*g, tree).empty());
}

// ---------------------------------------------------------------------------
// Dendrite type and tree depth (#236, SPEC.md M.7.18). Counts and input widths
// say nothing about either: a checkpoint of a proximal dendrite loads into an
// apical one of the same width, and a depth-3 tree's segment ids all fit a
// depth-4 tree. From format version 5 every dendrite records both and a load
// refuses a difference; older checkpoints are checked against what their
// configuration record holds.
// ---------------------------------------------------------------------------

namespace {

// Two dendrites per neuron on one shared channel: "d" of the given type and
// depth, and "p", always proximal (M.7.1: every neuron keeps one).
ngm2::cfg_store make_typed_store(ngm2::dendrite_type_t d_type, std::uint8_t d_depth)
{
    ngm2::cfg_store cs;
    ngm2::dendrite_params_t dp;
    dp.input_channels   = { "da" };
    dp.max_branch_level = d_depth;
    dp.type             = d_type;
    cs.dendrites["d"] = dp;
    dp.max_branch_level = 3;
    dp.type             = ngm2::dendrite_type_t::proximal;
    cs.dendrites["p"] = dp;

    ngm2::neuron_params_t np;
    np.input_channels   = { "na" };
    np.dendrites        = { { "d", "d", 1 }, { "p", "p", 1 } };
    np.channel_mappings = { { "na", "d", "da" }, { "na", "p", "da" } };
    cs.neurons["n"] = np;

    ngm2::neuron_group_params_t gp;
    gp.input_channels   = { "ga" };
    gp.neurons          = { { "pyr", "n", 3 } };
    gp.channel_mappings = { { "ga", "pyr", "na" } };
    cs.neuron_groups["g"] = gp;
    return cs;
}

std::unique_ptr<ngm2::neuron_group_t> make_typed_group(ngm2::dendrite_type_t d_type, std::uint8_t d_depth,
                                                       std::size_t width = 12)
{
    auto g = std::make_unique<ngm2::neuron_group_t>(make_typed_store(d_type, d_depth), "g", k_id);
    g->init_io({ { "ga", sim::output_channel_t{ k_in_id, width } } });
    return g;
}

constexpr auto k_prox = ngm2::dendrite_type_t::proximal;
constexpr auto k_apic = ngm2::dendrite_type_t::apical;

// The checkpoint as a format-4 build wrote it: no per-dendrite type or depth.
mem_group as_format_4(mem_group tree)
{
    for (auto& [ni, n] : tree.groups.at("neurons").groups)
        for (auto& [di, d] : n.groups.at("dendrites").groups) {
            d.fields.erase("apical");
            d.fields.erase("max_branch_level");
        }
    return tree;
}

// The group's own rng and neuron 0 are what load_state assigns first; a
// checkpoint that changes them shows whether a refusal came too late.
mem_group with_other_rng(mem_group tree)
{
    std::get<std::uint64_t>(tree.groups.at("rng").fields.at("mti")) += 1;
    std::get<std::uint64_t>(tree.groups.at("neurons").groups.at("0").fields.at("input_count")) += 1;
    return tree;
}

} // namespace

TEST_CASE("each dendrite records its type and its tree depth")
{
    const auto tree = saved(*make_typed_group(k_apic, 4));
    const auto& dends = tree.groups.at("neurons").groups.at("2").groups.at("dendrites").groups;
    CHECK(std::get<std::uint64_t>(dends.at("0").fields.at("apical")) == 1);
    CHECK(std::get<std::uint64_t>(dends.at("0").fields.at("max_branch_level")) == 4);
    CHECK(std::get<std::uint64_t>(dends.at("1").fields.at("apical")) == 0);
    CHECK(std::get<std::uint64_t>(dends.at("1").fields.at("max_branch_level")) == 3);
}

TEST_CASE("a dendrite type change at constant counts and width is refused, before anything is assigned")
{
    const auto tree = with_other_rng(saved(*make_typed_group(k_prox, 3)));
    auto g = make_typed_group(k_apic, 3);
    const auto before = saved(*g);
    const auto err = load_error(*g, tree);
    CHECK(contains(err, "does not fit"));
    CHECK(contains(err, "neuron 0: dendrite 0"));
    CHECK(contains(err, "apical 0 (checkpoint) vs 1 (configured)"));
    CHECK(saved(*g) == before);

    // And the other way round.
    auto h = make_typed_group(k_prox, 3);
    CHECK(contains(load_error(*h, saved(*make_typed_group(k_apic, 3))), "apical 1 (checkpoint) vs 0 (configured)"));
}

TEST_CASE("a tree depth change is refused although every saved segment id fits, before anything is assigned")
{
    // A fresh depth-3 tree holds segment 1 only, which a depth-4 tree has too, so
    // the segment-id check alone lets this through.
    const auto tree = with_other_rng(saved(*make_typed_group(k_prox, 3)));
    auto g = make_typed_group(k_prox, 4);
    const auto before = saved(*g);
    const auto err = load_error(*g, tree);
    CHECK(contains(err, "does not fit"));
    CHECK(contains(err, "max_branch_level 3 (checkpoint) vs 4 (configured)"));
    CHECK(saved(*g) == before);

    auto h = make_typed_group(k_prox, 2);
    CHECK(contains(load_error(*h, saved(*make_typed_group(k_prox, 3))), "max_branch_level 3 (checkpoint) vs 2 (configured)"));
}

TEST_CASE("a format-5 checkpoint without a dendrite's type or depth is refused")
{
    const auto tree = as_format_4(saved(*make_typed_group(k_prox, 3)));
    auto g = make_typed_group(k_prox, 3);
    const auto err = load_error(*g, tree);
    CHECK(contains(err, "apical missing"));
    CHECK(contains(err, "max_branch_level missing"));
}

// Older checkpoints: the depth was recorded as a setting in the configuration
// record (dendrite[<cfg>].max_branch_level), so a difference there refuses the
// load too, before anything is assigned. The type was recorded nowhere: such a
// checkpoint loads, unchecked for it, and the load says so.
TEST_CASE("an older checkpoint is refused if its configuration record shows another tree depth")
{
    const auto tree = with_other_rng(as_format_4(saved(*make_typed_group(k_prox, 3))));
    auto g = make_typed_group(k_prox, 4);
    const auto before = saved(*g);
    const auto err = load_error(*g, tree, 4);
    CHECK(contains(err, "does not fit"));
    CHECK(contains(err, "dendrite[d].max_branch_level 3 (checkpoint's configuration record) vs 4 (configured)"));
    CHECK(saved(*g) == before);
}

TEST_CASE("an older checkpoint of the same depth loads, its dendrite types unchecked")
{
    const auto tree = as_format_4(saved(*make_typed_group(k_prox, 3)));
    auto same = make_typed_group(k_prox, 3);
    CHECK(load_error(*same, tree, 4).empty());
    auto other_type = make_typed_group(k_apic, 3);   // not detectable from a format-4 checkpoint
    CHECK(load_error(*other_type, tree, 4).empty());

    // Without a configuration record (older still) the depth cannot be checked either.
    auto bare = tree;
    bare.groups.erase("config");
    auto deeper = make_typed_group(k_prox, 4);
    CHECK(load_error(*deeper, bare, 4).empty());
}

TEST_CASE("a checkpoint of the same types and depths loads")
{
    const auto tree = saved(*make_typed_group(k_apic, 4));
    auto g = make_typed_group(k_apic, 4);
    CHECK(load_error(*g, tree).empty());
}

// What an older checkpoint could not be checked for is reported once per
// checkpoint, as the load that owns the reader does after its entity loop.
TEST_CASE("an older checkpoint's unchecked dendrite types are reported in one notice")
{
    auto notices = [](ngm2::neuron_group_t& g, const mem_group& tree, std::uint32_t version) {
        coast::test::diag_capture cap;
        mem_state_reader r(tree, version);
        g.load_state(r);
        logging::report_unchecked_dendrite_types(r);
        std::vector<std::string> out;
        for (const auto& rec : cap.got)
            if (rec.cat == "checkpoint" && rec.lvl == coast::diag::level::info) out.push_back(rec.msg);
        return out;
    };
    const auto tree = as_format_4(saved(*make_typed_group(k_prox, 3)));
    auto g = make_typed_group(k_prox, 3);
    auto n = notices(*g, tree, 4);
    REQUIRE(n.size() == 1);
    CHECK(contains(n[0], "does not record the dendrite types (apical/proximal) (ngm2 neuron group 0)"));

    auto bare = tree;
    bare.groups.erase("config");          // also the "no configuration record" case
    n = notices(*g, bare, 4);
    REQUIRE(n.size() == 1);
    CHECK(contains(n[0], "(ngm2 neuron group 0 (nor the tree depth of dendrite cfg d, p))"));

    CHECK(notices(*g, saved(*make_typed_group(k_prox, 3)), logging::k_format_version).empty());
}
