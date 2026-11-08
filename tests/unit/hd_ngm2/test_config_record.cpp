// tests/unit/hd_ngm2/test_config_record.cpp
//
// ngm2's configuration record (SPEC.md M.7.8): what neuron_group_t::current_config
// records, and what a load reports. Constructing a group needs only a cfg_store —
// no IO wiring — and save/load run through the in-memory state IO.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "diag/diag.h"
#include "diag_capture.h"
#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "logging/config_record.h"
#include "logging/mem_state_io.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

using logging::mem_group;
using logging::mem_state_reader;
using logging::mem_state_writer;

namespace {

constexpr sim::output_id_t k_id = 0;

ngm2::cfg_store make_store()
{
    ngm2::cfg_store cs;
    cs.neurons["default"].dendrites            = { { "d", "default", 2 } };
    cs.neuron_groups["default"].neurons        = { { "n", "default", 4 } };
    return cs;
}

mem_group saved(const ngm2::neuron_group_t& g)
{
    mem_state_writer w;
    g.save_state(w);
    return w.take_root();
}

// The checkpoint warnings a load of `tree` into `g` emits.
std::vector<std::string> load_warnings(ngm2::neuron_group_t& g, const mem_group& tree)
{
    coast::test::diag_capture cap;   // resets the sink even if load_state throws
    mem_state_reader r(tree, logging::k_format_version);
    g.load_state(r);
    return cap.warnings("checkpoint");
}

// A setter changes one instance alone; the console and probes reach it through
// the element tree, as here.
ngm2::neuron_t& neuron(ngm2::neuron_group_t& g, std::size_t i) { return *g.child<ngm2::neuron_t>(i); }

const logging::config_value& entry(const logging::config_record& rec, const std::string& name)
{
    for (const auto& e : rec)
        if (e.name == name) return e.value;
    FAIL("no record entry " << name);
    throw;   // unreachable
}

bool has_entry(const logging::config_record& rec, const std::string& name)
{
    for (const auto& e : rec)
        if (e.name == name) return true;
    return false;
}

ngm2::cfg_store make_ridx_store(std::uint32_t seed, bool from_time)
{
    auto cs = make_store();
    auto& ridx = cs.inhibitions["ridx"];
    ridx.type              = inhibition::inhibition_type_t::random_index;
    ridx.ri_seed           = seed_t{ seed };
    ridx.ri_seed_from_time = from_time;
    cs.neuron_groups["default"].inhibition_cfg = "ridx";
    return cs;
}

} // namespace

// Finding 3 of the equivalence review: the record showed ri_seed as configured,
// not the seed the inhibitor drew and runs with. A seed drawn from time is random
// state, not a setting, so the record leaves it out and the state saves it.
TEST_CASE("a seed drawn from time is saved with the state, not recorded as a setting")
{
    const auto cs = make_ridx_store(/*seed=*/7, /*from_time=*/true);
    const ngm2::neuron_group_t g(cs, "default", k_id);
    CHECK_FALSE(has_entry(g.current_config(), "inhibition.ri_seed"));
    CHECK(std::get<bool>(entry(g.current_config(), "inhibition.ri_seed_from_time")));

    const auto root = saved(g);
    CHECK(std::get<std::uint64_t>(root.groups.at("inhibition").fields.at("seed")) != 7);
    CHECK_FALSE(root.groups.at("config").fields.contains("inhibition.ri_seed"));
}

// Every construction draws a fresh seed, so recording it made every resume of a
// from-time run report a seed change its configuration never made.
TEST_CASE("a from-time run resumes under its unchanged configuration without a warning")
{
    const auto cs = make_ridx_store(/*seed=*/7, /*from_time=*/true);
    const auto tree = saved(ngm2::neuron_group_t(cs, "default", k_id));

    ngm2::neuron_group_t resumed(cs, "default", k_id);
    CHECK(load_warnings(resumed, tree).empty());
}

TEST_CASE("a changed fixed seed is still reported, as not applied")
{
    const auto tree = saved(ngm2::neuron_group_t(make_ridx_store(7, false), "default", k_id));

    ngm2::neuron_group_t changed(make_ridx_store(8, false), "default", k_id);
    const auto w = load_warnings(changed, tree);
    REQUIRE(w.size() == 1);
    CHECK(w[0].find("inhibition.ri_seed 7 -> 8 (not applied") != std::string::npos);
}

// Finding 2 of the equivalence review: the record held only the first instance of
// each cfg, so a change to any other instance was neither recorded nor reported.
TEST_CASE("a neuron changed before the save is reported when a fresh group loads it")
{
    const auto cs = make_store();
    ngm2::neuron_group_t changed(cs, "default", k_id);
    neuron(changed, 2).set_branch_interval(7000);
    const auto tree = saved(changed);

    ngm2::neuron_group_t fresh(cs, "default", k_id);
    const auto w = load_warnings(fresh, tree);
    REQUIRE(w.size() == 1);
    CHECK(w[0].ends_with(": neuron[default]#2.branch_interval 7000 -> (not in the current configuration)"));
}

TEST_CASE("a neuron changed after the save is reported when it loads the checkpoint")
{
    const auto cs = make_store();
    const auto tree = saved(ngm2::neuron_group_t(cs, "default", k_id));

    ngm2::neuron_group_t changed(cs, "default", k_id);
    neuron(changed, 2).set_branch_interval(7000);
    const auto w = load_warnings(changed, tree);
    REQUIRE(w.size() == 1);
    CHECK(w[0].ends_with(": neuron[default]#2.branch_interval (not recorded) -> 7000"));
}

TEST_CASE("a dendrite changed before the save is named by its neuron and its index")
{
    const auto cs = make_store();
    ngm2::neuron_group_t changed(cs, "default", k_id);
    neuron(changed, 2).child<ngm2::dendrite_t>(1)->set_primary_learning_rate(0.02f);
    const auto tree = saved(changed);

    ngm2::neuron_group_t fresh(cs, "default", k_id);
    const auto w = load_warnings(fresh, tree);
    REQUIRE(w.size() == 1);
    CHECK(w[0].ends_with(": dendrite[default]#2:1.primary_learning_rate 0.02 -> (not in the current configuration)"));
}

// The first instance stays the cfg's reference: a change to it is a change of the
// cfg entry, and the instances that still hold the old value are recorded as
// differing from it.
TEST_CASE("the first instance is the cfg's reference")
{
    const auto cs = make_store();
    ngm2::neuron_group_t changed(cs, "default", k_id);
    neuron(changed, 0).set_branch_interval(7000);
    const auto rec = changed.current_config();
    CHECK(std::get<std::uint64_t>(entry(rec, "neuron[default].branch_interval")) == 7000);
    for (std::size_t i = 1; i < 4; ++i)
        CHECK(std::get<std::uint64_t>(entry(rec, "neuron[default]#" + std::to_string(i) + ".branch_interval")) == 5000);
}

// Nothing differs, nothing extra: the record is the cfg-level one alone, and a
// load of an unchanged group reports nothing.
TEST_CASE("an unchanged group records no instance entries and loads silently")
{
    const auto cs = make_store();
    const ngm2::neuron_group_t g(cs, "default", k_id);
    for (const auto& e : g.current_config()) {
        CAPTURE(e.name);
        CHECK(e.name.find('#') == std::string::npos);
    }
    ngm2::neuron_group_t fresh(cs, "default", k_id);
    CHECK(load_warnings(fresh, saved(g)).empty());
}

// #239 (SPEC.md M.2.4): a load restores the checkpoint's random state, so the
// seed that state came from is the checkpoint's too. effective_seed() — what the
// run manifest reports — and the next checkpoint must name it, not the seed the
// loading group was configured with.
TEST_CASE("after a load, the effective seed is the checkpoint's, and a re-save records it")
{
    const auto cs = make_store();
    const ngm2::neuron_group_t src(cs, "default", k_id, seed_offset_t{ 7 });
    REQUIRE(src.effective_seed() == seed_t{ 7 });
    const auto tree = saved(src);
    CHECK(std::get<std::uint32_t>(tree.fields.at("effective_seed")) == 7);

    ngm2::neuron_group_t dst(cs, "default", k_id, seed_offset_t{ 99 });
    REQUIRE(dst.effective_seed() == seed_t{ 99 });
    const auto w = load_warnings(dst, tree);
    // The configured seed is still reported as not applied.
    REQUIRE(w.size() == 1);
    CHECK(w[0].find("effective_seed 7 -> 99 (not applied") != std::string::npos);
    CHECK(dst.effective_seed() == seed_t{ 7 });

    const auto resaved = saved(dst);
    CHECK(std::get<std::uint32_t>(resaved.fields.at("effective_seed")) == 7);
    CHECK(std::get<std::uint64_t>(resaved.groups.at("config").fields.at("effective_seed")) == 7);
    // Loading the re-saved checkpoint back under the seed-7 configuration
    // reports no seed change: the provenance chain is unbroken.
    ngm2::neuron_group_t again(cs, "default", k_id, seed_offset_t{ 7 });
    CHECK(load_warnings(again, resaved).empty());
}

// A checkpoint written before the dedicated field (format 4 and older) still
// names its seed in its configuration record; one older still has neither, and
// the group keeps its constructed seed (state_checks.h, pre-field rule).
TEST_CASE("an older checkpoint's seed is taken from its configuration record, if it has one")
{
    const auto cs = make_store();
    auto tree = saved(ngm2::neuron_group_t(cs, "default", k_id, seed_offset_t{ 7 }));
    tree.fields.erase("effective_seed");

    ngm2::neuron_group_t dst(cs, "default", k_id, seed_offset_t{ 99 });
    (void)load_warnings(dst, tree);
    CHECK(dst.effective_seed() == seed_t{ 7 });

    tree.groups.erase("config");
    ngm2::neuron_group_t bare(cs, "default", k_id, seed_offset_t{ 99 });
    (void)load_warnings(bare, tree);
    CHECK(bare.effective_seed() == seed_t{ 99 });
}
