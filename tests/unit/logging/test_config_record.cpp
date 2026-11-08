// tests/unit/logging/test_config_record.cpp
//
// The configuration record a checkpoint carries (SPEC.md M.7.8): written by each
// neuron group, compared on load, every difference reported in one warning, seeds
// reported as not applied, and a checkpoint without a record reported once per
// load: its groups are noted on the load's own reader, never on shared state.
// Beside it, the channel record (SPEC.md M.7.5): a layout that differs refuses
// the load, and a checkpoint without one is reported once per load the same way.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "diag/diag.h"
#include "diag_capture.h"
#include "logging/config_record.h"
#include "logging/hdf5_utils.h"
#include "logging/mem_state_io.h"
#include "logging/state_logger.h"   // hdf5_state_writer / hdf5_state_reader
#include "hdf5/executor.h"
#include "sim_core/io_entity.h"
#include "sim_core/sim_env.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

using namespace logging;
namespace fs = std::filesystem;

namespace {

using capture = coast::test::diag_capture;

config_record sample()
{
    return {
        { "common_learning_rate", 0.0001f },
        { "weight_filter.steepness", 0.5 },
        { "branch_interval", std::uint64_t{ 5000 } },
        { "neuron[n].random_seed", std::int64_t{ -3 }, /*random_state=*/true },
        { "frozen", false },
        { "inhibition.type", std::string("max_softmax") },
        { "effective_seed", std::uint64_t{ 7 }, /*random_state=*/true },
    };
}

mem_group saved(const config_record& rec)
{
    mem_state_writer w;
    write_config_record(w, rec);
    return w.take_root();
}

config_record with(config_record rec, const std::string& name, config_value v)
{
    for (auto& e : rec)
        if (e.name == name) e.value = std::move(v);
    return rec;
}

// The notice for a checkpoint whose groups "group 1" and "group 2" have no record.
const std::string k_no_record_notice =
    "the checkpoint has no configuration record (group 1, group 2); settings were not compared";

// A group whose state has no configuration record, as one saved before records
// existed: load_state checks the record under its name.
struct recordless_entity : sim::io_entity {
    sim::output_id_t oid;
    std::string      who;

    recordless_entity(sim::output_id_t id, std::string name) : oid(id), who(std::move(name)) {}

    sim::step_result process() override { return {}; }
    void save_state(logging::state_writer& w) const override { w.write("x", std::uint32_t{ 1 }); }
    void load_state(logging::state_reader& r) override { check_config_record(r, sample(), who); }

    [[nodiscard]] sim::output_id_t get_outp_id() const override { return oid; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 0; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

} // namespace

TEST_CASE("an unchanged configuration matches its record and reports nothing")
{
    capture cap;
    const auto tree = saved(sample());
    mem_state_reader r(tree, logging::k_format_version);
    const auto d = diff_config_record(r, sample());
    REQUIRE(d.has_value());
    CHECK(d->empty());

    mem_state_reader r2(tree, logging::k_format_version);
    check_config_record(r2, sample(), "group 1");
    CHECK(cap.got.empty());
}

TEST_CASE("every differing setting is listed, recorded -> current, in one warning")
{
    capture cap;
    const auto tree = saved(sample());
    auto cur = with(sample(), "common_learning_rate", 0.001f);
    cur      = with(cur, "inhibition.type", std::string("random_index"));
    cur      = with(cur, "frozen", true);

    mem_state_reader r(tree, logging::k_format_version);
    const auto d = diff_config_record(r, cur);
    REQUIRE(d.has_value());
    CHECK(*d == std::vector<std::string>{ "common_learning_rate 1e-04 -> 0.001",
                                          "frozen false -> true",
                                          "inhibition.type max_softmax -> random_index" });

    mem_state_reader r2(tree, logging::k_format_version);
    check_config_record(r2, cur, "group 1");
    REQUIRE(cap.got.size() == 1);
    CHECK(cap.got[0].lvl == coast::diag::level::warn);
    CHECK(cap.got[0].cat == "checkpoint");
    CHECK(cap.got[0].msg.find("group 1") != std::string::npos);
    CHECK(cap.got[0].msg.find("common_learning_rate 1e-04 -> 0.001; frozen false -> true; "
                              "inhibition.type max_softmax -> random_index") != std::string::npos);
}

TEST_CASE("a differing seed is reported as not applied")
{
    const auto tree = saved(sample());
    mem_state_reader r(tree, logging::k_format_version);
    const auto d = diff_config_record(r, with(sample(), "effective_seed", std::uint64_t{ 8 }));
    REQUIRE(d.has_value());
    REQUIRE(d->size() == 1);
    CHECK((*d)[0] == "effective_seed 7 -> 8 (not applied: the random state comes from the checkpoint)");
}

TEST_CASE("a setting the record lacks is reported as not recorded")
{
    auto old = sample();
    old.pop_back();   // a record written before effective_seed existed
    const auto tree = saved(old);
    mem_state_reader r(tree, logging::k_format_version);
    const auto d = diff_config_record(r, sample());
    REQUIRE(d.has_value());
    REQUIRE(d->size() == 1);
    CHECK((*d)[0].starts_with("effective_seed (not recorded) -> 7"));
}

TEST_CASE("no record: each group is noted on its reader, in order, and nothing is logged")
{
    const mem_group empty;   // a checkpoint written before records existed
    capture cap;
    mem_state_reader r(empty, logging::k_format_version);
    CHECK_FALSE(diff_config_record(r, sample()).has_value());
    check_config_record(r, sample(), "group 1");
    check_config_record(r, sample(), "group 2");
    CHECK(cap.got.empty());   // held until the load reports them
    CHECK(r.take_unrecorded_configs() == std::vector<std::string>{ "group 1", "group 2" });
    CHECK(r.take_unrecorded_configs().empty());   // taking them empties the reader
}

TEST_CASE("no record: the groups of one load are reported in one notice")
{
    const mem_group empty;
    capture cap;
    mem_state_reader r(empty, logging::k_format_version);
    check_config_record(r, sample(), "group 1");
    check_config_record(r, sample(), "group 2");
    report_unrecorded_configs(r);
    REQUIRE(cap.got.size() == 1);
    CHECK(cap.got[0].lvl == coast::diag::level::info);
    CHECK(cap.got[0].cat == "checkpoint");
    CHECK(cap.got[0].msg == k_no_record_notice);

    report_unrecorded_configs(r);   // already taken: nothing more to report
    CHECK(cap.got.size() == 1);
}

TEST_CASE("a difference in settings is logged at once as a warning, not noted on the reader")
{
    capture cap;
    const auto tree = saved(sample());
    mem_state_reader r(tree, logging::k_format_version);
    check_config_record(r, with(sample(), "frozen", true), "group 1");
    REQUIRE(cap.got.size() == 1);
    CHECK(cap.got[0].lvl == coast::diag::level::warn);
    CHECK(r.take_unrecorded_configs().empty());
}

TEST_CASE("no record: readers used one after the other or interleaved keep their own groups")
{
    const mem_group empty;
    capture cap;
    {
        mem_state_reader a(empty, logging::k_format_version);
        check_config_record(a, sample(), "a 1");
        CHECK(a.take_unrecorded_configs() == std::vector<std::string>{ "a 1" });
        mem_state_reader b(empty, logging::k_format_version);
        check_config_record(b, sample(), "b 1");
        CHECK(b.take_unrecorded_configs() == std::vector<std::string>{ "b 1" });
    }
    {
        mem_state_reader a(empty, logging::k_format_version);
        mem_state_reader b(empty, logging::k_format_version);
        check_config_record(a, sample(), "a 1");
        check_config_record(b, sample(), "b 1");
        check_config_record(a, sample(), "a 2");
        check_config_record(b, sample(), "b 2");
        CHECK(a.take_unrecorded_configs() == std::vector<std::string>{ "a 1", "a 2" });
        CHECK(b.take_unrecorded_configs() == std::vector<std::string>{ "b 1", "b 2" });
    }
    CHECK(cap.got.empty());
}

TEST_CASE("no record: a load through state_logger reports every group in exactly one notice")
{
    // Two groups whose checkpoint has no record, loaded by both load paths.
    sim::env       env;
    hdf5::executor h5;
    env.emplace_back<recordless_entity>(sim::output_id_t{ 4 }, "group 1");
    env.emplace_back<recordless_entity>(sim::output_id_t{ 5 }, "group 2");
    const fs::path path = fs::temp_directory_path() /
                          ("coast_config_record_e2e_" + std::to_string(::getpid()) + ".h5");
    state_save_config cfg;
    cfg.output_path = path;
    REQUIRE(state_logger::serialize_state(state_logger::extract_state(env, 7), cfg, h5).get().success);

    const auto infos = [](const capture& cap) {
        std::vector<std::string> out;
        for (const auto& g : cap.got)
            if (g.lvl == coast::diag::level::info && g.cat == "checkpoint") out.push_back(g.msg);
        return out;
    };
    {
        capture cap;
        REQUIRE(state_logger{}.load_state(env, path, h5).success);
        CHECK(infos(cap) == std::vector<std::string>{ k_no_record_notice });
    }
    {
        capture cap;
        const auto pr = state_logger::parse_state(path, h5);
        REQUIRE(pr.result.success);
        REQUIRE(state_logger::apply_state(env, pr.tree).success);
        CHECK(infos(cap) == std::vector<std::string>{ k_no_record_notice });
    }
    fs::remove(path);
}

TEST_CASE("the record round-trips through HDF5 with every value type")
{
    const fs::path path = fs::temp_directory_path() / "coast_config_record.h5";
    std::error_code ec;
    fs::remove(path, ec);
    {
        auto file = hdf5::file::create(path.string());
        REQUIRE(static_cast<bool>(file));
        hdf5_state_writer w(file);
        write_config_record(w, sample());
        REQUIRE(file.flush() >= 0);
    }
    auto file = hdf5::file::open_read(path.string());
    REQUIRE(static_cast<bool>(file));
    {
        hdf5_state_reader r(file, logging::k_format_version);
        const auto d = diff_config_record(r, sample());
        REQUIRE(d.has_value());
        CHECK(d->empty());
    }
    {
        hdf5_state_reader r(file, logging::k_format_version);
        auto cur = with(sample(), "inhibition.type", std::string("random_index"));
        cur      = with(cur, "neuron[n].random_seed", std::int64_t{ 4 });
        const auto d = diff_config_record(r, cur);
        REQUIRE(d.has_value());
        CHECK(d->size() == 2);
    }
    fs::remove(path, ec);
}

namespace {

// sample() plus three entries the current configuration does not record: a
// numeric field, a float (it must print as written) and a string attribute.
config_record with_record_only()
{
    auto rec = sample();
    rec.push_back({ "neuron[n]#5.branch_interval", std::uint64_t{ 7000 } });
    rec.push_back({ "dendrite[d]#1:0.mismatch_smoothing", 0.0001f });
    rec.push_back({ "inhibition.extra", std::string("x") });
    return rec;
}

// Fields by name, then attributes by name.
const std::vector<std::string> k_record_only_lines = {
    "dendrite[d]#1:0.mismatch_smoothing 1e-04 -> (not in the current configuration)",
    "neuron[n]#5.branch_interval 7000 -> (not in the current configuration)",
    "inhibition.extra x -> (not in the current configuration)",
};

} // namespace

TEST_CASE("an entry only the record has is reported with its recorded value")
{
    capture cap;
    const auto tree = saved(with_record_only());
    mem_state_reader r(tree, logging::k_format_version);
    const auto d = diff_config_record(r, sample());
    REQUIRE(d.has_value());
    CHECK(*d == k_record_only_lines);

    mem_state_reader r2(tree, logging::k_format_version);
    check_config_record(r2, sample(), "group 1");
    REQUIRE(cap.got.size() == 1);
    CHECK(cap.got[0].lvl == coast::diag::level::warn);
    CHECK(cap.got[0].msg.find("neuron[n]#5.branch_interval 7000 -> (not in the current configuration)")
          != std::string::npos);
}

TEST_CASE("an entry only the record has is reported through HDF5 too, attributes included")
{
    const fs::path path = fs::temp_directory_path() / "coast_config_record_only.h5";
    std::error_code ec;
    fs::remove(path, ec);
    {
        auto file = hdf5::file::create(path.string());
        REQUIRE(static_cast<bool>(file));
        hdf5_state_writer w(file);
        write_config_record(w, with_record_only());
        REQUIRE(file.flush() >= 0);
    }
    auto file = hdf5::file::open_read(path.string());
    REQUIRE(static_cast<bool>(file));
    hdf5_state_reader r(file, logging::k_format_version);
    const auto d = diff_config_record(r, sample());
    REQUIRE(d.has_value());
    CHECK(*d == k_record_only_lines);
    fs::remove(path, ec);
}

// read_scalars is what lets the diff see those entries: the current group's
// scalars and string attributes, as written; no arrays, nothing from subgroups.
TEST_CASE("read_scalars lists a group's scalars and attributes with their types, in both readers")
{
    auto fill = [](state_writer& w) {
        w.begin_group("g");
        w.write("d", 0.1);
        w.write("f", 0.25f);
        w.write("i", std::int64_t{ -3 });
        w.write("u", std::uint64_t{ 18446744073709551615ull });
        w.write("b", true);
        w.write("v", std::span<const float>(std::vector<float>{ 1.0f, 2.0f }));
        w.write_attr("s", "text");
        w.begin_group("sub");
        w.write("hidden", 1.0);
        w.end_group();
        w.end_group();
    };
    const std::vector<std::pair<std::string, scalar_value>> expected = {
        { "b", true }, { "d", 0.1 }, { "f", 0.25f }, { "i", std::int64_t{ -3 } },
        { "u", std::uint64_t{ 18446744073709551615ull } }, { "s", std::string("text") },
    };

    mem_state_writer mw;
    fill(mw);
    const auto tree = mw.take_root();
    mem_state_reader mr(tree, logging::k_format_version);
    REQUIRE(mr.enter_group("g"));
    CHECK(mr.read_scalars() == expected);

    const fs::path path = fs::temp_directory_path() / "coast_read_scalars.h5";
    std::error_code ec;
    fs::remove(path, ec);
    {
        auto file = hdf5::file::create(path.string());
        REQUIRE(static_cast<bool>(file));
        hdf5_state_writer w(file);
        fill(w);
        REQUIRE(file.flush() >= 0);
    }
    auto file = hdf5::file::open_read(path.string());
    REQUIRE(static_cast<bool>(file));
    hdf5_state_reader hr(file, logging::k_format_version);
    REQUIRE(hr.enter_group("g"));
    CHECK(hr.read_scalars() == expected);
    fs::remove(path, ec);
}

// ---------------------------------------------------------------------------
// Channel record
// ---------------------------------------------------------------------------

namespace {

const channel_layout k_layout = { { "c0", 20 }, { "c1", 28 } };

mem_group saved_channels(const channel_layout& layout)
{
    mem_state_writer w;
    write_channel_record(w, "proximal_channels", layout);
    return w.take_root();
}

// The message check_channel_record throws for `tree` against `current`; empty if
// it does not throw.
std::string channel_error(const mem_group& tree, const channel_layout& current)
{
    mem_state_reader r(tree, logging::k_format_version);
    try {
        check_channel_record(r, "proximal_channels", current, "group 1");
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

} // namespace

TEST_CASE("channel record: the same layout passes; another order, width or name refuses, naming both")
{
    const auto tree = saved_channels(k_layout);
    CHECK(channel_error(tree, k_layout).empty());

    const auto reordered = channel_error(tree, { { "c1", 28 }, { "c0", 20 } });
    CHECK(has(reordered, "group 1: checkpoint does not fit this group — proximal_channels "
                         "[c0 (20), c1 (28)] (checkpoint) vs [c1 (28), c0 (20)] (configured)"));
    CHECK(has(channel_error(tree, { { "c0", 28 }, { "c1", 20 } }), "vs [c0 (28), c1 (20)] (configured)"));
    CHECK(has(channel_error(tree, { { "c0", 20 }, { "cx", 28 } }), "vs [c0 (20), cx (28)] (configured)"));
    CHECK(has(channel_error(tree, { { "c0", 20 } }),               "vs [c0 (20)] (configured)"));

    // An empty layout (a bank with no channels) records count 0 and no widths.
    const auto none = saved_channels({});
    CHECK(channel_error(none, {}).empty());
    CHECK(has(channel_error(none, k_layout), "[] (checkpoint) vs [c0 (20), c1 (28)] (configured)"));
}

TEST_CASE("channel record: a missing or incomplete record is a corrupt checkpoint")
{
    CHECK(has(channel_error(mem_group{}, k_layout), "group 1: corrupt checkpoint — proximal_channels missing"));

    auto no_count = saved_channels(k_layout);
    no_count.groups.at("proximal_channels").fields.erase("count");
    CHECK(has(channel_error(no_count, k_layout), "corrupt checkpoint — proximal_channels has no count"));

    auto no_widths = saved_channels(k_layout);
    no_widths.groups.at("proximal_channels").fields.erase("widths");
    CHECK(has(channel_error(no_widths, k_layout), "proximal_channels records 2 channels but 0 widths"));

    auto no_name = saved_channels(k_layout);
    no_name.groups.at("proximal_channels").attrs.erase("1");
    CHECK(has(channel_error(no_name, k_layout), "proximal_channels has no name for channel 1"));

    mem_state_writer w;
    CHECK_THROWS_AS(write_channel_record(w, "proximal_channels", { { "", 4 } }), std::invalid_argument);
}

TEST_CASE("channel record: round-trips through HDF5")
{
    const fs::path path = fs::temp_directory_path() / "coast_channel_record.h5";
    std::error_code ec;
    fs::remove(path, ec);
    {
        auto file = hdf5::file::create(path.string());
        REQUIRE(static_cast<bool>(file));
        hdf5_state_writer w(file);
        write_channel_record(w, "proximal_channels", k_layout);
        write_channel_record(w, "apical_channels", {});
        REQUIRE(file.flush() >= 0);
    }
    auto file = hdf5::file::open_read(path.string());
    REQUIRE(static_cast<bool>(file));
    hdf5_state_reader r(file, logging::k_format_version);
    CHECK_NOTHROW(check_channel_record(r, "proximal_channels", k_layout, "group 1"));
    CHECK_NOTHROW(check_channel_record(r, "apical_channels", {}, "group 1"));
    CHECK_THROWS_WITH_AS(check_channel_record(r, "proximal_channels", { { "c1", 28 }, { "c0", 20 } }, "group 1"),
                         doctest::Contains("[c0 (20), c1 (28)] (checkpoint) vs [c1 (28), c0 (20)] (configured)"),
                         std::runtime_error);
    fs::remove(path, ec);
}

TEST_CASE("channel record: the groups of one load without a channel record are reported in one notice")
{
    const mem_group empty;   // a checkpoint written before channel records existed
    capture cap;
    mem_state_reader r(empty, logging::k_format_version);
    r.note_unrecorded_channels("group 1");
    r.note_unrecorded_channels("group 2");
    check_config_record(r, sample(), "group 1");
    CHECK(cap.got.empty());   // held until the load reports them
    report_unrecorded_configs(r);
    report_unrecorded_channels(r);
    // One notice of each kind, each naming every group.
    REQUIRE(cap.got.size() == 2);
    CHECK(cap.got[1].lvl == coast::diag::level::info);
    CHECK(has(cap.got[0].msg, "no configuration record (group 1)"));
    CHECK(has(cap.got[1].msg, "does not record the channel order (group 1, group 2)"));

    report_unrecorded_channels(r);   // already taken: nothing more to report
    CHECK(cap.got.size() == 2);
}
