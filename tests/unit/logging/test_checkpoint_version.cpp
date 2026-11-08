// tests/unit/logging/test_checkpoint_version.cpp
//
// The checkpoint format version (logging::k_format_version, recorded as
// metadata/schema_version). A checkpoint this build writes records it; one
// written before the version was an integer ("1.0") is version 1 and loads
// without a warning; one whose version is newer, missing or malformed is refused
// by every load path before any entity reads it, as is one without a metadata
// group. Checks that need a newer format gate on state_reader::format_version();
// the first such gate comes with the first bump, together with its own test.
// Every reader is built with the version, so no load path can leave it unset.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"
#include "logging/mem_state_io.h"
#include "logging/state_io.h"
#include "logging/state_logger.h"
#include "sim_core/sim_env.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unistd.h>

namespace fs = std::filesystem;
using logging::mem_group;
using logging::state_logger;

// A reader takes the checkpoint's version at construction, with no default and no
// setter: a load path that forgot it would read an old checkpoint as the newest
// format, and the checks gated on newer versions would refuse it.
static_assert(!std::is_constructible_v<logging::mem_state_reader, const mem_group&>);
static_assert(!std::is_constructible_v<logging::hdf5_state_reader, hdf5::file&>);

namespace {

// An io_entity with state that counts its load_state calls.
struct counting_entity : sim::io_entity {
    sim::output_id_t oid;
    int&             loads;

    counting_entity(sim::output_id_t id, int& n) : oid(id), loads(n) {}

    sim::step_result process() override { return {}; }
    void save_state(logging::state_writer& w) const override { w.write("x", std::uint32_t{1}); }
    void load_state(logging::state_reader&) override { ++loads; }

    [[nodiscard]] sim::output_id_t get_outp_id() const override { return oid; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 0; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

struct fixture {
    int            loads = 0;
    sim::env       env;
    hdf5::executor h5;
    fs::path       path = fs::temp_directory_path() /
                          ("coast_ckpt_version_" + std::to_string(::getpid()) + ".h5");

    fixture() { env.emplace_back<counting_entity>(sim::output_id_t{4}, loads); }
    ~fixture() { fs::remove(path); }

    // A checkpoint of the env, its recorded version replaced by `version`, or
    // removed when `version` is nullopt.
    mem_group tree(std::optional<std::string_view> version) const
    {
        auto t = state_logger::extract_state(const_cast<sim::env&>(env), 7);
        auto& attrs = t.groups.at("metadata").attrs;
        if (version) attrs.insert_or_assign("schema_version", std::string(*version));
        else         attrs.erase("schema_version");
        return t;
    }

    void write(mem_group t)
    {
        logging::state_save_config cfg;
        cfg.output_path = path;
        REQUIRE(state_logger::serialize_state(std::move(t), cfg, h5).get().success);
    }
};

} // namespace

TEST_CASE("a checkpoint records this build's format version and loads without a warning")
{
    fixture f;
    const auto t = state_logger::extract_state(f.env, 7);
    CHECK(t.groups.at("metadata").attrs.at("schema_version") == std::to_string(logging::k_format_version));

    f.write(t);
    const auto res = state_logger{}.load_state(f.env, f.path, f.h5);
    CHECK(res.success);
    CHECK(res.warnings.empty());
    CHECK(f.loads == 1);
}

TEST_CASE("a checkpoint written as \"1.0\" is version 1 and loads without a warning")
{
    fixture f;
    f.write(f.tree("1.0"));

    const auto pr = state_logger::parse_state(f.path, f.h5);
    REQUIRE(pr.result.success);
    CHECK(pr.result.warnings.empty());
    CHECK(state_logger::apply_state(f.env, pr.tree).success);
    CHECK(f.loads == 1);
}

TEST_CASE("a newer, missing or malformed format version is refused by every load path, before any entity reads")
{
    const std::string newer = std::to_string(logging::k_format_version + 1);
    for (const std::optional<std::string_view> v :
         { std::optional<std::string_view>{ newer }, std::optional<std::string_view>{},
           std::optional<std::string_view>{ "abc" }, std::optional<std::string_view>{ "0" },
           std::optional<std::string_view>{ "2.5" } }) {
        CAPTURE(v.value_or("(missing)"));
        fixture f;
        f.write(f.tree(v));

        const auto startup = state_logger{}.load_state(f.env, f.path, f.h5);
        CHECK_FALSE(startup.success);
        CHECK(startup.error_message.find("format version") != std::string::npos);

        const auto pr = state_logger::parse_state(f.path, f.h5);
        CHECK_FALSE(pr.result.success);

        // apply_state checks on its own, for a tree that skipped parse_state.
        CHECK_FALSE(state_logger::apply_state(f.env, f.tree(v)).success);

        CHECK(f.loads == 0);
    }
}

TEST_CASE("a newer format version names both versions")
{
    fixture f;
    f.write(f.tree(std::to_string(logging::k_format_version + 1)));
    const auto pr = state_logger::parse_state(f.path, f.h5);
    CHECK(pr.result.error_message.find(std::format("format version {}, newer than this build reads ({})",
                                                   logging::k_format_version + 1, logging::k_format_version))
          != std::string::npos);
}

TEST_CASE("a checkpoint without a metadata group is refused by every load path, before any entity reads")
{
    fixture f;
    // serialize_state always writes a metadata group, so remove it from the file.
    f.write(state_logger::extract_state(f.env, 7));
    {
        auto file = hdf5::file::open_readwrite(f.path.string());
        REQUIRE(file);
        REQUIRE(H5Ldelete(file.id(), "metadata", H5P_DEFAULT) >= 0);
    }
    auto t = state_logger::extract_state(f.env, 7);
    REQUIRE(t.groups.erase("metadata") == 1);

    const auto startup = state_logger{}.load_state(f.env, f.path, f.h5);
    CHECK_FALSE(startup.success);
    CHECK(startup.error_message == "Missing metadata group in state file");

    const auto pr = state_logger::parse_state(f.path, f.h5);
    CHECK_FALSE(pr.result.success);
    CHECK(pr.result.error_message == "Missing metadata group in state file");

    // apply_state checks on its own, for a tree that skipped parse_state.
    const auto applied = state_logger::apply_state(f.env, t);
    CHECK_FALSE(applied.success);
    CHECK(applied.error_message == "Missing metadata group in state file");

    CHECK(f.loads == 0);
}
