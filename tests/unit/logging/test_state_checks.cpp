// tests/unit/logging/test_state_checks.cpp
//
// check_shape: a checkpoint's structural fields must equal the group's, or the
// load throws naming every field that differs or is absent.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "logging/hdf5_utils.h"
#include "logging/mem_state_io.h"
#include "logging/state_checks.h"
#include "logging/state_logger.h"   // hdf5_state_writer / hdf5_state_reader

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

using namespace logging;
namespace fs = std::filesystem;

namespace {

// neuron_cnt = 5 as uint64, max_seg_id = 7 as uint32 (ngm25 writes it so).
mem_group sample()
{
    mem_state_writer w;
    w.write("neuron_cnt", std::uint64_t{ 5 });
    w.write("max_seg_id", std::uint32_t{ 7 });
    return w.take_root();
}

// The message check_shape throws, or "" when it returns.
std::string refusal(state_reader& r, std::initializer_list<shape_field> fields)
{
    try {
        check_shape(r, "ngm3 neuron group 4", fields);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

} // namespace

TEST_CASE("matching fields pass, whatever integer type the checkpoint stored them as")
{
    const auto tree = sample();
    mem_state_reader r(tree, logging::k_format_version);
    CHECK(refusal(r, { { "neuron_cnt", 5 }, { "max_seg_id", 7 } }).empty());
}

TEST_CASE("a differing field throws, naming the group, the field and both values")
{
    const auto tree = sample();
    mem_state_reader r(tree, logging::k_format_version);
    const auto what = refusal(r, { { "neuron_cnt", 3 }, { "max_seg_id", 7 } });
    INFO("message: ", what);
    CHECK(what == "ngm3 neuron group 4: checkpoint does not fit this group — "
                  "neuron_cnt 5 (checkpoint) vs 3 (configured)");
}

TEST_CASE("an absent field throws as missing, not as a zero")
{
    const auto tree = sample();
    mem_state_reader r(tree, logging::k_format_version);
    const auto what = refusal(r, { { "neuron_cnt", 5 }, { "dend_count", 0 } });
    INFO("message: ", what);
    CHECK(what.ends_with("— dend_count missing"));
}

TEST_CASE("every differing field is listed, in the order given")
{
    const auto tree = sample();
    mem_state_reader r(tree, logging::k_format_version);
    const auto what = refusal(r, { { "neuron_cnt", 3 }, { "dend_count", 2 }, { "max_seg_id", 9 } });
    INFO("message: ", what);
    CHECK(what.ends_with("— neuron_cnt 5 (checkpoint) vs 3 (configured), dend_count missing, "
                         "max_seg_id 7 (checkpoint) vs 9 (configured)"));
}

TEST_CASE("an HDF5 checkpoint reads the same: a uint32 field compares as its value")
{
    const fs::path path = fs::temp_directory_path() / "coast_state_checks.h5";
    std::error_code ec;
    fs::remove(path, ec);
    {
        auto file = hdf5::file::create(path.string());
        REQUIRE(static_cast<bool>(file));
        hdf5_state_writer w(file);
        w.write("neuron_cnt", std::uint64_t{ 5 });
        w.write("max_seg_id", std::uint32_t{ 7 });
        REQUIRE(file.flush() >= 0);
    }
    {
        auto file = hdf5::file::open_read(path.string());
        REQUIRE(static_cast<bool>(file));
        hdf5_state_reader r(file, logging::k_format_version);
        CHECK(refusal(r, { { "neuron_cnt", 5 }, { "max_seg_id", 7 } }).empty());
        CHECK(refusal(r, { { "max_seg_id", 8 } }).ends_with("max_seg_id 7 (checkpoint) vs 8 (configured)"));
    }
    fs::remove(path, ec);
}
