// tests/unit/logging/test_mem_state_io.cpp
//
// Step-1 verification gate for the in-memory state tree + HDF5 adapters
// (design/quiesced-state-access-plan.md §"Step 1").
//
// Two obligations, both single-threaded (no sim thread, no executor — this test
// is the sole thread touching libhdf5, which satisfies the one-thread contract):
//
//   1. Round-trip across EVERY supported state_io dtype:
//        build a tree via mem_state_writer  ->  replay into hdf5_state_writer
//        ->  parse_hdf5_to_tree  ->  read via mem_state_reader  ->  assert equal,
//      type-by-type, including empty arrays, nested groups, 2D-f32, string attrs.
//
//   2. Structural equivalence with the inline path:
//        the same call sequence written (a) directly through hdf5_state_writer
//        and (b) through the tree then replay()ed into hdf5_state_writer must
//        produce the SAME on-disk structure. coast_compare operates on A-tier
//        JSON blobs, not raw HDF5, so the faithful Step-1 realization of
//        "coast_compare-clean" is structural/logical tree identity (parse both
//        files, assert the trees are equal) plus loadability — exactly the
//        contract the plan names (not byte-identity, which HDF5 never gives).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "logging/mem_state_io.h"
#include "logging/state_logger.h"   // hdf5_state_writer
#include "logging/hdf5_utils.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

using namespace logging;
namespace fs = std::filesystem;

namespace {

// A unique temp path per name; removed if it already exists.
fs::path temp_h5(const char* tag)
{
    fs::path p = fs::temp_directory_path() / (std::string("coast_mem_state_") + tag + ".h5");
    std::error_code ec;
    fs::remove(p, ec);
    return p;
}

// Write a fixed call sequence that exercises every dtype, shape, and nesting.
// Used both to build the in-memory tree and to drive the inline hdf5 writer.
void write_sample(state_writer& w)
{
    // Scalars at the root.
    w.write_attr("note", std::string_view("hello world"));
    w.write("s_double", 3.141592653589793);
    w.write("s_float", 2.71828f);
    w.write("s_i32", std::int32_t{-12345});
    w.write("s_i64", std::int64_t{-9000000000LL});
    w.write("s_u32", std::uint32_t{4000000000u});
    w.write("s_u64", std::uint64_t{18000000000000000000ULL});
    w.write("s_bool_t", true);
    w.write("s_bool_f", false);

    // Vectors in a nested group.
    w.begin_group("vectors");
    const std::vector<float>         vf  = {1.0f, -2.5f, 3.25f};
    const std::vector<double>        vd  = {1.5, 2.5, 3.5, 4.5};
    const std::vector<std::int32_t>  vi  = {-1, 0, 1, 2, 3};
    const std::vector<std::uint32_t> vu  = {0u, 100u, 4000000000u};
    const std::vector<std::uint64_t> vu64 = {0ull, 4000000000ull, 18000000000000000000ull};
    const std::vector<std::uint16_t> v16 = {0, 256, 65535};
    const std::vector<std::uint8_t>  v8  = {0, 127, 255};
    w.write("vf",  std::span<const float>(vf));
    w.write("vd",  std::span<const double>(vd));
    w.write("vi",  std::span<const std::int32_t>(vi));
    w.write("vu",  std::span<const std::uint32_t>(vu));
    w.write("vu64", std::span<const std::uint64_t>(vu64));
    w.write("v16", std::span<const std::uint16_t>(v16));
    w.write("v8",  std::span<const std::uint8_t>(v8));
    // Empty array: hdf5_state_writer skips it (no dataset). On read it must
    // come back as an empty vector, indistinguishable from "absent".
    w.write("vempty", std::span<const float>{});
    w.end_group();

    // 2D float in another nested group, plus a deeper nesting level.
    w.begin_group("matrices");
    const std::vector<float> m = {1, 2, 3, 4, 5, 6};   // 2x3
    w.write_2d("mat", std::span<const float>(m), 2, 3);
    w.begin_group("deeper");
    w.write("leaf", std::int32_t{42});
    w.write_attr("tag", std::string_view("nested-attr"));
    w.end_group();
    w.end_group();
}

} // namespace

TEST_CASE("mem_state tree round-trips every dtype through HDF5")
{
    // Build the tree.
    mem_state_writer mw;
    write_sample(mw);

    // Serialize: tree -> hdf5_state_writer -> file.
    const fs::path path = temp_h5("roundtrip");
    {
        auto file = hdf5::file::create(path.string());
        REQUIRE(static_cast<bool>(file));
        hdf5_state_writer hw(file);
        replay(mw.root(), hw);
        REQUIRE(file.flush() >= 0);
    }

    // Parse the file back to a tree.
    mem_group parsed;
    {
        auto file = hdf5::file::open_read(path.string());
        REQUIRE(static_cast<bool>(file));
        parsed = parse_hdf5_to_tree(file);
    }

    // Read back via mem_state_reader and assert equality, type-by-type.
    mem_state_reader r(parsed, logging::k_format_version);

    CHECK(r.read_attr("note") == "hello world");
    CHECK(r.read_double("s_double", 0.0) == doctest::Approx(3.141592653589793));
    CHECK(r.read_float("s_float", 0.0f) == doctest::Approx(2.71828f));
    CHECK(r.read_int32("s_i32", 0) == -12345);
    CHECK(r.read_int64("s_i64", 0) == -9000000000LL);
    CHECK(r.read_uint32("s_u32", 0) == 4000000000u);
    CHECK(r.read_uint64("s_u64", 0) == 18000000000000000000ULL);
    CHECK(r.read_bool("s_bool_t", false) == true);
    CHECK(r.read_bool("s_bool_f", true) == false);

    REQUIRE(r.enter_group("vectors"));
    CHECK(r.read_float_vec("vf")    == std::vector<float>{1.0f, -2.5f, 3.25f});
    CHECK(r.read_double_vec("vd")   == std::vector<double>{1.5, 2.5, 3.5, 4.5});
    CHECK(r.read_int32_vec("vi")    == std::vector<std::int32_t>{-1, 0, 1, 2, 3});
    CHECK(r.read_uint32_vec("vu")   == std::vector<std::uint32_t>{0u, 100u, 4000000000u});
    CHECK(r.read_uint64_vec("vu64") == std::vector<std::uint64_t>{0ull, 4000000000ull, 18000000000000000000ull});
    CHECK(r.read_uint16_vec("v16")  == std::vector<std::uint16_t>{0, 256, 65535});
    CHECK(r.read_uint8_vec("v8")    == std::vector<std::uint8_t>{0, 127, 255});
    CHECK(r.read_float_vec("vempty").empty());   // empty array round-trips as empty
    CHECK_FALSE(r.has_field("vempty"));          // ... and was never materialized
    r.leave_group();

    REQUIRE(r.enter_group("matrices"));
    state_reader::array_2d_info info{};
    const auto mat = r.read_float_2d("mat", info);
    CHECK(info.rows == 2);
    CHECK(info.cols == 3);
    CHECK(mat == std::vector<float>{1, 2, 3, 4, 5, 6});
    REQUIRE(r.enter_group("deeper"));
    CHECK(r.read_int32("leaf", 0) == 42);
    CHECK(r.read_attr("tag") == "nested-attr");
    r.leave_group();
    r.leave_group();

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("tree-driven serialize is structurally identical to the inline path")
{
    // (a) Inline: same call sequence written directly through hdf5_state_writer.
    const fs::path path_inline = temp_h5("equiv_inline");
    {
        auto file = hdf5::file::create(path_inline.string());
        REQUIRE(static_cast<bool>(file));
        hdf5_state_writer hw(file);
        write_sample(hw);
        REQUIRE(file.flush() >= 0);
    }

    // (b) Via tree: build the tree from the same sequence, then replay it.
    const fs::path path_tree = temp_h5("equiv_tree");
    {
        mem_state_writer mw;
        write_sample(mw);
        auto file = hdf5::file::create(path_tree.string());
        REQUIRE(static_cast<bool>(file));
        hdf5_state_writer hw(file);
        replay(mw.root(), hw);
        REQUIRE(file.flush() >= 0);
    }

    // Parse both files and assert the trees are structurally identical
    // (name-keyed, type-sensitive equality over the whole hierarchy). This is
    // the structural/logical "coast_compare-clean" contract for Step 1.
    mem_group a;
    mem_group b;
    {
        auto fa = hdf5::file::open_read(path_inline.string());
        auto fb = hdf5::file::open_read(path_tree.string());
        REQUIRE(static_cast<bool>(fa));
        REQUIRE(static_cast<bool>(fb));
        a = parse_hdf5_to_tree(fa);
        b = parse_hdf5_to_tree(fb);
    }

    CHECK(a == b);

    std::error_code ec;
    fs::remove(path_inline, ec);
    fs::remove(path_tree, ec);
}
