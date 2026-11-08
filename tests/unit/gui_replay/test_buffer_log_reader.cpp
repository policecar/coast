// tests/unit/gui_replay/test_buffer_log_reader.cpp
//
// The replay reader refuses a --log-buffers file whose datasets or attributes
// have a shape buffer_logger never writes (#150), checked once at open(). The
// optional vis_hints only steer the display: another shape is skipped. Before the fix it sized its
// buffers from dims[0] (or assumed a width) and read the whole dataset into
// them: an [n, 3] connectivity, a rank-3 vis_hints or a rank-2 buffer_ids
// overflowed the heap, and a rank-3 data dataset made a frame read take one
// offset/count entry past its two-entry stack arrays.
//
// Each case writes a minimal valid buffer log by hand (the layout buffer_logger
// writes), optionally with one object replaced, and opens it.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "gui_replay/buffer_log_reader.h"
#include "hdf5/executor.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using coast::replay::buffer_log_reader;

namespace {

hid_t simple_space(std::initializer_list<hsize_t> dims)
{
    const std::vector<hsize_t> d(dims);
    return H5Screate_simple(static_cast<int>(d.size()), d.data(), nullptr);
}

// Creates dataset `name` under `loc` with `dims`, filled from `data`.
void put(hid_t loc, const char* name, hid_t type, std::initializer_list<hsize_t> dims,
         const void* data)
{
    const hid_t sp = simple_space(dims);
    const hid_t ds = H5Dcreate2(loc, name, type, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(ds >= 0);
    REQUIRE(H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0);
    H5Dclose(ds);
    H5Sclose(sp);
}

// Which object of the valid layout to replace with a malformed one.
enum class bad { none, connectivity_3_cols, vis_hints_rank_3, vis_hints_4_cols, vis_hints_strings,
                 buffer_ids_rank_2, data_rank_3, data_5_cols, data_unstored,
                 start_step_2_elements, name_fills_slot };

// One buffer (id 1, size 4) fed by buffer 2, two frames.
struct log_file {
    hdf5::executor h5;
    fs::path       path = fs::temp_directory_path() /
                          ("coast_replay_guards_" + std::to_string(::getpid()) + ".h5");

    explicit log_file(bad b)
    {
        h5.submit_sync([&] {
            const hid_t f = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            REQUIRE(f >= 0);
            const hid_t md   = H5Gcreate2(f, "metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            const hid_t topo = H5Gcreate2(md, "topology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            if (b == bad::start_step_2_elements) {
                const hid_t sp = simple_space({2});
                const hid_t a  = H5Acreate2(md, "start_step", H5T_NATIVE_UINT64, sp, H5P_DEFAULT, H5P_DEFAULT);
                const std::uint64_t v[2] = {1, 2};
                REQUIRE(H5Awrite(a, H5T_NATIVE_UINT64, v) >= 0);
                H5Aclose(a);
                H5Sclose(sp);
            } else {
                REQUIRE(hdf5::write_attr(md, "start_step", std::uint64_t{0}) >= 0);
            }

            const std::uint64_t ids[2] = {1, 1};
            if (b == bad::buffer_ids_rank_2) put(topo, "buffer_ids", H5T_NATIVE_UINT64, {1, 2}, ids);
            else                             put(topo, "buffer_ids", H5T_NATIVE_UINT64, {1}, ids);
            // data_unstored: frames of 2^34 floats that the file never stores.
            constexpr std::uint64_t huge     = std::uint64_t{1} << 34;
            const std::uint64_t     sizes[1] = {b == bad::data_unstored ? huge : 4};
            put(topo, "buffer_sizes", H5T_NATIVE_UINT64, {1}, sizes);

            // Names as buffer_logger writes them: fixed-length, NULLTERM. With
            // name_fills_slot the name fills all 4 bytes with no NUL, which the
            // old reader's std::string(char*) read past.
            const hid_t st = H5Tcopy(H5T_C_S1);
            H5Tset_size(st, 4);
            H5Tset_strpad(st, H5T_STR_NULLTERM);
            put(topo, "buffer_names", st, {1}, b == bad::name_fills_slot ? "abcd" : "abc\0");
            H5Tclose(st);

            const std::uint64_t edges[3] = {2, 1, 0};
            if (b == bad::connectivity_3_cols) put(topo, "connectivity", H5T_NATIVE_UINT64, {1, 3}, edges);
            else                               put(topo, "connectivity", H5T_NATIVE_UINT64, {1, 2}, edges);

            // {1, 4, 3}, unlike the computed default for size 4 (2 x 2, 1 channel).
            const std::int32_t hints[6] = {1, 4, 3, 0, 0, 0};
            if (b == bad::vis_hints_rank_3)      put(topo, "vis_hints", H5T_NATIVE_INT32, {1, 3, 2}, hints);
            else if (b == bad::vis_hints_4_cols) put(topo, "vis_hints", H5T_NATIVE_INT32, {1, 4}, hints);
            else if (b == bad::vis_hints_strings) {
                const hid_t hs = H5Tcopy(H5T_C_S1);
                H5Tset_size(hs, 2);
                put(topo, "vis_hints", hs, {1, 3}, "a\0b\0c\0");
                H5Tclose(hs);
            }
            else                                 put(topo, "vis_hints", H5T_NATIVE_INT32, {1, 3}, hints);

            const hid_t steps = H5Gcreate2(f, "steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            const std::uint64_t step_numbers[2] = {10, 11};
            const double        timestamps[2]   = {0.5, 1.5};
            put(steps, "step_numbers", H5T_NATIVE_UINT64, {2}, step_numbers);
            put(steps, "timestamps", H5T_NATIVE_DOUBLE, {2}, timestamps);
            const hid_t bufs = H5Gcreate2(steps, "buffers", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            const hid_t b1   = H5Gcreate2(bufs, "1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            const float data[16] = {1, 2, 3, 4, 5, 6, 7, 8};
            if (b == bad::data_rank_3)      put(b1, "data", H5T_NATIVE_FLOAT, {2, 4, 2}, data);
            else if (b == bad::data_5_cols) put(b1, "data", H5T_NATIVE_FLOAT, {2, 5}, data);
            else if (b == bad::data_unstored) {
                const hid_t sp = simple_space({2, huge});
                const hid_t ds = H5Dcreate2(b1, "data", H5T_NATIVE_FLOAT, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                REQUIRE(ds >= 0);
                H5Dclose(ds);
                H5Sclose(sp);
            }
            else                            put(b1, "data", H5T_NATIVE_FLOAT, {2, 4}, data);

            for (const hid_t g : {b1, bufs, steps, topo, md}) H5Gclose(g);
            H5Fclose(f);
        });
    }
    ~log_file() { fs::remove(path); }
};

} // namespace

TEST_CASE("a well-formed buffer log opens and reads")
{
    log_file          lf(bad::none);
    buffer_log_reader r(lf.h5);
    REQUIRE(r.open(lf.path));
    CHECK(r.frame_count() == 2);
    REQUIRE(r.buffer_count() == 1);
    const auto* info = r.get_buffer(1);
    REQUIRE(info != nullptr);
    CHECK(info->name == "abc");
    CHECK(info->size == 4);
    CHECK(info->input_ids == std::vector<std::size_t>{2});
    CHECK(info->vis_hint.width == 1);
    CHECK(info->vis_hint.height == 4);
    CHECK(info->vis_hint.num_channels == 3);
    CHECK(r.get_step(1) == 11);
    CHECK(r.get_timestamp(1) == 1.5);

    std::vector<float> row;
    REQUIRE(r.read_frame_data(1, 1, row));
    CHECK(row == std::vector<float>{5, 6, 7, 8});
    REQUIRE(r.read_frame_range(1, 0, 2, row));
    CHECK(row == std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
}

TEST_CASE("a malformed topology, metadata or data object refuses the file")
{
    // data is checked at open, so a frame read never meets a bad shape (and
    // playback does not log one error per buffer per frame).
    // data_unstored: a frame read sized its buffer from buffer_sizes, 64 GiB
    // here, for a dataset the file does not store (#159).
    for (const bad b : {bad::connectivity_3_cols, bad::buffer_ids_rank_2,
                        bad::start_step_2_elements, bad::data_rank_3, bad::data_5_cols,
                        bad::data_unstored}) {
        CAPTURE(static_cast<int>(b));
        log_file          lf(b);
        buffer_log_reader r(lf.h5);
        CHECK_FALSE(r.open(lf.path));
        CHECK_FALSE(r.is_open());
    }
}

TEST_CASE("vis_hints of another shape or a non-numeric type are skipped for the computed defaults")
{
    for (const bad b : {bad::vis_hints_rank_3, bad::vis_hints_4_cols, bad::vis_hints_strings}) {
        CAPTURE(static_cast<int>(b));
        log_file          lf(b);
        buffer_log_reader r(lf.h5);
        REQUIRE(r.open(lf.path));
        const auto* info = r.get_buffer(1);
        REQUIRE(info != nullptr);
        CHECK(info->vis_hint.width == 2);
        CHECK(info->vis_hint.height == 2);
        CHECK(info->vis_hint.num_channels == 1);
        std::vector<float> row;
        CHECK(r.read_frame_data(1, 1, row));
    }
}

TEST_CASE("a name that fills its fixed-length slot is bounded by the slot")
{
    log_file          lf(bad::name_fills_slot);
    buffer_log_reader r(lf.h5);
    REQUIRE(r.open(lf.path));
    REQUIRE(r.get_buffer(1) != nullptr);
    CHECK(r.get_buffer(1)->name == "abcd");
}
