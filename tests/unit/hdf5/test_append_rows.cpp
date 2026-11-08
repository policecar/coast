// tests/unit/hdf5/test_append_rows.cpp
//
// hdf5::append_rows (logging/hdf5_utils.h) — the one append path of the buffer
// logger and the observe writer: grow an extensible dataset along dimension 0
// and write rows at a given index, reporting failure instead of writing past.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hdf5/executor.h"
#include "hdf5/native_type.h"
#include "logging/hdf5_utils.h"

#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct scratch {
    fs::path path = fs::temp_directory_path() / "coast_test_append_rows.h5";
    ~scratch() { std::error_code ec; fs::remove(path, ec); }
};

// An empty dataset [0, row_shape...], extensible along dimension 0.
hdf5::dataset make_extensible(hid_t loc, const char* name, std::vector<hsize_t> row_shape)
{
    std::vector<hsize_t> dims{ 0 }, chunk{ 4 };
    dims.insert(dims.end(), row_shape.begin(), row_shape.end());
    chunk.insert(chunk.end(), row_shape.begin(), row_shape.end());
    auto space = hdf5::dataspace::extensible(dims);
    auto dcpl  = hdf5::plist::dataset_create();
    dcpl.set_chunk(chunk);
    return hdf5::dataset::create(loc, name, hdf5::native_type<float>(), space.id(), dcpl.id());
}

std::vector<float> read_all(hdf5::dataset& ds, std::size_t n)
{
    std::vector<float> v(n);
    REQUIRE(ds.read(hdf5::native_type<float>(), H5S_ALL, H5S_ALL, v.data()) >= 0);
    return v;
}

hsize_t rows_of(hdf5::dataset& ds)
{
    auto space = hdf5::dataspace::from_dataset(ds.id());
    hsize_t dims[H5S_MAX_RANK]{};
    H5Sget_simple_extent_dims(space.id(), dims, nullptr);
    return dims[0];
}

} // namespace

TEST_CASE("append_rows: 2-D rows land at their index, batch after batch") {
    scratch s;
    hdf5::executor exec;
    exec.submit_sync([&] {
        auto f  = hdf5::file::create(s.path.string());
        auto ds = make_extensible(f.id(), "d", { 3 });
        const hsize_t row_shape[1] = { 3 };
        const std::vector<float> a{ 1, 2, 3, 4, 5, 6 };      // rows 0, 1
        const std::vector<float> b{ 7, 8, 9 };               // row 2
        CHECK(hdf5::append_rows(ds.id(), 0, 2, row_shape, hdf5::native_type<float>(), a.data()) >= 0);
        CHECK(hdf5::append_rows(ds.id(), 2, 1, row_shape, hdf5::native_type<float>(), b.data()) >= 0);
        CHECK(rows_of(ds) == 3);
        CHECK(read_all(ds, 9) == std::vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });

        // A retry at the same index overwrites it (what a caller that did not
        // count a failed row relies on), it does not append.
        const std::vector<float> c{ 0, 0, 1 };
        CHECK(hdf5::append_rows(ds.id(), 2, 1, row_shape, hdf5::native_type<float>(), c.data()) >= 0);
        CHECK(rows_of(ds) == 3);
        CHECK(read_all(ds, 9) == std::vector<float>{ 1, 2, 3, 4, 5, 6, 0, 0, 1 });
    });
}

TEST_CASE("append_rows: a 1-D dataset takes an empty row shape") {
    scratch s;
    hdf5::executor exec;
    exec.submit_sync([&] {
        auto f  = hdf5::file::create(s.path.string());
        auto ds = make_extensible(f.id(), "d", {});
        const float v[2] = { 0.5f, 1.5f };
        CHECK(hdf5::append_rows(ds.id(), 0, 2, {}, hdf5::native_type<float>(), v) >= 0);
        CHECK(read_all(ds, 2) == std::vector<float>{ 0.5f, 1.5f });
    });
}

TEST_CASE("append_rows: a row shape that does not match the dataset is refused") {
    scratch s;
    hdf5::executor exec;
    exec.submit_sync([&] {
        auto f  = hdf5::file::create(s.path.string());
        auto ds = make_extensible(f.id(), "d", { 3 });
        // The refusals below would also "pass" on an invalid id.
        CHECK((f && ds));
        // Two rows already written: a refusal must leave them exactly as they are.
        const hsize_t row_shape[1] = { 3 };
        const std::vector<float> a{ 1, 2, 3, 4, 5, 6 };
        REQUIRE(hdf5::append_rows(ds.id(), 0, 2, row_shape, hdf5::native_type<float>(), a.data()) >= 0);
        const float v[4] = {};
        // rank 3 against a rank-2 dataset
        const hsize_t wrong_rank[2] = { 2, 2 };
        H5E_BEGIN_TRY {
            CHECK(hdf5::append_rows(ds.id(), 2, 1, wrong_rank, hdf5::native_type<float>(), v) < 0);
        } H5E_END_TRY;
        CHECK(rows_of(ds) == 2);
        // right rank, wrong width: H5Dset_extent would accept {3, 2} and reshape
        // every row already written — refused before the extent is touched
        const hsize_t wrong_width[1] = { 2 };
        H5E_BEGIN_TRY {
            CHECK(hdf5::append_rows(ds.id(), 2, 1, wrong_width, hdf5::native_type<float>(), v) < 0);
        } H5E_END_TRY;
        CHECK(rows_of(ds) == 2);
        CHECK(hdf5::dataspace::from_dataset(ds.id()).dims() == std::vector<hsize_t>{ 2, 3 });
        CHECK(read_all(ds, 6) == a);
        // a fixed-size (non-extensible) dataset cannot grow
        const hsize_t dims[1] = { 2 };
        auto fixed_space = hdf5::dataspace::simple(dims);
        auto fixed = hdf5::dataset::create(f.id(), "fixed", hdf5::native_type<float>(), fixed_space.id());
        CHECK(static_cast<bool>(fixed));
        H5E_BEGIN_TRY {
            CHECK(hdf5::append_rows(fixed.id(), 2, 1, {}, hdf5::native_type<float>(), v) < 0);
        } H5E_END_TRY;
    });
}
