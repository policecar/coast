// tests/unit/io/test_dsprites_io.cpp
//
// dsprites_io sizes its image buffer from the dims of the file's `imgs`
// dataset. Dims are metadata, so a small file can declare far more images than
// it stores; the loader must refuse it before allocating (#159). Before the fix
// it also multiplied dims[0] by the image size unchecked and read the whole
// dataset (H5S_ALL) into the result, so a wrapping product overflowed the heap.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hdf5/executor.h"
#include "io/dsprites_cfg.h"
#include "io/dsprites_io.h"

#include <hdf5.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr hsize_t k_dim = 64;   // dsprites_io::image_dim (private)

// A file with an `imgs` dataset of `count` 64x64 uint8 images, written with
// `count` images' worth of ones when `write` is set and left unwritten otherwise.
struct imgs_file {
    hdf5::executor& h5;
    fs::path        path = fs::temp_directory_path() /
                           ("coast_dsprites_" + std::to_string(::getpid()) + ".h5");

    imgs_file(hdf5::executor& exec, hsize_t count, bool write) : h5(exec)
    {
        h5.submit_sync([&] {
            const hid_t   f       = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            const hsize_t dims[3] = {count, k_dim, k_dim};
            const hid_t   sp      = H5Screate_simple(3, dims, nullptr);
            const hid_t   ds      = H5Dcreate2(f, "imgs", H5T_NATIVE_UINT8, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            REQUIRE(ds >= 0);
            if (write) {
                const std::vector<std::uint8_t> ones(count * k_dim * k_dim, 1);
                REQUIRE(H5Dwrite(ds, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, ones.data()) >= 0);
            }
            H5Dclose(ds);
            H5Sclose(sp);
            H5Fclose(f);
        });
    }
    ~imgs_file() { fs::remove(path); }

    [[nodiscard]] dsprites::dsprites_param_t params() const
    {
        dsprites::dsprites_param_t p;
        p.hdf5_file = path.string();
        return p;
    }
};

} // namespace

TEST_CASE("a stored image set loads")
{
    hdf5::executor h5;
    imgs_file      file(h5, 3, /*write=*/true);
    CHECK_NOTHROW(dsprites::dsprites_io(file.params(), sim::output_id_t{1}, h5));
}

TEST_CASE("an image set the file declares but does not store is refused before allocating")
{
    // 2^24 images of 4 KiB: 64 GiB declared, none stored.
    hdf5::executor h5;
    imgs_file      file(h5, hsize_t{1} << 24, /*write=*/false);
    CHECK_THROWS_WITH_AS(dsprites::dsprites_io(file.params(), sim::output_id_t{1}, h5),
                         doctest::Contains("/imgs: declares 68719476736 bytes, the file stores at most 0"),
                         std::runtime_error);
}
