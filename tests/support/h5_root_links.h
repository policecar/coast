// tests/support/h5_root_links.h
//
// Header-only: a test using this already links HDF5 through coast_lib.

#ifndef COAST_TEST_H5_ROOT_LINKS_H
#define COAST_TEST_H5_ROOT_LINKS_H

#include <string>
#include <vector>

#include <hdf5.h>

namespace coast::test {

// Names of every link under the file root; empty on any open/inspect failure.
inline std::vector<std::string> root_link_names(hid_t file) {
    std::vector<std::string> names;
    const hid_t root = H5Gopen2(file, "/", H5P_DEFAULT);
    if (root < 0) return names;
    H5G_info_t ginfo{};
    if (H5Gget_info(root, &ginfo) >= 0) {
        for (hsize_t i = 0; i < ginfo.nlinks; ++i) {
            char buf[512];
            const ssize_t len = H5Lget_name_by_idx(
                root, ".", H5_INDEX_NAME, H5_ITER_INC, i, buf, sizeof(buf), H5P_DEFAULT);
            if (len > 0) names.emplace_back(buf);
        }
    }
    H5Gclose(root);
    return names;
}

} // namespace coast::test

#endif // COAST_TEST_H5_ROOT_LINKS_H
