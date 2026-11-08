//
// HDF5 utilities implementation
//

#include "logging/hdf5_utils.h"

#include <array>
#include <cstring>

namespace hdf5 {

group group::create_or_open(hid_t parent, const char* name) {
    // Check if it already exists
    if (link_exists(parent, name)) {
        return group::open(parent, name);
    }

    // Create the path recursively if needed
    std::string path(name);
    std::size_t pos = 0;
    hid_t current = parent;
    std::vector<hid_t> opened_groups;

    while ((pos = path.find('/', pos)) != std::string::npos) {
        std::string segment = path.substr(0, pos);
        if (!segment.empty()) {
            if (!link_exists(current, segment.c_str())) {
                hid_t g = H5Gcreate2(current, segment.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (g < 0) {
                    // Cleanup
                    for (auto it = opened_groups.rbegin(); it != opened_groups.rend(); ++it) {
                        H5Gclose(*it);
                    }
                    return group{};
                }
                opened_groups.push_back(g);
                current = g;
            } else {
                hid_t g = H5Gopen2(current, segment.c_str(), H5P_DEFAULT);
                if (g < 0) {
                    for (auto it = opened_groups.rbegin(); it != opened_groups.rend(); ++it) {
                        H5Gclose(*it);
                    }
                    return group{};
                }
                opened_groups.push_back(g);
                current = g;
            }
        }
        ++pos;
    }

    // Create or open the final group
    group result;
    if (!link_exists(current, name)) {
        result = group::create(current, name);
    } else {
        result = group::open(current, name);
    }

    // Close intermediate groups
    for (auto it = opened_groups.rbegin(); it != opened_groups.rend(); ++it) {
        H5Gclose(*it);
    }

    return result;
}

herr_t append_rows(hid_t ds, hsize_t first_row, hsize_t n_rows,
                   std::span<const hsize_t> row_shape, hid_t mem_type, const void* data)
{
    // No heap per call: observe appends one row per site per step.
    std::array<hsize_t, H5S_MAX_RANK> dims{}, offset{}, count{};
    const std::size_t rank = 1 + row_shape.size();
    if (rank > H5S_MAX_RANK) return -1;
    dims[0]   = first_row + n_rows;
    offset[0] = first_row;
    count[0]  = n_rows;
    for (std::size_t i = 0; i < row_shape.size(); ++i) dims[1 + i] = count[1 + i] = row_shape[i];

    // The row shape must be the dataset's own before anything is touched:
    // H5Dset_extent takes the whole extent, so a wrong row shape would silently
    // reshape (and truncate) every row already written, not just this batch.
    {
        auto cur = dataspace::from_dataset(ds);
        if (!cur) return -1;
        std::array<hsize_t, H5S_MAX_RANK> cur_dims{};
        const int cur_rank = H5Sget_simple_extent_ndims(cur.id());
        if (cur_rank < 0 || static_cast<std::size_t>(cur_rank) != rank) return -1;
        if (H5Sget_simple_extent_dims(cur.id(), cur_dims.data(), nullptr) < 0) return -1;
        for (std::size_t i = 1; i < rank; ++i)
            if (cur_dims[i] != dims[i]) return -1;
    }

    if (H5Dset_extent(ds, dims.data()) < 0) return -1;
    auto file_space = dataspace::from_dataset(ds);
    if (!file_space ||
        file_space.select_hyperslab({ offset.data(), rank }, { count.data(), rank }) < 0)
        return -1;
    auto mem_space = dataspace::simple({ count.data(), rank });
    if (!mem_space) return -1;
    return H5Dwrite(ds, mem_type, mem_space.id(), file_space.id(), H5P_DEFAULT, data);
}

} // namespace hdf5
