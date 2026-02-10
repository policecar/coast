//
// HDF5 utilities implementation
//

#include "logging/hdf5_utils.h"
#include <cstring>

namespace logging::hdf5 {

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

} // namespace logging::hdf5
