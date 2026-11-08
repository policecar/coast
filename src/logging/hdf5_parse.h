//
// HDF5 -> mem_group parsing, internal to src/logging: shared by
// parse_hdf5_to_tree (whole files) and hdf5_state_reader::read_scalars (one
// group). Calls libhdf5 directly — invoke only on the hdf5::executor worker.
//

#ifndef COAST_LOGGING_HDF5_PARSE_H
#define COAST_LOGGING_HDF5_PARSE_H

#include <string>
#include <utility>
#include <vector>

#include <hdf5.h>

#include "logging/mem_state_io.h"

namespace logging::detail {

// Reads the attributes and datasets of `gid` into `out`, element type and shape
// preserved; with `recurse`, its subgroups too, else they are skipped. With
// `out` null, only checks them. Throws on anything a state writer never
// produces (see mem_state_io.cpp).
void parse_group(hid_t gid, mem_group* out, bool recurse);

// The scalar fields of `g` (fields first, then its string attributes), by name.
std::vector<std::pair<std::string, scalar_value>> scalars_of(const mem_group& g);

} // namespace logging::detail

#endif // COAST_LOGGING_HDF5_PARSE_H
