//
// Created for the COAST observe system
//

#ifndef COAST_OBSERVE_SITE_H
#define COAST_OBSERVE_SITE_H

#include "hdf5/handle.h"

#include <hdf5.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace observe {

/// Per-site state for one observed (group, id) pair on one entity element.
///
/// Under the post-process observation pass the value is handed straight to the
/// group writer (which copies it into an owned buffer), so the site holds no
/// scratch / fill-flag / group back-pointer — only the resolved identity and
/// the lazily-opened HDF5 dataset handles.
struct observe_site_t {
    std::string                       full_id;          // e.g. "neuron[3]/dendrite[7]/activity"
    hid_t                             h5_dtype;         // HDF5 native type id (e.g. H5T_NATIVE_FLOAT)
    std::vector<hsize_t>              shape;            // empty for scalar; [N] / [d0,d1,...] otherwise
    std::size_t                       value_bytes;      // sizeof(scalar) * product(shape)

    // Lazily-opened HDF5 dataset handles. Both touched only from the
    // executor's worker thread (inside lambdas submitted by the writer).
    // Lifetime: opened on first commit; closed explicitly by the writer
    // before the file handle, to honour HDF5's child-before-parent close
    // requirement. The destructor path is a safety net for orphan sites.
    hdf5::dataset_h                 h5_dataset;        // value dataset
    hdf5::dataset_h                 h5_step_dataset;   // sibling step-index dataset

    std::uint64_t                     rows_written{0};  // number of rows already in dataset
};

} // namespace observe

#endif // COAST_OBSERVE_SITE_H
