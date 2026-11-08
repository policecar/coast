#ifndef COAST_DATASET_ACCESS_H
#define COAST_DATASET_ACCESS_H

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "gui_vis/core/color_types.h"

namespace sim { class env; class sim_builder; }

namespace coast {

/**
 * A fully-resolved view onto one loaded dataset entity: its geometry plus a
 * type-erased per-sample accessor. This is the single place dataset-specific
 * knowledge (the io_entity_type switch, the color_mnist special case) is turned
 * into a uniform handle; both the dataset_browser and the Metric Explorer consume
 * these instead of repeating the switch.
 *
 * The get_sample closure captures the concrete io_entity pointer, so a handle is
 * valid only for the lifetime of the sim::env it was enumerated from (i.e. the
 * sim_instance or dataset_loader that owns that env).
 */
struct dataset_handle {
    std::string name;
    std::size_t entity_id = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::size_t sample_count = 0;
    std::function<std::span<const float>(std::size_t)> get_sample;

    /// Feature dimension = w * h * c. Two datasets are metric-comparable iff equal.
    [[nodiscard]] std::size_t feature_dim() const {
        return static_cast<std::size_t>(width) * height * channels;
    }
    [[nodiscard]] bool valid() const { return static_cast<bool>(get_sample); }
};

/**
 * Enumerate every loaded dataset input entity as a resolved handle.
 * This is the one true dataset->accessor switch (relocated from dataset_browser).
 */
std::vector<dataset_handle> enumerate_datasets(sim::env& env, const sim::sim_builder& builder);

/**
 * Resample one normalized sample (row-major floats) into an RGBA
 * thumb_size x thumb_size image via nearest-neighbour. channels==1 -> grayscale,
 * channels==3 -> RGB; pixels past the end of `sample` are filled magenta so a
 * malformed sample is visibly wrong rather than silently black. `out` is resized.
 */
void convert_sample_to_rgba(std::span<const float> sample,
                            int w, int h, int channels, int thumb_size,
                            std::vector<color_t>& out);

} // namespace coast

#endif // COAST_DATASET_ACCESS_H
