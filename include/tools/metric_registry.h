//
// metric_registry.h — enumerable metadata over the metric.h functions (D9b).
//
// The registry gives tools (h5-browser pairwise view; Metric Explorer in its
// planned refactor) a single table to build metric dropdowns and parameter
// widgets from, without hard-coding metric_kind knowledge per panel.
// Computation stays in metric.h (compute_metric); this header only describes.
//
// Header-only like its metric.h neighbor; lives in tools/ so it stays free of
// both GUI and h5browse dependencies.
//

#ifndef COAST_METRIC_REGISTRY_H
#define COAST_METRIC_REGISTRY_H

#include "tools/metric.h"

#include <optional>
#include <span>

namespace coast {

struct metric_def {
    const char* name;  // UI display name
    metric_kind kind;

    // Data-independent histogram domain, when the metric has one
    // (cosine: [-1, 1]). nullopt = unbounded above, adaptive binning.
    std::optional<value_range> fixed_domain;

    // Scalar parameter descriptor (Minkowski p); has_param == false means
    // metric_params is ignored.
    bool        has_param     = false;
    const char* param_name    = nullptr;
    float       param_default = 0.0f;
    float       param_min     = 0.0f; // valid range for UI clamping
    float       param_max     = 0.0f;
};

inline std::span<const metric_def> metric_registry()
{
    static constexpr metric_def defs[] = {
        {
            .name          = "Cosine similarity",
            .kind          = metric_kind::cosine_similarity,
            .fixed_domain  = value_range{-1.0f, 1.0f},
            .has_param     = false,
            .param_name    = nullptr,
            .param_default = 0.0f,
            .param_min     = 0.0f,
            .param_max     = 0.0f,
        },
        {
            .name          = "Minkowski distance",
            .kind          = metric_kind::minkowski_distance,
            .fixed_domain  = std::nullopt,
            .has_param     = true,
            .param_name    = "p",
            .param_default = 2.0f,
            .param_min     = 0.1f,
            .param_max     = 16.0f,
        },
    };
    return defs;
}

} // namespace coast

#endif // COAST_METRIC_REGISTRY_H
