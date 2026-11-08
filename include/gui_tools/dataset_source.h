#ifndef COAST_DATASET_SOURCE_H
#define COAST_DATASET_SOURCE_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "gui_core/dataset_access.h"        // dataset_handle
#include "tools/ensemble_code.h"   // ensemble_codes (value member of synthetic_source)

namespace hdf5 { class executor; }

namespace coast {

class dataset_loader;

/**
 * A "source" of datasets for the Metric Explorer. Imported configs and generated
 * synthetic code-sets are interchangeable behind this interface: each yields a list
 * of dataset_handles and a display label (the explorer derives each source's memory
 * footprint from its handles). The explorer assigns each source a stable `id` so an
 * axis can reference a dataset by (source_id, local_idx) and survive other sources
 * being added/removed.
 *
 * Non-copyable: handles enumerated from a source capture pointers into storage the
 * source owns, so the source must outlive every handle derived from it (the
 * explorer holds sources by unique_ptr, giving them a stable address/lifetime).
 */
class dataset_source {
public:
    dataset_source() = default;
    virtual ~dataset_source() = default;
    dataset_source(const dataset_source&) = delete;
    dataset_source& operator=(const dataset_source&) = delete;

    /// Row label in the sources panel; also the basis for dataset names.
    [[nodiscard]] virtual std::string label() const = 0;
    /// Enumerate this source's datasets as ready-to-use, already-named handles.
    [[nodiscard]] virtual std::vector<dataset_handle> datasets() const = 0;

    std::size_t id = 0;   // stable, assigned by metric_explorer (0 = unassigned)
};

/**
 * A source backed by a config file loaded as CPU dataset entities (wraps
 * dataset_loader). datasets() are stem-prefixed so the same dataset imported from
 * two configs stays distinguishable.
 */
class imported_source : public dataset_source {
public:
    /// @throws on parse/build failure (propagated from dataset_loader).
    imported_source(const std::string& config_path, hdf5::executor& h5_exec);
    ~imported_source() override;

    [[nodiscard]] std::string label() const override;
    [[nodiscard]] std::vector<dataset_handle> datasets() const override;

    /// Canonicalized path, used to block duplicate imports.
    [[nodiscard]] const std::string& canonical_path() const { return canonical_; }

private:
    std::unique_ptr<dataset_loader> loader_;
    std::string canonical_;
};

/**
 * A source backed by an in-memory set of synthetic ensemble codes. Exposes exactly
 * one dataset whose geometry follows the explorer's global 1D/2D display toggle
 * (S*S square vs S^2 x 1 strip) — feature_dim is S^2 either way, so the toggle
 * never changes metric comparability.
 */
class synthetic_source : public dataset_source {
public:
    /// `show_2d` points at the explorer's global toggle (stable for its lifetime).
    synthetic_source(std::string label, ensemble_codes codes, const bool* show_2d);

    [[nodiscard]] std::string label() const override { return label_; }
    [[nodiscard]] std::vector<dataset_handle> datasets() const override;

private:
    std::string    label_;
    ensemble_codes codes_;
    const bool*    show_2d_ = nullptr;
};

} // namespace coast

#endif // COAST_DATASET_SOURCE_H
