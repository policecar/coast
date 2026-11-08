#include "gui_tools/dataset_source.h"

#include <filesystem>
#include <span>
#include <utility>

#include "tools/dataset_loader.h"

namespace coast {

// ---------------------------------------------------------------------------
// imported_source
// ---------------------------------------------------------------------------

imported_source::imported_source(const std::string& config_path, hdf5::executor& h5_exec)
    : loader_(std::make_unique<dataset_loader>(config_path, h5_exec))
{
    std::error_code ec;
    auto canon = std::filesystem::weakly_canonical(config_path, ec);
    canonical_ = ec ? config_path : canon.string();
}

imported_source::~imported_source() = default;   // dataset_loader complete here

std::string imported_source::label() const
{
    return loader_->config_path();
}

std::vector<dataset_handle> imported_source::datasets() const
{
    auto handles = enumerate_datasets(loader_->env(), loader_->builder());
    const std::string stem = std::filesystem::path(loader_->config_path()).stem().string();
    for (auto& h : handles)
        h.name = stem + ": " + h.name;   // keep same-named datasets distinguishable
    return handles;
}

// ---------------------------------------------------------------------------
// synthetic_source
// ---------------------------------------------------------------------------

synthetic_source::synthetic_source(std::string label, ensemble_codes codes, const bool* show_2d)
    : label_(std::move(label)), codes_(std::move(codes)), show_2d_(show_2d)
{
}

std::vector<dataset_handle> synthetic_source::datasets() const
{
    dataset_handle h;
    h.name = label_;
    h.entity_id = 0;

    const int S      = codes_.side;
    const int stride = S * S;
    if (show_2d_ && *show_2d_) {           // square: S x S x 1
        h.width = S; h.height = S; h.channels = 1;
    } else {                                // strip: S^2 x 1 x 1
        h.width = stride; h.height = 1; h.channels = 1;
    }
    h.sample_count = static_cast<std::size_t>(codes_.count);

    // Capture the (stable) data pointer + stride; the buffer lives as long as this
    // source does, matching the dataset_handle lifetime contract.
    const float* base = codes_.data.data();
    h.get_sample = [base, stride](std::size_t idx) {
        return std::span<const float>(base + idx * static_cast<std::size_t>(stride),
                                      static_cast<std::size_t>(stride));
    };

    std::vector<dataset_handle> out;
    out.push_back(std::move(h));
    return out;
}

} // namespace coast
