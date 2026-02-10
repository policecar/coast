//
// Probe Logger implementation
//

#include "logging/probe_logger.h"
#include "logging/file_observer.h"
#include "probe/probe_instance.h"

#include <algorithm>
#include <chrono>
#include <iomanip>

namespace logging {

probe_logger::probe_logger(probe_logger_config config)
    : config_(std::move(config))
{
}

probe_logger::~probe_logger()
{
    if (logging_active_) {
        stop();
    }
}

log_result probe_logger::start()
{
    if (logging_active_) {
        return log_result::error("Probe logger already active");
    }

    stats_ = {};

    if (config_.format == probe_log_format::hdf5) {
        auto result = start_hdf5();
        if (!result) return result;
    } else {
        auto result = start_csv();
        if (!result) return result;
    }

    logging_active_ = true;
    return log_result::ok();
}

log_result probe_logger::stop()
{
    if (!logging_active_) {
        return log_result::error("Probe logger not active");
    }

    // Flush remaining data
    flush();

    // Remove all file observers from probes
    for (auto& [probe, index] : probe_to_index_) {
        // Find and remove our observer
        auto& observers = probe->observers;
        auto it = std::remove_if(observers.begin(), observers.end(),
            [](const std::unique_ptr<probe::probe_observer>& obs) {
                return dynamic_cast<file_observer*>(obs.get()) != nullptr;
            });
        observers.erase(it, observers.end());
    }
    probe_to_index_.clear();
    probe_buffers_.clear();

    if (config_.format == probe_log_format::hdf5) {
        stop_hdf5();
    } else {
        stop_csv();
    }

    logging_active_ = false;
    return log_result::ok();
}

void probe_logger::flush()
{
    if (!logging_active_) return;

    if (config_.format == probe_log_format::hdf5) {
        flush_hdf5();
    } else {
        flush_csv();
    }
}

bool probe_logger::add_probe(probe::probe_instance* probe)
{
    if (!logging_active_ || !probe) {
        return false;
    }

    // Check if already logging this probe
    if (probe_to_index_.contains(probe)) {
        return false;
    }

    // Create buffer for this probe
    std::size_t index = probe_buffers_.size();
    probe_buffers_.push_back({
        .path = probe->full_path,
        .steps = {},
        .values = {},
        .total_samples = 0
    });
    probe_buffers_.back().steps.reserve(config_.buffer_size);
    probe_buffers_.back().values.reserve(config_.buffer_size);

    probe_to_index_[probe] = index;

    // Create HDF5 dataset if needed
    if (config_.format == probe_log_format::hdf5) {
        ensure_hdf5_dataset(index);
    }

    // Add file observer to the probe
    auto observer = std::make_unique<file_observer>(this, probe->full_path);
    probe->add_observer(std::move(observer));

    return true;
}

void probe_logger::remove_probe(probe::probe_instance* probe)
{
    if (!probe) return;

    auto it = probe_to_index_.find(probe);
    if (it == probe_to_index_.end()) {
        return;
    }

    // Flush data for this probe before removing
    flush();

    // Remove file observer from probe
    auto& observers = probe->observers;
    auto obs_it = std::remove_if(observers.begin(), observers.end(),
        [](const std::unique_ptr<probe::probe_observer>& obs) {
            return dynamic_cast<file_observer*>(obs.get()) != nullptr;
        });
    observers.erase(obs_it, observers.end());

    probe_to_index_.erase(it);
}

std::vector<probe::probe_instance*> probe_logger::active_probes() const
{
    std::vector<probe::probe_instance*> result;
    result.reserve(probe_to_index_.size());
    for (const auto& [probe, _] : probe_to_index_) {
        result.push_back(probe);
    }
    return result;
}

void probe_logger::record_value(const std::string& probe_path, std::uint64_t step, double value)
{
    // Find the buffer for this probe
    for (auto& buf : probe_buffers_) {
        if (buf.path == probe_path) {
            buf.steps.push_back(step);
            buf.values.push_back(value);
            buf.total_samples++;
            stats_.samples_written++;

            // Auto-flush if buffer is full
            if (buf.values.size() >= config_.buffer_size) {
                flush();
            }
            return;
        }
    }
}

// --- HDF5 implementation ---

log_result probe_logger::start_hdf5()
{
    hdf5_file_ = hdf5::file::create(config_.output_path.string());
    if (!hdf5_file_) {
        return log_result::error("Failed to create HDF5 file: " + config_.output_path.string());
    }

    // Write metadata
    auto metadata = hdf5::group::create(hdf5_file_.id(), "metadata");
    if (metadata) {
        hdf5::write_attr(metadata.id(), "schema_version", std::string_view("1.0"));
        hdf5::write_attr(metadata.id(), "type", std::string_view("probe_log"));

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::string timestamp = std::ctime(&time_t);
        if (!timestamp.empty() && timestamp.back() == '\n') {
            timestamp.pop_back();
        }
        hdf5::write_attr(metadata.id(), "created", std::string_view(timestamp));
    }

    probes_group_ = hdf5::group::create(hdf5_file_.id(), "probes");
    if (!probes_group_) {
        return log_result::error("Failed to create probes group");
    }

    hdf5_datasets_.clear();
    return log_result::ok();
}

void probe_logger::ensure_hdf5_dataset(std::size_t probe_index)
{
    if (probe_index >= probe_buffers_.size()) return;

    // Check if dataset already exists for this index
    for (const auto& ds : hdf5_datasets_) {
        if (ds.path == probe_buffers_[probe_index].path) {
            return;  // Already exists
        }
    }

    const auto& path = probe_buffers_[probe_index].path;

    // Create sanitized group name (replace / with _)
    std::string safe_name = path;
    std::replace(safe_name.begin(), safe_name.end(), '/', '_');
    std::replace(safe_name.begin(), safe_name.end(), '[', '_');
    std::replace(safe_name.begin(), safe_name.end(), ']', '_');

    auto probe_group = hdf5::group::create(probes_group_.id(), safe_name.c_str());
    if (!probe_group) return;

    // Store the original path as an attribute
    hdf5::write_attr(probe_group.id(), "probe_path", std::string_view(path));

    // Create extensible datasets for steps and values
    hsize_t initial_dims[1] = {0};
    hsize_t chunk_dims[1] = {config_.buffer_size};

    auto dcpl = hdf5::plist::dataset_create();
    dcpl.set_chunk(std::span(chunk_dims));
    if (config_.compression.algorithm == compression_config::type::gzip) {
        dcpl.set_shuffle();
        dcpl.set_deflate(config_.compression.level);
    }

    auto space = hdf5::dataspace::extensible(std::span(initial_dims));

    // Create steps dataset (uint64)
    auto steps_dataset = hdf5::dataset::create(probe_group.id(), "steps",
                                                H5T_NATIVE_UINT64, space.id(), dcpl.id());

    // Create values dataset (double)
    auto space2 = hdf5::dataspace::extensible(std::span(initial_dims));
    auto values_dataset = hdf5::dataset::create(probe_group.id(), "values",
                                                 H5T_NATIVE_DOUBLE, space2.id(), dcpl.id());

    hdf5_datasets_.push_back({
        .path = path,
        .steps = std::move(steps_dataset),
        .values = std::move(values_dataset),
        .current_rows = 0
    });
}

void probe_logger::flush_hdf5()
{
    for (std::size_t i = 0; i < probe_buffers_.size(); ++i) {
        auto& buf = probe_buffers_[i];
        if (buf.values.empty()) continue;

        // Find corresponding dataset
        hdf5_probe_dataset* ds = nullptr;
        for (auto& d : hdf5_datasets_) {
            if (d.path == buf.path) {
                ds = &d;
                break;
            }
        }

        if (!ds || !ds->values || !ds->steps) continue;

        std::size_t num_values = buf.values.size();
        std::size_t new_rows = ds->current_rows + num_values;

        // Extend both datasets
        hsize_t new_dims[1] = {new_rows};
        ds->steps.extend(std::span(new_dims));
        ds->values.extend(std::span(new_dims));

        // Write steps data
        hsize_t offset[1] = {ds->current_rows};
        hsize_t count[1] = {num_values};
        auto steps_file_space = hdf5::dataspace::from_dataset(ds->steps.id());
        steps_file_space.select_hyperslab(std::span(offset), std::span(count));
        auto mem_space = hdf5::dataspace::simple(std::span(count));
        ds->steps.write(H5T_NATIVE_UINT64, mem_space.id(), steps_file_space.id(), buf.steps.data());

        // Write values data
        auto values_file_space = hdf5::dataspace::from_dataset(ds->values.id());
        values_file_space.select_hyperslab(std::span(offset), std::span(count));
        auto values_mem_space = hdf5::dataspace::simple(std::span(count));
        ds->values.write(H5T_NATIVE_DOUBLE, values_mem_space.id(), values_file_space.id(), buf.values.data());

        ds->current_rows = new_rows;
        stats_.bytes_written += num_values * (sizeof(double) + sizeof(std::uint64_t));
        buf.steps.clear();
        buf.values.clear();
    }

    hdf5_file_.flush();
}

void probe_logger::stop_hdf5()
{
    hdf5_datasets_.clear();
    probes_group_ = {};
    hdf5_file_ = {};
}

// --- CSV implementation ---

log_result probe_logger::start_csv()
{
    csv_file_.open(config_.output_path, std::ios::out | std::ios::trunc);
    if (!csv_file_.is_open()) {
        return log_result::error("Failed to create CSV file: " + config_.output_path.string());
    }

    csv_file_ << std::setprecision(15);
    csv_header_written_ = false;

    return log_result::ok();
}

void probe_logger::flush_csv()
{
    if (!csv_file_.is_open()) return;

    // Write header if needed
    if (!csv_header_written_ && !probe_buffers_.empty()) {
        csv_file_ << "step";
        for (const auto& buf : probe_buffers_) {
            csv_file_ << "," << buf.path;
        }
        csv_file_ << "\n";
        csv_header_written_ = true;
    }

    // Find max samples across all buffers
    std::size_t max_samples = 0;
    for (const auto& buf : probe_buffers_) {
        max_samples = std::max(max_samples, buf.values.size());
    }

    if (max_samples == 0) return;

    // Write rows using actual step values from first buffer
    // Note: This is a simple approach - all probes should have same step values
    // For mismatched steps, this may produce incorrect output
    for (std::size_t i = 0; i < max_samples; ++i) {
        // Use step from first buffer that has this index
        std::uint64_t step = 0;
        for (const auto& buf : probe_buffers_) {
            if (i < buf.steps.size()) {
                step = buf.steps[i];
                break;
            }
        }
        csv_file_ << step;

        for (const auto& buf : probe_buffers_) {
            csv_file_ << ",";
            if (i < buf.values.size()) {
                csv_file_ << buf.values[i];
            }
            // Empty if no value (NaN equivalent in CSV)
        }
        csv_file_ << "\n";
    }

    // Clear buffers
    for (auto& buf : probe_buffers_) {
        stats_.bytes_written += buf.values.size() * 20;  // Rough estimate for CSV
        buf.steps.clear();
        buf.values.clear();
    }

    csv_file_.flush();
}

void probe_logger::stop_csv()
{
    if (csv_file_.is_open()) {
        csv_file_.close();
    }
    csv_header_written_ = false;
}

} // namespace logging
