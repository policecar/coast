//
// File Observer - probe_observer implementation that logs to file
//

#ifndef COAST_FILE_OBSERVER_H
#define COAST_FILE_OBSERVER_H

#include "probe/probe_observer.h"
#include <string>
#include <string_view>
#include <cstdint>

namespace logging {

class probe_logger;

/// A probe_observer that writes values to a file sink instead of visualizing them.
/// This observer does not render anything - it's purely for data capture.
class file_observer : public probe::probe_observer {
public:
    explicit file_observer(probe_logger* logger, std::string probe_path);
    ~file_observer() override;

    file_observer(const file_observer&) = delete;
    file_observer& operator=(const file_observer&) = delete;

    /// Receive a new timestamped value (called during update_observers from GUI thread).
    void push(std::uint64_t step, double value) override;

    /// No-op for file observer - nothing to render.
    void render() override {}

    /// Get the observer type name for display.
    [[nodiscard]] std::string_view name() const override { return "File Logger"; }

    /// Get the probe path being logged
    [[nodiscard]] const std::string& probe_path() const { return probe_path_; }

    /// Get number of samples logged
    [[nodiscard]] std::uint64_t sample_count() const { return sample_count_; }

private:
    probe_logger* logger_;
    std::string probe_path_;
    std::uint64_t sample_count_ = 0;
};

} // namespace logging

#endif // COAST_FILE_OBSERVER_H
