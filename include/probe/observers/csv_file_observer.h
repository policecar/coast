//
// CSV File Observer - writes probe values directly to a CSV file
//

#ifndef COAST_CSV_FILE_OBSERVER_H
#define COAST_CSV_FILE_OBSERVER_H

#include "probe/probe_observer.h"
#include <string>
#include <string_view>
#include <fstream>
#include <cstdint>

namespace probe {

/// A probe_observer that writes values directly to a CSV file.
/// Each observer manages its own file (one probe per file).
/// CSV format: step,value (where step is the simulation step number)
class csv_file_observer : public probe_observer {
public:
    /// Create observer with file path
    /// @param file_path Path to the CSV file to write
    explicit csv_file_observer(const std::string& file_path);
    ~csv_file_observer() override;

    csv_file_observer(const csv_file_observer&) = delete;
    csv_file_observer& operator=(const csv_file_observer&) = delete;

    /// Receive a new timestamped value (called during update_observers from GUI thread).
    void push(std::uint64_t step, double value) override;

    /// Render status display.
    void render() override;

    /// Get the observer type name for display.
    [[nodiscard]] std::string_view name() const override { return "CSV File"; }

    /// Get the file path
    [[nodiscard]] const std::string& file_path() const { return file_path_; }

    /// Get number of samples written
    [[nodiscard]] std::uint64_t sample_count() const { return sample_count_; }

    /// Check if file is open and ready
    [[nodiscard]] bool is_open() const { return file_.is_open(); }

private:
    std::string file_path_;
    std::ofstream file_;
    std::uint64_t sample_count_ = 0;
    std::string error_message_;
};

} // namespace probe

#endif // COAST_CSV_FILE_OBSERVER_H
