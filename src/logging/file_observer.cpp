//
// File Observer implementation
//

#include "logging/file_observer.h"
#include "logging/probe_logger.h"

namespace logging {

file_observer::file_observer(probe_logger* logger, std::string probe_path)
    : logger_(logger)
    , probe_path_(std::move(probe_path))
{
}

file_observer::~file_observer() = default;

void file_observer::push(std::uint64_t step, double value)
{
    if (logger_) {
        logger_->record_value(probe_path_, step, value);
        ++sample_count_;
    }
}

} // namespace logging
