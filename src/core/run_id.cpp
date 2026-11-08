#include "core/run_id.h"

#include <cstdint>
#include <format>
#include <random>

namespace coast {

std::string new_run_id()
{
    std::random_device rd;
    std::string id;
    for (int i = 0; i < 4; ++i)
        id += std::format("{:08x}", static_cast<std::uint32_t>(rd()));
    return id;
}

std::string resolve_run_id(std::string_view configured)
{
    return configured == k_auto_run_id ? new_run_id() : std::string(configured);
}

} // namespace coast
