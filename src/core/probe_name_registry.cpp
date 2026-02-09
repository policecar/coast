//
// Created for the COAST probe system
//

#include "core/probe_name_registry.h"

namespace core {

std::mutex probe_name_registry::mutex_;
std::unordered_map<std::type_index, std::set<std::string>> probe_name_registry::registry_;

void probe_name_registry::register_probe(const std::type_info& type, std::string_view name) {
    std::lock_guard<std::mutex> lock(mutex_);
    registry_[std::type_index(type)].emplace(name);
}

std::set<std::string> probe_name_registry::get_probes(const std::type_info& type) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(std::type_index(type));
    if (it != registry_.end()) {
        return it->second;
    }
    return {};
}

} // namespace core
