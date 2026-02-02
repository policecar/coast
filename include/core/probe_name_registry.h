//
// Created for the COAST probe system
//
// Global registry mapping element types to their discovered probe names.
// Thread-safe, populated lazily via PROBE macro on first method execution.
//

#ifndef COAST_PROBE_NAME_REGISTRY_H
#define COAST_PROBE_NAME_REGISTRY_H

#include <mutex>
#include <set>
#include <string>
#include <typeindex>
#include <unordered_map>

namespace core {

/// Thread-safe registry that maps element types to their probe names.
/// Probe names are registered at runtime via the PROBE macro's static initialization.
class probe_name_registry {
public:
    /// Register a probe name for a given element type.
    /// Thread-safe; called once per probe site via static initialization.
    static void register_probe(const std::type_info& type, std::string_view name);

    /// Get all registered probe names for a given element type.
    /// Returns empty set if type has no registered probes.
    static std::set<std::string> get_probes(const std::type_info& type);

private:
    static std::mutex mutex_;
    static std::unordered_map<std::type_index, std::set<std::string>> registry_;
};

} // namespace core

#endif // COAST_PROBE_NAME_REGISTRY_H
