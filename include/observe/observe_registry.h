//
// Created for the COAST observe system
//

#ifndef COAST_OBSERVE_REGISTRY_H
#define COAST_OBSERVE_REGISTRY_H

#include "observe/observe_group.h"
#include "observe/observe_cfg.h"
#include "observe/observe_trigger.h"

#include <hdf5.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sim  { class env; }
namespace core { class entity_element; }
namespace hdf5 { class executor; }

namespace observe {

/// Process-wide registry of observe groups and sites.
///
/// Lifecycle is bound to a `sim::env` via attach() / detach(). On attach the
/// registry installs ONE post-process hook -> run_observation_pass(), which
/// every step (device idle, single-threaded):
///   1. recomputes each group's `active_this_step` from its ranges/combine/
///      trigger state,
///   2. walks every entity calling entity_element::perform_observations(),
///   3. each active group's observe() calls hand their value straight to the
///      group writer.
/// There is no pre-process hook and no global hot-path gate. Site lookup is a
/// single-threaded resolution cache (`site_cache_`, keyed on owner + group + id
/// value), not the old per-source-line magic-static caches: the pass is the only
/// producer and it runs after all process() work.
class observe_registry {
public:
    static observe_registry& instance();

    observe_registry(const observe_registry&)            = delete;
    observe_registry& operator=(const observe_registry&) = delete;

    // Group lookup. Thread-safe (locks the registry mutex).
    observe_group_t* group(std::string_view id);  // nullptr if no such group
    observe_group_t& get_or_create_group(std::string_view id);

    // Lazy site discovery. Locks dtype/shape/value_bytes on first call.
    observe_site_t* resolve_site(core::entity_element*    owner,
                                 std::string_view         group_id,
                                 std::string_view         local_id,
                                 hid_t                    h5_dtype,
                                 std::span<const hsize_t> shape,
                                 std::size_t              value_bytes);

    // ---- observe_accessor back-end (called only during the pass) ----
    // Gate on the group's per-step activity; if active, resolve+commit the value.
    void emit(core::entity_element&    owner,
              std::string_view         group_id,
              std::string_view         local_id,
              hid_t                    h5_dtype,
              std::span<const hsize_t> shape,
              const void*              value,
              std::size_t              value_bytes);
    [[nodiscard]] bool group_active(std::string_view group_id) const;
    void trigger(std::string_view group_id, trigger_mode_t mode);

    // Output directory accessor. attach() sets output_dir_ from its caller-
    // resolved argument (the single source of truth), so there is no setter.
    const std::filesystem::path&  output_dir() const noexcept { return output_dir_; }

    // Lifecycle. `cfg` carries the parsed observe section (groups + triggers),
    // `output_dir` is the caller-resolved final output directory (the config's
    // observe leaf placed under config_base, or an absolute override — resolution
    // lives in the app layer, coast::paths::resolve_config_output), and `h5_exec`
    // is the process-wide HDF5 executor that observe writers submit work to. The
    // executor must outlive the registry; detach() flushes all writers first.
    void attach(sim::env&                          env,
                const observe_cfg_t&               cfg,
                const std::filesystem::path&       output_dir,
                hdf5::executor&                    h5_exec);
    void detach();

    hdf5::executor& h5_exec() const noexcept { return *h5_exec_; }

    // Inspection (GUI / config).
    std::vector<observe_group_t*> all_groups();

    // Run `fn` against the given group while holding the registry mutex.
    // Used by the GUI panel to safely mutate a group's `ranges` / `combine`
    // without racing the observation pass. No-op if `g` is null.
    void with_group_locked(observe_group_t* g, const std::function<void(observe_group_t&)>& fn);

private:
    observe_registry() = default;

    // The single post-process hook: recompute activity, then walk entities.
    void run_observation_pass();

    // (entity_element*, group value, id value) -> resolved site. Single-threaded
    // (pass-only) read/write; valid for the registry's attached lifetime since
    // entities are stable across steps. The group and id are stored by VALUE
    // (owning std::string), so the cache does not depend on the caller passing a
    // stable string literal — a dynamically-built id would still resolve
    // correctly. The group is part of the key so the same owner may legitimately
    // emit the same id under two different groups without one group's value being
    // routed into the other's writer. Steady-state lookups (cache hits) use the
    // non-owning cache_lookup view and allocate nothing; the std::strings are
    // materialised only once, on the first insert for a site.
    struct cache_key {
        const void* owner;
        std::string group;
        std::string id;
    };
    struct cache_lookup {            // non-owning query type for heterogeneous find
        const void*      owner;
        std::string_view group;
        std::string_view id;
    };
    struct cache_hash {
        using is_transparent = void;
        static std::size_t mix(const void* owner, std::string_view group,
                               std::string_view id) noexcept {
            auto a = reinterpret_cast<std::uintptr_t>(owner);
            std::size_t h = std::hash<std::uintptr_t>{}(a);
            h = h * 1099511628211ull ^ std::hash<std::string_view>{}(group);
            h = h * 1099511628211ull ^ std::hash<std::string_view>{}(id);
            return h;
        }
        std::size_t operator()(const cache_key&    k) const noexcept { return mix(k.owner, k.group, k.id); }
        std::size_t operator()(const cache_lookup& k) const noexcept { return mix(k.owner, k.group, k.id); }
    };
    struct cache_eq {
        using is_transparent = void;
        bool operator()(const cache_key& a, const cache_key& b) const noexcept {
            return a.owner == b.owner && a.group == b.group && a.id == b.id;
        }
        bool operator()(const cache_key& a, const cache_lookup& b) const noexcept {
            return a.owner == b.owner && a.group == b.group && a.id == b.id;
        }
        bool operator()(const cache_lookup& a, const cache_key& b) const noexcept {
            return a.owner == b.owner && a.group == b.group && a.id == b.id;
        }
    };

    std::mutex                                                          groups_mtx_;
    std::unordered_map<std::string, std::unique_ptr<observe_group_t>>   groups_;
    // Name -> group fast index for the pass's lock-free group_active()/emit()
    // reads. Written ONLY on the sim thread at the start of run_observation_pass()
    // (never by the GUI), so those reads stay race-free within a pass. It is
    // rebuilt from groups_ whenever index_dirty_ is set — get_or_create_group()
    // (config seeding and the GUI Add-Group button both funnel through it) flips
    // the flag so a group created after attach() becomes visible to the pass.
    std::unordered_map<std::string, observe_group_t*>                   group_index_;
    std::atomic<bool>                                                   index_dirty_{false};
    std::unordered_map<cache_key, observe_site_t*, cache_hash, cache_eq> site_cache_;

    // Set by attach() to the caller-resolved output directory (config's
    // observe leaf under config_base, or an absolute override). Empty until
    // attach; never a hardcoded relative default.
    std::filesystem::path                                               output_dir_{};
    sim::env*                                                           env_{nullptr};
    hdf5::executor*                                                     h5_exec_{nullptr};
    std::size_t                                                         post_hook_id_{0};
    // Always written under groups_mtx_ (by the sim thread each pass; reset by
    // detach()). Read under groups_mtx_ by trigger(), which may run on the GUI
    // thread, and unlocked by emit(), which runs on the sim thread inside the pass.
    std::uint64_t                                                       pass_step_{0};
    bool                                                                attached_{false};
};

} // namespace observe

#endif // COAST_OBSERVE_REGISTRY_H
