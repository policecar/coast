//
// Created for the COAST observe system
//

#include "observe/observe_registry.h"
#include "observe/observe_accessor.h"
#include "observe/observe_writer.h"

#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"
#include "probe/probe_instance.h"
#include "core/structure_registry.h"
#include "core/entity_element.h"
#include "diag/diag.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace observe {

// Configuration applier lives in observe_config.cpp (separate translation
// unit). Forward-declare here so attach() can call it without pulling its
// header in.
void apply_config(observe_registry&, const observe_cfg_t&);

namespace {

// Enforce the per-site dtype/shape/size lock (first observe wins). A later observe
// at the same (owner, group, id) that disagrees on dtype/shape/size is a
// programming error — silently forwarding a value shorter than the site was
// registered with would make the writer over-read the caller's buffer. Abort with
// a clear message (matching the debug assert). Shared by resolve_site (cache-miss)
// and emit (cache-hit), so BOTH paths enforce the same contract.
void enforce_site_lock(const observe_site_t& s, hid_t h5_dtype,
                       std::span<const hsize_t> shape, std::size_t value_bytes)
{
    const bool dtype_ok = (s.h5_dtype    == h5_dtype);
    const bool size_ok  = (s.value_bytes == value_bytes);
    const bool shape_ok = (s.shape.size() == shape.size())
                        && std::equal(s.shape.begin(), s.shape.end(),
                                      shape.begin(), shape.end());
    if (!(dtype_ok && size_ok && shape_ok)) {
        COAST_ERROR("observe", "dtype/shape lock violated for site '{}' "
                               "(dtype_ok={} size_ok={} shape_ok={})",
                    s.full_id, dtype_ok, size_ok, shape_ok);
        assert(false && "observe site dtype/shape mismatch");
        std::abort();
    }
}

std::string resolve_full_id(core::entity_element* owner,
                            std::string_view      local_id)
{
    if (owner == nullptr) {
        return std::string(local_id);
    }

    sim::io_entity* root = owner->root_entity();
    if (root != nullptr) {
        const auto& structure = root->structure();
        if (const auto* entry = structure.find(owner)) {
            const auto* root_entry = structure.root();
            const std::string& path = entry->path();
            std::string qualified_path;
            if (root_entry != nullptr) {
                const std::string& root_path = root_entry->path();
                qualified_path.reserve(path.size() + 32);
                qualified_path.append(root_path);
                qualified_path.push_back('[');
                qualified_path.append(std::to_string(root->get_outp_id()));
                qualified_path.push_back(']');
                if (path.size() > root_path.size()) {
                    qualified_path.append(path.substr(root_path.size()));
                }
            } else {
                qualified_path = path;
            }

            qualified_path.push_back('/');
            qualified_path.append(local_id);
            return qualified_path;
        }
    }

    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
        COAST_WARN("observe", "owner path unavailable; site '{}' will not be namespaced",
                   local_id);
    }
    return std::string(local_id);
}

// Per-step trigger evaluation. `active` is the strict commit decision
// (combine-mode-aware); identical to the pre-refactor evaluation so committed
// rows are unchanged.
struct trigger_eval_t { bool range_hit; bool active; };

trigger_eval_t evaluate_triggers(const observe_group_t& g,
                                 std::uint64_t step,
                                 bool prog_active, bool prog_single)
{
    bool range_hit = false;
    for (const auto& r : g.ranges) {
        if (r.start_step <= step && step <= r.stop_step) { range_hit = true; break; }
    }
    const bool prog_input = prog_active || prog_single;

    bool active;
    if (g.combine == combine_mode_t::AND_) {
        const bool has_range = !g.ranges.empty();
        active = has_range ? (range_hit && prog_input) : prog_input;
    } else {  // OR
        active = range_hit || prog_input;
    }
    return { range_hit, active };
}

} // namespace

observe_registry& observe_registry::instance() {
    static observe_registry inst;
    return inst;
}

observe_group_t* observe_registry::group(std::string_view id) {
    std::lock_guard<std::mutex> lk(groups_mtx_);
    auto it = groups_.find(std::string(id));
    return it == groups_.end() ? nullptr : it->second.get();
}

observe_group_t& observe_registry::get_or_create_group(std::string_view id) {
    std::lock_guard<std::mutex> lk(groups_mtx_);
    std::string key(id);
    auto it = groups_.find(key);
    if (it == groups_.end()) {
        auto g       = std::make_unique<observe_group_t>();
        g->group_id  = key;
        it = groups_.emplace(std::move(key), std::move(g)).first;
        // A new group appeared (config seeding or the GUI Add-Group button).
        // Flag the name index for rebuild so the next pass makes this group
        // visible to group_active()/emit(); without this a GUI-armed group
        // would show ACTIVE in the panel yet capture nothing.
        index_dirty_.store(true, std::memory_order_release);
    }
    return *it->second;
}

observe_site_t* observe_registry::resolve_site(core::entity_element*    owner,
                                               std::string_view         group_id,
                                               std::string_view         local_id,
                                               hid_t                    h5_dtype,
                                               std::span<const hsize_t> shape,
                                               std::size_t              value_bytes)
{
    // Build the namespaced full identifier from the observed owner. Each root
    // entity owns its own structure_registry, so a single registry pointer from
    // attach() cannot namespace multi-root simulations correctly.
    std::string full_id = resolve_full_id(owner, local_id);

    observe_group_t& g = get_or_create_group(group_id);

    // Guard the plain map: the sim thread inserts here while the GUI may read
    // site_count() under the same lock. Returned observe_site_t* is heap-stable
    // (held by unique_ptr), so it remains valid after the lock is released.
    std::lock_guard<site_spin_lock> lk(g.sites_lock);

    auto it = g.sites.find(full_id);
    if (it == g.sites.end()) {
        auto site         = std::make_unique<observe_site_t>();
        site->full_id     = full_id;
        site->h5_dtype    = h5_dtype;
        site->shape.assign(shape.begin(), shape.end());
        site->value_bytes = value_bytes;
        it = g.sites.emplace(std::move(full_id), std::move(site)).first;
    } else {
        // Shape/dtype lock: a second observe against the same full_id that
        // disagrees on dtype, shape, or size is a programming error.
        enforce_site_lock(*it->second, h5_dtype, shape, value_bytes);
    }

    return it->second.get();
}

bool observe_registry::group_active(std::string_view group_id) const {
    // group_index_ is refreshed only on the sim thread at pass start;
    // active_this_step is written only on the sim thread (pass start / trigger),
    // so this lock-free read is safe from within the pass.
    // NOTE: the std::string(group_id) is a per-call construction (group_index_
    // has no transparent hash). SSO covers all current group names, so it does
    // not allocate in practice; switch group_index_ to a heterogeneous-lookup
    // map (as site_cache_ already does) if a non-SSO group name ever appears.
    auto it = group_index_.find(std::string(group_id));
    return it != group_index_.end() &&
           it->second->active_this_step.load(std::memory_order_acquire);
}

void observe_registry::emit(core::entity_element&    owner,
                            std::string_view         group_id,
                            std::string_view         local_id,
                            hid_t                    h5_dtype,
                            std::span<const hsize_t> shape,
                            const void*              value,
                            std::size_t              value_bytes)
{
    auto it = group_index_.find(std::string(group_id));
    if (it == group_index_.end()) return;          // unconfigured group -> drop
    observe_group_t* g = it->second;
    if (!g->active_this_step.load(std::memory_order_acquire)) return;  // dormant this step

    // Resolve (and cache) the site for this (owner, group, id). The cache is
    // keyed on the group + id VALUE, not the id pointer, so it stays correct even
    // if the caller passes a non-literal id, and the same owner emitting the same
    // id under two groups resolves to two distinct sites/writers. The hit path
    // uses a non-owning view (no alloc); the owning std::strings are built only on
    // the first insert for this site.
    observe_site_t* site = nullptr;
    if (auto cit = site_cache_.find(cache_lookup{ &owner, group_id, local_id });
        cit != site_cache_.end()) {
        site = cit->second;
        // Cache hit skips resolve_site's validation, so enforce the dtype/shape/
        // size lock here too — otherwise a later emit with a smaller value_bytes
        // than the site was registered with is handed to the writer unchecked and
        // over-reads the caller's buffer in H5Dwrite.
        enforce_site_lock(*site, h5_dtype, shape, value_bytes);
    } else {
        site = resolve_site(&owner, group_id, local_id, h5_dtype, shape, value_bytes);
        if (site == nullptr) return;
        site_cache_.emplace(cache_key{ &owner, std::string(group_id), std::string(local_id) }, site);
    }

    // Lazily open the group writer on first commit (sim thread only).
    if (!g->writer) {
        g->h5_path = output_dir_ / (g->group_id + std::string(".observe.h5"));
        std::error_code ec;
        std::filesystem::create_directories(g->h5_path.parent_path(), ec);
        // The run this file belongs to: a continued run continues its files
        // (observe_run_t, SPEC.md S.5.14).
        g->writer = std::make_unique<observe_writer_t>(
            g->h5_path, *h5_exec_,
            env_ != nullptr ? observe_run_t{ env_->run_id(), env_->continued_from_step() }
                            : observe_run_t{});
        g->writer->start();
    }

    g->writer->enqueue(site, value, value_bytes, pass_step_);
}

void observe_registry::trigger(std::string_view group_id, trigger_mode_t mode) {
    // Look up in groups_ (authoritative; populated by both attach() and direct
    // get_or_create_group()) under the registry mutex. Callable from the sim
    // thread inside the pass, or the GUI thread.
    std::lock_guard<std::mutex> lk(groups_mtx_);
    auto it = groups_.find(std::string(group_id));
    if (it == groups_.end()) return;
    observe_group_t& g = *it->second;

    switch (mode) {
        case trigger_mode_t::START:          g.prog_active.store(true,  std::memory_order_release); break;
        case trigger_mode_t::STOP:           g.prog_active.store(false, std::memory_order_release); break;
        case trigger_mode_t::SINGLE_CAPTURE: g.prog_single_pending.store(true, std::memory_order_release); break;
        case trigger_mode_t::NONE:           break;
    }

    // Re-evaluate this group's commit decision live so a subsequent observe()
    // in the same pass sees the change.
    const bool pa = g.prog_active.load(std::memory_order_acquire);
    const bool ps = g.prog_single_pending.load(std::memory_order_acquire);
    g.active_this_step.store(evaluate_triggers(g, pass_step_, pa, ps).active,
                             std::memory_order_release);
}

void observe_registry::attach(sim::env&                          env,
                              const observe_cfg_t&               cfg,
                              const std::filesystem::path&       output_dir,
                              hdf5::executor&                    h5_exec)
{
    if (attached_) {
        COAST_WARN("observe", "attach() called while already attached; ignoring");
        return;
    }

    // `output_dir` is the caller-resolved final directory (the config's observe
    // leaf placed under config_base, or an absolute override — resolution lives
    // in the app layer, coast::paths::resolve_config_output). It is authoritative
    // here; apply_config() below no longer touches output_dir_, only seeds groups.
    output_dir_ = output_dir;
    env_        = &env;
    h5_exec_    = &h5_exec;

    apply_config(*this, cfg);

    // Index the configured groups for lock-free name lookup during the pass.
    // apply_config() created the groups via get_or_create_group(), which set
    // index_dirty_; we clear it here now that the index reflects groups_.
    {
        std::lock_guard<std::mutex> lk(groups_mtx_);
        group_index_.clear();
        for (auto& [gid, g_up] : groups_) {
            group_index_[gid] = g_up.get();
        }
        index_dirty_.store(false, std::memory_order_release);
    }

    post_hook_id_ = env.set_post_process_hook([this]{ this->run_observation_pass(); });
    attached_     = true;
}

void observe_registry::detach() {
    if (!attached_) {
        return;
    }

    if (env_ != nullptr) {
        env_->remove_post_process_hook(post_hook_id_);
    }

    {
        std::lock_guard<std::mutex> lk(groups_mtx_);
        for (auto& [gid, g_up] : groups_) {
            if (g_up->writer) {
                g_up->writer->stop_and_join();
                g_up->writer.reset();
            }
        }
        groups_.clear();
        group_index_.clear();
        index_dirty_.store(false, std::memory_order_release);
        pass_step_ = 0;
    }

    site_cache_.clear();

    attached_     = false;
    env_          = nullptr;
    h5_exec_      = nullptr;
    post_hook_id_ = 0;
}

void observe_registry::with_group_locked(observe_group_t* g,
                                         const std::function<void(observe_group_t&)>& fn)
{
    if (g == nullptr) return;
    std::lock_guard<std::mutex> lk(groups_mtx_);
    fn(*g);
}

std::vector<observe_group_t*> observe_registry::all_groups() {
    std::lock_guard<std::mutex> lk(groups_mtx_);
    std::vector<observe_group_t*> out;
    out.reserve(groups_.size());
    for (auto& [gid, g_up] : groups_) {
        out.push_back(g_up.get());
    }
    return out;
}

void observe_registry::run_observation_pass() {
    // Single post-process hook: sim thread, end of step, device idle, after
    // every process() and GPU rendezvous/read_probes.
    //
    // Idle fast-path: nothing configured AND nothing newly added -> one atomic
    // load and return. index_dirty_ lets a group armed after attach() (GUI
    // Add-Group) wake the pass even while group_index_ is still empty.
    if (env_ == nullptr) return;
    if (group_index_.empty() && !index_dirty_.load(std::memory_order_acquire)) {
        return;  // nothing configured -> zero cost
    }

    const std::uint64_t step = probe::get_current_step();

    // Under the lock: publish this pass's step, refresh the name index if a
    // group was added after attach() (rebuilt on the sim thread so the pass's
    // lock-free reads stay race-free), then recompute each group's commit
    // decision for this step.
    {
        std::lock_guard<std::mutex> lk(groups_mtx_);
        // Written under groups_mtx_ because trigger() reads it under the same
        // lock from the GUI thread (#215). emit() reads it unlocked, which is
        // fine: emit() runs on the sim thread, which is the only writer while
        // attached (detach() also writes it, after removing this hook).
        pass_step_ = step;
        if (index_dirty_.exchange(false, std::memory_order_acq_rel)) {
            group_index_.clear();
            group_index_.reserve(groups_.size());
            for (auto& [gid, g_up] : groups_) {
                group_index_[gid] = g_up.get();
            }
        }
        for (auto& [gid, g_up] : groups_) {
            observe_group_t& g = *g_up;
            const bool pa = g.prog_active.load(std::memory_order_acquire);
            const bool ps = g.prog_single_pending.load(std::memory_order_acquire);
            g.active_this_step.store(evaluate_triggers(g, step, pa, ps).active,
                                     std::memory_order_release);
        }
    }

    // Always walk every entity so optional trigger() calls inside an override
    // are evaluated every step; each override short-circuits on group_active().
    observe_accessor acc(*this);
    for (auto& root : env_->iterate_entities()) {
        root.perform_observations(acc);
    }

    // Consume single-capture pulses now that this pass has honoured them.
    {
        std::lock_guard<std::mutex> lk(groups_mtx_);
        for (auto& [gid, g_up] : groups_) {
            g_up->prog_single_pending.store(false, std::memory_order_release);
        }
    }
}

} // namespace observe
