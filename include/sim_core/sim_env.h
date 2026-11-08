//
// Created by jk on 04.09.25.
//

#ifndef SIM_ENV_H
#define SIM_ENV_H

#include <functional>
#include <initializer_list>
#include <unordered_map>
#include <vector>
#include <typeindex>
#include <utility>
#include <memory>
#include <atomic>
#include <mutex>
#include <optional>

#include <oneapi/tbb/partitioner.h>

#include "sim_core/io_entity.h"
#include "sim_core/io_buffer.h"

#ifdef COAST_WITH_GPU
namespace coast::gpu { struct gpu_device; }
#endif

namespace sim {

#ifdef COAST_WITH_GPU
// fwd: the per-step GPU dispatch scratch (below) holds pointers only.
class gpu_io_entity;
#endif

class entity_vec {
public:
    virtual ~entity_vec() = default;
    [[nodiscard]] virtual std::size_t get_size() const { std::unreachable(); };
    [[nodiscard]] virtual io_entity&  as_base(std::size_t) { std::unreachable(); };
};

template<class T>
requires std::is_base_of_v<io_entity, T>
class typed_entity_vec : public entity_vec, public std::vector<T> {
public:
    [[nodiscard]] std::size_t get_size() const override
    {
        return std::vector<T>::size();
    }
    [[nodiscard]] io_entity& as_base(std::size_t idx) override
    {
        return std::vector<T>::at(idx);
    }
};

// Owning forward range over io_entity references. Replaces the previous
// std::generator<io_entity&> coroutine used by iterate_entities(): libc++ 22
// ships no <generator>, so the macOS clang/libc++ console build needs a
// non-coroutine iteration. The pointer vector is snapshotted at construction
// and owned here, so iteration yields io_entity& exactly as before and callers
// stay unchanged. Not on the hot path (env::process() uses cpu_dispatch_).
class entity_range {
public:
    explicit entity_range(std::vector<io_entity*> ptrs) : ptrs_(std::move(ptrs)) {}

    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = io_entity;
        using difference_type   = std::ptrdiff_t;
        using pointer           = io_entity*;
        using reference         = io_entity&;

        explicit iterator(std::vector<io_entity*>::const_iterator it) : it_(it) {}
        io_entity& operator*() const { return **it_; }
        iterator& operator++() { ++it_; return *this; }
        iterator operator++(int) { auto tmp = *this; ++it_; return tmp; }
        bool operator==(const iterator& o) const { return it_ == o.it_; }
        bool operator!=(const iterator& o) const { return it_ != o.it_; }
    private:
        std::vector<io_entity*>::const_iterator it_;
    };

    [[nodiscard]] iterator begin() const { return iterator(ptrs_.begin()); }
    [[nodiscard]] iterator end()   const { return iterator(ptrs_.end());   }

private:
    std::vector<io_entity*> ptrs_;
};

/**
 * Simulation environment that manages entities and I/O buffers.
 *
 * ── Simulation step semantics (the contract process()/swap_io() implement) ──
 *
 * A step is bracketed by two GLOBAL barriers. Both are mandatory in this
 * "robust" formulation; do not weaken them to chase throughput.
 *
 *   … BARRIER 2 of the previous step ─┐
 *   step begins                        │
 *     all entities process — ORDER-INDEPENDENT, therefore parallelized
 *       (CPU entities run concurrently; the GPU batch is submitted no-wait and
 *        runs alongside them). [env::process()]
 *
 *       Independence is REAL only because the io_buffers are DOUBLE-BUFFERED:
 *       every entity reads the stable read-side and writes the write-side. The
 *       double-buffering is what licenses "any order"; the barriers are what
 *       keep the double-buffering invariant true. The barriers are FEATURES,
 *       not overhead — removing one does not speed the sim up, it makes the
 *       independence claim false.
 *
 *     BARRIER 1: every entity has finished processing, before any swap.
 *       (the GPU rendezvous — wait_idle — joined with the CPU dispatch join.)
 *
 *     swap: each io_buffer flips its pages; some buffers move data across a
 *       boundary DURING the flip (CPU↔GPU today; a future io_buffer variant
 *       might move it over a network). The swap is not complete until that
 *       movement has landed on the read-side. [env::swap_io()]
 *
 *     BARRIER 2: every buffer has swapped, before the next step may start.
 *       Enforced structurally by the SERIAL sim loop (process → swap_io →
 *       process on the sim thread): swap_io() does not return until the
 *       cross-device transfers have completed and the pages have flipped, so
 *       the next process() provably cannot begin until the sim is in a defined
 *       state.
 *
 *   step ends → simulation is in a DEFINED state: the previous step's results
 *       sit on the read-side of every io_buffer. GUI / logging / state-save
 *       read THIS stable state. They need no mandatory extra sync; the only
 *       optional sync is an on-demand readback of a buffer that is normally
 *       GPU-resident and keeps no CPU mirror (io_buffer::read_back_sync), when
 *       a consumer reaches for it.
 *
 * Out of scope (deliberately): DECENTRALIZED stepping, where an entity restarts
 * as soon as its OWN inputs have swapped instead of waiting on the global
 * barrier 2. That relaxation is interesting for very large / machine-distributed
 * graphs but is overkill here, and any GPU keep-warm (cross-step pipelining)
 * would require it — hence keep-warm is NOT a design driver for the current
 * two-barrier model.
 *
 * Future consideration: If the simulation structure (entities) can change
 * during runtime, add a callback/observer system to notify interested
 * components (e.g., GUI entity browsers) about structural changes.
 * This would allow them to invalidate caches and refresh their views.
 */
class env {

#ifdef COAST_WITH_GPU
    // Set by enable_gpu() (before init_io_buffers()) when a GPU entity is present;
    // false in CPU-only runs. Gates lazy device creation in init_io_buffers().
    bool gpu_enabled_ = false;

    // The agnostic GPU device (gpu-abstraction-layer). Created in init_io_buffers()
    // via make_gpu_device() once the buffer topology is known; the backend builds
    // and privately owns its own context inside the device (no SDK type reaches
    // this header). Declared here (before entities / io_buffers) so it is destroyed
    // LAST — every entity and io_buffer releases its GPU resources against the
    // device this owns. Null in CPU-only runs.
    std::unique_ptr<coast::gpu::gpu_device> gpu_dev_;
#endif

    std::unordered_map<std::type_index,std::unique_ptr<entity_vec>> entities;
    // unique_ptr-of-base: io_buffer is abstract since S3; the map owns one of the
    // concrete variants (io_buffer_cpu / _cpu_to_gpu / _gpu_to_cpu / _gpu) per id,
    // selected by topology in init_io_buffers(). Pointers into mapped values stay
    // address-stable across rehash (unique_ptr move only moves the pointer).
    std::unordered_map<std::size_t,std::unique_ptr<io_buffer>> io_buffers;

    // Per-step dispatch scratch, reused across process() calls (cleared and
    // refilled each step). Materializes the iterate_entities() generator into a
    // splittable range TBB can parallelize, and partitions CPU vs GPU entities.
    // Rebuilt every step (not cached) so a future dynamic entity set stays
    // correct for free; the cost is nanoseconds for a handful of entities.
    std::vector<io_entity*> cpu_dispatch_;
#ifdef COAST_WITH_GPU
    std::vector<gpu_io_entity*> gpu_dispatch_;
#endif

    // Persistent partitioner for the per-step CPU dispatch. affinity_partitioner
    // is STATEFUL: it records which worker ran which subrange and replays that
    // mapping on the next step, so an entity (and its data) stays on the same core
    // across steps. This must outlive single process() calls for that map to
    // survive — hence a member, not a temporary. Critically, it stops the outer
    // dispatch from scrambling the thread placement the ngm2/ngm3 inner
    // static_partitioner loops rely on; without it, nesting the inner loops under
    // a fresh outer partition each step regressed the entity-sparse mnist config
    // ~7%. With it, the dispatch is a net win over the old serial loop (mnist
    // +9%, philox +4%) and bit-identical (each entity writes only its own buffer,
    // so chunking/ordering cannot change results).
    tbb::affinity_partitioner cpu_partitioner_;
    // Fans the CPU entities out over the partitioner and reduces their per-entity
    // step_results into one (deterministic: smallest entity_id wins). The happy
    // path does one relaxed atomic reset+load per step and NO allocation — the
    // fault detail is written under cpu_fault_mutex_ on the cold branch only.
    // See design/error_handling_plan.md.
    step_result dispatch_cpu_entities();
    std::atomic<bool>               cpu_fault_{false};
    std::mutex                      cpu_fault_mutex_;
    std::optional<core::step_error> cpu_fault_err_;

    // Hook management. Additions are deferred: set_*_hook() queues the hook and
    // the sim thread applies it at the start of its next process()/swap_io().
    // Removal is SYNCHRONOUS with respect to invocation (#235): see the public
    // remove_*_hook() contract below.
    //
    // Two locks, always taken in this order when both are held:
    //  1. active_hooks_mutex_ guards the active maps. The sim thread holds it
    //     while it applies pending changes and for the whole of each hook phase
    //     (pre_process; post_process; pre_swap; post_swap + barrier as ONE
    //     phase). A removal from another thread takes it, so it waits for a
    //     running phase to end and then erases the hook directly.
    //  2. pending_hooks_mutex_ guards the pending vectors (any thread).
    std::mutex active_hooks_mutex_;
    std::mutex pending_hooks_mutex_;
    // Removals from other threads currently waiting for active_hooks_mutex_.
    std::atomic<int> removals_waiting_{0};
    std::atomic<bool> has_pending_hooks_{false};  // Fast check to avoid mutex in hot path
    std::size_t next_hook_id_{};

    using hook_fn_t  = std::function<void()>;
    using hook_map_t = std::unordered_map<std::size_t, hook_fn_t>;

    struct pending_hook_t {
        std::size_t id;
        hook_fn_t func;
    };
    std::vector<pending_hook_t> pending_pre_process_hooks_;
    std::vector<pending_hook_t> pending_post_process_hooks_;
    std::vector<pending_hook_t> pending_pre_swap_hooks_;
    std::vector<pending_hook_t> pending_post_swap_hooks_;
    std::vector<pending_hook_t> pending_post_swap_barrier_hooks_;
    // Removals requested from inside a hook of this env (the only deferred case).
    std::vector<std::size_t> pending_remove_pre_process_;
    std::vector<std::size_t> pending_remove_post_process_;
    std::vector<std::size_t> pending_remove_pre_swap_;
    std::vector<std::size_t> pending_remove_post_swap_;
    std::vector<std::size_t> pending_remove_post_swap_barrier_;

    // Active hooks (guarded by active_hooks_mutex_)
    hook_map_t pre_process_hooks_;
    hook_map_t post_process_hooks_;
    hook_map_t pre_swap_hooks_;
    hook_map_t post_swap_hooks_;
    // Barrier hooks run after all post_swap_hooks, enabling two-phase signal+wait
    // pattern: all post_swap_hooks signal simultaneously, then all barrier hooks wait.
    // This ensures N visualizations block for only 1 GUI frame instead of N frames.
    // They share the post_swap phase lock: a removal can never fall between a
    // visualization's signal and its wait.
    hook_map_t post_swap_barrier_hooks_;

    // Process pending hook additions/removals (called from sim thread)
    void process_pending_hooks();
    // Run the given maps' hooks in order as one phase (sim thread).
    void run_hook_phase(std::initializer_list<hook_map_t*> maps);
    // Shared body of the remove_*_hook() functions.
    void remove_hook(hook_map_t& active, std::vector<pending_hook_t>& pending_add,
                     std::vector<std::size_t>& pending_remove, std::size_t id);

    // The run's id (core/run_id.h), set by whoever builds the env from a config
    // (coast::sim_instance). Written into every checkpoint (metadata/run_id) by
    // logging::state_logger::extract_state; a load compares it with the checkpoint's.
    std::string run_id_;
    // --resume of a run whose config says "auto": the next load adopts the
    // checkpoint's run_id (build_env).
    bool adopt_checkpoint_run_id_ = false;
    std::optional<std::uint64_t> continued_from_step_;

public:

    void set_run_id(std::string id) { run_id_ = std::move(id); }
    // The checkpoint step this run continues from (the last load continued the run
    // that wrote it), or none: observe writers continue their files from there
    // (observe::observe_run_t). Set by state_logger::apply_state.
    void set_continued_from_step(std::optional<std::uint64_t> step) noexcept { continued_from_step_ = step; }
    [[nodiscard]] std::optional<std::uint64_t> continued_from_step() const noexcept { return continued_from_step_; }
    [[nodiscard]] const std::string& run_id() const noexcept { return run_id_; }
    void adopt_checkpoint_run_id() noexcept { adopt_checkpoint_run_id_ = true; }
    // Adopts `checkpoint_id` if asked to (once); state_logger::apply_state.
    void maybe_adopt_run_id(const std::string& checkpoint_id)
    {
        if (adopt_checkpoint_run_id_ && !checkpoint_id.empty()) run_id_ = checkpoint_id;
        adopt_checkpoint_run_id_ = false;
    }

    // Out-of-line: gpu_dev_ is a unique_ptr to an incomplete type here, so the
    // compiler-generated ctor/dtor (which must be able to destroy the member) can
    // only be emitted where gpu_device is complete (sim_env.cpp).
    env();
    ~env();

#ifdef COAST_WITH_GPU
    // Mark this run as GPU-enabled. The same injection seam build_env uses (S5)
    // when a config contains a GPU entity. Must be called before init_io_buffers()
    // so the topology pre-pass constructs the GPU io_buffer variants and the device
    // is created. Throws if already enabled.
    void enable_gpu();
    [[nodiscard]] bool has_gpu() const noexcept { return gpu_enabled_; }

    // Backend/adapter identity for the run manifest ("<device> [Vulkan|Metal]" or
    // "null"); empty before the device is built or in CPU-only runs. Delegates to
    // gpu_dev_->adapter_name().
    [[nodiscard]] std::string gpu_adapter_name() const;

    // The agnostic GPU device (null until init_io_buffers() creates it, and in
    // CPU-only runs). The per-step path and observers (boundary readback) drive the
    // GPU exclusively through this seam.
    [[nodiscard]] coast::gpu::gpu_device* gpu_dev() noexcept { return gpu_dev_.get(); }
#endif

    template<class T, class... Ps>
    requires std::is_base_of_v<io_entity, T>
    void emplace_back(Ps&&... params)
    {
        auto it = entities.find(std::type_index(typeid(T)));
        if (it == entities.end()) {
            auto [new_it,inserted] = entities.insert( {std::type_index(typeid(T)), std::make_unique<typed_entity_vec<T>>() } );
            it = new_it;
        }
        auto ev = dynamic_cast<typed_entity_vec<T>*>(it->second.get());
        ev->emplace_back(std::forward<Ps>(params)...);
    }

    template<class T>
    requires std::is_base_of_v<io_entity, T>
    std::optional<typed_entity_vec<T>*> get_entities()
    {
        const auto it = entities.find(std::type_index(typeid(T)));
        if (it == entities.end()) {
            return {};
        }
        return dynamic_cast<typed_entity_vec<T>*>(it->second.get());
    }

    void init_io_buffers();

    [[nodiscard]] std::unordered_map<std::size_t,std::unique_ptr<io_buffer>>& get_io_buffers() { return io_buffers; }
    [[nodiscard]] std::optional<io_buffer*> get_io_buffer(std::size_t id)
    {
        auto it = io_buffers.find(id);
        if (it == io_buffers.end())
            return {};
        return it->second.get();
    }

    entity_range iterate_entities();

    io_entity* get_entity(output_id_t output_id);

    // Run one per-step pass over all entities. Returns the first (deterministic)
    // hot-path fault as std::unexpected, or success. Does not throw on entity
    // faults — the controller halts on the returned value.
    step_result process();
    // Like process(): returns the first fault as std::unexpected (a GPU device
    // fault surfaces here as device_fault), or success. Does not throw.
    step_result swap_io();

    // Hook registration - adds to pending queue, returns hook ID immediately
    // Actual registration happens on sim thread at next process()/swap_io() call
    std::size_t set_pre_process_hook(std::function<void()> func);
    std::size_t set_post_process_hook(std::function<void()> func);
    std::size_t set_pre_swap_hook(std::function<void()> func);
    std::size_t set_post_swap_hook(std::function<void()> func);
    std::size_t set_post_swap_barrier_hook(std::function<void()> func);

    // Hook removal (#235). Called from any thread other than one running a hook
    // of this env, it is synchronous: when it returns the hook is not running
    // and will never run again (whether it was active or still pending), so the
    // caller may free whatever the hook captures. It blocks for at most the
    // current hook phase on the sim thread, and not at all if the sim is not
    // stepping (paused, halted, between steps, controller gone). A hook that
    // waits on the remover's thread must give up when hook_removal_waiting()
    // is set; one that waited without a timeout would deadlock the remover.
    // Called from inside a hook of this env (sim thread), it cannot wait for
    // itself: the removal takes effect at the next process()/swap_io() boundary,
    // and the hook may still run once more in the current phase.
    void remove_pre_process_hook(std::size_t id);
    void remove_post_process_hook(std::size_t id);
    void remove_pre_swap_hook(std::size_t id);
    void remove_post_swap_hook(std::size_t id);
    void remove_post_swap_barrier_hook(std::size_t id);

    // True while a remove_*_hook() call from another thread is waiting for the
    // running hook phase. A hook that waits for another thread (the
    // visualizations' barrier hooks wait for the GUI) must stop waiting when
    // this is set: the remover may be the very thread it waits for, which would
    // otherwise stall the remover for every such wait's full timeout.
    [[nodiscard]] bool hook_removal_waiting() const noexcept {
        return removals_waiting_.load(std::memory_order_acquire) > 0;
    }

};

} // sim

#endif //SIM_ENV_H
