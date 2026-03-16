//
// Created by jk on 04.09.25.
//

#ifndef SIM_ENV_H
#define SIM_ENV_H

#include <unordered_map>
#include <vector>
#include <typeindex>
#include <utility>
#include <generator>
#include <memory>
#include <atomic>
#include <mutex>

#include "io_entity.h"
#include "io_buffer.h"

namespace sim {

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

/**
 * Simulation environment that manages entities and I/O buffers.
 *
 * Future consideration: If the simulation structure (entities) can change
 * during runtime, add a callback/observer system to notify interested
 * components (e.g., GUI entity browsers) about structural changes.
 * This would allow them to invalidate caches and refresh their views.
 */
class env {

    std::unordered_map<std::type_index,std::unique_ptr<entity_vec>> entities;
    std::unordered_map<std::size_t,io_buffer> io_buffers;

    // Hook management - deferred registration pattern
    // GUI thread adds to pending vectors (mutex protected)
    // Sim thread processes pending vectors at start of process()/swap_io() only when non-empty
    std::mutex pending_hooks_mutex_;
    std::atomic<bool> has_pending_hooks_{false};  // Fast check to avoid mutex in hot path
    std::size_t next_hook_id_{};

    struct pending_hook_t {
        std::size_t id;
        std::function<void()> func;
    };
    std::vector<pending_hook_t> pending_pre_process_hooks_;
    std::vector<pending_hook_t> pending_post_process_hooks_;
    std::vector<pending_hook_t> pending_pre_swap_hooks_;
    std::vector<pending_hook_t> pending_post_swap_hooks_;
    std::vector<pending_hook_t> pending_post_swap_barrier_hooks_;
    std::vector<std::size_t> pending_remove_pre_process_;
    std::vector<std::size_t> pending_remove_post_process_;
    std::vector<std::size_t> pending_remove_pre_swap_;
    std::vector<std::size_t> pending_remove_post_swap_;
    std::vector<std::size_t> pending_remove_post_swap_barrier_;

    // Active hooks (only modified by sim thread)
    std::unordered_map<std::size_t,std::function<void()>> pre_process_hooks_;
    std::unordered_map<std::size_t,std::function<void()>> post_process_hooks_;
    std::unordered_map<std::size_t,std::function<void()>> pre_swap_hooks_;
    std::unordered_map<std::size_t,std::function<void()>> post_swap_hooks_;
    // Barrier hooks run after all post_swap_hooks, enabling two-phase signal+wait
    // pattern: all post_swap_hooks signal simultaneously, then all barrier hooks wait.
    // This ensures N visualizations block for only 1 GUI frame instead of N frames.
    std::unordered_map<std::size_t,std::function<void()>> post_swap_barrier_hooks_;

    // Process pending hook additions/removals (called from sim thread)
    void process_pending_hooks();

public:

    template<class T, class... Ps>
    requires std::is_base_of_v<io_entity, T>
    void emplace_back(Ps... params)
    {
        auto it = entities.find(std::type_index(typeid(T)));
        if (it == entities.end()) {
            auto [new_it,inserted] = entities.insert( {std::type_index(typeid(T)), std::make_unique<typed_entity_vec<T>>() } );
            it = new_it;
        }
        auto ev = dynamic_cast<typed_entity_vec<T>*>(it->second.get());
        ev->emplace_back(params...);
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

    [[nodiscard]] std::unordered_map<std::size_t,io_buffer>& get_io_buffers() { return io_buffers; }
    [[nodiscard]] std::optional<io_buffer*> get_io_buffer(std::size_t id)
    {
        auto it = io_buffers.find(id);
        if (it == io_buffers.end())
            return {};
        return &it->second;
    }

    std::generator<io_entity&> iterate_entities();

    io_entity* get_entity(output_id_t output_id);

    void process();
    void swap_io();

    // Hook registration - adds to pending queue, returns hook ID immediately
    // Actual registration happens on sim thread at next process()/swap_io() call
    std::size_t set_pre_process_hook(std::function<void()> func);
    std::size_t set_post_process_hook(std::function<void()> func);
    std::size_t set_pre_swap_hook(std::function<void()> func);
    std::size_t set_post_swap_hook(std::function<void()> func);
    std::size_t set_post_swap_barrier_hook(std::function<void()> func);

    // Hook removal - adds to pending removal queue
    // Actual removal happens on sim thread at next process()/swap_io() call
    void remove_pre_process_hook(std::size_t id);
    void remove_post_process_hook(std::size_t id);
    void remove_pre_swap_hook(std::size_t id);
    void remove_post_swap_hook(std::size_t id);
    void remove_post_swap_barrier_hook(std::size_t id);

};

} // sim

#endif //SIM_ENV_H
