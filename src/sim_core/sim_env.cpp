//
// Created by jk on 04.09.25.
//

#include <ranges>
#include <stdexcept>

#include "sim_env.h"

namespace sim {

void env::init_io_buffers()
{
    // construct buffers and set output functions
    for (auto &io_ent : iterate_entities()) {
        const std::size_t id = io_ent.get_outp_id();
        auto [buf_it,success] = io_buffers.emplace(id, io_buffer(io_ent.get_outp_size()));
        if (success == false) {
            throw std::runtime_error("Duplicate io_entity ID: " + std::to_string(id));
        }
        io_ent.set_outp_func(buf_it->second.outp_buffer_func());
    }
    // set input funcs
    for (auto &io_ent : iterate_entities()) {
        const auto inp_ids = io_ent.get_inp_ids();
        for (auto inp_id : inp_ids) {
            auto it = io_buffers.find(inp_id);
            if (it == io_buffers.end()) {
                throw std::runtime_error("Missing io_entity ID: " + std::to_string(inp_id));
            }
            io_ent.set_inp_func(inp_id, it->second.inp_buffer_func() );
        }
    }
}

std::generator<io_entity &> env::iterate_entities()
{
    for (auto &ev : entities | std::views::values ) {
        std::size_t ev_size = ev->get_size();
        for (std::size_t i = 0; i < ev_size; ++i)
            co_yield ev->as_base(i);
    }
}

io_entity* env::get_entity(output_id_t output_id)
{
    for (auto &ev : entities | std::views::values ) {
        std::size_t ev_size = ev->get_size();
        for (std::size_t i = 0; i < ev_size; ++i) {
            io_entity &cur_entity = ev->as_base(i);
            if (cur_entity.get_outp_id() == output_id)
                return &cur_entity;
        }
    }
    return nullptr;
}

void env::process_pending_hooks()
{
    // Fast path: no pending work
    if (!has_pending_hooks_.load(std::memory_order_acquire)) {
        return;
    }

    std::lock_guard lock(pending_hooks_mutex_);

    // Process additions
    for (auto& [id, func] : pending_pre_process_hooks_) {
        pre_process_hooks_.emplace(id, std::move(func));
    }
    pending_pre_process_hooks_.clear();

    for (auto& [id, func] : pending_post_process_hooks_) {
        post_process_hooks_.emplace(id, std::move(func));
    }
    pending_post_process_hooks_.clear();

    for (auto& [id, func] : pending_pre_swap_hooks_) {
        pre_swap_hooks_.emplace(id, std::move(func));
    }
    pending_pre_swap_hooks_.clear();

    for (auto& [id, func] : pending_post_swap_hooks_) {
        post_swap_hooks_.emplace(id, std::move(func));
    }
    pending_post_swap_hooks_.clear();

    // Process removals
    for (auto id : pending_remove_pre_process_) {
        pre_process_hooks_.erase(id);
    }
    pending_remove_pre_process_.clear();

    for (auto id : pending_remove_post_process_) {
        post_process_hooks_.erase(id);
    }
    pending_remove_post_process_.clear();

    for (auto id : pending_remove_pre_swap_) {
        pre_swap_hooks_.erase(id);
    }
    pending_remove_pre_swap_.clear();

    for (auto id : pending_remove_post_swap_) {
        post_swap_hooks_.erase(id);
    }
    pending_remove_post_swap_.clear();

    has_pending_hooks_.store(false, std::memory_order_release);
}

void env::process()
{
    // Process any pending hook changes before iterating
    process_pending_hooks();

    for (auto &[id, func] : pre_process_hooks_) {
        func();
    }

    for (auto &io_ent : iterate_entities()) {
        io_ent.process();
    }

    for (auto &[id, func] : post_process_hooks_) {
        func();
    }
}

void env::swap_io()
{
    // Process any pending hook changes before iterating
    process_pending_hooks();

    for (auto &[id, func] : pre_swap_hooks_) {
        func();
    }

    for (auto &buf : io_buffers | std::views::values ) {
        buf.swap_buffer();
    }

    for (auto &[id, func] : post_swap_hooks_) {
        func();
    }
}

std::size_t env::set_pre_process_hook(std::function<void()> func)
{
    std::lock_guard lock(pending_hooks_mutex_);
    std::size_t id = next_hook_id_++;
    pending_pre_process_hooks_.push_back({id, std::move(func)});
    has_pending_hooks_.store(true, std::memory_order_release);
    return id;
}

std::size_t env::set_post_process_hook(std::function<void()> func)
{
    std::lock_guard lock(pending_hooks_mutex_);
    std::size_t id = next_hook_id_++;
    pending_post_process_hooks_.push_back({id, std::move(func)});
    has_pending_hooks_.store(true, std::memory_order_release);
    return id;
}

std::size_t env::set_pre_swap_hook(std::function<void()> func)
{
    std::lock_guard lock(pending_hooks_mutex_);
    std::size_t id = next_hook_id_++;
    pending_pre_swap_hooks_.push_back({id, std::move(func)});
    has_pending_hooks_.store(true, std::memory_order_release);
    return id;
}

std::size_t env::set_post_swap_hook(std::function<void()> func)
{
    std::lock_guard lock(pending_hooks_mutex_);
    std::size_t id = next_hook_id_++;
    pending_post_swap_hooks_.push_back({id, std::move(func)});
    has_pending_hooks_.store(true, std::memory_order_release);
    return id;
}

void env::remove_pre_process_hook(const std::size_t id)
{
    std::lock_guard lock(pending_hooks_mutex_);
    pending_remove_pre_process_.push_back(id);
    has_pending_hooks_.store(true, std::memory_order_release);
}

void env::remove_post_process_hook(const std::size_t id)
{
    std::lock_guard lock(pending_hooks_mutex_);
    pending_remove_post_process_.push_back(id);
    has_pending_hooks_.store(true, std::memory_order_release);
}

void env::remove_pre_swap_hook(const std::size_t id)
{
    std::lock_guard lock(pending_hooks_mutex_);
    pending_remove_pre_swap_.push_back(id);
    has_pending_hooks_.store(true, std::memory_order_release);
}

void env::remove_post_swap_hook(const std::size_t id)
{
    std::lock_guard lock(pending_hooks_mutex_);
    pending_remove_post_swap_.push_back(id);
    has_pending_hooks_.store(true, std::memory_order_release);
}

} // sim
