//
// Created by jk on 04.09.25.
//

#include <ranges>
#include <cstdio>

#include "sim_env.h"

namespace sim {
void env::init_io_buffers()
{
    // construct buffers and set output functions
    for (auto &io_ent : iterate_entities()) {
        const std::size_t id = io_ent.get_outp_id();
        auto [buf_it,success] = io_buffers.emplace(id, io_buffer(io_ent.get_outp_size()));
        if (success == false) {
            std::fprintf(stderr,"duplicate io_entity ID!\n");
            std::terminate();
        }
        io_ent.set_outp_func(buf_it->second.outp_buffer_func());
    }
    // set input funcs
    for (auto &io_ent : iterate_entities()) {
        const auto inp_ids = io_ent.get_inp_ids();
        for (auto inp_id : inp_ids) {
            auto it = io_buffers.find(inp_id);
            if (it == io_buffers.end()) {
                std::fprintf(stderr,"missing io_entity ID!\n");
                std::terminate();
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

void env::process()
{
    for (auto &pre_proc : pre_process_hooks | std::views::values ) {
        pre_proc();
    }

    for (auto &io_ent : iterate_entities()) {
        io_ent.process();
    }

    for (auto &post_proc : post_process_hooks | std::views::values ) {
        post_proc();
    }
}

void env::swap_io()
{
    for (auto &pre_proc : pre_swap_hooks | std::views::values) {
        pre_proc();
    }

    for (auto &buf : io_buffers | std::views::values ) {
        buf.swap_buffer();
    }

    for (auto &post_proc : post_swap_hooks | std::views::values) {
        post_proc();
    }
}

std::size_t env::set_pre_process_hook(std::function<void()> func)
{
    pre_process_hooks.emplace(next_hook_id, func);
    return next_hook_id++;
}

std::size_t env::set_post_process_hook(std::function<void()> func)
{
    post_process_hooks.emplace(next_hook_id, func);
    return next_hook_id++;
}

std::size_t env::set_pre_swap_hook(std::function<void()> func)
{
    pre_swap_hooks.emplace(next_hook_id, func);
    return next_hook_id++;
}

std::size_t env::set_post_swap_hook(std::function<void()> func)
{
    post_swap_hooks.emplace(next_hook_id, func);
    return next_hook_id++;
}

void env::remove_pre_process_hook(std::size_t id)
{
    pre_process_hooks.erase(id);
}

void env::remove_post_process_hook(std::size_t id)
{
    post_process_hooks.erase(id);
}

void env::remove_pre_swap_hook(std::size_t id)
{
    pre_swap_hooks.erase(id);
}

void env::remove_post_swap_hook(std::size_t id)
{
    post_swap_hooks.erase(id);
}

} // sim