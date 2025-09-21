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

class env {

    std::unordered_map<std::type_index,std::unique_ptr<entity_vec>> entities;
    std::unordered_map<std::size_t,io_buffer> io_buffers;

    std::size_t next_hook_id {};
    std::unordered_map<std::size_t,std::function<void()>> pre_process_hooks;
    std::unordered_map<std::size_t,std::function<void()>> post_process_hooks;
    std::unordered_map<std::size_t,std::function<void()>> pre_swap_hooks;
    std::unordered_map<std::size_t,std::function<void()>> post_swap_hooks;

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
        typed_entity_vec<T> *ev = dynamic_cast<typed_entity_vec<T>*>(it->second.get());
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

    void process();
    void swap_io();

    std::size_t set_pre_process_hook(std::function<void()> func);
    std::size_t set_post_process_hook(std::function<void()> func);
    std::size_t set_pre_swap_hook(std::function<void()> func);
    std::size_t set_post_swap_hook(std::function<void()> func);

    void remove_pre_process_hook(std::size_t id);
    void remove_post_process_hook(std::size_t id);
    void remove_pre_swap_hook(std::size_t id);
    void remove_post_swap_hook(std::size_t id);

};

} // sim

#endif //SIM_ENV_H
