//
// Created by jk on 31.07.25.
//

#ifndef RAYAPP_H
#define RAYAPP_H

#include <map>
#include <functional>

namespace coast {

class ray_app {

    std::map<std::size_t,std::function<void()>> state_fs;
    std::map<std::size_t,std::function<void()>> draw_fs;
    std::map<std::size_t,std::function<void()>> gui_fs;
    std::map<std::size_t,std::function<void()>> shutdown_fs;

    std::size_t next_free_id;

public:
    ray_app();
    ~ray_app();

    ray_app(const ray_app &other) = delete;
    ray_app(ray_app &&other) = delete;

    ray_app& operator=(const ray_app &other) = delete;
    ray_app& operator=(ray_app &&other) = delete;

    void run();

    std::size_t register_draw_func(const std::function<void()> &func);
    void deregister_draw_func(std::size_t id);

    std::size_t register_state_func(const std::function<void()> &func);
    void deregister_state_func(std::size_t id);

    std::size_t register_gui_func(const std::function<void()> &func);
    void deregister_gui_func(std::size_t id);

    std::size_t register_shutdown_func(const std::function<void()> &func);
    void deregister_shutdown_func(std::size_t id);

};

} // ngm

#endif //RAYAPP_H
