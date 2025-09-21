//
// Created by jk on 31.07.25.
//
#include <ranges>
#include <raylib.h>

#include "rlImGui.h"

#include "ray_app.h"


namespace coast {

ray_app::ray_app() :
    state_fs(),
    draw_fs(),
    gui_fs(),
    next_free_id(1)
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT /*| FLAG_WINDOW_HIGHDPI*/);
    InitWindow(800, 600, "");
    SetTargetFPS(60);
    MaximizeWindow();
    rlImGuiSetup(false);
}

ray_app::~ray_app()
{
    rlImGuiShutdown();
    CloseWindow();
}

void ray_app::run()
{
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        for (auto &val: state_fs | std::views::values)
            val();

        BeginDrawing();

        for (auto &val: draw_fs | std::views::values)
            val();

        rlImGuiBegin();
        for (auto &val: gui_fs | std::views::values)
            val();
        rlImGuiEnd();

        EndDrawing();
    }
    for (auto &val: shutdown_fs | std::views::values)
        val();
}

std::size_t ray_app::register_draw_func(const std::function<void()> &func)
{
    const std::size_t id = next_free_id++;
    draw_fs[id] = func;
    return id;
}

void ray_app::deregister_draw_func(std::size_t id)
{
    draw_fs.erase(id);
}

std::size_t ray_app::register_state_func(const std::function<void()> &func)
{
    const std::size_t id = next_free_id++;
    state_fs[id] = func;
    return id;
}

void ray_app::deregister_state_func(std::size_t id)
{
    state_fs.erase(id);
}

std::size_t ray_app::register_gui_func(const std::function<void()> &func)
{
    const std::size_t id = next_free_id++;
    gui_fs[id] = func;
    return id;
}

void ray_app::deregister_gui_func(std::size_t id)
{
    gui_fs.erase(id);
}

std::size_t ray_app::register_shutdown_func(const std::function<void()> &func)
{
    const std::size_t id = next_free_id++;
    shutdown_fs[id] = func;
    return id;
}

void ray_app::deregister_shutdown_func(std::size_t id)
{
    shutdown_fs.erase(id);
}
} // ngm