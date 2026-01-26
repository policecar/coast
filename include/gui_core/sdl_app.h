#ifndef COAST_SDL_APP_H
#define COAST_SDL_APP_H

#include <map>
#include <functional>
#include <cstddef>

// Forward declarations to avoid heavy SDL includes in header
struct SDL_Window;
using SDL_GLContext = void*;

namespace coast {

/**
 * SDL2 + OpenGL3 application framework with Dear ImGui integration.
 *
 * Provides a callback-based architecture for building GUI applications:
 * - state_fs: State update callbacks (simulation, data processing)
 * - gui_fs: ImGui rendering callbacks (all visualization happens here)
 * - shutdown_fs: Cleanup callbacks
 *
 * Note: Unlike ray_app, there is no separate draw_fs since all rendering
 * is done through ImGui windows.
 */
class sdl_app {
public:
    explicit sdl_app(const char* title = "COAST");
    ~sdl_app();

    // Non-copyable, non-movable
    sdl_app(const sdl_app&) = delete;
    sdl_app(sdl_app&&) = delete;
    sdl_app& operator=(const sdl_app&) = delete;
    sdl_app& operator=(sdl_app&&) = delete;

    /**
     * Run the main application loop.
     * Blocks until the window is closed.
     */
    void run();

    /**
     * Request the application to stop.
     */
    void quit();

    // State update callbacks (called before ImGui frame)
    std::size_t register_state_func(const std::function<void()>& func);
    void deregister_state_func(std::size_t id);

    // ImGui rendering callbacks
    std::size_t register_gui_func(const std::function<void()>& func);
    void deregister_gui_func(std::size_t id);

    // Shutdown callbacks (called after main loop ends)
    std::size_t register_shutdown_func(const std::function<void()>& func);
    void deregister_shutdown_func(std::size_t id);

private:
    SDL_Window* window = nullptr;
    SDL_GLContext gl_context = nullptr;

    std::map<std::size_t, std::function<void()>> state_fs;
    std::map<std::size_t, std::function<void()>> gui_fs;
    std::map<std::size_t, std::function<void()>> shutdown_fs;

    std::size_t next_free_id = 1;
    bool running = false;

    void init_sdl();
    void init_imgui();
    void cleanup_imgui();
    void cleanup_sdl();
};

} // namespace coast

#endif // COAST_SDL_APP_H
