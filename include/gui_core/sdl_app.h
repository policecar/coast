#ifndef COAST_SDL_APP_H
#define COAST_SDL_APP_H

#include <map>
#include <functional>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations to avoid heavy SDL includes in header
struct SDL_Window;
using SDL_GLContext = void*;

namespace coast {

// Application-provided paths for the SDL/ImGui app. Kept as plain resolved
// strings so sdl_app stays free of the app-config type: run_gui resolves them
// from the app config (font falls back to the compile-time COAST_FONT_PATH,
// imgui.ini to ImGui's default) and passes the results here.
struct sdl_app_paths {
    std::string font;       // TTF to load; empty -> ImGui's built-in default font
    std::string imgui_ini;  // imgui.ini location; empty -> ImGui's default "imgui.ini"
};

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
    explicit sdl_app(const char* title = "COAST", sdl_app_paths paths = {});
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

    // Single-shot callbacks: run ONCE in the state phase (outside the ImGui frame),
    // then cleared. Thread-safe to register from any thread. For deferred work that
    // must run on the GUI thread with the GL context current but OUTSIDE a frame —
    // e.g. freeing a GL resource whose owner was dropped on another thread.
    void register_single_shot_func(std::function<void()> func);

private:
    SDL_Window* window = nullptr;
    SDL_GLContext gl_context = nullptr;

    // Resolved font / imgui.ini paths (see sdl_app_paths). Stored because
    // ImGui's io.IniFilename holds a borrowed const char* that must outlive the
    // ImGui context — imgui_ini's std::string owns that storage.
    sdl_app_paths paths_;

    std::map<std::size_t, std::function<void()>> state_fs;
    std::map<std::size_t, std::function<void()>> gui_fs;
    std::map<std::size_t, std::function<void()>> shutdown_fs;

    // Run-once callbacks, guarded because they are registered from other threads.
    std::mutex                         single_shot_mutex_;
    std::vector<std::function<void()>> single_shot_fs;

    std::size_t next_free_id = 1;
    bool running = false;

    void init_sdl();
    void init_imgui();
    void cleanup_imgui();
    void cleanup_sdl();
    void run_single_shots(); // drain single_shot_fs (GUI thread, state phase)
};

} // namespace coast

#endif // COAST_SDL_APP_H
