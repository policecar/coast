#include "gui_core/sdl_app.h"

#include <SDL.h>
#include <SDL_opengl.h>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include "diag/diag.h"

#include <stdexcept>
#include <ranges>
#include <cstdlib>
#include <cstdio>
#include <filesystem>

namespace coast {

// Get display scale factor for HiDPI support
// On Wayland with fractional scaling, SDL2 doesn't report DPI correctly,
// so we read Xft.dpi from X resources (set by KDE/GNOME for compatibility)
static float get_display_scale()
{
#ifdef __linux__
    const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
    if (wayland_display) {
        // Try to get DPI from Xft.dpi (X resources, set by desktop environments)
        FILE* pipe = popen("xrdb -query 2>/dev/null | grep 'Xft.dpi' | cut -f2", "r");
        if (pipe) {
            char buffer[32];
            if (fgets(buffer, sizeof(buffer), pipe)) {
                float dpi = std::atof(buffer);
                pclose(pipe);
                if (dpi > 0) {
                    return dpi / 96.0f;
                }
            } else {
                pclose(pipe);
            }
        }

        // Fallback: check QT_SCALE_FACTOR environment variable
        const char* qt_scale = std::getenv("QT_SCALE_FACTOR");
        if (qt_scale) {
            float scale = std::atof(qt_scale);
            if (scale > 0) {
                return scale;
            }
        }
    }
#endif
    return 1.0f;
}

sdl_app::sdl_app(const char* title, sdl_app_paths paths)
    : paths_(std::move(paths))
{
    init_sdl();

    // Create window with OpenGL context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
#ifdef __APPLE__
    // macOS requires OpenGL 3.2+ for core profile
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif

    SDL_WindowFlags window_flags = static_cast<SDL_WindowFlags>(
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI
    );

    window = SDL_CreateWindow(
        title,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        1280, 800,
        window_flags
    );

    if (window == nullptr) {
        throw std::runtime_error("Failed to create SDL window: " + std::string(SDL_GetError()));
    }

    gl_context = SDL_GL_CreateContext(window);
    if (gl_context == nullptr) {
        SDL_DestroyWindow(window);
        throw std::runtime_error("Failed to create OpenGL context: " + std::string(SDL_GetError()));
    }

    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1);  // Enable vsync

    SDL_MaximizeWindow(window);

    init_imgui();
}

sdl_app::~sdl_app()
{
    cleanup_imgui();
    cleanup_sdl();
}

void sdl_app::init_sdl()
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        throw std::runtime_error("Failed to initialize SDL: " + std::string(SDL_GetError()));
    }

#ifdef SDL_HINT_IME_SHOW_UI
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif
}

void sdl_app::init_imgui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Persisted UI layout file. io.IniFilename borrows this const char*, so it
    // must point at storage that outlives the ImGui context — paths_.imgui_ini
    // (an sdl_app member) provides it. Empty keeps ImGui's default ("imgui.ini").
    if (!paths_.imgui_ini.empty()) {
        io.IniFilename = paths_.imgui_ini.c_str();
    }

    // Get display scale factor for HiDPI support (especially for Wayland fractional scaling)
    float scale = get_display_scale();

    // Setup style
    ImGui::StyleColorsDark();

    // Load the custom font with a DPI-aware size. paths_.font is an absolute path
    // (the build-copied COAST_FONT_PATH, or an app-config override), so the GUI
    // finds its font from any CWD. If it is empty or missing, fall back to ImGui's
    // built-in default font rather than asserting inside AddFontFromFileTTF.
    float base_font_size = 16.0f;
    if (!paths_.font.empty() && std::filesystem::exists(paths_.font)) {
        io.Fonts->AddFontFromFileTTF(paths_.font.c_str(), base_font_size * scale);
    } else {
        if (!paths_.font.empty()) {
            COAST_WARN("gui", "GUI font '{}' not found; using the built-in default.",
                       paths_.font);
        }
        io.Fonts->AddFontDefault();
    }

    // Setup platform/renderer backends
#ifdef __APPLE__
    const char* glsl_version = "#version 150";  // macOS requires OpenGL 3.2 core profile
#else
    const char* glsl_version = "#version 130";
#endif
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

void sdl_app::cleanup_imgui()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

void sdl_app::cleanup_sdl()
{
    if (gl_context) {
        SDL_GL_DeleteContext(gl_context);
        gl_context = nullptr;
    }
    if (window) {
        SDL_DestroyWindow(window);
        window = nullptr;
    }
    SDL_Quit();
}

void sdl_app::run()
{
    running = true;

    while (running) {
        // Event handling
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);

            if (event.type == SDL_QUIT) {
                running = false;
            }
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window)) {
                running = false;
            }
        }

        // Skip rendering when minimized
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
            SDL_Delay(10);
            continue;
        }

        // State update phase — outside the ImGui frame. Single-shot deferrals
        // (e.g. GL frees posted from other threads) run here, before the frame.
        run_single_shots();
        for (auto& [id, func] : state_fs) {
            func();
        }

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // GUI callbacks
        for (auto& [id, func] : gui_fs) {
            func();
        }

        // Render
        ImGui::Render();
        ImGuiIO& io = ImGui::GetIO();
        glViewport(0, 0, static_cast<int>(io.DisplaySize.x), static_cast<int>(io.DisplaySize.y));
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window);
    }

    // Shutdown callbacks
    for (auto& [id, func] : shutdown_fs) {
        func();
    }
    // Drain any single-shots posted during the final frame while the GL context is
    // still current (it is destroyed later in ~sdl_app).
    run_single_shots();
}

void sdl_app::quit()
{
    running = false;
}

std::size_t sdl_app::register_state_func(const std::function<void()>& func)
{
    std::size_t id = next_free_id++;
    state_fs[id] = func;
    return id;
}

void sdl_app::deregister_state_func(std::size_t id)
{
    state_fs.erase(id);
}

void sdl_app::register_single_shot_func(std::function<void()> func)
{
    if (!func) return;
    std::scoped_lock lk(single_shot_mutex_);
    single_shot_fs.push_back(std::move(func));
}

void sdl_app::run_single_shots()
{
    // Swap out under the lock, then run unlocked — a callback may register another
    // single-shot (it lands in the now-empty list and runs next drain).
    std::vector<std::function<void()>> pending;
    {
        std::scoped_lock lk(single_shot_mutex_);
        pending.swap(single_shot_fs);
    }
    for (auto& fn : pending) fn();
}

std::size_t sdl_app::register_gui_func(const std::function<void()>& func)
{
    std::size_t id = next_free_id++;
    gui_fs[id] = func;
    return id;
}

void sdl_app::deregister_gui_func(std::size_t id)
{
    gui_fs.erase(id);
}

std::size_t sdl_app::register_shutdown_func(const std::function<void()>& func)
{
    std::size_t id = next_free_id++;
    shutdown_fs[id] = func;
    return id;
}

void sdl_app::deregister_shutdown_func(std::size_t id)
{
    shutdown_fs.erase(id);
}

} // namespace coast
