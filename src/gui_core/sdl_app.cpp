#include "sdl_app.h"

#include <SDL.h>
#include <SDL_opengl.h>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "imnodes.h"

#include <stdexcept>
#include <ranges>

namespace coast {

sdl_app::sdl_app(const char* title)
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
    ImNodes::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup style
    ImGui::StyleColorsDark();

    // Load custom font
    io.Fonts->AddFontFromFileTTF("./3rd_party/DejaVuSans.ttf", 16.0f);

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
    ImNodes::DestroyContext();
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

        // State update phase
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
