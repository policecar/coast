#include "app/run_gui.h"

#include <memory>
#include <string>

#include "cli/options.h"
#include "core/app_config.h"
#include "diag/diag.h"
#include "gui_core/sdl_app.h"
#include "gui_core/gui_manager.h"
#include "gui_plot/console_plot.h"
#include "gui_plot/plot_manager.h"
#include "gui_plot/plot_tools.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_thread_controller.h"
#include "hdf5/executor.h"

namespace coast::app {

int run_gui(std::unique_ptr<sim_instance>  sim,
            const cli::options&            opts,
            const paths::app_config&              app_cfg,
            hdf5::executor&                h5_exec,
            std::unique_ptr<console_iface> console) {
    // Resolve the GUI's asset/state paths from the app config (app layer keeps the
    // policy; sdl_app takes plain strings). Font default is the build-copied,
    // absolute COAST_FONT_PATH so the GUI finds it from any CWD; an app-config
    // assets.font overrides it. imgui.ini defaults to app_cfg.state.imgui_ini.
    sdl_app_paths gui_paths;
    gui_paths.font = app_cfg.assets.font.empty() ? std::string(COAST_FONT_PATH)
                                                 : app_cfg.assets.font;
    gui_paths.imgui_ini = app_cfg.state.imgui_ini;

    sdl_app     main_app("COAST - Neural Simulation", std::move(gui_paths));

    // Point plot-theme persistence at the configured file before any theme is
    // accessed (themes load lazily and need the ImPlot context sdl_app just
    // created). Default unchanged unless app_cfg.state.plot_themes overrides it.
    coast::plot::set_theme_file(app_cfg.state.plot_themes);
    // Default directory the save-PNG / save-HDF5 dialogs seed into (CWD unless
    // app_cfg.output.plot_dir overrides it).
    coast::plot::set_export_dir(app_cfg.output.plot_dir);

    gui_manager gui(main_app, app_cfg, h5_exec);

    // Wire the console plot bridge's HDF5 access (used by coast::plot::load).
    coast::plot::set_executor(h5_exec);
    // Wire the GUI host so the plot bridge runs deferred GUI-thread work (freeing an
    // image texture whose owner was dropped on the console thread) via gui_manager's
    // single-shot registry — in the state phase, outside the ImGui frame.
    coast::plot::plot_manager::instance().set_gui_host(gui);

    if (console) gui.set_console(std::move(console));

    if (sim) {
        gui.set_sim_instance(sim.get());
        COAST_INFO("gui", "loaded configuration: {}", opts.config_path);

        if (opts.auto_start) {
            if (auto* controller = sim->get_thread_controller()) {
                controller->set_mode(sim::sim_mode::throttled);
                COAST_INFO("gui", "simulation started");
            }
        }
    } else if (opts.config_path.empty()) {
        COAST_INFO("gui", "no configuration file specified; "
                          "use System > Open Configuration to load one");
    }

    // Handle user-driven configuration loads from the GUI menu.
    main_app.register_state_func(
        [&] {
            if (gui.should_start_loading()) {
                std::string path = gui.get_pending_load_path();
                try {
                    gui.set_sim_instance(nullptr);
                    sim.reset();

                    sim = std::make_unique<sim_instance>(path, h5_exec, app_cfg);
                    gui.set_sim_instance(sim.get());
                    gui.clear_pending_load();
                    COAST_INFO("gui", "loaded configuration: {}", path);
                } catch (const std::exception& e) {
                    gui.clear_pending_load();
                    gui.set_load_error(e.what());
                    COAST_ERROR("gui", "failed to load configuration: {}", e.what());
                }
            }
        }
    );

    // Service queued console-plot PNG exports. Runs in the state phase, i.e.
    // OUTSIDE the main ImGui frame, because each export runs its own off-screen
    // ImGui render pass (ImGui has a single global frame).
    main_app.register_state_func([&] { coast::plot::flush_captures(h5_exec); });

    main_app.register_gui_func(
        [&] {
            gui.render_menu_bar();
            gui.render_dialogs();
            gui.render_status_bar();
        }
    );

    main_app.register_shutdown_func(
        [&]() {
            gui.set_sim_instance(nullptr);
            gui.free_resources();   // free persistent tool GPU textures (context still current)
            sim.reset();
        }
    );

    main_app.run();

    // Tear the console down FIRST: destroying it joins the interpreter worker
    // thread, so no macro can still be mid-flight creating figures or freeing image
    // textures when the plot singleton is destroyed just below. Without this, `gui`
    // (which owns the console) outlives deinit() — it is a local destroyed only at
    // function-scope exit — so the "console can no longer create figures" invariant
    // the deinit() below relies on would be unenforced.
    gui.shutdown_console();

    // Destroy the plot subsystem singleton HERE — the GL context is still current
    // (torn down later in ~sdl_app), the console thread is joined (above), and it
    // happens before `gui`/`main_app` (which it references via set_gui_host) go out
    // of scope. Avoids relying on unspecified static-teardown ordering. deinit()
    // tombstones the singleton, so any stray post-shutdown access throws rather than
    // resurrecting a GUI-host-less manager.
    coast::plot::plot_manager::deinit();
    return 0;
}

} // namespace coast::app
