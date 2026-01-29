#include <iostream>
#include <memory>

#include "sdl_app.h"
#include "gui_manager.h"
#include "sim_instance.h"

using namespace coast;

int main(int argc, char **argv)
{
    /*
     * This object represents the main GUI-application that we use for visualization.
     * It provides a number of hooks to which we can register functions that are
     * called at the respective parts of the GUI-loop.
     */
    sdl_app main_app("COAST - Neural Simulation");

    /*
     * Set up GUI manager for menu bar and dialogs.
     * Initially starts without a simulation loaded.
     */
    gui_manager gui(main_app);

    /*
     * Simulation instance managed via unique_ptr for proper RAII lifecycle.
     * Can be nullptr when no configuration is loaded.
     */
    std::unique_ptr<sim_instance> sim;

    /*
     * Optional: load configuration from command line argument.
     */
    if (argc >= 2) {
        try {
            sim = std::make_unique<sim_instance>(argv[1]);
            gui.set_sim_instance(sim.get());
            std::cout << "Loaded configuration: " << argv[1] << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to load configuration: " << e.what() << "\n";
            std::cerr << "Starting without simulation. Use System > Open Configuration to load.\n";
        }
    } else {
        std::cout << "No configuration file specified.\n";
        std::cout << "Use System > Open Configuration to load a configuration.\n";
    }

    /*
     * Registering a state update function to be called during the state-update-phase
     * of the GUI-loop. Handles loading new configurations.
     * Note: Simulation stepping now runs in a dedicated thread managed by sim_thread_controller.
     */
    main_app.register_state_func(
        [&] {
            // Handle pending configuration load (after loading dialog has been shown)
            if (gui.should_start_loading()) {
                std::string path = gui.get_pending_load_path();
                try {
                    // Clear old simulation first (this stops the simulation thread)
                    gui.set_sim_instance(nullptr);
                    sim.reset();

                    // Load new simulation (this starts a new simulation thread in paused mode)
                    sim = std::make_unique<sim_instance>(path);
                    gui.set_sim_instance(sim.get());
                    gui.clear_pending_load();
                    std::cout << "Loaded configuration: " << path << "\n";
                } catch (const std::exception& e) {
                    gui.clear_pending_load();
                    gui.set_load_error(e.what());
                    std::cerr << "Failed to load configuration: " << e.what() << "\n";
                }
            }
        }
    );

    /*
     * Registering GUI callback for menu bar and dialogs.
     */
    main_app.register_gui_func(
        [&] {
            gui.render_menu_bar();
            gui.render_dialogs();
            gui.render_status_bar();
        }
    );

    /*
     * Registering a function that performs some clean-up operations after the
     * application window is closed.
     */
    main_app.register_shutdown_func(
        [&]() {
            // Reset gui_manager first to free browser resources while GL context is valid
            gui.set_sim_instance(nullptr);
            sim.reset();
        }
    );

    /*
     * Finally it is time to run the GUI application.
     * This call will block until the window is closed.
     */
    main_app.run();

    return 0;
}
