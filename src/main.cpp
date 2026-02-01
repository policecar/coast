#include <iostream>
#include <memory>
#include <string>
#include <cstdint>
#include <chrono>
#include <thread>

#include "sdl_app.h"
#include "gui_manager.h"
#include "sim_instance.h"
#include "sim_env.h"

using namespace coast;

namespace {

struct cli_options {
    std::string config_path;
    bool auto_start = false;
    bool headless = false;
    std::uint64_t max_steps = 0;  // 0 = unlimited
    bool show_help = false;
};

void print_usage(const char* program_name) {
    std::cout << "COAST - Neural Simulation\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS] [CONFIG_FILE]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --config <path>   Path to configuration file\n";
    std::cout << "  -r, --run             Start simulation immediately\n";
    std::cout << "  -H, --headless        Run without GUI (implies --run)\n";
    std::cout << "  -n, --steps <count>   Run for N steps then exit (implies --headless)\n";
    std::cout << "  -h, --help            Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " config.json                    # Load config, start GUI\n";
    std::cout << "  " << program_name << " -r config.json                 # Load config, auto-start\n";
    std::cout << "  " << program_name << " --headless -n 1000 config.json # Run 1000 steps, no GUI\n";
}

cli_options parse_args(int argc, char** argv) {
    cli_options opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            opts.show_help = true;
        } else if (arg == "-r" || arg == "--run") {
            opts.auto_start = true;
        } else if (arg == "-H" || arg == "--headless") {
            opts.headless = true;
            opts.auto_start = true;
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            opts.config_path = argv[++i];
        } else if ((arg == "-n" || arg == "--steps") && i + 1 < argc) {
            opts.max_steps = std::stoull(argv[++i]);
            opts.headless = true;
            opts.auto_start = true;
        } else if (arg[0] != '-' && opts.config_path.empty()) {
            // Positional argument: treat as config path
            opts.config_path = arg;
        } else if (arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }

    return opts;
}

// Headless simulation runner
int run_headless(const cli_options& opts) {
    std::cout << "COAST Headless Mode\n";
    std::cout << "Loading configuration: " << opts.config_path << "\n";

    try {
        auto sim = std::make_unique<sim_instance>(opts.config_path);
        std::cout << "Configuration loaded successfully.\n";

        auto* controller = sim->get_thread_controller();
        if (!controller) {
            std::cerr << "Error: No simulation thread controller available.\n";
            return 1;
        }

        std::cout << "Starting simulation";
        if (opts.max_steps > 0) {
            std::cout << " for " << opts.max_steps << " steps";
        }
        std::cout << "...\n";

        // Start simulation in unlimited mode
        controller->set_mode(sim::sim_mode::unlimited);

        // Run until max_steps reached (or forever if max_steps == 0)
        while (opts.max_steps == 0 || controller->get_step_count() < opts.max_steps) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Update probe observers (like GUI does each frame)
            for (auto& entity : sim->get_sim_env().iterate_entities()) {
                entity.probes().update_observers();
            }

            // Print progress every 10000 steps
            auto steps = controller->get_step_count();
            if (steps > 0 && steps % 10000 == 0) {
                std::cout << "  Steps: " << steps
                          << " | SPS: " << controller->get_actual_sps() << "\n";
            }
        }

        // Stop simulation
        controller->set_mode(sim::sim_mode::paused);
        std::cout << "Simulation completed. Total steps: " << controller->get_step_count() << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

} // anonymous namespace

int main(int argc, char **argv)
{
    cli_options opts = parse_args(argc, argv);

    if (opts.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    // Headless mode: run without GUI
    if (opts.headless) {
        if (opts.config_path.empty()) {
            std::cerr << "Error: Configuration file required for headless mode.\n";
            std::cerr << "Use --help for usage information.\n";
            return 1;
        }
        return run_headless(opts);
    }

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
    if (!opts.config_path.empty()) {
        try {
            sim = std::make_unique<sim_instance>(opts.config_path);
            gui.set_sim_instance(sim.get());
            std::cout << "Loaded configuration: " << opts.config_path << "\n";

            // Auto-start if requested
            if (opts.auto_start) {
                if (auto* controller = sim->get_thread_controller()) {
                    controller->set_mode(sim::sim_mode::throttled);
                    std::cout << "Simulation started.\n";
                }
            }
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
