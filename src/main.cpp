#include <iostream>
#include <memory>
#include <string>
#include <cstdint>
#include <chrono>
#include <thread>
#include <filesystem>

#include "sdl_app.h"
#include "gui_manager.h"
#include "sim_instance.h"
#include "sim_env.h"
#include "logging/logging.h"

using namespace coast;

namespace {

struct cli_options {
    std::string config_path;
    bool auto_start = false;
    bool headless = false;
    std::uint64_t max_steps = 0;  // 0 = unlimited
    bool show_help = false;

    // Logging options
    std::string buffer_log_path;      // Path for buffer logging
    std::string state_save_path;      // Path to save final state
    std::string state_load_path;      // Path to load initial state
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
    std::cout << "\nLogging options:\n";
    std::cout << "  --log-buffers <path>  Log IO buffers to HDF5 file\n";
    std::cout << "  --save-state <path>   Save final state to HDF5 file (headless only)\n";
    std::cout << "  --load-state <path>   Load initial state from HDF5 file\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " config.json                    # Load config, start GUI\n";
    std::cout << "  " << program_name << " -r config.json                 # Load config, auto-start\n";
    std::cout << "  " << program_name << " --headless -n 1000 config.json # Run 1000 steps, no GUI\n";
    std::cout << "  " << program_name << " -n 50000 --log-buffers run.h5 config.json\n";
    std::cout << "                                           # Log buffer data during run\n";
    std::cout << "  " << program_name << " -n 10000 --save-state checkpoint.h5 config.json\n";
    std::cout << "                                           # Save state after 10k steps\n";
    std::cout << "  " << program_name << " --load-state checkpoint.h5 -n 10000 config.json\n";
    std::cout << "                                           # Resume from checkpoint\n";
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
        } else if (arg == "--log-buffers" && i + 1 < argc) {
            opts.buffer_log_path = argv[++i];
        } else if (arg == "--save-state" && i + 1 < argc) {
            opts.state_save_path = argv[++i];
        } else if (arg == "--load-state" && i + 1 < argc) {
            opts.state_load_path = argv[++i];
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

        auto& logging_mgr = sim->get_logging_manager();

        // Load state if requested
        if (!opts.state_load_path.empty()) {
            std::cout << "Loading state from: " << opts.state_load_path << "\n";
            auto result = logging_mgr.load_state(opts.state_load_path);
            if (result) {
                std::cout << "State loaded. Restored step: " << result.restored_step << "\n";
                // Restore the step counter to the checkpoint value
                controller->set_step_count(result.restored_step);
                for (const auto& w : result.warnings) {
                    std::cout << "  Warning: " << w << "\n";
                }
            } else {
                std::cerr << "Failed to load state: " << result.error_message << "\n";
                return 1;
            }
        }

        // Start buffer logging if requested
        if (!opts.buffer_log_path.empty()) {
            std::cout << "Starting buffer logging to: " << opts.buffer_log_path << "\n";
            logging::buffer_logger_config cfg;
            cfg.output_path = opts.buffer_log_path;
            cfg.chunk_size = 1000;
            cfg.compression.algorithm = logging::compression_config::type::gzip;
            cfg.compression.level = 4;
            cfg.include_stats = true;

            auto result = logging_mgr.start_buffer_logging(cfg);
            if (!result) {
                std::cerr << "Failed to start buffer logging: " << result.error_message << "\n";
                return 1;
            }
        }

        // Calculate target step count (current + max_steps for relative mode)
        const std::uint64_t start_step = controller->get_step_count();
        const std::uint64_t target_step = (opts.max_steps > 0) ? (start_step + opts.max_steps) : 0;

        std::cout << "Starting simulation";
        if (opts.max_steps > 0) {
            std::cout << " for " << opts.max_steps << " steps (from step " << start_step << " to " << target_step << ")";
            // Set step limit for precise stopping
            controller->set_step_limit(target_step);
        }
        std::cout << "...\n";

        // Start simulation in unlimited mode
        controller->set_mode(sim::sim_mode::unlimited);

        // Run until target_step reached (or forever if max_steps == 0)
        // The controller will auto-pause when hitting the step limit
        std::uint64_t last_reported_steps = start_step;
        while ((target_step == 0 || controller->get_step_count() < target_step) &&
               controller->get_mode() != sim::sim_mode::paused) {
            // Adaptive polling: use shorter intervals when close to target
            auto steps = controller->get_step_count();
            std::chrono::milliseconds sleep_time{100};
            if (target_step > 0) {
                auto remaining = (target_step > steps) ? (target_step - steps) : 0;
                if (remaining < 1000) {
                    sleep_time = std::chrono::milliseconds{1};
                } else if (remaining < 10000) {
                    sleep_time = std::chrono::milliseconds{10};
                }
            }
            std::this_thread::sleep_for(sleep_time);

            // Update probe observers (like GUI does each frame)
            for (auto& entity : sim->get_sim_env().iterate_entities()) {
                entity.probes().update_observers();
            }

            // Print progress every 10000 steps
            steps = controller->get_step_count();
            if (steps >= last_reported_steps + 10000) {
                std::cout << "  Steps: " << steps
                          << " | SPS: " << controller->get_actual_sps();

                // Report logging stats if active
                if (logging_mgr.is_buffer_logging()) {
                    auto stats = logging_mgr.combined_stats();
                    std::cout << " | Logged: " << stats.frames_written << " frames";
                }
                std::cout << "\n";
                last_reported_steps = steps;
            }
        }

        // Stop simulation (may already be paused if step limit was reached)
        controller->set_mode(sim::sim_mode::paused);
        controller->set_step_limit(0);  // Clear step limit
        std::cout << "Simulation completed. Total steps: " << controller->get_step_count() << "\n";

        // Stop buffer logging
        if (logging_mgr.is_buffer_logging()) {
            logging_mgr.stop_buffer_logging();
            std::cout << "Buffer logging stopped.\n";
        }

        // Save state if requested
        if (!opts.state_save_path.empty()) {
            std::cout << "Saving state to: " << opts.state_save_path << "\n";
            logging::state_save_config cfg;
            cfg.output_path = opts.state_save_path;
            cfg.compression.algorithm = logging::compression_config::type::gzip;
            cfg.compression.level = 4;

            auto result = logging_mgr.save_state(controller->get_step_count(), cfg);
            if (result) {
                std::cout << "State saved successfully.\n";
            } else {
                std::cerr << "Failed to save state: " << result.error_message << "\n";
                return 1;
            }
        }

        // Report final logging stats
        auto stats = logging_mgr.combined_stats();
        if (stats.frames_written > 0 || stats.samples_written > 0) {
            std::cout << "Logging summary: " << stats.frames_written << " frames, "
                      << stats.samples_written << " samples, "
                      << (stats.bytes_written / (1024.0 * 1024.0)) << " MB\n";
        }

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
