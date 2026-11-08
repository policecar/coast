#include <cctype>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "cli/options.h"
#include "cli/parse.h"
#include "diag/diag.h"
#include "core/app_config.h"
#include "app/build_env.h"
#include "app/run_headless.h"
#include "sim_core/sim_instance.h"
#include "hdf5/executor.h"

// The GUI layer (coast_gui) is only linked in when the build was configured with
// -DCOAST_WITH_GUI=ON (the default). A headless build has no run_gui, no
// console_iface and no console: this file is the single place that knows it.
#ifdef COAST_WITH_GUI
#include "app/run_gui.h"
#include "gui_core/console_iface.h"
#ifdef COAST_WITH_CONSOLE
#include "console/console_panel.h"
#endif
#endif

int main(int argc, char** argv) {
    coast::cli::options opts;
    try {
        opts = coast::cli::parse(argc, argv);
    } catch (const coast::cli::parse_error& e) {
        // Pre-config: opts aren't trusted yet, so report straight to stderr.
        std::cerr << e.what() << "\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }

    // --help is pure stdout output that needs no diagnostics configured; handle
    // it before touching the log sink so `--help --log-file /bad/path` prints
    // usage without a spurious sink-open warning ahead of it.
    if (opts.show_help) {
        coast::cli::print_usage(std::cout, argv[0]);
        return 0;
    }

    // --init: bootstrap a ./coast.json template and exit. A pure CLI action (like
    // --help), handled before diagnostics/executor/build_env. The never-clobber
    // guarantee lives in the writer (atomic exclusive create), so there is no
    // check-then-write race here — we only translate its outcome to a message.
    if (opts.init) {
        const std::filesystem::path p = "coast.json";
        switch (coast::paths::write_app_config_file(p)) {
            case coast::paths::write_status::exists:
                std::cout << "coast.json already exists; leaving it untouched.\n";
                return 0;
            case coast::paths::write_status::error:
                std::cerr << "error: could not write coast.json\n";
                return 1;
            case coast::paths::write_status::written:
                std::cout << "wrote coast.json (built-in defaults) — edit it to "
                             "customize output locations and assets.\n";
                return 0;
        }
    }

    // Configure diagnostics before anything else can log.
    coast::diag::set_level(opts.log_level);
    // Match the stdout/stderr keywords case-insensitively (as --log-level is);
    // anything else is a real path, used verbatim (opts.log_file).
    std::string log_target = opts.log_file;
    for (char& c : log_target)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (!log_target.empty() && log_target != "stderr") {
        if (log_target == "stdout") {
            coast::diag::set_stream_sink(stdout);
        } else if (!coast::diag::set_log_file(opts.log_file.c_str())) {
            // Intentionally raw std::cerr, not COAST_WARN: this reports the
            // diagnostics sink's own open failure, so it must be visible
            // regardless of --log-level (a -q-raised threshold would otherwise
            // hide the reason the log file is empty) and can't go to the sink
            // we just failed to install.
            std::cerr << "Warning: could not open --log-file '" << opts.log_file
                      << "'; logging to stderr.\n";
        }
    }

    // Load the application-level config (WHERE outputs go + app assets). A file
    // named explicitly (--app-config / $COAST_APP_CONFIG) that is missing or malformed
    // is a hard error; an absent auto-discovered ./coast.json falls back to
    // built-in defaults. Distinct from the per-experiment config loaded below.
    coast::paths::app_config app_cfg;
    try {
        app_cfg = coast::paths::load_app_config(opts.app_config_path);
    } catch (const std::exception& e) {
        COAST_ERROR("app_config", "{}", e.what());
        return 1;
    }

#ifndef COAST_WITH_GUI
    // Headless-only build: there is no GUI to fall back to, so every run is
    // headless, including one whose command line never said so. cli::parse is
    // build-independent and only rejects an explicit -H with no length, so the
    // implicit case is enforced here — otherwise `coast config.json`, the command
    // that opens a window in a GUI build, would start an unbounded run instead.
    if (opts.steps_mode == coast::cli::options::run_length::unset) {
        std::cerr << coast::cli::headless_needs_length_msg << "\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }
    // Announce the headless-only build once, unless the user said -H/--headless
    // (that flag means they already expect this, so it silences the notice).
    if (!opts.headless_flag_given) {
        COAST_INFO("cli", "this binary was built without GUI support "
                          "(-DCOAST_WITH_GUI=OFF); running headless");
    }
    opts.headless = true;
#endif

    // The HDF5 executor owns the single thread that is ever allowed to enter
    // libhdf5 (built without --enable-threadsafe). Must outlive every
    // sim_instance, logging_manager, and observe_registry — construct first,
    // let RAII destroy last.
    hdf5::executor h5_exec;

    std::unique_ptr<coast::sim_instance> sim;
    try {
        // Arm the --save-on-term catcher BEFORE build_env. Loading datasets,
        // restoring --load-state and initialising the GPU take seconds, and until
        // now a SIGTERM in that window hit the default disposition: the job died
        // with no manifest, which coastd cannot tell apart from a crash, so an
        // operator's cancel during startup counted toward the crash-loop guard.
        if (opts.headless && opts.save_on_term) coast::app::arm_term_signal();
        sim = coast::app::build_env(opts, app_cfg, h5_exec);
    } catch (const coast::app::state_load_error& e) {
        // The configuration loaded; the state it was asked to restore did not.
        COAST_ERROR("build", "{}", e.what());
        if (opts.headless) return 1;
        COAST_WARN("build", "starting without simulation; "
                            "use System > Open Configuration to load");
    } catch (const std::exception& e) {
        COAST_ERROR("build", "failed to load configuration: {}", e.what());
        if (opts.headless) return 1;
        COAST_WARN("build", "starting without simulation; "
                            "use System > Open Configuration to load");
        // Fall through to GUI with no sim loaded.
    }

    if (opts.headless) {
        if (!sim) {
            COAST_ERROR("cli", "configuration file required for headless mode "
                               "(use --help for usage)");
            return 1;
        }
        return coast::app::run_headless(*sim, opts, app_cfg);
    }

#ifdef COAST_WITH_GUI
    // The C++ console is compiled in only when COAST_WITH_CONSOLE is enabled (it
    // carries the LLVM/Clang dependency); coast_lib and run_gui remain agnostic
    // via coast::console_iface.
    std::unique_ptr<coast::console_iface> console;
#ifdef COAST_WITH_CONSOLE
    console = std::make_unique<console::console_panel>(h5_exec);
#endif

    return coast::app::run_gui(std::move(sim), opts, app_cfg, h5_exec, std::move(console));
#else
    // Unreachable: opts.headless was forced true above in a headless build, so
    // the run_headless branch always returns. Present so the function has a
    // return on every path without a GUI symbol in sight.
    return 1;
#endif
}
