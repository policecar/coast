#include "app/build_env.h"

#include <exception>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

#include "cli/options.h"
#include "core/app_config.h"
#include "diag/diag.h"
#include "sim_core/sim_instance.h"
#include "json_cfg/json_loader.h"
#include "sim_core/sim_thread_controller.h"
#include "logging/logging_manager.h"
#include "hdf5/executor.h"

namespace coast::app {

namespace {

// Find the latest snapshot file for a given config in the already-resolved
// snapshot directory. Returns empty string if no snapshot found. The caller
// resolves dir_path via resolve_snapshot_output_dir so resume and write agree.
std::string find_latest_snapshot(const std::string&           config_path,
                                 const std::filesystem::path&  dir_path) {
    if (!std::filesystem::exists(dir_path) ||
        !std::filesystem::is_directory(dir_path)) {
        return "";
    }

    const std::string config_name =
        std::filesystem::path(config_path).stem().string();

    std::filesystem::path latest_path;
    std::filesystem::file_time_type latest_time;
    bool found = false;

    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;

        const std::string filename = entry.path().filename().string();
        if (!filename.starts_with(config_name + "_")) continue;
        if (!filename.ends_with(".h5") && !filename.ends_with(".hdf5")) continue;

        const auto write_time = entry.last_write_time();
        if (!found || write_time > latest_time) {
            latest_path = entry.path();
            latest_time = write_time;
            found = true;
        }
    }

    return found ? latest_path.string() : "";
}

} // namespace

std::unique_ptr<sim_instance> build_env(const cli::options& opts,
                                        const paths::app_config&   app_cfg,
                                        hdf5::executor&     h5_exec) {
    if (opts.config_path.empty()) {
        return nullptr;
    }

    auto sim = std::make_unique<sim_instance>(opts.config_path,
                                              h5_exec,
                                              app_cfg,
                                              opts.seed_offset);

    // Determine state to load. --load-state wins over --resume.
    std::string state_to_load = opts.state_load_path;
    if (!state_to_load.empty() && opts.resume_latest) {
        COAST_INFO("build", "--load-state takes precedence over --resume");
    }
    const bool resumed = state_to_load.empty() && opts.resume_latest;
    std::filesystem::path snap_dir;
    if (resumed) {
        snap_dir = paths::resolve_snapshot_output_dir(
            opts.config_path, sim->get_config().snapshot_dir, app_cfg);
        state_to_load = find_latest_snapshot(opts.config_path, snap_dir);
        if (state_to_load.empty()) {
            COAST_INFO("build", "no existing snapshot to resume from; starting fresh");
        } else {
            COAST_INFO("build", "found snapshot to resume: {}", state_to_load);
            // --resume says "continue this config's newest checkpoint".
            sim->prepare_resume();
        }
    }

    if (!state_to_load.empty()) {
        COAST_INFO("build", "loading state from: {}", state_to_load);
        // A snapshot found by --resume was not named by the user, so the error
        // says where it came from and how to get past it.
        auto failed = [&](std::string_view why) {
            auto msg = std::format("Failed to load state from {}: {}", state_to_load, why);
            if (resumed)
                msg += std::format(" — the snapshot was found by --resume in {}; drop --resume "
                                   "to start fresh, or move that directory aside", snap_dir.string());
            return state_load_error(msg);
        };
        auto& logging_mgr = sim->get_logging_manager();
        logging::state_load_result result;
        try {
            result = logging_mgr.load_state_startup(state_to_load);
        } catch (const std::exception& e) {
            throw failed(e.what());
        }
        if (!result) {
            throw failed(result.error_message);
        }
        COAST_INFO("build", "state loaded; restored step: {}", result.restored_step);
        if (auto* controller = sim->get_thread_controller()) {
            controller->set_step_count(result.restored_step);
        }
        for (const auto& w : result.warnings) {
            COAST_WARN("build", "{}", w);
        }
    }

    return sim;
}

std::unique_ptr<sim_instance> build_env(const cli::options& opts,
                                        hdf5::executor&     h5_exec) {
    return build_env(opts, paths::app_config{}, h5_exec);
}

} // namespace coast::app
