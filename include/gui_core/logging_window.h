#ifndef COAST_LOGGING_WINDOW_H
#define COAST_LOGGING_WINDOW_H

#include <string>
#include <cstdint>
#include <memory>
#include "logging/logging_common.h"
#include "imgui.h"
#include "imfilebrowser.h"

namespace coast::paths { struct app_config; }

namespace coast {

class sim_instance;

// Shared state for the deterministic auto-checkpoint hook. Defined in
// logging_window.cpp (kept out of the header so the heavy sim/logging types stay
// behind the .cpp). Held via shared_ptr so the sim-thread hook that captures it
// is decoupled from the logging_window lifetime — mirrors gui_manager's status
// publisher.
struct logging_checkpoint_ctx;

/**
 * Window for controlling data logging functionality.
 * Provides controls for:
 * - IO Buffer logging (HDF5)
 * - State save/load
 */
class logging_window {
public:
    logging_window();

    /**
     * Set the simulation instance to control logging for.
     */
    void set_sim_instance(sim_instance* instance);

    /**
     * Inject the application location config (WHERE outputs go). Set once by
     * gui_manager's constructor before any sim is attached; the auto-checkpoint
     * path resolves snapshot_dir under config_base with it. Held by pointer
     * (this is a default-constructed member; the setter mirrors set_sim_instance).
     */
    void set_app_config(const paths::app_config& cfg) { app_cfg_ = &cfg; }

    /**
     * Render the logging control window.
     */
    void render();

    /**
     * Per-frame GUI-thread tick: surface completed async checkpoint writes.
     * The checkpoint cadence itself runs on the sim thread (see the .cpp); this
     * only reaps results for the status line. Call once per GUI frame.
     */
    void update();

    /**
     * Access visibility state for menu toggle.
     */
    bool& visible() { return is_visible; }

    /**
     * Trigger state save dialog (can be called from menu).
     */
    void open_save_state_dialog();

    /**
     * Trigger state load dialog (can be called from menu).
     */
    void open_load_state_dialog();

    /**
     * Check if state was just loaded (for external handling).
     */
    bool has_loaded_state() const { return state_just_loaded; }

    /**
     * Get the step count from loaded state.
     */
    std::uint64_t get_loaded_step_count() const { return loaded_step_count; }

    /**
     * Clear the loaded state flag.
     */
    void clear_loaded_state() { state_just_loaded = false; }

private:
    sim_instance* sim = nullptr;
    // Application location config (owned by main; set by gui_manager's ctor).
    // nullptr only before that injection, which happens before any sim attaches.
    const paths::app_config* app_cfg_ = nullptr;
    bool is_visible = false;

    // Default log/state file names. On set_sim_instance these are (re)seeded
    // under the config's <base>/logs directory (paths.log_dir under config_base),
    // UNLESS the user has edited the field — the *_edited_ flags below latch on
    // any InputText/file-browser change so a config reload never clobbers a
    // user-chosen path (design §3.1). Until a sim is attached they hold the bare
    // CWD name.

    // Buffer logging state
    std::string buffer_log_path = "buffer_log.h5";
    bool buffer_log_edited_ = false;        // user picked/typed a path -> stop reseeding
    bool buffer_logging_active = false;
    bool buffer_confirm_overwrite = false;  // Show overwrite confirmation popup

    // State save/load
    std::string state_path = "state.h5";
    bool state_path_edited_ = false;

    // config_path the log/state defaults were last seeded for. Used to reset the
    // *_edited_ latches on a config SWITCH (different path) but not on a reload of
    // the same config — see seed_log_defaults().
    std::string last_seeded_config_;
    bool state_just_loaded = false;
    std::uint64_t loaded_step_count = 0;
    std::string last_error;
    std::string last_success;

    // Auto-checkpoint settings. The cadence is enforced on the SIM thread by a
    // deterministic post_swap hook (see logging_window.cpp), NOT by polling the
    // step count each GUI frame — so `auto_checkpoint_interval` is an exact
    // guarantee, mirroring the headless `--checkpoint` path. The two fields below
    // are GUI widget state (checkbox + input); the live controls the hook reads
    // (enabled flag, interval, modulo phase) and the pending async-write queue
    // live in the shared `logging_checkpoint_ctx`.
    bool auto_checkpoint_enabled = false;
    int  auto_checkpoint_interval = 10000;  // Steps between checkpoints
    std::string last_checkpoint_path;       // last completed checkpoint (display)

    std::shared_ptr<logging_checkpoint_ctx> checkpoint_ctx_;
    std::size_t checkpoint_hook_id_ = 0;
    bool        checkpoint_hook_registered_ = false;

    // File browsers
    ImGui::FileBrowser buffer_file_browser;
    ImGui::FileBrowser state_save_browser;
    ImGui::FileBrowser state_load_browser;

    // Internal rendering helpers
    void render_buffer_logging_section();
    void render_state_section();
    void render_status_section();

    // (Re)seed unedited default log/state paths under the new sim's <base>/logs.
    void seed_log_defaults();

    // Stop buffer logging through the sim's controller (quiesced, issue #197).
    logging::log_result stop_buffer_logging();

    // Auto-checkpoint helpers (see logging_window.cpp for the threading model).
    void setup_checkpoint();      // build ctx + install the sim-thread cadence hook
    void teardown_checkpoint();   // drain pending writes + remove hook (old sim alive)
    void reap_checkpoints();      // GUI thread: surface completed writes (non-blocking)
    void drain_checkpoints();     // GUI thread: block on all pending writes (teardown)
};

} // namespace coast

#endif // COAST_LOGGING_WINDOW_H
