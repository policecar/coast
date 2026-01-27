#ifndef COAST_CONFIG_EDITOR_H
#define COAST_CONFIG_EDITOR_H

#include <string>
#include <functional>
#include <memory>

#include "imgui.h"
#include "imfilebrowser.h"
#include "json_loader.h"
#include "network_editor.h"

namespace coast {

/**
 * Configuration editor dialog for editing global_cfg_t.
 * Provides UI for editing dataset configs, neural model configs,
 * and simulation graph nodes.
 *
 * Features a File menu with Save, Save As, and Restore options.
 */
class config_editor {
public:
    /**
     * Callback type for reload requests after saving.
     * The callback receives the path of the saved config file.
     */
    using reload_callback_t = std::function<void(const std::string&)>;

    /**
     * Construct a configuration editor.
     * @param config Reference to the configuration to edit
     * @param config_path Path to the current configuration file
     */
    config_editor(loader::global_cfg_t& config, const std::string& config_path);

    /**
     * Render the configuration editor window.
     * @param p_open Pointer to visibility flag (window can be closed via X button)
     */
    void render(bool* p_open);

    /**
     * Check if configuration has unsaved changes.
     */
    [[nodiscard]] bool has_unsaved_changes() const { return is_dirty; }

    /**
     * Set callback to be invoked when user wants to reload after saving.
     */
    void set_reload_callback(reload_callback_t callback) { reload_callback = std::move(callback); }

    /**
     * Mark configuration as dirty (has unsaved changes).
     * Called internally when edits are detected.
     */
    void mark_dirty() { is_dirty = true; }

private:
    loader::global_cfg_t& config;
    loader::global_cfg_t original_config;  // Copy for restore
    std::string config_path;

    // File browser for Save As
    ImGui::FileBrowser save_file_browser;

    // Dirty flag - true if config differs from original
    bool is_dirty = false;

    // Dialog states
    bool show_save_confirm_dialog = false;
    bool reload_after_save = true;  // Checkbox state for reload option
    std::string pending_save_path;  // Path to save to (for Save As)

    // Reload callback
    reload_callback_t reload_callback;

    // Render menu bar
    void render_menu_bar();

    // Render dialogs
    void render_save_confirm_dialog();

    // Save configuration to file
    bool save_config(const std::string& path);

    // Restore configuration to original
    void restore_config();

    // Render sections
    void render_dataset_configs();
    void render_ngm2_configs();
    void render_simulation_graph();

    // Network editor
    std::unique_ptr<network_editor> net_editor;
    bool show_network_editor = false;
};

} // namespace coast

#endif // COAST_CONFIG_EDITOR_H
