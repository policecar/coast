#include "config_editor.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "mnist_cfg.h"
#include "dsprites_cfg.h"
#include "hd_ngm2_cfg.h"
#include "sim_builder_cfg.h"

// Include JSON serialization headers
#include "ngm2_json.h"
#include "sim_json.h"
#include "dataset_json.h"
#include "gui_vis_json.h"

namespace coast {

namespace {

// Buffer size for ImGui text input fields
constexpr std::size_t INPUT_BUFFER_SIZE = 512;

// Helper: InputText wrapper for std::string
// Returns true if the value was modified
bool InputTextString(const char* label, std::string& str, std::size_t max_size = INPUT_BUFFER_SIZE)
{
    // Ensure string has enough capacity for editing
    str.resize(max_size);

    bool modified = ImGui::InputText(label, str.data(), str.size());

    // Resize to actual content length
    str.resize(std::strlen(str.c_str()));

    return modified;
}

// Helper: InputText wrapper with item width
bool InputTextStringWidth(const char* label, std::string& str, float width, std::size_t max_size = INPUT_BUFFER_SIZE)
{
    ImGui::SetNextItemWidth(width);
    return InputTextString(label, str, max_size);
}

} // anonymous namespace

// Helper: Render string list editor (returns true if modified)
static bool render_string_list(const char* label, std::vector<std::string>& list)
{
    bool modified = false;
    if (ImGui::TreeNode(label)) {
        for (std::size_t i = 0; i < list.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            if (InputTextStringWidth("##item", list[i], 200)) {
                modified = true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("X")) {
                list.erase(list.begin() + static_cast<std::ptrdiff_t>(i));
                modified = true;
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
        if (ImGui::SmallButton("+ Add")) {
            list.emplace_back("");
            modified = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render MNIST config (returns true if modified, sets remove_requested if remove clicked)
static bool render_mnist_config(const std::string& name, mnist::mnist_param_t& cfg, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        if (InputTextString("Image File", cfg.image_file)) {
            modified = true;
        }
        if (InputTextString("Label File", cfg.label_file)) {
            modified = true;
        }

        int change_interval = static_cast<int>(cfg.change_interval);
        if (ImGui::InputInt("Change Interval", &change_interval)) {
            cfg.change_interval = static_cast<std::size_t>(std::max(1, change_interval));
            modified = true;
        }

        if (ImGui::DragFloatRange2("Noise Range", &cfg.noise_range.first, &cfg.noise_range.second,
                               0.01f, 0.0f, 1.0f)) {
            modified = true;
        }
        if (ImGui::InputInt("Random Seed", &cfg.random_seed)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove Config")) {
            remove_requested = true;
        }

        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render dSprites config (returns true if modified, sets remove_requested if remove clicked)
static bool render_dsprites_config(const std::string& name, dsprites::dsprites_param_t& cfg, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        if (InputTextString("HDF5 File", cfg.hdf5_file)) {
            modified = true;
        }

        int change_interval = static_cast<int>(cfg.change_interval);
        if (ImGui::InputInt("Change Interval", &change_interval)) {
            cfg.change_interval = static_cast<std::size_t>(std::max(1, change_interval));
            modified = true;
        }

        if (ImGui::DragFloatRange2("Noise Range", &cfg.noise_range.first, &cfg.noise_range.second,
                               0.01f, 0.0f, 1.0f)) {
            modified = true;
        }
        if (ImGui::InputInt("Random Seed", &cfg.random_seed)) {
            modified = true;
        }
        if (ImGui::Checkbox("Randomize Readout", &cfg.randomize_readout)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove Config")) {
            remove_requested = true;
        }

        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render dendrite config (returns true if modified, sets remove_requested if remove clicked)
static bool render_dendrite_config(const std::string& name, ngm2::dendrite_params_t& cfg, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        // Dendrite type dropdown
        static constexpr std::array type_names = { "Apical", "Proximal" };
        int type_idx = static_cast<int>(cfg.type);
        if (ImGui::Combo("Type", &type_idx, type_names.data(), static_cast<int>(type_names.size()))) {
            cfg.type = static_cast<ngm2::dendrite_type_t>(type_idx);
            modified = true;
        }

        if (render_string_list("Input Channels", cfg.input_channels)) {
            modified = true;
        }

        if (ImGui::SliderFloat("Permanence Threshold", &cfg.permanence_threshold, 0.0f, 1.0f)) {
            modified = true;
        }

        int max_branch = cfg.max_branch_level;
        if (ImGui::SliderInt("Max Branch Level", &max_branch, 1, 10)) {
            cfg.max_branch_level = static_cast<uint8_t>(max_branch);
            modified = true;
        }

        if (ImGui::InputInt("Random Seed", &cfg.rnd_seed)) {
            modified = true;
        }

        if (ImGui::TreeNode("Learning Parameters")) {
            if (ImGui::SliderFloat("Primary Learning Rate", &cfg.default_primary_learning_rate, 0.0001f, 0.1f, "%.4f")) {
                modified = true;
            }
            if (ImGui::SliderFloat("Secondary Learning Rate", &cfg.default_secondary_learning_rate, 0.00001f, 0.01f, "%.5f")) {
                modified = true;
            }
            if (ImGui::SliderFloat("Mismatch Act Threshold", &cfg.default_mismatch_act_thres, 0.0f, 1.0f)) {
                modified = true;
            }
            if (ImGui::SliderFloat("Mismatch Smoothing", &cfg.default_mismatch_smoothing, 0.0001f, 0.01f, "%.4f")) {
                modified = true;
            }
            if (ImGui::SliderFloat("Accumulated Theta Threshold", &cfg.default_accumulated_theta_thres, 0.0f, 10.0f)) {
                modified = true;
            }
            if (ImGui::SliderFloat("Min Mismatch Deviation", &cfg.default_min_mismatch_deviation, 0.0f, 5.0f)) {
                modified = true;
            }
            if (ImGui::SliderFloat("Min Mismatch Percentage", &cfg.default_min_mismatch_percentage, 0.0f, 0.1f, "%.4f")) {
                modified = true;
            }
            ImGui::TreePop();
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove Config")) {
            remove_requested = true;
        }

        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render dendrite set (returns true if modified)
static bool render_dendrite_set(ngm2::dendrite_set_t& ds, const ngm2::cfg_store& store)
{
    bool modified = false;

    if (InputTextString("Name", ds.name)) {
        modified = true;
    }

    // Dropdown for dendrite_cfg from available dendrite configs
    if (ImGui::BeginCombo("Dendrite Config", ds.dendrite_cfg.c_str())) {
        for (const auto& [key, val] : store.dendrites) {
            bool selected = (ds.dendrite_cfg == key);
            if (ImGui::Selectable(key.c_str(), selected)) {
                ds.dendrite_cfg = key;
                modified = true;
            }
        }
        ImGui::EndCombo();
    }

    int count = static_cast<int>(ds.dendrite_count);
    if (ImGui::InputInt("Count", &count)) {
        ds.dendrite_count = static_cast<std::size_t>(std::max(1, count));
        modified = true;
    }
    return modified;
}

// Helper: Render channel mapping (returns true if modified)
static bool render_channel_mapping(ngm2::channel_mapping_t& cm)
{
    bool modified = false;

    if (InputTextStringWidth("From", cm.from_channel, 100)) {
        modified = true;
    }
    ImGui::SameLine();
    if (InputTextStringWidth("To Name", cm.to_name, 100)) {
        modified = true;
    }
    ImGui::SameLine();
    if (InputTextStringWidth("To Channel", cm.to_channel, 100)) {
        modified = true;
    }
    return modified;
}

// Helper: Render neuron config (returns true if modified, sets remove_requested if remove clicked)
static bool render_neuron_config(const std::string& name, ngm2::neuron_params_t& cfg,
                                  const ngm2::cfg_store& store, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        if (render_string_list("Input Channels", cfg.input_channels)) {
            modified = true;
        }

        // Dendrite sets
        if (ImGui::TreeNode("Dendrite Sets")) {
            for (std::size_t i = 0; i < cfg.dendrites.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                std::string set_label = std::format("Set {}", i + 1);
                if (ImGui::TreeNode("##ds", "%s", set_label.c_str())) {
                    if (render_dendrite_set(cfg.dendrites[i], store)) {
                        modified = true;
                    }
                    if (ImGui::SmallButton("Remove")) {
                        cfg.dendrites.erase(cfg.dendrites.begin() + static_cast<std::ptrdiff_t>(i));
                        modified = true;
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
            if (ImGui::SmallButton("+ Add Dendrite Set")) {
                cfg.dendrites.push_back({});
                modified = true;
            }
            ImGui::TreePop();
        }

        // Channel mappings
        if (ImGui::TreeNode("Channel Mappings")) {
            for (std::size_t i = 0; i < cfg.channel_mappings.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (render_channel_mapping(cfg.channel_mappings[i])) {
                    modified = true;
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) {
                    cfg.channel_mappings.erase(cfg.channel_mappings.begin() + static_cast<std::ptrdiff_t>(i));
                    modified = true;
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
            if (ImGui::SmallButton("+ Add Mapping")) {
                cfg.channel_mappings.push_back({});
                modified = true;
            }
            ImGui::TreePop();
        }

        int branch_interval = static_cast<int>(cfg.default_branch_interval);
        if (ImGui::InputInt("Branch Interval", &branch_interval)) {
            cfg.default_branch_interval = static_cast<std::size_t>(std::max(1, branch_interval));
            modified = true;
        }

        if (ImGui::TreeNode("Activity Learning Window")) {
            if (ImGui::SliderFloat("Pre Steepness", &cfg.default_activity_learning_window.first.steepness, 0.0f, 1.0f)) {
                modified = true;
            }
            if (ImGui::SliderFloat("Pre Transition", &cfg.default_activity_learning_window.first.transition_point, 0.0f, 1.0f)) {
                modified = true;
            }
            if (ImGui::SliderFloat("Post Steepness", &cfg.default_activity_learning_window.second.steepness, 0.0f, 1.0f)) {
                modified = true;
            }
            if (ImGui::SliderFloat("Post Transition", &cfg.default_activity_learning_window.second.transition_point, 0.0f, 1.0f)) {
                modified = true;
            }
            ImGui::TreePop();
        }

        if (ImGui::InputInt("Random Seed", &cfg.random_seed)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove Config")) {
            remove_requested = true;
        }

        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render neuron set (returns true if modified)
static bool render_neuron_set(ngm2::neuron_set_t& ns, const ngm2::cfg_store& store)
{
    bool modified = false;

    if (InputTextString("Name", ns.name)) {
        modified = true;
    }

    // Dropdown for neuron_cfg
    if (ImGui::BeginCombo("Neuron Config", ns.neuron_cfg.c_str())) {
        for (const auto& [key, val] : store.neurons) {
            bool selected = (ns.neuron_cfg == key);
            if (ImGui::Selectable(key.c_str(), selected)) {
                ns.neuron_cfg = key;
                modified = true;
            }
        }
        ImGui::EndCombo();
    }

    int count = static_cast<int>(ns.neuron_count);
    if (ImGui::InputInt("Count", &count)) {
        ns.neuron_count = static_cast<std::size_t>(std::max(1, count));
        modified = true;
    }
    return modified;
}

// Helper: Render neuron group config (returns true if modified, sets remove_requested if remove clicked)
static bool render_neuron_group_config(const std::string& name, ngm2::neuron_group_params_t& cfg,
                                        const ngm2::cfg_store& store, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        if (render_string_list("Input Channels", cfg.input_channels)) {
            modified = true;
        }

        // Neuron sets
        if (ImGui::TreeNode("Neuron Sets")) {
            for (std::size_t i = 0; i < cfg.neurons.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                std::string set_label = std::format("Set {}", i + 1);
                if (ImGui::TreeNode("##ns", "%s", set_label.c_str())) {
                    if (render_neuron_set(cfg.neurons[i], store)) {
                        modified = true;
                    }
                    if (ImGui::SmallButton("Remove")) {
                        cfg.neurons.erase(cfg.neurons.begin() + static_cast<std::ptrdiff_t>(i));
                        modified = true;
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
            if (ImGui::SmallButton("+ Add Neuron Set")) {
                cfg.neurons.push_back({});
                modified = true;
            }
            ImGui::TreePop();
        }

        // Channel mappings
        if (ImGui::TreeNode("Channel Mappings")) {
            for (std::size_t i = 0; i < cfg.channel_mappings.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (render_channel_mapping(cfg.channel_mappings[i])) {
                    modified = true;
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) {
                    cfg.channel_mappings.erase(cfg.channel_mappings.begin() + static_cast<std::ptrdiff_t>(i));
                    modified = true;
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
            if (ImGui::SmallButton("+ Add Mapping")) {
                cfg.channel_mappings.push_back({});
                modified = true;
            }
            ImGui::TreePop();
        }

        if (ImGui::SliderFloat("Local Inhibition", &cfg.default_local_inhibition_strength, 0.1f, 20.0f)) {
            modified = true;
        }
        if (ImGui::SliderFloat("Common Learning Rate", &cfg.default_common_learning_rate, 0.00001f, 0.01f, "%.5f")) {
            modified = true;
        }

        if (ImGui::TreeNode("Weight Filter")) {
            if (ImGui::SliderFloat("Steepness", &cfg.default_weight_filter.steepness, 0.0f, 1.0f)) {
                modified = true;
            }
            if (ImGui::SliderFloat("Transition Point", &cfg.default_weight_filter.transition_point, 0.0f, 1.0f)) {
                modified = true;
            }
            ImGui::TreePop();
        }

        if (ImGui::SliderFloat("Stochastic Win Threshold", &cfg.default_stochastic_win_thres, 0.0f, 1.0f)) {
            modified = true;
        }
        if (ImGui::InputInt("Random Seed", &cfg.random_seed)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove Config")) {
            remove_requested = true;
        }

        ImGui::TreePop();
    }
    return modified;
}

// Helper: Get entity type name
static std::string_view get_entity_type_name(sim::io_entity_type_t type)
{
    switch (type) {
        case sim::io_entity_type_t::ngm2_neuron_group: return "Neuron Group";
        case sim::io_entity_type_t::mnist_input: return "MNIST Input";
        case sim::io_entity_type_t::dsprites_input: return "dSprites Input";
        default: return "Unknown";
    }
}

// Helper: Render simulation node (returns true if modified)
static bool render_sim_node(sim::sim_node_t& node, const loader::global_cfg_t& cfg)
{
    bool modified = false;

    // ID
    int id = static_cast<int>(node.id);
    if (ImGui::InputInt("ID", &id)) {
        node.id = static_cast<sim::sim_node_id_t>(id);
        modified = true;
    }

    // Label
    if (InputTextString("Label", node.label)) {
        modified = true;
    }

    // Entity type dropdown
    static constexpr std::array type_names = { "Neuron Group", "MNIST Input", "dSprites Input" };
    int type_idx = static_cast<int>(node.entity_type);
    if (ImGui::Combo("Entity Type", &type_idx, type_names.data(), static_cast<int>(type_names.size()))) {
        node.entity_type = static_cast<sim::io_entity_type_t>(type_idx);
        modified = true;
    }

    // Entity config dropdown - show configs from appropriate store
    if (ImGui::BeginCombo("Entity Config", node.entity_cfg.c_str())) {
        switch (node.entity_type) {
            case sim::io_entity_type_t::ngm2_neuron_group:
                for (const auto& [key, val] : cfg.ngm2.neuron_groups) {
                    if (ImGui::Selectable(key.c_str(), node.entity_cfg == key)) {
                        node.entity_cfg = key;
                        modified = true;
                    }
                }
                break;
            case sim::io_entity_type_t::mnist_input:
                for (const auto& [key, val] : cfg.datasets.mnist) {
                    if (ImGui::Selectable(key.c_str(), node.entity_cfg == key)) {
                        node.entity_cfg = key;
                        modified = true;
                    }
                }
                break;
            case sim::io_entity_type_t::dsprites_input:
                for (const auto& [key, val] : cfg.datasets.dsprites) {
                    if (ImGui::Selectable(key.c_str(), node.entity_cfg == key)) {
                        node.entity_cfg = key;
                        modified = true;
                    }
                }
                break;
            default:
                break;
        }
        ImGui::EndCombo();
    }

    // Inputs list
    if (ImGui::TreeNode("Inputs")) {
        for (std::size_t i = 0; i < node.inputs.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            int src_id = static_cast<int>(node.inputs[i].first);

            ImGui::SetNextItemWidth(80);
            if (ImGui::InputInt("##src", &src_id)) {
                node.inputs[i].first = static_cast<sim::sim_node_id_t>(src_id);
                modified = true;
            }
            ImGui::SameLine();
            if (InputTextStringWidth("##chan", node.inputs[i].second, 120)) {
                modified = true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("X")) {
                node.inputs.erase(node.inputs.begin() + static_cast<std::ptrdiff_t>(i));
                modified = true;
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
        if (ImGui::SmallButton("+ Add Input")) {
            node.inputs.emplace_back(0, "");
            modified = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

// ============================================================================
// config_editor implementation
// ============================================================================

config_editor::config_editor(loader::global_cfg_t& config, const std::string& config_path)
    : config(config)
    , original_config(config)  // Make a copy for restore
    , config_path(config_path)
    , save_file_browser(ImGuiFileBrowserFlags_EnterNewFilename
                       | ImGuiFileBrowserFlags_CloseOnEsc
                       | ImGuiFileBrowserFlags_ConfirmOnEnter
                       | ImGuiFileBrowserFlags_CreateNewDir)
{
    save_file_browser.SetTitle("Save Configuration As");
    save_file_browser.SetTypeFilters({ ".json" });

    // Create network editor (imnodes-based)
    net_editor = std::make_unique<network_editor>(config);
    net_editor->set_dirty_callback([this]() { mark_dirty(); });

    // Create network editor 2 (ImNodeFlow-based, read-only)
    net_editor2 = std::make_unique<network_editor2>(config);
    net_editor2->set_dirty_callback([this]() { mark_dirty(); });
}

void config_editor::render(bool* p_open)
{
    ImGui::SetNextWindowSize(ImVec2(550, 700), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);

    // Window title shows dirty indicator
    // Use ###id to keep stable window ID when title changes
    std::string title = "Configuration Editor";
    if (is_dirty) {
        title += " *";
    }
    title += "###ConfigEditor";

    if (ImGui::Begin(title.c_str(), p_open, ImGuiWindowFlags_MenuBar)) {
        render_menu_bar();
        render_dataset_configs();
        render_ngm2_configs();
        render_simulation_graph();
    }
    ImGui::End();

    // Render file browser (outside main window)
    save_file_browser.Display();
    if (save_file_browser.HasSelected()) {
        pending_save_path = save_file_browser.GetSelected().string();
        // Ensure .json extension
        if (!pending_save_path.ends_with(".json")) {
            pending_save_path += ".json";
        }
        save_file_browser.ClearSelected();

        // Check if file exists - if so, show overwrite confirmation
        if (std::filesystem::exists(pending_save_path)) {
            show_save_as_confirm_dialog = true;
        } else {
            // File doesn't exist, save directly
            if (save_config(pending_save_path)) {
                config_path = pending_save_path;
                original_config = config;
                is_dirty = false;
            }
            pending_save_path.clear();
        }
    }

    // Render confirmation dialogs
    render_save_confirm_dialog();
    render_save_as_confirm_dialog();

    // Render network editor (imnodes-based)
    if (net_editor) {
        net_editor->render(&show_network_editor);
    }

    // Render network editor 2 (ImNodeFlow-based)
    if (net_editor2) {
        net_editor2->render(&show_network_editor2);
    }
}

void config_editor::render_menu_bar()
{
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            // Save - only enabled if dirty
            if (ImGui::MenuItem("Save", "Ctrl+S", false, is_dirty)) {
                show_save_confirm_dialog = true;
                pending_save_path = config_path;
            }

            // Save As - always enabled
            if (ImGui::MenuItem("Save As...", "Ctrl+Shift+S")) {
                save_file_browser.Open();
            }

            ImGui::Separator();

            // Restore - only enabled if dirty
            if (ImGui::MenuItem("Restore", nullptr, false, is_dirty)) {
                restore_config();
            }

            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
}

void config_editor::render_save_confirm_dialog()
{
    if (!show_save_confirm_dialog) return;

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::Begin("Confirm Save", &show_save_confirm_dialog,
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {
        ImGui::Text("Save configuration to:");
        ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "%s", pending_save_path.c_str());
        ImGui::Spacing();

        ImGui::Checkbox("Reload simulation with saved configuration", &reload_after_save);
        ImGui::Spacing();

        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Save", ImVec2(100, 0))) {
            if (save_config(pending_save_path)) {
                original_config = config;
                is_dirty = false;

                if (reload_after_save && reload_callback) {
                    reload_callback(pending_save_path);
                }
            }
            show_save_confirm_dialog = false;
            pending_save_path.clear();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(100, 0))) {
            show_save_confirm_dialog = false;
            pending_save_path.clear();
        }
    }
    ImGui::End();
}

void config_editor::render_save_as_confirm_dialog()
{
    if (!show_save_as_confirm_dialog) return;

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::Begin("Confirm Overwrite", &show_save_as_confirm_dialog,
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "File already exists:");
        ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "%s", pending_save_path.c_str());
        ImGui::Spacing();
        ImGui::Text("Do you want to overwrite it?");
        ImGui::Spacing();

        ImGui::Checkbox("Reload simulation with saved configuration", &reload_after_save_as);
        ImGui::Spacing();

        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Overwrite", ImVec2(100, 0))) {
            if (save_config(pending_save_path)) {
                config_path = pending_save_path;
                original_config = config;
                is_dirty = false;

                if (reload_after_save_as && reload_callback) {
                    reload_callback(pending_save_path);
                }
            }
            show_save_as_confirm_dialog = false;
            pending_save_path.clear();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(100, 0))) {
            show_save_as_confirm_dialog = false;
            pending_save_path.clear();
        }
    }
    ImGui::End();
}

bool config_editor::save_config(const std::string& path)
{
    try {
        nlohmann::json j = config;
        std::ofstream file(path);
        if (!file.is_open()) {
            return false;
        }
        file << j.dump(4);  // Pretty print with 4-space indent
        return true;
    } catch (...) {
        return false;
    }
}

void config_editor::restore_config()
{
    config = original_config;
    is_dirty = false;
}

void config_editor::render_dataset_configs()
{
    if (ImGui::CollapsingHeader("Dataset Configurations", ImGuiTreeNodeFlags_DefaultOpen)) {

        // MNIST configs
        if (ImGui::TreeNode("MNIST Configs")) {
            std::string mnist_to_remove;
            for (auto& [name, cfg] : config.datasets.mnist) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_mnist_config(name, cfg, remove_requested)) {
                    is_dirty = true;
                }
                if (remove_requested) {
                    mnist_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!mnist_to_remove.empty()) {
                config.datasets.mnist.erase(mnist_to_remove);
                is_dirty = true;
            }
            // Add new config
            static std::string new_mnist_name;
            if (InputTextString("##newmnist", new_mnist_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add MNIST Config") && !new_mnist_name.empty()) {
                config.datasets.mnist[new_mnist_name] = {};
                new_mnist_name.clear();
                is_dirty = true;
            }
            ImGui::TreePop();
        }

        // dSprites configs
        if (ImGui::TreeNode("dSprites Configs")) {
            std::string dsprites_to_remove;
            for (auto& [name, cfg] : config.datasets.dsprites) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_dsprites_config(name, cfg, remove_requested)) {
                    is_dirty = true;
                }
                if (remove_requested) {
                    dsprites_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!dsprites_to_remove.empty()) {
                config.datasets.dsprites.erase(dsprites_to_remove);
                is_dirty = true;
            }
            // Add new config
            static std::string new_dsprites_name;
            if (InputTextString("##newdsprites", new_dsprites_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add dSprites Config") && !new_dsprites_name.empty()) {
                config.datasets.dsprites[new_dsprites_name] = {};
                new_dsprites_name.clear();
                is_dirty = true;
            }
            ImGui::TreePop();
        }
    }
}

void config_editor::render_ngm2_configs()
{
    if (ImGui::CollapsingHeader("Neural Model Configurations")) {

        // Dendrite configs
        if (ImGui::TreeNode("Dendrite Configs")) {
            std::string dendrite_to_remove;
            for (auto& [name, cfg] : config.ngm2.dendrites) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_dendrite_config(name, cfg, remove_requested)) {
                    is_dirty = true;
                }
                if (remove_requested) {
                    dendrite_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!dendrite_to_remove.empty()) {
                config.ngm2.dendrites.erase(dendrite_to_remove);
                is_dirty = true;
            }
            static std::string new_dendrite_name;
            if (InputTextString("##newdendrite", new_dendrite_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add Dendrite Config") && !new_dendrite_name.empty()) {
                config.ngm2.dendrites[new_dendrite_name] = {};
                new_dendrite_name.clear();
                is_dirty = true;
            }
            ImGui::TreePop();
        }

        // Neuron configs
        if (ImGui::TreeNode("Neuron Configs")) {
            std::string neuron_to_remove;
            for (auto& [name, cfg] : config.ngm2.neurons) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_neuron_config(name, cfg, config.ngm2, remove_requested)) {
                    is_dirty = true;
                }
                if (remove_requested) {
                    neuron_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!neuron_to_remove.empty()) {
                config.ngm2.neurons.erase(neuron_to_remove);
                is_dirty = true;
            }
            static std::string new_neuron_name;
            if (InputTextString("##newneuron", new_neuron_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add Neuron Config") && !new_neuron_name.empty()) {
                config.ngm2.neurons[new_neuron_name] = {};
                new_neuron_name.clear();
                is_dirty = true;
            }
            ImGui::TreePop();
        }

        // Neuron group configs
        if (ImGui::TreeNode("Neuron Group Configs")) {
            std::string group_to_remove;
            for (auto& [name, cfg] : config.ngm2.neuron_groups) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_neuron_group_config(name, cfg, config.ngm2, remove_requested)) {
                    is_dirty = true;
                }
                if (remove_requested) {
                    group_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!group_to_remove.empty()) {
                config.ngm2.neuron_groups.erase(group_to_remove);
                is_dirty = true;
            }
            static std::string new_group_name;
            if (InputTextString("##newgroup", new_group_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add Group Config") && !new_group_name.empty()) {
                config.ngm2.neuron_groups[new_group_name] = {};
                new_group_name.clear();
                is_dirty = true;
            }
            ImGui::TreePop();
        }
    }
}

void config_editor::render_simulation_graph()
{
    if (ImGui::CollapsingHeader("Simulation Graph")) {
        if (ImGui::TreeNode("Nodes")) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Graph View...")) {
                show_network_editor = true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Graph View 2...")) {
                show_network_editor2 = true;
            }
            for (std::size_t i = 0; i < config.sim.nodes.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                auto& node = config.sim.nodes[i];
                std::string header = std::format("Node {}: {} ({})",
                         i, node.label, get_entity_type_name(node.entity_type));
                if (ImGui::TreeNode("##node", "%s", header.c_str())) {
                    if (render_sim_node(node, config)) {
                        is_dirty = true;
                    }
                    if (ImGui::SmallButton("Remove Node")) {
                        config.sim.nodes.erase(config.sim.nodes.begin() + static_cast<std::ptrdiff_t>(i));
                        is_dirty = true;
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
            if (ImGui::SmallButton("+ Add Node")) {
                config.sim.nodes.push_back({});
                is_dirty = true;
            }
            ImGui::TreePop();
        }
    }
}

} // namespace coast
