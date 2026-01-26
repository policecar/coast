#include "config_editor.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "imgui.h"

#include "mnist_cfg.h"
#include "dsprites_cfg.h"
#include "hd_ngm2_cfg.h"
#include "sim_builder_cfg.h"

namespace coast {

// Helper: Render string list editor
static void render_string_list(const char* label, std::vector<std::string>& list)
{
    if (ImGui::TreeNode(label)) {
        for (std::size_t i = 0; i < list.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            char buf[256];
            strncpy(buf, list[i].c_str(), sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            ImGui::SetNextItemWidth(200);
            if (ImGui::InputText("##item", buf, sizeof(buf))) {
                list[i] = buf;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("X")) {
                list.erase(list.begin() + static_cast<std::ptrdiff_t>(i));
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
        if (ImGui::SmallButton("+ Add")) {
            list.emplace_back("");
        }
        ImGui::TreePop();
    }
}

// Helper: Render MNIST config
static void render_mnist_config(const std::string& name, mnist::mnist_param_t& cfg)
{
    if (ImGui::TreeNode(name.c_str())) {
        char image_buf[512], label_buf[512];
        strncpy(image_buf, cfg.image_file.c_str(), sizeof(image_buf) - 1);
        strncpy(label_buf, cfg.label_file.c_str(), sizeof(label_buf) - 1);
        image_buf[sizeof(image_buf) - 1] = '\0';
        label_buf[sizeof(label_buf) - 1] = '\0';

        if (ImGui::InputText("Image File", image_buf, sizeof(image_buf))) {
            cfg.image_file = image_buf;
        }
        if (ImGui::InputText("Label File", label_buf, sizeof(label_buf))) {
            cfg.label_file = label_buf;
        }

        int change_interval = static_cast<int>(cfg.change_interval);
        if (ImGui::InputInt("Change Interval", &change_interval)) {
            cfg.change_interval = static_cast<std::size_t>(std::max(1, change_interval));
        }

        ImGui::DragFloatRange2("Noise Range", &cfg.noise_range.first, &cfg.noise_range.second,
                               0.01f, 0.0f, 1.0f);
        ImGui::InputInt("Random Seed", &cfg.random_seed);

        ImGui::TreePop();
    }
}

// Helper: Render dSprites config
static void render_dsprites_config(const std::string& name, dsprites::dsprites_param_t& cfg)
{
    if (ImGui::TreeNode(name.c_str())) {
        char file_buf[512];
        strncpy(file_buf, cfg.hdf5_file.c_str(), sizeof(file_buf) - 1);
        file_buf[sizeof(file_buf) - 1] = '\0';

        if (ImGui::InputText("HDF5 File", file_buf, sizeof(file_buf))) {
            cfg.hdf5_file = file_buf;
        }

        int change_interval = static_cast<int>(cfg.change_interval);
        if (ImGui::InputInt("Change Interval", &change_interval)) {
            cfg.change_interval = static_cast<std::size_t>(std::max(1, change_interval));
        }

        ImGui::DragFloatRange2("Noise Range", &cfg.noise_range.first, &cfg.noise_range.second,
                               0.01f, 0.0f, 1.0f);
        ImGui::InputInt("Random Seed", &cfg.random_seed);
        ImGui::Checkbox("Randomize Readout", &cfg.randomize_readout);

        ImGui::TreePop();
    }
}

// Helper: Render dendrite config
static void render_dendrite_config(const std::string& name, ngm2::dendrite_params_t& cfg)
{
    if (ImGui::TreeNode(name.c_str())) {
        // Dendrite type dropdown
        const char* type_names[] = { "Apical", "Proximal" };
        int type_idx = static_cast<int>(cfg.type);
        if (ImGui::Combo("Type", &type_idx, type_names, 2)) {
            cfg.type = static_cast<ngm2::dendrite_type_t>(type_idx);
        }

        render_string_list("Input Channels", cfg.input_channels);

        ImGui::SliderFloat("Permanence Threshold", &cfg.permanence_threshold, 0.0f, 1.0f);

        int max_branch = cfg.max_branch_level;
        if (ImGui::SliderInt("Max Branch Level", &max_branch, 1, 10)) {
            cfg.max_branch_level = static_cast<uint8_t>(max_branch);
        }

        ImGui::InputInt("Random Seed", &cfg.rnd_seed);

        if (ImGui::TreeNode("Learning Parameters")) {
            ImGui::SliderFloat("Primary Learning Rate", &cfg.default_primary_learning_rate, 0.0001f, 0.1f, "%.4f");
            ImGui::SliderFloat("Secondary Learning Rate", &cfg.default_secondary_learning_rate, 0.00001f, 0.01f, "%.5f");
            ImGui::SliderFloat("Mismatch Act Threshold", &cfg.default_mismatch_act_thres, 0.0f, 1.0f);
            ImGui::SliderFloat("Mismatch Smoothing", &cfg.default_mismatch_smoothing, 0.0001f, 0.01f, "%.4f");
            ImGui::SliderFloat("Accumulated Theta Threshold", &cfg.default_accumulated_theta_thres, 0.0f, 10.0f);
            ImGui::SliderFloat("Min Mismatch Deviation", &cfg.default_min_mismatch_deviation, 0.0f, 5.0f);
            ImGui::SliderFloat("Min Mismatch Percentage", &cfg.default_min_mismatch_percentage, 0.0f, 0.1f, "%.4f");
            ImGui::TreePop();
        }

        ImGui::TreePop();
    }
}

// Helper: Render dendrite set
static void render_dendrite_set(ngm2::dendrite_set_t& ds, const ngm2::cfg_store& store)
{
    char name_buf[128];
    strncpy(name_buf, ds.name.c_str(), sizeof(name_buf) - 1);
    name_buf[sizeof(name_buf) - 1] = '\0';
    if (ImGui::InputText("Name", name_buf, sizeof(name_buf))) {
        ds.name = name_buf;
    }

    // Dropdown for dendrite_cfg from available dendrite configs
    if (ImGui::BeginCombo("Dendrite Config", ds.dendrite_cfg.c_str())) {
        for (const auto& [key, val] : store.dendrites) {
            bool selected = (ds.dendrite_cfg == key);
            if (ImGui::Selectable(key.c_str(), selected)) {
                ds.dendrite_cfg = key;
            }
        }
        ImGui::EndCombo();
    }

    int count = static_cast<int>(ds.dendrite_count);
    if (ImGui::InputInt("Count", &count)) {
        ds.dendrite_count = static_cast<std::size_t>(std::max(1, count));
    }
}

// Helper: Render channel mapping
static void render_channel_mapping(ngm2::channel_mapping_t& cm)
{
    char from_buf[128], to_name_buf[128], to_chan_buf[128];
    strncpy(from_buf, cm.from_channel.c_str(), sizeof(from_buf) - 1);
    strncpy(to_name_buf, cm.to_name.c_str(), sizeof(to_name_buf) - 1);
    strncpy(to_chan_buf, cm.to_channel.c_str(), sizeof(to_chan_buf) - 1);
    from_buf[sizeof(from_buf) - 1] = '\0';
    to_name_buf[sizeof(to_name_buf) - 1] = '\0';
    to_chan_buf[sizeof(to_chan_buf) - 1] = '\0';

    ImGui::SetNextItemWidth(100);
    if (ImGui::InputText("From", from_buf, sizeof(from_buf))) {
        cm.from_channel = from_buf;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputText("To Name", to_name_buf, sizeof(to_name_buf))) {
        cm.to_name = to_name_buf;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputText("To Channel", to_chan_buf, sizeof(to_chan_buf))) {
        cm.to_channel = to_chan_buf;
    }
}

// Helper: Render neuron config
static void render_neuron_config(const std::string& name, ngm2::neuron_params_t& cfg,
                                  const ngm2::cfg_store& store)
{
    if (ImGui::TreeNode(name.c_str())) {
        render_string_list("Input Channels", cfg.input_channels);

        // Dendrite sets
        if (ImGui::TreeNode("Dendrite Sets")) {
            for (std::size_t i = 0; i < cfg.dendrites.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::TreeNode("##ds", "Set %zu", i + 1)) {
                    render_dendrite_set(cfg.dendrites[i], store);
                    if (ImGui::SmallButton("Remove")) {
                        cfg.dendrites.erase(cfg.dendrites.begin() + static_cast<std::ptrdiff_t>(i));
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
            }
            ImGui::TreePop();
        }

        // Channel mappings
        if (ImGui::TreeNode("Channel Mappings")) {
            for (std::size_t i = 0; i < cfg.channel_mappings.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                render_channel_mapping(cfg.channel_mappings[i]);
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) {
                    cfg.channel_mappings.erase(cfg.channel_mappings.begin() + static_cast<std::ptrdiff_t>(i));
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
            if (ImGui::SmallButton("+ Add Mapping")) {
                cfg.channel_mappings.push_back({});
            }
            ImGui::TreePop();
        }

        int branch_interval = static_cast<int>(cfg.default_branch_interval);
        if (ImGui::InputInt("Branch Interval", &branch_interval)) {
            cfg.default_branch_interval = static_cast<std::size_t>(std::max(1, branch_interval));
        }

        if (ImGui::TreeNode("Activity Learning Window")) {
            ImGui::SliderFloat("Pre Steepness", &cfg.default_activity_learning_window.first.steepness, 0.0f, 1.0f);
            ImGui::SliderFloat("Pre Transition", &cfg.default_activity_learning_window.first.transition_point, 0.0f, 1.0f);
            ImGui::SliderFloat("Post Steepness", &cfg.default_activity_learning_window.second.steepness, 0.0f, 1.0f);
            ImGui::SliderFloat("Post Transition", &cfg.default_activity_learning_window.second.transition_point, 0.0f, 1.0f);
            ImGui::TreePop();
        }

        ImGui::InputInt("Random Seed", &cfg.random_seed);

        ImGui::TreePop();
    }
}

// Helper: Render neuron set
static void render_neuron_set(ngm2::neuron_set_t& ns, const ngm2::cfg_store& store)
{
    char name_buf[128];
    strncpy(name_buf, ns.name.c_str(), sizeof(name_buf) - 1);
    name_buf[sizeof(name_buf) - 1] = '\0';
    if (ImGui::InputText("Name", name_buf, sizeof(name_buf))) {
        ns.name = name_buf;
    }

    // Dropdown for neuron_cfg
    if (ImGui::BeginCombo("Neuron Config", ns.neuron_cfg.c_str())) {
        for (const auto& [key, val] : store.neurons) {
            bool selected = (ns.neuron_cfg == key);
            if (ImGui::Selectable(key.c_str(), selected)) {
                ns.neuron_cfg = key;
            }
        }
        ImGui::EndCombo();
    }

    int count = static_cast<int>(ns.neuron_count);
    if (ImGui::InputInt("Count", &count)) {
        ns.neuron_count = static_cast<std::size_t>(std::max(1, count));
    }
}

// Helper: Render neuron group config
static void render_neuron_group_config(const std::string& name, ngm2::neuron_group_params_t& cfg,
                                        const ngm2::cfg_store& store)
{
    if (ImGui::TreeNode(name.c_str())) {
        render_string_list("Input Channels", cfg.input_channels);

        // Neuron sets
        if (ImGui::TreeNode("Neuron Sets")) {
            for (std::size_t i = 0; i < cfg.neurons.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::TreeNode("##ns", "Set %zu", i + 1)) {
                    render_neuron_set(cfg.neurons[i], store);
                    if (ImGui::SmallButton("Remove")) {
                        cfg.neurons.erase(cfg.neurons.begin() + static_cast<std::ptrdiff_t>(i));
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
            }
            ImGui::TreePop();
        }

        // Channel mappings
        if (ImGui::TreeNode("Channel Mappings")) {
            for (std::size_t i = 0; i < cfg.channel_mappings.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                render_channel_mapping(cfg.channel_mappings[i]);
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) {
                    cfg.channel_mappings.erase(cfg.channel_mappings.begin() + static_cast<std::ptrdiff_t>(i));
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
            if (ImGui::SmallButton("+ Add Mapping")) {
                cfg.channel_mappings.push_back({});
            }
            ImGui::TreePop();
        }

        ImGui::SliderFloat("Local Inhibition", &cfg.default_local_inhibition_strength, 0.1f, 20.0f);
        ImGui::SliderFloat("Common Learning Rate", &cfg.default_common_learning_rate, 0.00001f, 0.01f, "%.5f");

        if (ImGui::TreeNode("Weight Filter")) {
            ImGui::SliderFloat("Steepness", &cfg.default_weight_filter.steepness, 0.0f, 1.0f);
            ImGui::SliderFloat("Transition Point", &cfg.default_weight_filter.transition_point, 0.0f, 1.0f);
            ImGui::TreePop();
        }

        ImGui::SliderFloat("Stochastic Win Threshold", &cfg.default_stochastic_win_thres, 0.0f, 1.0f);
        ImGui::InputInt("Random Seed", &cfg.random_seed);

        ImGui::TreePop();
    }
}

// Helper: Get entity type name
static const char* get_entity_type_name(sim::io_entity_type_t type)
{
    switch (type) {
        case sim::io_entity_type_t::ngm2_neuron_group: return "Neuron Group";
        case sim::io_entity_type_t::mnist_input: return "MNIST Input";
        case sim::io_entity_type_t::dsprites_input: return "dSprites Input";
        default: return "Unknown";
    }
}

// Helper: Render simulation node
static void render_sim_node(sim::sim_node_t& node, const loader::global_cfg_t& cfg)
{
    // ID
    int id = static_cast<int>(node.id);
    if (ImGui::InputInt("ID", &id)) {
        node.id = static_cast<sim::sim_node_id_t>(id);
    }

    // Label
    char label_buf[128];
    strncpy(label_buf, node.label.c_str(), sizeof(label_buf) - 1);
    label_buf[sizeof(label_buf) - 1] = '\0';
    if (ImGui::InputText("Label", label_buf, sizeof(label_buf))) {
        node.label = label_buf;
    }

    // Entity type dropdown
    const char* type_names[] = { "Neuron Group", "MNIST Input", "dSprites Input" };
    int type_idx = static_cast<int>(node.entity_type);
    if (ImGui::Combo("Entity Type", &type_idx, type_names, 3)) {
        node.entity_type = static_cast<sim::io_entity_type_t>(type_idx);
    }

    // Entity config dropdown - show configs from appropriate store
    const char* current_cfg = node.entity_cfg.c_str();
    if (ImGui::BeginCombo("Entity Config", current_cfg)) {
        switch (node.entity_type) {
            case sim::io_entity_type_t::ngm2_neuron_group:
                for (const auto& [key, val] : cfg.ngm2.neuron_groups) {
                    if (ImGui::Selectable(key.c_str(), node.entity_cfg == key)) {
                        node.entity_cfg = key;
                    }
                }
                break;
            case sim::io_entity_type_t::mnist_input:
                for (const auto& [key, val] : cfg.datasets.mnist) {
                    if (ImGui::Selectable(key.c_str(), node.entity_cfg == key)) {
                        node.entity_cfg = key;
                    }
                }
                break;
            case sim::io_entity_type_t::dsprites_input:
                for (const auto& [key, val] : cfg.datasets.dsprites) {
                    if (ImGui::Selectable(key.c_str(), node.entity_cfg == key)) {
                        node.entity_cfg = key;
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
            char chan_buf[128];
            strncpy(chan_buf, node.inputs[i].second.c_str(), sizeof(chan_buf) - 1);
            chan_buf[sizeof(chan_buf) - 1] = '\0';

            ImGui::SetNextItemWidth(80);
            if (ImGui::InputInt("##src", &src_id)) {
                node.inputs[i].first = static_cast<sim::sim_node_id_t>(src_id);
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120);
            if (ImGui::InputText("##chan", chan_buf, sizeof(chan_buf))) {
                node.inputs[i].second = chan_buf;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("X")) {
                node.inputs.erase(node.inputs.begin() + static_cast<std::ptrdiff_t>(i));
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
        if (ImGui::SmallButton("+ Add Input")) {
            node.inputs.emplace_back(0, "");
        }
        ImGui::TreePop();
    }
}

// ============================================================================
// config_editor implementation
// ============================================================================

config_editor::config_editor(loader::global_cfg_t& config)
    : config(config)
{
}

void config_editor::render(bool* p_open)
{
    ImGui::SetNextWindowSize(ImVec2(550, 700), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Configuration Editor", p_open)) {
        render_dataset_configs();
        render_ngm2_configs();
        render_simulation_graph();
    }
    ImGui::End();
}

void config_editor::render_dataset_configs()
{
    if (ImGui::CollapsingHeader("Dataset Configurations", ImGuiTreeNodeFlags_DefaultOpen)) {

        // MNIST configs
        if (ImGui::TreeNode("MNIST Configs")) {
            for (auto& [name, cfg] : config.datasets.mnist) {
                ImGui::PushID(name.c_str());
                render_mnist_config(name, cfg);
                ImGui::PopID();
            }
            // Add new config
            static char new_mnist_name[64] = "";
            ImGui::InputText("##newmnist", new_mnist_name, sizeof(new_mnist_name));
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add MNIST Config") && new_mnist_name[0] != '\0') {
                config.datasets.mnist[new_mnist_name] = {};
                new_mnist_name[0] = '\0';
            }
            ImGui::TreePop();
        }

        // dSprites configs
        if (ImGui::TreeNode("dSprites Configs")) {
            for (auto& [name, cfg] : config.datasets.dsprites) {
                ImGui::PushID(name.c_str());
                render_dsprites_config(name, cfg);
                ImGui::PopID();
            }
            // Add new config
            static char new_dsprites_name[64] = "";
            ImGui::InputText("##newdsprites", new_dsprites_name, sizeof(new_dsprites_name));
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add dSprites Config") && new_dsprites_name[0] != '\0') {
                config.datasets.dsprites[new_dsprites_name] = {};
                new_dsprites_name[0] = '\0';
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
            for (auto& [name, cfg] : config.ngm2.dendrites) {
                ImGui::PushID(name.c_str());
                render_dendrite_config(name, cfg);
                ImGui::PopID();
            }
            static char new_dendrite_name[64] = "";
            ImGui::InputText("##newdendrite", new_dendrite_name, sizeof(new_dendrite_name));
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add Dendrite Config") && new_dendrite_name[0] != '\0') {
                config.ngm2.dendrites[new_dendrite_name] = {};
                new_dendrite_name[0] = '\0';
            }
            ImGui::TreePop();
        }

        // Neuron configs
        if (ImGui::TreeNode("Neuron Configs")) {
            for (auto& [name, cfg] : config.ngm2.neurons) {
                ImGui::PushID(name.c_str());
                render_neuron_config(name, cfg, config.ngm2);
                ImGui::PopID();
            }
            static char new_neuron_name[64] = "";
            ImGui::InputText("##newneuron", new_neuron_name, sizeof(new_neuron_name));
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add Neuron Config") && new_neuron_name[0] != '\0') {
                config.ngm2.neurons[new_neuron_name] = {};
                new_neuron_name[0] = '\0';
            }
            ImGui::TreePop();
        }

        // Neuron group configs
        if (ImGui::TreeNode("Neuron Group Configs")) {
            for (auto& [name, cfg] : config.ngm2.neuron_groups) {
                ImGui::PushID(name.c_str());
                render_neuron_group_config(name, cfg, config.ngm2);
                ImGui::PopID();
            }
            static char new_group_name[64] = "";
            ImGui::InputText("##newgroup", new_group_name, sizeof(new_group_name));
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add Group Config") && new_group_name[0] != '\0') {
                config.ngm2.neuron_groups[new_group_name] = {};
                new_group_name[0] = '\0';
            }
            ImGui::TreePop();
        }
    }
}

void config_editor::render_simulation_graph()
{
    if (ImGui::CollapsingHeader("Simulation Graph")) {
        if (ImGui::TreeNode("Nodes")) {
            for (std::size_t i = 0; i < config.sim.nodes.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                auto& node = config.sim.nodes[i];
                char header[128];
                snprintf(header, sizeof(header), "Node %zu: %s (%s)",
                         i, node.label.c_str(), get_entity_type_name(node.entity_type));
                if (ImGui::TreeNode("##node", "%s", header)) {
                    render_sim_node(node, config);
                    if (ImGui::SmallButton("Remove Node")) {
                        config.sim.nodes.erase(config.sim.nodes.begin() + static_cast<std::ptrdiff_t>(i));
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
            }
            ImGui::TreePop();
        }
    }
}

} // namespace coast
