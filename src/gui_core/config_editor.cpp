#include "gui_core/config_editor.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "io/mnist_cfg.h"
#include "io/dsprites_cfg.h"
#include "io/emnist_cfg.h"
#include "io/kmnist_cfg.h"
#include "io/cifar10_cfg.h"
#include "io/cifar100_cfg.h"
#include "io/chairs3d_cfg.h"
#include "io/color_mnist_cfg.h"
#include "hd_ngm2/hd_ngm2_cfg.h"
#include "inhibition/inhibition_cfg.h"
#include "tools/sim_builder_cfg.h"

// Include JSON serialization headers
#include "json_cfg/ngm2_json.h"
#include "json_cfg/ngm3_json.h"
#include "json_cfg/ngm25_json.h"
#include "json_cfg/sim_json.h"
#include "json_cfg/dataset_json.h"
#include "json_cfg/retina_json.h"
#include "retina/retina.h"
#include "json_cfg/vis_hints_json.h"

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

// Helper: SliderFloat wrapper for double values
bool SliderDouble(const char* label, double& val, float v_min, float v_max, const char* format = "%.3f")
{
    float temp = static_cast<float>(val);
    if (ImGui::SliderFloat(label, &temp, v_min, v_max, format)) {
        val = static_cast<double>(temp);
        return true;
    }
    return false;
}

// Helper: DragFloatRange2 wrapper for double pair values
bool DragDoubleRange2(const char* label, double& v_min, double& v_max,
                      float v_speed, float min_limit, float max_limit)
{
    float temp_min = static_cast<float>(v_min);
    float temp_max = static_cast<float>(v_max);
    if (ImGui::DragFloatRange2(label, &temp_min, &temp_max, v_speed, min_limit, max_limit)) {
        v_min = static_cast<double>(temp_min);
        v_max = static_cast<double>(temp_max);
        return true;
    }
    return false;
}

// Helper: InputScalar wrapper for a seed_t or seed_offset_t, edited as its uint32
// value (core/seed.h)
template <class Seed>
bool InputSeed(const char* label, Seed& seed)
{
    const std::uint32_t step = 1;
    auto v = std::to_underlying(seed);
    if (ImGui::InputScalar(label, ImGuiDataType_U32, &v, &step)) {
        seed = Seed{ v };
        return true;
    }
    return false;
}

// Helper: Get scaled width for input widgets (accounts for HiDPI font scaling)
float scaled_width(float base_width)
{
    return base_width * (ImGui::GetFontSize() / 16.0f);
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

// Helper: Render the sequencing fields every sequenced dataset loader shares.
// All seven loaders publicly derive from dataset::sequence_params_t, so they pass
// their cfg here by base reference. Rendering the block once keeps a new
// cross-cutting field a single edit — mirroring the single serialiser in
// dataset_json.h. chairs3d is deliberately excluded (no sequencer) and keeps its
// own inline fields.
static bool render_sequence_params(dataset::sequence_params_t& cfg)
{
    bool modified = false;

    int change_interval = static_cast<int>(cfg.change_interval);
    if (ImGui::InputInt("Change Interval", &change_interval)) {
        cfg.change_interval = static_cast<std::size_t>(std::max(1, change_interval));
        modified = true;
    }

    if (DragDoubleRange2("Noise Range", cfg.noise_range.first, cfg.noise_range.second,
                           0.01f, 0.0f, 1.0f)) {
        modified = true;
    }
    if (InputSeed("Random Seed", cfg.random_seed)) {
        modified = true;
    }
    if (ImGui::Checkbox("Shuffle", &cfg.shuffle)) {
        modified = true;
    }

    if (InputTextString("Index Manifest", cfg.index_manifest)) {
        modified = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Path to a text file of sample indices restricting presentation\n"
                          "to that subset. Empty = full dataset.");
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

        if (render_sequence_params(cfg)) {
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

        if (render_sequence_params(cfg)) {
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

// Helper: Render EMNIST config (returns true if modified, sets remove_requested if remove clicked)
static bool render_emnist_config(const std::string& name, emnist::emnist_param_t& cfg, bool& remove_requested)
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

        if (render_sequence_params(cfg)) {
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

// Helper: Render KMNIST config (returns true if modified, sets remove_requested if remove clicked)
static bool render_kmnist_config(const std::string& name, kmnist::kmnist_param_t& cfg, bool& remove_requested)
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

        if (render_sequence_params(cfg)) {
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

// Helper: Render CIFAR-10 config (returns true if modified, sets remove_requested if remove clicked)
static bool render_cifar10_config(const std::string& name, cifar10::cifar10_param_t& cfg, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        if (InputTextString("Data Directory", cfg.data_dir)) {
            modified = true;
        }

        if (render_sequence_params(cfg)) {
            modified = true;
        }
        if (ImGui::Checkbox("Use Test Set", &cfg.use_test_set)) {
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

// Helper: Render CIFAR-100 config (returns true if modified, sets remove_requested if remove clicked)
static bool render_cifar100_config(const std::string& name, cifar100::cifar100_param_t& cfg, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        if (InputTextString("Data Directory", cfg.data_dir)) {
            modified = true;
        }

        if (render_sequence_params(cfg)) {
            modified = true;
        }
        if (ImGui::Checkbox("Use Test Set", &cfg.use_test_set)) {
            modified = true;
        }
        if (ImGui::Checkbox("Use Coarse Labels", &cfg.use_coarse_labels)) {
            modified = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Use 20 superclasses instead of 100 fine classes");
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove Config")) {
            remove_requested = true;
        }

        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render retina config (returns true if modified, sets remove_requested if remove clicked)
static bool render_retina_config(const std::string& name, retina::retina_param_t& cfg, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        if (ImGui::InputInt("Width", &cfg.width))       { cfg.width    = std::max(1, cfg.width);    modified = true; }
        if (ImGui::InputInt("Height", &cfg.height))     { cfg.height   = std::max(1, cfg.height);   modified = true; }
        if (ImGui::InputInt("Channels", &cfg.channels)) { cfg.channels = std::max(1, cfg.channels); modified = true; }
        if (ImGui::Checkbox("Luminance (RGB -> Y)", &cfg.luminance)) modified = true;
        int pol = cfg.polarity == retina::polarity_t::off ? 1 : 0;
        if (ImGui::Combo("Polarity", &pol, "on\0off\0")) {
            cfg.polarity = pol == 1 ? retina::polarity_t::off : retina::polarity_t::on;
            modified = true;
        }
        // unclamped, like k: validate() is the range authority (see below)
        if (ImGui::InputDouble("Center Sigma", &cfg.center_sigma, 0.1, 1.0, "%.3f"))     modified = true;
        if (ImGui::InputDouble("Surround Sigma", &cfg.surround_sigma, 0.1, 1.0, "%.3f")) modified = true;
        // unclamped: validate() accepts any k >= 0, and a slider would snap k > 1 back
        if (ImGui::InputDouble("Surround Weight (k)", &cfg.surround_weight, 0.05, 0.1, "%.3f")) modified = true;
        if (SliderDouble("Gain", cfg.gain, 0.1f, 20.0f)) modified = true;
        if (ImGui::Checkbox("Divisive Normalisation", &cfg.normalize)) modified = true;
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("d = gain * x / (norm_sigma + RMS(x)), per output plane");
        }
        ImGui::BeginDisabled(!cfg.normalize);
        // unclamped, like the sigmas: validate() is the range authority
        if (ImGui::InputDouble("Norm Sigma", &cfg.norm_sigma, 0.01, 0.1, "%.4f")) modified = true;
        ImGui::EndDisabled();
        if (DragDoubleRange2("Noise Range", cfg.noise_range.first, cfg.noise_range.second,
                               0.001f, 0.0f, static_cast<float>(retina::max_noise))) {
            modified = true;
        }
        if (InputSeed("Random Seed", cfg.random_seed)) modified = true;

        // validate() is the one authority on what the node accepts; show its verdict
        // here instead of mirroring its ranges in the widgets.
        try {
            retina::validate(cfg);
        } catch (const std::invalid_argument& e) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", e.what());
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove Config")) {
            remove_requested = true;
        }

        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render 3D Chairs config (returns true if modified, sets remove_requested if remove clicked)
static bool render_chairs3d_config(const std::string& name, chairs3d::chairs3d_param_t& cfg, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        if (InputTextString("Data Directory", cfg.data_dir)) {
            modified = true;
        }

        if (ImGui::InputInt("Image Width", &cfg.image_width)) {
            cfg.image_width = std::max(1, cfg.image_width);
            modified = true;
        }
        if (ImGui::InputInt("Image Height", &cfg.image_height)) {
            cfg.image_height = std::max(1, cfg.image_height);
            modified = true;
        }

        int change_interval = static_cast<int>(cfg.change_interval);
        if (ImGui::InputInt("Change Interval", &change_interval)) {
            cfg.change_interval = static_cast<std::size_t>(std::max(1, change_interval));
            modified = true;
        }

        if (DragDoubleRange2("Noise Range", cfg.noise_range.first, cfg.noise_range.second,
                               0.01f, 0.0f, 1.0f)) {
            modified = true;
        }
        if (InputSeed("Random Seed", cfg.random_seed)) {
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

// Helper: Render Color MNIST config (returns true if modified, sets remove_requested if remove clicked)
static bool render_color_mnist_config(const std::string& name, color_mnist::color_mnist_param_t& cfg, bool& remove_requested)
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

        if (render_sequence_params(cfg)) {
            modified = true;
        }

        ImGui::TextDisabled("(Colors digits randomly as Red, Green, or Blue)");

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

        if (SliderDouble("Permanence Threshold", cfg.permanence_threshold, 0.0f, 1.0f)) {
            modified = true;
        }

        int max_branch = cfg.max_branch_level;
        if (ImGui::SliderInt("Max Branch Level", &max_branch, 0, 4)) {
            cfg.max_branch_level = static_cast<uint8_t>(max_branch);
            modified = true;
        }

        if (InputSeed("Random Seed", cfg.rnd_seed)) {
            modified = true;
        }

        if (ImGui::TreeNode("Learning Parameters")) {
            if (SliderDouble("Primary Learning Rate", cfg.default_primary_learning_rate, 0.0001f, 0.1f, "%.4f")) {
                modified = true;
            }
            if (SliderDouble("Secondary Learning Rate", cfg.default_secondary_learning_rate, 0.00001f, 0.01f, "%.5f")) {
                modified = true;
            }
            if (SliderDouble("Mismatch Act Threshold", cfg.default_mismatch_act_thres, 0.0f, 1.0f)) {
                modified = true;
            }
            if (SliderDouble("Mismatch Smoothing", cfg.default_mismatch_smoothing, 0.0001f, 0.01f, "%.4f")) {
                modified = true;
            }
            if (SliderDouble("Accumulated Theta Threshold", cfg.default_accumulated_theta_thres, 0.0f, 10.0f)) {
                modified = true;
            }
            if (SliderDouble("Min Mismatch Deviation", cfg.default_min_mismatch_deviation, 0.0f, 5.0f)) {
                modified = true;
            }
            if (SliderDouble("Min Mismatch Percentage", cfg.default_min_mismatch_percentage, 0.0f, 0.1f, "%.4f")) {
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

        if (InputSeed("Random Seed", cfg.random_seed)) {
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
// ---- inhibition (shared between ngm2 and ngm3; same inhibition_params_t) ----

static const char* inhibition_type_name(inhibition::inhibition_type_t t)
{
    switch (t) {
        case inhibition::inhibition_type_t::random_index: return "random_index";
        case inhibition::inhibition_type_t::max_softmax: return "max_softmax";
        case inhibition::inhibition_type_t::TYPE_COUNT: break;
    }
    return "?";
}

static const char* random_index_variant_name(inhibition::random_index_variant_t v)
{
    switch (v) {
        case inhibition::random_index_variant_t::threshold_linear: return "threshold_linear";
        case inhibition::random_index_variant_t::TYPE_COUNT: break;
    }
    return "?";
}

// Render one named inhibition config entry (type combo + type-specific fields).
static bool render_inhibition_config(const std::string& name,
                                     inhibition::inhibition_params_t& cfg,
                                     bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        if (ImGui::BeginCombo("Type", inhibition_type_name(cfg.type))) {
            for (int i = 0; i < static_cast<int>(inhibition::inhibition_type_t::TYPE_COUNT); ++i) {
                const auto t = static_cast<inhibition::inhibition_type_t>(i);
                const bool selected = (cfg.type == t);
                if (ImGui::Selectable(inhibition_type_name(t), selected)) { cfg.type = t; modified = true; }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        // type-specific fields
        if (cfg.type == inhibition::inhibition_type_t::max_softmax) {
            if (SliderDouble("Beta", cfg.strength, 0.1f, 50.0f)) modified = true;
        }
        if (cfg.type == inhibition::inhibition_type_t::random_index) {
            if (ImGui::BeginCombo("Variant", random_index_variant_name(cfg.ri_variant))) {
                for (int i = 0; i < static_cast<int>(inhibition::random_index_variant_t::TYPE_COUNT); ++i) {
                    const auto v = static_cast<inhibition::random_index_variant_t>(i);
                    const bool selected = (cfg.ri_variant == v);
                    if (ImGui::Selectable(random_index_variant_name(v), selected)) { cfg.ri_variant = v; modified = true; }
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (SliderDouble("Reach", cfg.ri_reach, 0.0f, 1.0f)) modified = true;
            if (SliderDouble("RI Strength", cfg.ri_strength, 0.0f, 1.0f)) modified = true;
            if (InputSeed("Seed", cfg.ri_seed)) modified = true;
            if (ImGui::Checkbox("Seed from time", &cfg.ri_seed_from_time)) modified = true;
        }
        ImGui::Separator();
        if (ImGui::SmallButton("Remove Config")) remove_requested = true;
        ImGui::TreePop();
    }
    return modified;
}

// Name-reference combo for a neuron group's inhibition_cfg. The empty/default
// entry "(legacy strength)" selects the synthesized max_softmax (beta = the group's
// local inhibition strength).
static bool render_inhibition_ref_combo(
    std::string& ref,
    const std::unordered_map<std::string, inhibition::inhibition_params_t>& dict)
{
    bool modified = false;
    const char* preview = ref.empty() ? "(legacy strength)" : ref.c_str();
    if (ImGui::BeginCombo("Inhibition Config", preview)) {
        if (ImGui::Selectable("(legacy strength)", ref.empty())) { ref.clear(); modified = true; }
        for (const auto& [key, val] : dict) {
            const bool selected = (ref == key);
            if (ImGui::Selectable(key.c_str(), selected)) { ref = key; modified = true; }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return modified;
}

// Render a named inhibition-config dictionary (iterate / edit / remove / add).
static bool render_inhibition_dict(
    std::unordered_map<std::string, inhibition::inhibition_params_t>& dict,
    const char* add_id)
{
    bool modified = false;
    std::string to_remove;
    for (auto& [name, cfg] : dict) {
        ImGui::PushID(name.c_str());
        bool remove_requested = false;
        if (render_inhibition_config(name, cfg, remove_requested)) modified = true;
        if (remove_requested) to_remove = name;
        ImGui::PopID();
    }
    if (!to_remove.empty()) { dict.erase(to_remove); modified = true; }
    static std::unordered_map<std::string, std::string> new_names;  // per add_id buffer
    std::string& new_name = new_names[add_id];
    ImGui::PushID(add_id);
    InputTextString("##newinhib", new_name, 64);
    ImGui::SameLine();
    if (ImGui::SmallButton("+ Add Inhibition Config") && !new_name.empty()) {
        dict[new_name] = {};
        new_name.clear();
        modified = true;
    }
    ImGui::PopID();
    return modified;
}

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

        // The slider only feeds the synthesized max_softmax on the legacy/empty
        // selection; a named config supplies its own strength, so grey the
        // slider out then rather than leave a draggable no-op.
        ImGui::BeginDisabled(!cfg.inhibition_cfg.empty());
        if (SliderDouble("Local Inhibition (beta)", cfg.default_local_inhibition_strength, 0.1f, 50.0f)) {
            modified = true;
        }
        ImGui::EndDisabled();
        // Inhibition selector: empty => "(legacy strength)" uses the slider above
        // (synthesized max_softmax); a named entry overrides it.
        if (render_inhibition_ref_combo(cfg.inhibition_cfg, store.inhibitions)) {
            modified = true;
        }
        if (SliderDouble("Common Learning Rate", cfg.default_common_learning_rate, 0.00001f, 0.01f, "%.5f")) {
            modified = true;
        }

        if (ImGui::TreeNode("Weight Filter")) {
            if (SliderDouble("Steepness", cfg.default_weight_filter.steepness, 0.0f, 0.999f)) {
                modified = true;
            }
            if (SliderDouble("Transition Point", cfg.default_weight_filter.transition_point, 0.0f, 1.0f)) {
                modified = true;
            }
            ImGui::TreePop();
        }

        if (SliderDouble("Stochastic Win Threshold", cfg.default_stochastic_win_thres, 0.0f, 1.0f)) {
            modified = true;
        }
        if (InputSeed("Random Seed", cfg.random_seed)) {
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

// ---- ngm3 ------------------------------------------------------------------

// One input-route row, shared by the ngm3 and the ngm2.5 form (both use
// ngm::input_route_t, ngm/bank.h).
static bool render_input_route(ngm::input_route_t& r)
{
    bool modified = false;
    if (InputTextStringWidth("##channel_name", r.channel_name, 200)) modified = true;
    ImGui::SameLine();
    int bank_idx = (r.bank == ngm::bank_t::apical) ? 0 : 1;
    static constexpr std::array bank_labels = { "apical", "proximal" };
    if (ImGui::Combo("##bank", &bank_idx, bank_labels.data(),
                     static_cast<int>(bank_labels.size())))
    {
        r.bank = (bank_idx == 0) ? ngm::bank_t::apical : ngm::bank_t::proximal;
        modified = true;
    }
    return modified;
}

static bool render_ngm3_group_config(const std::string& name,
                                     ngm3::neuron_group_params_t& cfg,
                                     const ngm3::cfg_store& store,
                                     bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {

        // structural ints
        int neuron_cnt    = static_cast<int>(cfg.neuron_cnt);
        int apical_cnt    = static_cast<int>(cfg.apical_cnt);
        int proximal_cnt  = static_cast<int>(cfg.proximal_cnt);
        int branch_depth  = static_cast<int>(cfg.max_branch_level);
        if (ImGui::InputInt("Neuron Count",        &neuron_cnt))   { cfg.neuron_cnt   = static_cast<std::size_t>(std::max(0, neuron_cnt));   modified = true; }
        if (ImGui::InputInt("Apical Count",        &apical_cnt))   { cfg.apical_cnt   = static_cast<std::size_t>(std::max(0, apical_cnt));   modified = true; }
        if (ImGui::InputInt("Proximal Count",      &proximal_cnt)) { cfg.proximal_cnt = static_cast<std::size_t>(std::max(1, proximal_cnt)); modified = true; }
        if (ImGui::SliderInt("Branch Depth (1-4)", &branch_depth, 1, 4)) { cfg.max_branch_level = static_cast<uint8_t>(branch_depth); modified = true; }

        // algorithmic floats
        if (ImGui::SliderFloat("Permanence Threshold",     &cfg.permanence_threshold,     0.0f, 1.0f))                modified = true;
        // Slider only feeds the synthesized max_softmax on the empty selection; a
        // named config overrides it, so grey it out then (see ngm2 path above).
        ImGui::BeginDisabled(!cfg.inhibition_cfg.empty());
        if (ImGui::SliderFloat("Local Inhibition (beta)",  &cfg.local_inhibition_strength, 0.1f, 50.0f))               modified = true;
        ImGui::EndDisabled();
        // Inhibition selector: empty => "(legacy strength)" uses the slider above.
        if (render_inhibition_ref_combo(cfg.inhibition_cfg, store.inhibitions))                                       modified = true;
        if (ImGui::SliderFloat("Stochastic Win Threshold", &cfg.stochastic_win_thres,      0.0f, 1.0f))                modified = true;
        if (ImGui::SliderFloat("Common Learning Rate",     &cfg.common_learning_rate,      0.00001f, 0.01f, "%.5f"))   modified = true;
        if (ImGui::SliderFloat("Primary Learning Rate",    &cfg.primary_learning_rate,     0.0001f, 0.1f,  "%.5f"))    modified = true;
        if (ImGui::SliderFloat("Secondary Learning Rate",  &cfg.secondary_learning_rate,   0.00001f, 0.01f,"%.5f"))    modified = true;
        if (ImGui::SliderFloat("Mismatch Act Threshold",   &cfg.mismatch_act_thres,        0.0f, 1.0f))                modified = true;
        if (ImGui::SliderFloat("Mismatch Smoothing",       &cfg.mismatch_smoothing,        0.0001f, 0.1f, "%.5f"))     modified = true;

        int branch_interval = static_cast<int>(cfg.branch_interval);
        if (ImGui::InputInt("Branch Interval", &branch_interval)) { cfg.branch_interval = static_cast<std::size_t>(std::max(1, branch_interval)); modified = true; }

        if (ImGui::SliderFloat("Min Mismatch Deviation",   &cfg.min_mismatch_deviation,   0.0f, 5.0f))                 modified = true;
        if (ImGui::SliderFloat("Accumulated Theta Thres",  &cfg.accumulated_theta_thres,  0.0f, 10.0f))                modified = true;
        if (ImGui::SliderFloat("Min Mismatch Percentage",  &cfg.min_mismatch_percentage,  0.0f, 0.1f, "%.4f"))         modified = true;


        // weight filter
        if (ImGui::TreeNode("Weight Filter")) {
            if (SliderDouble("Steepness",         cfg.weight_filter.steepness,         0.0f, 0.999f)) modified = true;
            if (SliderDouble("Transition Point",  cfg.weight_filter.transition_point,  0.0f, 1.0f)) modified = true;
            ImGui::TreePop();
        }

        // input routes
        if (ImGui::TreeNode("Input Routes")) {
            for (std::size_t i = 0; i < cfg.inputs.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (render_input_route(cfg.inputs[i])) modified = true;
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) {
                    cfg.inputs.erase(cfg.inputs.begin() + static_cast<std::ptrdiff_t>(i));
                    modified = true;
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
            if (ImGui::SmallButton("+ Add Route")) {
                cfg.inputs.push_back({});
                modified = true;
            }
            ImGui::TreePop();
        }

        if (InputSeed("Random Seed", cfg.random_seed)) modified = true;

        ImGui::Separator();
        if (ImGui::SmallButton("Remove Config")) {
            remove_requested = true;
        }

        ImGui::TreePop();
    }
    return modified;
}

// --- ngm2.5 group config form (mirrors the ngm3 form; flat schema differs only
//     in max_branch_level range, the penalty_policy selector, and neuron noise) ---
static bool render_ngm25_group_config(const std::string& name,
                                      ngm25::neuron_group_params_t& cfg,
                                      const ngm25::cfg_store& store,
                                      bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        int neuron_cnt   = static_cast<int>(cfg.neuron_cnt);
        int apical_cnt   = static_cast<int>(cfg.apical_cnt);
        int proximal_cnt = static_cast<int>(cfg.proximal_cnt);
        int branch_depth = static_cast<int>(cfg.max_branch_level);
        if (ImGui::InputInt("Neuron Count",   &neuron_cnt))   { cfg.neuron_cnt   = static_cast<std::size_t>(std::max(0, neuron_cnt));   modified = true; }
        if (ImGui::InputInt("Apical Count",   &apical_cnt))   { cfg.apical_cnt   = static_cast<std::size_t>(std::max(0, apical_cnt));   modified = true; }
        if (ImGui::InputInt("Proximal Count", &proximal_cnt)) { cfg.proximal_cnt = static_cast<std::size_t>(std::max(1, proximal_cnt)); modified = true; }
        if (ImGui::SliderInt("Branch Depth (1-6)", &branch_depth, 1, 6)) { cfg.max_branch_level = static_cast<uint8_t>(branch_depth); modified = true; }

        // Penalty-draw policy (the §8 research seam).
        int policy_idx = static_cast<int>(cfg.penalty_policy);
        static constexpr std::array policy_labels = { "stochastic", "expected_value", "none" };
        if (ImGui::Combo("Penalty Policy", &policy_idx, policy_labels.data(),
                         static_cast<int>(policy_labels.size())))
        { cfg.penalty_policy = static_cast<ngm25::penalty_policy_t>(policy_idx); modified = true; }

        if (ImGui::SliderFloat("Permanence Threshold", &cfg.permanence_threshold, 0.0f, 1.0f)) modified = true;
        ImGui::BeginDisabled(!cfg.inhibition_cfg.empty());
        if (ImGui::SliderFloat("Local Inhibition (beta)", &cfg.local_inhibition_strength, 0.1f, 50.0f)) modified = true;
        ImGui::EndDisabled();
        if (render_inhibition_ref_combo(cfg.inhibition_cfg, store.inhibitions)) modified = true;
        if (ImGui::SliderFloat("Stochastic Win Threshold", &cfg.stochastic_win_thres, 0.0f, 1.0f)) modified = true;
        if (ImGui::SliderFloat("Common Learning Rate",  &cfg.common_learning_rate,  0.00001f, 0.01f, "%.5f")) modified = true;
        if (ImGui::SliderFloat("Primary Learning Rate", &cfg.primary_learning_rate, 0.0001f, 0.1f, "%.5f")) modified = true;
        if (ImGui::SliderFloat("Secondary Learning Rate", &cfg.secondary_learning_rate, 0.00001f, 0.01f, "%.5f")) modified = true;
        if (ImGui::SliderFloat("Mismatch Act Threshold", &cfg.mismatch_act_thres, 0.0f, 1.0f)) modified = true;
        if (ImGui::SliderFloat("Mismatch Smoothing", &cfg.mismatch_smoothing, 0.0001f, 0.1f, "%.5f")) modified = true;

        int branch_interval = static_cast<int>(cfg.branch_interval);
        if (ImGui::InputInt("Branch Interval", &branch_interval)) { cfg.branch_interval = static_cast<std::size_t>(std::max(1, branch_interval)); modified = true; }

        if (ImGui::SliderFloat("Min Mismatch Deviation",  &cfg.min_mismatch_deviation,  0.0f, 5.0f)) modified = true;
        if (ImGui::SliderFloat("Accumulated Theta Thres", &cfg.accumulated_theta_thres, 0.0f, 10.0f)) modified = true;
        if (ImGui::SliderFloat("Min Mismatch Percentage", &cfg.min_mismatch_percentage, 0.0f, 0.1f, "%.4f")) modified = true;

        if (ImGui::SliderFloat("Neuron Noise Min", &cfg.neuron_noise_range.first,  0.0f, 0.2f, "%.3f")) modified = true;
        if (ImGui::SliderFloat("Neuron Noise Max", &cfg.neuron_noise_range.second, 0.0f, 0.2f, "%.3f")) modified = true;


        if (ImGui::TreeNode("Weight Filter")) {
            if (SliderDouble("Steepness",        cfg.weight_filter.steepness,        0.0f, 0.999f)) modified = true;
            if (SliderDouble("Transition Point", cfg.weight_filter.transition_point, 0.0f, 1.0f)) modified = true;
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Input Routes")) {
            for (std::size_t i = 0; i < cfg.inputs.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (render_input_route(cfg.inputs[i])) modified = true;
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) {
                    cfg.inputs.erase(cfg.inputs.begin() + static_cast<std::ptrdiff_t>(i));
                    modified = true;
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
            if (ImGui::SmallButton("+ Add Route")) { cfg.inputs.push_back({}); modified = true; }
            ImGui::TreePop();
        }

        if (InputSeed("Random Seed", cfg.random_seed)) modified = true;

        ImGui::Separator();
        if (ImGui::SmallButton("Remove Config")) remove_requested = true;
        ImGui::TreePop();
    }
    return modified;
}

// Helper: Get entity type name
static std::string_view get_entity_type_name(sim::io_entity_type_t type)
{
    return sim::entity_type_display_name(type);   // canonical (sim_builder_cfg.h)
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

    // Entity type dropdown. Built from the canonical label fn in enum order so it
    // is index-aligned with io_entity_type_t by construction — the previous
    // hand-maintained array omitted the GPU types and so misaligned every entry
    // from index 2 on (a GPU group showed as "MNIST Input", and editing the combo
    // wrote back the wrong type).
    static const auto type_names = [] {
        std::array<const char*, static_cast<std::size_t>(sim::io_entity_type_t::ENTITY_COUNT)> names{};
        for (std::size_t i = 0; i < names.size(); ++i)
            names[i] = sim::entity_type_display_name(static_cast<sim::io_entity_type_t>(i));
        return names;
    }();
    int type_idx = static_cast<int>(node.entity_type);
    if (ImGui::Combo("Entity Type", &type_idx, type_names.data(), static_cast<int>(type_names.size()))) {
        node.entity_type = static_cast<sim::io_entity_type_t>(type_idx);
        modified = true;
    }

    // Entity config dropdown - show configs from appropriate store. A dataset's
    // store is its family's map (entity_types); a GPU type shares its CPU
    // sibling's configs.
    if (ImGui::BeginCombo("Entity Config", node.entity_cfg.c_str())) {
        auto list = [&](const auto& configs) {
            for (const auto& [key, val] : configs) {
                if (ImGui::Selectable(key.c_str(), node.entity_cfg == key)) {
                    node.entity_cfg = key;
                    modified = true;
                }
            }
        };
        const auto* info = sim::find_entity_type(node.entity_type);
        if (info != nullptr && info->dataset_family != nullptr) {
            dataset::for_each_family(cfg.datasets, [&](std::string_view family, const auto& configs) {
                if (family == info->dataset_family) list(configs);
            });
        } else {
            switch (node.entity_type) {
                case sim::io_entity_type_t::ngm2_neuron_group:
                    list(cfg.ngm2.neuron_groups);
                    break;
                case sim::io_entity_type_t::ngm3_neuron_group:
                case sim::io_entity_type_t::ngm3_neuron_group_gpu:
                    list(cfg.ngm3.neuron_groups);
                    break;
                case sim::io_entity_type_t::ngm25_neuron_group:
                case sim::io_entity_type_t::ngm25_neuron_group_gpu:
                    list(cfg.ngm25.neuron_groups);
                    break;
                case sim::io_entity_type_t::retina:
                    list(cfg.retina);
                    break;
                default:   // datasets: above
                    break;
            }
        }
        ImGui::EndCombo();
    }

    // Random seed offset
    if (InputSeed("Seed Offset", node.random_seed_offset)) {
        modified = true;
    }
    ImGui::SetItemTooltip("Additive offset applied to all RNG seeds for this node");

    // Inputs list
    if (ImGui::TreeNode("Inputs")) {
        for (std::size_t i = 0; i < node.inputs.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            int src_id = static_cast<int>(node.inputs[i].first);

            ImGui::SetNextItemWidth(scaled_width(80));
            if (ImGui::InputInt("##src", &src_id)) {
                node.inputs[i].first = static_cast<sim::sim_node_id_t>(src_id);
                modified = true;
            }
            ImGui::SameLine();
            if (InputTextStringWidth("##chan", node.inputs[i].second, scaled_width(120))) {
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
        // sim_builder refuses a node that repeats a channel name; say so before the rebuild
        for (std::size_t i = 0; i < node.inputs.size(); ++i)
            for (std::size_t j = i + 1; j < node.inputs.size(); ++j)
                if (node.inputs[i].second == node.inputs[j].second) {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "channel \"%s\" is listed more than once",
                                       node.inputs[i].second.c_str());
                    i = j = node.inputs.size();   // one message is enough
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

    // Create network editor (ImNodeFlow-based)
    net_editor = std::make_unique<network_editor>(config);
    net_editor->set_dirty_callback([this]() { mark_dirty(); });
    net_editor->set_edit_callbacks({
        .add_link = [this](auto src, auto dst, const auto& ch) { return request_add_link(src, dst, ch); },
        .remove_link = [this](auto src, auto dst, const auto& ch) { return request_remove_link(src, dst, ch); },
        .remove_node = [this](auto id) { return request_remove_node(id); },
        .add_node = [this](auto type, const auto& label) { return request_add_node(type, label); }
    });
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
        render_retina_configs();
        render_ngm2_configs();
        render_ngm3_configs();
        render_ngm25_configs();
        render_simulation_graph();
        render_observe_configs();
        render_run_settings();
        render_logging_settings();
        render_visualization_hints();
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

    // Render network editor
    if (net_editor) {
        net_editor->render(&show_network_editor);
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

        if (ImGui::Button("Save", ImVec2(scaled_width(100), 0))) {
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
        if (ImGui::Button("Cancel", ImVec2(scaled_width(100), 0))) {
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

        if (ImGui::Button("Overwrite", ImVec2(scaled_width(100), 0))) {
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
        if (ImGui::Button("Cancel", ImVec2(scaled_width(100), 0))) {
            show_save_as_confirm_dialog = false;
            pending_save_path.clear();
        }
    }
    ImGui::End();
}

bool config_editor::save_config(const std::string& path)
{
    try {
        nlohmann::ordered_json j = config;
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
                    mark_dirty();
                }
                if (remove_requested) {
                    mnist_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!mnist_to_remove.empty()) {
                config.datasets.mnist.erase(mnist_to_remove);
                mark_dirty();
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
                mark_dirty();
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
                    mark_dirty();
                }
                if (remove_requested) {
                    dsprites_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!dsprites_to_remove.empty()) {
                config.datasets.dsprites.erase(dsprites_to_remove);
                mark_dirty();
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
                mark_dirty();
            }
            ImGui::TreePop();
        }

        // EMNIST configs
        if (ImGui::TreeNode("EMNIST Configs")) {
            std::string emnist_to_remove;
            for (auto& [name, cfg] : config.datasets.emnist) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_emnist_config(name, cfg, remove_requested)) {
                    mark_dirty();
                }
                if (remove_requested) {
                    emnist_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!emnist_to_remove.empty()) {
                config.datasets.emnist.erase(emnist_to_remove);
                mark_dirty();
            }
            // Add new config
            static std::string new_emnist_name;
            if (InputTextString("##newemnist", new_emnist_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add EMNIST Config") && !new_emnist_name.empty()) {
                config.datasets.emnist[new_emnist_name] = {};
                new_emnist_name.clear();
                mark_dirty();
            }
            ImGui::TreePop();
        }

        // KMNIST configs
        if (ImGui::TreeNode("KMNIST Configs")) {
            std::string kmnist_to_remove;
            for (auto& [name, cfg] : config.datasets.kmnist) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_kmnist_config(name, cfg, remove_requested)) {
                    mark_dirty();
                }
                if (remove_requested) {
                    kmnist_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!kmnist_to_remove.empty()) {
                config.datasets.kmnist.erase(kmnist_to_remove);
                mark_dirty();
            }
            // Add new config
            static std::string new_kmnist_name;
            if (InputTextString("##newkmnist", new_kmnist_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add KMNIST Config") && !new_kmnist_name.empty()) {
                config.datasets.kmnist[new_kmnist_name] = {};
                new_kmnist_name.clear();
                mark_dirty();
            }
            ImGui::TreePop();
        }

        // CIFAR-10 configs
        if (ImGui::TreeNode("CIFAR-10 Configs")) {
            std::string cifar10_to_remove;
            for (auto& [name, cfg] : config.datasets.cifar10) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_cifar10_config(name, cfg, remove_requested)) {
                    mark_dirty();
                }
                if (remove_requested) {
                    cifar10_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!cifar10_to_remove.empty()) {
                config.datasets.cifar10.erase(cifar10_to_remove);
                mark_dirty();
            }
            // Add new config
            static std::string new_cifar10_name;
            if (InputTextString("##newcifar10", new_cifar10_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add CIFAR-10 Config") && !new_cifar10_name.empty()) {
                config.datasets.cifar10[new_cifar10_name] = {};
                new_cifar10_name.clear();
                mark_dirty();
            }
            ImGui::TreePop();
        }

        // CIFAR-100 configs
        if (ImGui::TreeNode("CIFAR-100 Configs")) {
            std::string cifar100_to_remove;
            for (auto& [name, cfg] : config.datasets.cifar100) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_cifar100_config(name, cfg, remove_requested)) {
                    mark_dirty();
                }
                if (remove_requested) {
                    cifar100_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!cifar100_to_remove.empty()) {
                config.datasets.cifar100.erase(cifar100_to_remove);
                mark_dirty();
            }
            // Add new config
            static std::string new_cifar100_name;
            if (InputTextString("##newcifar100", new_cifar100_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add CIFAR-100 Config") && !new_cifar100_name.empty()) {
                config.datasets.cifar100[new_cifar100_name] = {};
                new_cifar100_name.clear();
                mark_dirty();
            }
            ImGui::TreePop();
        }

        // 3D Chairs configs
        if (ImGui::TreeNode("3D Chairs Configs")) {
            std::string chairs3d_to_remove;
            for (auto& [name, cfg] : config.datasets.chairs3d) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_chairs3d_config(name, cfg, remove_requested)) {
                    mark_dirty();
                }
                if (remove_requested) {
                    chairs3d_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!chairs3d_to_remove.empty()) {
                config.datasets.chairs3d.erase(chairs3d_to_remove);
                mark_dirty();
            }
            // Add new config
            static std::string new_chairs3d_name;
            if (InputTextString("##newchairs3d", new_chairs3d_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add 3D Chairs Config") && !new_chairs3d_name.empty()) {
                config.datasets.chairs3d[new_chairs3d_name] = {};
                new_chairs3d_name.clear();
                mark_dirty();
            }
            ImGui::TreePop();
        }

        // Color MNIST configs
        if (ImGui::TreeNode("Color MNIST Configs")) {
            std::string color_mnist_to_remove;
            for (auto& [name, cfg] : config.datasets.color_mnist) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_color_mnist_config(name, cfg, remove_requested)) {
                    mark_dirty();
                }
                if (remove_requested) {
                    color_mnist_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!color_mnist_to_remove.empty()) {
                config.datasets.color_mnist.erase(color_mnist_to_remove);
                mark_dirty();
            }
            // Add new config
            static std::string new_color_mnist_name;
            if (InputTextString("##newcolormnist", new_color_mnist_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add Color MNIST Config") && !new_color_mnist_name.empty()) {
                config.datasets.color_mnist[new_color_mnist_name] = {};
                new_color_mnist_name.clear();
                mark_dirty();
            }
            ImGui::TreePop();
        }
    }
}

void config_editor::render_retina_configs()
{
    if (ImGui::CollapsingHeader("Retina Configurations")) {
        std::string to_remove;
        for (auto& [name, cfg] : config.retina) {
            ImGui::PushID(name.c_str());
            bool remove_requested = false;
            if (render_retina_config(name, cfg, remove_requested)) {
                mark_dirty();
            }
            if (remove_requested) {
                to_remove = name;
            }
            ImGui::PopID();
        }
        if (!to_remove.empty()) {
            config.retina.erase(to_remove);
            mark_dirty();
        }
        static std::string new_retina_name;
        InputTextString("##newretina", new_retina_name, 64);
        ImGui::SameLine();
        if (ImGui::SmallButton("+ Add Retina Config") && !new_retina_name.empty()) {
            config.retina[new_retina_name] = {};
            new_retina_name.clear();
            mark_dirty();
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
                    mark_dirty();
                }
                if (remove_requested) {
                    dendrite_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!dendrite_to_remove.empty()) {
                config.ngm2.dendrites.erase(dendrite_to_remove);
                mark_dirty();
            }
            static std::string new_dendrite_name;
            if (InputTextString("##newdendrite", new_dendrite_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add Dendrite Config") && !new_dendrite_name.empty()) {
                config.ngm2.dendrites[new_dendrite_name] = {};
                new_dendrite_name.clear();
                mark_dirty();
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
                    mark_dirty();
                }
                if (remove_requested) {
                    neuron_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!neuron_to_remove.empty()) {
                config.ngm2.neurons.erase(neuron_to_remove);
                mark_dirty();
            }
            static std::string new_neuron_name;
            if (InputTextString("##newneuron", new_neuron_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add Neuron Config") && !new_neuron_name.empty()) {
                config.ngm2.neurons[new_neuron_name] = {};
                new_neuron_name.clear();
                mark_dirty();
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
                    mark_dirty();
                }
                if (remove_requested) {
                    group_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!group_to_remove.empty()) {
                config.ngm2.neuron_groups.erase(group_to_remove);
                mark_dirty();
            }
            static std::string new_group_name;
            if (InputTextString("##newgroup", new_group_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add Group Config") && !new_group_name.empty()) {
                config.ngm2.neuron_groups[new_group_name] = {};
                new_group_name.clear();
                mark_dirty();
            }
            ImGui::TreePop();
        }

        // Inhibition configs (referenced by a group's "Inhibition Config" combo)
        if (ImGui::TreeNode("Inhibition Configs")) {
            if (render_inhibition_dict(config.ngm2.inhibitions, "ngm2_inhib")) {
                mark_dirty();
            }
            ImGui::TreePop();
        }
    }
}

void config_editor::render_ngm3_configs()
{
    if (ImGui::CollapsingHeader("NGM3 Neural Model Configurations")) {

        // Neuron group configs
        if (ImGui::TreeNode("NGM3 Group Configs")) {
            std::string group_to_remove;
            for (auto& [name, cfg] : config.ngm3.neuron_groups) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_ngm3_group_config(name, cfg, config.ngm3, remove_requested)) {
                    mark_dirty();
                }
                if (remove_requested) {
                    group_to_remove = name;
                }
                ImGui::PopID();
            }
            if (!group_to_remove.empty()) {
                config.ngm3.neuron_groups.erase(group_to_remove);
                mark_dirty();
            }
            static std::string new_ngm3_group_name;
            if (InputTextString("##newngm3group", new_ngm3_group_name, 64)) {
                // Text changed
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add NGM3 Group Config") && !new_ngm3_group_name.empty()) {
                config.ngm3.neuron_groups[new_ngm3_group_name] = {};
                new_ngm3_group_name.clear();
                mark_dirty();
            }
            ImGui::TreePop();
        }

        // Inhibition configs (referenced by a group's "Inhibition Config" combo)
        if (ImGui::TreeNode("NGM3 Inhibition Configs")) {
            if (render_inhibition_dict(config.ngm3.inhibitions, "ngm3_inhib")) {
                mark_dirty();
            }
            ImGui::TreePop();
        }
    }
}

void config_editor::render_ngm25_configs()
{
    if (ImGui::CollapsingHeader("NGM2.5 Configuration")) {
        if (ImGui::TreeNode("NGM2.5 Group Configs")) {
            std::string group_to_remove;
            for (auto& [name, cfg] : config.ngm25.neuron_groups) {
                ImGui::PushID(name.c_str());
                bool remove_requested = false;
                if (render_ngm25_group_config(name, cfg, config.ngm25, remove_requested))
                    mark_dirty();
                if (remove_requested) group_to_remove = name;
                ImGui::PopID();
            }
            if (!group_to_remove.empty()) {
                config.ngm25.neuron_groups.erase(group_to_remove);
                mark_dirty();
            }
            static std::string new_ngm25_group_name;
            if (InputTextString("##newngm25group", new_ngm25_group_name, 64)) { /* text changed */ }
            ImGui::SameLine();
            if (ImGui::SmallButton("+ Add NGM2.5 Group Config") && !new_ngm25_group_name.empty()) {
                config.ngm25.neuron_groups[new_ngm25_group_name] = {};
                new_ngm25_group_name.clear();
                mark_dirty();
            }
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("NGM2.5 Inhibition Configs")) {
            if (render_inhibition_dict(config.ngm25.inhibitions, "ngm25_inhib"))
                mark_dirty();
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
            for (std::size_t i = 0; i < config.sim.nodes.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                auto& node = config.sim.nodes[i];
                std::string header = std::format("Node ID {}: {} ({})",
                         node.id, node.label, get_entity_type_name(node.entity_type));
                if (ImGui::TreeNode("##node", "%s", header.c_str())) {
                    if (render_sim_node(node, config)) {
                        mark_dirty();
                    }
                    if (ImGui::SmallButton("Remove Node")) {
                        config.sim.nodes.erase(config.sim.nodes.begin() + static_cast<std::ptrdiff_t>(i));
                        mark_dirty();
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
                mark_dirty();
            }
            ImGui::TreePop();
        }
    }
}

// ============================================================================
// Delegation methods for network_editor edit requests
// ============================================================================

bool config_editor::request_add_link(sim::sim_node_id_t src, sim::sim_node_id_t dst, const std::string& channel) {
    // Find destination node
    sim::sim_node_t* dst_node = nullptr;
    for (auto& node : config.sim.nodes) {
        if (node.id == dst) {
            dst_node = &node;
            break;
        }
    }
    if (!dst_node) return false;

    // Check if source node exists
    bool src_exists = false;
    for (const auto& node : config.sim.nodes) {
        if (node.id == src) {
            src_exists = true;
            break;
        }
    }
    if (!src_exists) return false;

    // One link per channel name: sim_builder refuses a node that repeats a
    // channel name, so a second source on a wired channel is refused here too
    for (const auto& [existing_src, existing_ch] : dst_node->inputs) {
        if (existing_ch == channel) {
            return false;
        }
    }

    // Add the link
    dst_node->inputs.emplace_back(src, channel);
    mark_dirty();
    return true;
}

bool config_editor::request_remove_link(sim::sim_node_id_t src, sim::sim_node_id_t dst, const std::string& channel) {
    // Find destination node
    for (auto& node : config.sim.nodes) {
        if (node.id == dst) {
            // Find and remove the link
            for (auto it = node.inputs.begin(); it != node.inputs.end(); ++it) {
                if (it->first == src && it->second == channel) {
                    node.inputs.erase(it);
                    mark_dirty();
                    return true;
                }
            }
            break;
        }
    }
    return false;
}

bool config_editor::request_remove_node(sim::sim_node_id_t id) {
    // Find and remove the node
    for (auto it = config.sim.nodes.begin(); it != config.sim.nodes.end(); ++it) {
        if (it->id == id) {
            config.sim.nodes.erase(it);
            mark_dirty();

            // Also remove any links pointing to this node
            for (auto& node : config.sim.nodes) {
                auto& inputs = node.inputs;
                inputs.erase(
                    std::remove_if(inputs.begin(), inputs.end(),
                        [id](const auto& link) { return link.first == id; }),
                    inputs.end()
                );
            }
            return true;
        }
    }
    return false;
}

sim::sim_node_id_t config_editor::request_add_node(sim::io_entity_type_t type, const std::string& label) {
    // Generate a new unique ID
    sim::sim_node_id_t new_id = 1;
    for (const auto& node : config.sim.nodes) {
        if (node.id >= new_id) {
            new_id = node.id + 1;
        }
    }

    // Create the new node
    sim::sim_node_t new_node;
    new_node.id = new_id;
    new_node.label = label;
    new_node.entity_type = type;
    new_node.entity_cfg = "";  // Empty config initially

    config.sim.nodes.push_back(new_node);
    mark_dirty();
    return new_id;
}

// ============================================================================
// Visualization Hints
// ============================================================================

// Helper: Edit RGBA color stored as int (0-255) using ImGui ColorEdit4
static bool ColorEdit4FromInts(const char* label, int& r, int& g, int& b, int& a, ImGuiColorEditFlags flags = 0)
{
    float col[4] = {
        r / 255.0f,
        g / 255.0f,
        b / 255.0f,
        a / 255.0f
    };
    if (ImGui::ColorEdit4(label, col, flags)) {
        r = static_cast<int>(col[0] * 255.0f + 0.5f);
        g = static_cast<int>(col[1] * 255.0f + 0.5f);
        b = static_cast<int>(col[2] * 255.0f + 0.5f);
        a = static_cast<int>(col[3] * 255.0f + 0.5f);
        return true;
    }
    return false;
}

// Helper: Render ngm_flat_vis_hint_t (returns true if modified)
static bool render_ngm_flat_vis_hint(ngm_flat_vis_hint_t& hint)
{
    bool modified = false;

    ImGui::SetNextItemWidth(scaled_width(80));
    if (ImGui::InputInt("Width", &hint.width)) {
        hint.width = std::max(1, hint.width);
        modified = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(scaled_width(80));
    if (ImGui::InputInt("Height", &hint.height)) {
        hint.height = std::max(1, hint.height);
        modified = true;
    }

    ImGui::SetNextItemWidth(scaled_width(80));
    if (ImGui::InputInt("Channels", &hint.num_channels)) {
        hint.num_channels = std::clamp(hint.num_channels, 1, 3);
        modified = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("1 = grayscale (colormap), 3 = RGB");
    }

    ImGui::SetNextItemWidth(scaled_width(120));
    if (ImGui::InputInt("Update Interval", &hint.update_interval)) {
        hint.update_interval = std::max(0, hint.update_interval);
        modified = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Simulation steps between updates (0 = disabled)");
    }

    return modified;
}

// Helper: Render MNIST hint (returns true if modified, sets remove_requested if remove clicked)
static bool render_mnist_hint(const std::string& name, mnist_hint_t& hint, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        ImGui::Text("Flat Visualization Dimensions:");
        if (render_ngm_flat_vis_hint(hint.ngm_flat_vis)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove")) {
            remove_requested = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render dSprites hint (returns true if modified, sets remove_requested if remove clicked)
static bool render_dsprites_hint(const std::string& name, dsprites_hint_t& hint, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        ImGui::Text("Flat Visualization Dimensions:");
        if (render_ngm_flat_vis_hint(hint.ngm_flat_vis)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove")) {
            remove_requested = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render EMNIST hint (returns true if modified, sets remove_requested if remove clicked)
static bool render_emnist_hint(const std::string& name, emnist_hint_t& hint, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        ImGui::Text("Flat Visualization Dimensions:");
        if (render_ngm_flat_vis_hint(hint.ngm_flat_vis)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove")) {
            remove_requested = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render KMNIST hint (returns true if modified, sets remove_requested if remove clicked)
static bool render_kmnist_hint(const std::string& name, kmnist_hint_t& hint, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        ImGui::Text("Flat Visualization Dimensions:");
        if (render_ngm_flat_vis_hint(hint.ngm_flat_vis)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove")) {
            remove_requested = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render CIFAR-10 hint (returns true if modified, sets remove_requested if remove clicked)
static bool render_cifar10_hint(const std::string& name, cifar10_hint_t& hint, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        ImGui::Text("Flat Visualization Dimensions:");
        if (render_ngm_flat_vis_hint(hint.ngm_flat_vis)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove")) {
            remove_requested = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render CIFAR-100 hint (returns true if modified, sets remove_requested if remove clicked)
static bool render_cifar100_hint(const std::string& name, cifar100_hint_t& hint, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        ImGui::Text("Flat Visualization Dimensions:");
        if (render_ngm_flat_vis_hint(hint.ngm_flat_vis)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove")) {
            remove_requested = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render 3D Chairs hint (returns true if modified, sets remove_requested if remove clicked)
static bool render_chairs3d_hint(const std::string& name, chairs3d_hint_t& hint, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        ImGui::Text("Flat Visualization Dimensions:");
        if (render_ngm_flat_vis_hint(hint.ngm_flat_vis)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove")) {
            remove_requested = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render Color MNIST hint (returns true if modified, sets remove_requested if remove clicked)
static bool render_color_mnist_hint(const std::string& name, color_mnist_hint_t& hint, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        ImGui::Text("Flat Visualization Dimensions:");
        if (render_ngm_flat_vis_hint(hint.ngm_flat_vis)) {
            modified = true;
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove")) {
            remove_requested = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

// Helper: Render ngm_tree_vis_hint_t (returns true if modified)
static bool render_ngm_tree_vis_hint(ngm_tree_vis_hint_t& hint)
{
    bool modified = false;

    // Representation dimensions
    ImGui::SetNextItemWidth(scaled_width(80));
    if (ImGui::InputInt("Rep Width", &hint.rep_width)) {
        hint.rep_width = std::max(1, hint.rep_width);
        modified = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(scaled_width(80));
    if (ImGui::InputInt("Rep Height", &hint.rep_height)) {
        hint.rep_height = std::max(1, hint.rep_height);
        modified = true;
    }

    ImGui::SetNextItemWidth(scaled_width(80));
    if (ImGui::InputInt("Channels##tree", &hint.num_channels)) {
        hint.num_channels = std::clamp(hint.num_channels, 1, 3);
        modified = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("1 = grayscale (colormap), 3 = RGB");
    }

    ImGui::SetNextItemWidth(scaled_width(120));
    if (ImGui::InputInt("Update Interval##tree", &hint.update_interval)) {
        hint.update_interval = std::max(0, hint.update_interval);
        modified = true;
    }

    // Colors
    if (ImGui::TreeNode("Colors")) {
        if (ColorEdit4FromInts("Background", hint.bg_r, hint.bg_g, hint.bg_b, hint.bg_a)) {
            modified = true;
        }
        if (ColorEdit4FromInts("Root Node", hint.root_r, hint.root_g, hint.root_b, hint.root_a)) {
            modified = true;
        }
        if (ColorEdit4FromInts("Branch Node", hint.branch_r, hint.branch_g, hint.branch_b, hint.branch_a)) {
            modified = true;
        }
        if (ColorEdit4FromInts("Edge", hint.edge_r, hint.edge_g, hint.edge_b, hint.edge_a)) {
            modified = true;
        }
        ImGui::TreePop();
    }

    // Node appearance
    if (ImGui::TreeNode("Node Appearance")) {
        ImGui::SetNextItemWidth(scaled_width(80));
        if (ImGui::InputFloat("Root Radius", &hint.root_node_radius, 0.5f)) {
            hint.root_node_radius = std::max(0.5f, hint.root_node_radius);
            modified = true;
        }
        ImGui::SetNextItemWidth(scaled_width(80));
        if (ImGui::InputFloat("Branch Radius", &hint.branch_node_radius, 0.5f)) {
            hint.branch_node_radius = std::max(0.5f, hint.branch_node_radius);
            modified = true;
        }
        ImGui::SetNextItemWidth(scaled_width(120));
        if (ImGui::InputFloat("Texture Scale Div", &hint.texture_scale_divisor, 100.0f)) {
            hint.texture_scale_divisor = std::max(100.0f, hint.texture_scale_divisor);
            modified = true;
        }
        ImGui::TreePop();
    }

    // Edge appearance
    if (ImGui::TreeNode("Edge Appearance")) {
        ImGui::SetNextItemWidth(scaled_width(80));
        if (ImGui::InputFloat("Min Thickness", &hint.edge_thickness_min, 0.5f)) {
            hint.edge_thickness_min = std::max(0.5f, hint.edge_thickness_min);
            modified = true;
        }
        ImGui::SetNextItemWidth(scaled_width(80));
        if (ImGui::InputFloat("Max Thickness", &hint.edge_thickness_max, 0.5f)) {
            hint.edge_thickness_max = std::max(hint.edge_thickness_min, hint.edge_thickness_max);
            modified = true;
        }
        ImGui::TreePop();
    }

    return modified;
}

// Helper: Render neuron group hint (returns true if modified, sets remove_requested if remove clicked)
static bool render_neuron_group_hint(const std::string& name, neuron_group_hints_t& hint, bool& remove_requested)
{
    bool modified = false;
    remove_requested = false;
    if (ImGui::TreeNode(name.c_str())) {
        if (ImGui::TreeNode("Dendritic Representations")) {
            if (render_ngm_flat_vis_hint(hint.ngm_flat_vis)) {
                modified = true;
            }
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Neuron Tree")) {
            if (render_ngm_tree_vis_hint(hint.ngm_tree_vis)) {
                modified = true;
            }
            ImGui::TreePop();
        }

        ImGui::Separator();
        if (ImGui::SmallButton("Remove")) {
            remove_requested = true;
        }
        ImGui::TreePop();
    }
    return modified;
}

void config_editor::render_observe_configs()
{
    if (!ImGui::CollapsingHeader("Observe Configuration")) {
        return;
    }

    auto& obs_cfg = config.observe;

    // ---- output_dir ----
    if (InputTextString("Output Directory", obs_cfg.output_dir)) {
        mark_dirty();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Directory where per-group <name>.observe.h5 files are written.\n"
            "A relative name is placed under the app config's config_base\n"
            "(default <config_dir>/<config>/); an absolute path is used as-is.");
    }

    ImGui::Separator();

    // ---- groups ----
    if (obs_cfg.groups.empty()) {
        ImGui::TextDisabled("(no groups configured)");
    }

    // Collect mutations to apply after the iteration (cannot erase from
    // std::map while iterating it).
    std::string group_to_remove;

    for (auto& [group_name, group_cfg] : obs_cfg.groups) {
        ImGui::PushID(group_name.c_str());

        if (ImGui::TreeNodeEx(group_name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {

            // Combine mode combo
            {
                const char* items[] = { "or", "and" };
                int current = (group_cfg.combine == observe::combine_mode_t::AND_) ? 1 : 0;
                ImGui::SetNextItemWidth(scaled_width(120.0f));
                if (ImGui::Combo("Combine", &current, items, IM_ARRAYSIZE(items))) {
                    group_cfg.combine = (current == 1)
                        ? observe::combine_mode_t::AND_
                        : observe::combine_mode_t::OR_;
                    mark_dirty();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "OR  : capture when range_hit OR programmatic trigger is active.\n"
                        "AND : capture when both are active simultaneously.");
                }
            }

            // Ranges list
            ImGui::TextUnformatted("Ranges (inclusive [start, stop]):");
            ImGui::Indent();

            int range_to_remove = -1;
            for (std::size_t i = 0; i < group_cfg.ranges.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));

                std::uint64_t start_v = group_cfg.ranges[i].start_step;
                std::uint64_t stop_v  = group_cfg.ranges[i].stop_step;

                ImGui::SetNextItemWidth(scaled_width(140.0f));
                if (ImGui::InputScalar("##start", ImGuiDataType_U64, &start_v)) {
                    group_cfg.ranges[i].start_step = start_v;
                    mark_dirty();
                }
                ImGui::SameLine();
                ImGui::TextUnformatted("..");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(scaled_width(140.0f));
                if (ImGui::InputScalar("##stop", ImGuiDataType_U64, &stop_v)) {
                    group_cfg.ranges[i].stop_step = stop_v;
                    mark_dirty();
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) {
                    range_to_remove = static_cast<int>(i);
                }

                ImGui::PopID();
            }

            if (ImGui::SmallButton("+ Add Range")) {
                group_cfg.ranges.push_back(observe::step_range_t{0, 0});
                mark_dirty();
            }

            if (range_to_remove >= 0) {
                group_cfg.ranges.erase(group_cfg.ranges.begin() + range_to_remove);
                mark_dirty();
            }

            ImGui::Unindent();

            // Remove group button
            ImGui::Spacing();
            if (ImGui::SmallButton("Remove Group")) {
                group_to_remove = group_name;
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    if (!group_to_remove.empty()) {
        obs_cfg.groups.erase(group_to_remove);
        mark_dirty();
    }

    // ---- add new group ----
    ImGui::Separator();
    static std::string new_group_name;
    ImGui::SetNextItemWidth(scaled_width(220.0f));
    InputTextString("##new_observe_group", new_group_name, 64);
    ImGui::SameLine();
    if (ImGui::SmallButton("+ Add Group") && !new_group_name.empty()) {
        if (!obs_cfg.groups.contains(new_group_name)) {
            obs_cfg.groups.emplace(new_group_name, observe::group_cfg_t{});
            mark_dirty();
        }
        new_group_name.clear();
    }
}

void config_editor::render_run_settings()
{
    if (ImGui::CollapsingHeader("Run Settings")) {
        if (ImGui::Checkbox("Frozen (inference only)", &config.frozen)) {
            mark_dirty();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("A frozen run: every neuron group makes each step a pure forward pass\n"
                              "(no winner selection, adaptation, branching or running-activity-max\n"
                              "update).");
        }
        if (InputTextString("Run ID", config.run_id)) {
            mark_dirty();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Datasets continue their sequences from a checkpoint only when it\n"
                              "was written by this run ID; otherwise they start fresh. \"auto\"\n"
                              "draws a fresh ID. Enter a checkpoint's ID to continue it.");
        }
    }
}

void config_editor::render_logging_settings()
{
    if (ImGui::CollapsingHeader("Logging Settings")) {
        if (InputTextString("Snapshot Directory", config.snapshot_dir)) {
            mark_dirty();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Directory for auto-save snapshots. A relative name is placed under\n"
                "the app config's config_base (default <config_dir>/<config>/); an\n"
                "absolute path is used as-is.");
        }
    }
}

void config_editor::render_visualization_hints()
{
    if (ImGui::CollapsingHeader("Visualization Hints")) {

        // Dataset hints
        if (ImGui::TreeNode("Dataset Hints")) {
            // MNIST hints
            if (ImGui::TreeNode("MNIST")) {
                std::string to_remove;
                for (auto& [name, hint] : config.vis_hints.datasets.mnist) {
                    ImGui::PushID(name.c_str());
                    bool remove_requested = false;
                    if (render_mnist_hint(name, hint, remove_requested)) {
                        mark_dirty();
                    }
                    if (remove_requested) {
                        to_remove = name;
                    }
                    ImGui::PopID();
                }
                if (!to_remove.empty()) {
                    config.vis_hints.datasets.mnist.erase(to_remove);
                    mark_dirty();
                }
                // Add new hint
                static std::string new_mnist_hint_name;
                if (InputTextString("##newmnisthint", new_mnist_hint_name, 64)) {
                    // Text changed
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("+ Add MNIST Hint") && !new_mnist_hint_name.empty()) {
                    config.vis_hints.datasets.mnist[new_mnist_hint_name] = {{28, 28}};
                    new_mnist_hint_name.clear();
                    mark_dirty();
                }
                ImGui::TreePop();
            }

            // dSprites hints
            if (ImGui::TreeNode("dSprites")) {
                std::string to_remove;
                for (auto& [name, hint] : config.vis_hints.datasets.dsprites) {
                    ImGui::PushID(name.c_str());
                    bool remove_requested = false;
                    if (render_dsprites_hint(name, hint, remove_requested)) {
                        mark_dirty();
                    }
                    if (remove_requested) {
                        to_remove = name;
                    }
                    ImGui::PopID();
                }
                if (!to_remove.empty()) {
                    config.vis_hints.datasets.dsprites.erase(to_remove);
                    mark_dirty();
                }
                // Add new hint
                static std::string new_dsprites_hint_name;
                if (InputTextString("##newdspriteshint", new_dsprites_hint_name, 64)) {
                    // Text changed
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("+ Add dSprites Hint") && !new_dsprites_hint_name.empty()) {
                    config.vis_hints.datasets.dsprites[new_dsprites_hint_name] = {{64, 64}};
                    new_dsprites_hint_name.clear();
                    mark_dirty();
                }
                ImGui::TreePop();
            }

            // EMNIST hints
            if (ImGui::TreeNode("EMNIST")) {
                std::string to_remove;
                for (auto& [name, hint] : config.vis_hints.datasets.emnist) {
                    ImGui::PushID(name.c_str());
                    bool remove_requested = false;
                    if (render_emnist_hint(name, hint, remove_requested)) {
                        mark_dirty();
                    }
                    if (remove_requested) {
                        to_remove = name;
                    }
                    ImGui::PopID();
                }
                if (!to_remove.empty()) {
                    config.vis_hints.datasets.emnist.erase(to_remove);
                    mark_dirty();
                }
                // Add new hint
                static std::string new_emnist_hint_name;
                if (InputTextString("##newemnisthint", new_emnist_hint_name, 64)) {
                    // Text changed
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("+ Add EMNIST Hint") && !new_emnist_hint_name.empty()) {
                    config.vis_hints.datasets.emnist[new_emnist_hint_name] = {{28, 28, 1}};
                    new_emnist_hint_name.clear();
                    mark_dirty();
                }
                ImGui::TreePop();
            }

            // KMNIST hints
            if (ImGui::TreeNode("KMNIST")) {
                std::string to_remove;
                for (auto& [name, hint] : config.vis_hints.datasets.kmnist) {
                    ImGui::PushID(name.c_str());
                    bool remove_requested = false;
                    if (render_kmnist_hint(name, hint, remove_requested)) {
                        mark_dirty();
                    }
                    if (remove_requested) {
                        to_remove = name;
                    }
                    ImGui::PopID();
                }
                if (!to_remove.empty()) {
                    config.vis_hints.datasets.kmnist.erase(to_remove);
                    mark_dirty();
                }
                // Add new hint
                static std::string new_kmnist_hint_name;
                if (InputTextString("##newkmnisthint", new_kmnist_hint_name, 64)) {
                    // Text changed
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("+ Add KMNIST Hint") && !new_kmnist_hint_name.empty()) {
                    config.vis_hints.datasets.kmnist[new_kmnist_hint_name] = {{28, 28, 1}};
                    new_kmnist_hint_name.clear();
                    mark_dirty();
                }
                ImGui::TreePop();
            }

            // CIFAR-10 hints
            if (ImGui::TreeNode("CIFAR-10")) {
                std::string to_remove;
                for (auto& [name, hint] : config.vis_hints.datasets.cifar10) {
                    ImGui::PushID(name.c_str());
                    bool remove_requested = false;
                    if (render_cifar10_hint(name, hint, remove_requested)) {
                        mark_dirty();
                    }
                    if (remove_requested) {
                        to_remove = name;
                    }
                    ImGui::PopID();
                }
                if (!to_remove.empty()) {
                    config.vis_hints.datasets.cifar10.erase(to_remove);
                    mark_dirty();
                }
                // Add new hint
                static std::string new_cifar10_hint_name;
                if (InputTextString("##newcifar10hint", new_cifar10_hint_name, 64)) {
                    // Text changed
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("+ Add CIFAR-10 Hint") && !new_cifar10_hint_name.empty()) {
                    config.vis_hints.datasets.cifar10[new_cifar10_hint_name] = {{32, 32, 3}};
                    new_cifar10_hint_name.clear();
                    mark_dirty();
                }
                ImGui::TreePop();
            }

            // CIFAR-100 hints
            if (ImGui::TreeNode("CIFAR-100")) {
                std::string to_remove;
                for (auto& [name, hint] : config.vis_hints.datasets.cifar100) {
                    ImGui::PushID(name.c_str());
                    bool remove_requested = false;
                    if (render_cifar100_hint(name, hint, remove_requested)) {
                        mark_dirty();
                    }
                    if (remove_requested) {
                        to_remove = name;
                    }
                    ImGui::PopID();
                }
                if (!to_remove.empty()) {
                    config.vis_hints.datasets.cifar100.erase(to_remove);
                    mark_dirty();
                }
                // Add new hint
                static std::string new_cifar100_hint_name;
                if (InputTextString("##newcifar100hint", new_cifar100_hint_name, 64)) {
                    // Text changed
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("+ Add CIFAR-100 Hint") && !new_cifar100_hint_name.empty()) {
                    config.vis_hints.datasets.cifar100[new_cifar100_hint_name] = {{32, 32, 3}};
                    new_cifar100_hint_name.clear();
                    mark_dirty();
                }
                ImGui::TreePop();
            }

            // 3D Chairs hints
            if (ImGui::TreeNode("3D Chairs")) {
                std::string to_remove;
                for (auto& [name, hint] : config.vis_hints.datasets.chairs3d) {
                    ImGui::PushID(name.c_str());
                    bool remove_requested = false;
                    if (render_chairs3d_hint(name, hint, remove_requested)) {
                        mark_dirty();
                    }
                    if (remove_requested) {
                        to_remove = name;
                    }
                    ImGui::PopID();
                }
                if (!to_remove.empty()) {
                    config.vis_hints.datasets.chairs3d.erase(to_remove);
                    mark_dirty();
                }
                // Add new hint
                static std::string new_chairs3d_hint_name;
                if (InputTextString("##newchairs3dhint", new_chairs3d_hint_name, 64)) {
                    // Text changed
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("+ Add 3D Chairs Hint") && !new_chairs3d_hint_name.empty()) {
                    config.vis_hints.datasets.chairs3d[new_chairs3d_hint_name] = {{64, 64, 3}};
                    new_chairs3d_hint_name.clear();
                    mark_dirty();
                }
                ImGui::TreePop();
            }

            // Color MNIST hints
            if (ImGui::TreeNode("Color MNIST")) {
                std::string to_remove;
                for (auto& [name, hint] : config.vis_hints.datasets.color_mnist) {
                    ImGui::PushID(name.c_str());
                    bool remove_requested = false;
                    if (render_color_mnist_hint(name, hint, remove_requested)) {
                        mark_dirty();
                    }
                    if (remove_requested) {
                        to_remove = name;
                    }
                    ImGui::PopID();
                }
                if (!to_remove.empty()) {
                    config.vis_hints.datasets.color_mnist.erase(to_remove);
                    mark_dirty();
                }
                // Add new hint
                static std::string new_color_mnist_hint_name;
                if (InputTextString("##newcolormnisthint", new_color_mnist_hint_name, 64)) {
                    // Text changed
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("+ Add Color MNIST Hint") && !new_color_mnist_hint_name.empty()) {
                    config.vis_hints.datasets.color_mnist[new_color_mnist_hint_name] = {{28, 28, 3}};
                    new_color_mnist_hint_name.clear();
                    mark_dirty();
                }
                ImGui::TreePop();
            }

            ImGui::TreePop();
        }

        // NGM2 / NGM3 hints
        if (ImGui::TreeNode("Neural Model Hints")) {
            // NGM2 Neuron group hints
            if (ImGui::TreeNode("NGM2 Neuron Groups")) {
                std::string to_remove;
                for (auto& [name, hint] : config.vis_hints.ngm2.neuron_groups) {
                    ImGui::PushID(name.c_str());
                    bool remove_requested = false;
                    if (render_neuron_group_hint(name, hint, remove_requested)) {
                        mark_dirty();
                    }
                    if (remove_requested) {
                        to_remove = name;
                    }
                    ImGui::PopID();
                }
                if (!to_remove.empty()) {
                    config.vis_hints.ngm2.neuron_groups.erase(to_remove);
                    mark_dirty();
                }
                static std::string new_ngm2_hint_name;
                if (InputTextString("##newngm2hint", new_ngm2_hint_name, 64)) {}
                ImGui::SameLine();
                if (ImGui::SmallButton("+ Add NGM2 Group Hint") && !new_ngm2_hint_name.empty()) {
                    config.vis_hints.ngm2.neuron_groups[new_ngm2_hint_name] = {{50, 1}};
                    new_ngm2_hint_name.clear();
                    mark_dirty();
                }
                ImGui::TreePop();
            }

            // NGM3 Neuron group hints
            if (ImGui::TreeNode("NGM3 Neuron Groups")) {
                std::string to_remove;
                for (auto& [name, hint] : config.vis_hints.ngm3.neuron_groups) {
                    ImGui::PushID(name.c_str());
                    bool remove_requested = false;
                    if (render_neuron_group_hint(name, hint, remove_requested)) {
                        mark_dirty();
                    }
                    if (remove_requested) {
                        to_remove = name;
                    }
                    ImGui::PopID();
                }
                if (!to_remove.empty()) {
                    config.vis_hints.ngm3.neuron_groups.erase(to_remove);
                    mark_dirty();
                }
                static std::string new_ngm3_hint_name;
                if (InputTextString("##newngm3hint", new_ngm3_hint_name, 64)) {}
                ImGui::SameLine();
                if (ImGui::SmallButton("+ Add NGM3 Group Hint") && !new_ngm3_hint_name.empty()) {
                    config.vis_hints.ngm3.neuron_groups[new_ngm3_hint_name] = {{28, 28}};
                    new_ngm3_hint_name.clear();
                    mark_dirty();
                }
                ImGui::TreePop();
            }
            ImGui::TreePop();
        }

        // IO Buffer hints
        if (ImGui::TreeNode("IO Buffer Hints")) {
            ImGui::SetNextItemWidth(scaled_width(120));
            if (ImGui::InputInt("Update Interval", &config.vis_hints.buffer_vis.update_interval)) {
                config.vis_hints.buffer_vis.update_interval = std::max(0, config.vis_hints.buffer_vis.update_interval);
                mark_dirty();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Simulation steps between updates (0 = disabled)");
            }

            ImGui::SetNextItemWidth(scaled_width(120));
            if (ImGui::InputInt("Ring Length", &config.vis_hints.buffer_vis.ring_length)) {
                config.vis_hints.buffer_vis.ring_length = std::clamp(config.vis_hints.buffer_vis.ring_length, 10, 10000);
                mark_dirty();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Number of time steps to show in time series visualization");
            }
            ImGui::TreePop();
        }
    }
}

} // namespace coast
