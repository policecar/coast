#include "network_editor2.h"
#include "ImNodeFlow.h"

#include <algorithm>

namespace coast {

namespace {

// Connection filter that accepts any connection
auto filter_any = [](ImFlow::Pin*, ImFlow::Pin*) { return true; };

// Node style based on entity type
std::shared_ptr<ImFlow::NodeStyle> get_node_style(sim::io_entity_type_t type) {
    switch (type) {
        case sim::io_entity_type_t::ngm2_neuron_group:
            return ImFlow::NodeStyle::cyan();
        case sim::io_entity_type_t::mnist_input:
            return ImFlow::NodeStyle::green();
        case sim::io_entity_type_t::dsprites_input:
            return ImFlow::NodeStyle::brown();
        default:
            return ImFlow::NodeStyle::cyan();
    }
}

// Entity type names for display
const char* get_entity_type_name(sim::io_entity_type_t type) {
    switch (type) {
        case sim::io_entity_type_t::ngm2_neuron_group: return "NGM2 Group";
        case sim::io_entity_type_t::mnist_input: return "MNIST Input";
        case sim::io_entity_type_t::dsprites_input: return "dSprites Input";
        default: return "Unknown";
    }
}

} // anonymous namespace

/**
 * Custom node class for simulation nodes.
 * Stores a copy of node data for display.
 */
class SimNode : public ImFlow::BaseNode {
public:
    SimNode(sim::sim_node_id_t node_id,
            const std::string& label,
            sim::io_entity_type_t entity_type,
            const std::string& entity_cfg,
            const std::vector<std::string>& input_channels)
        : node_id_(node_id)
        , label_(label)
        , entity_type_(entity_type)
        , entity_cfg_(entity_cfg)
        , input_channels_(input_channels)
    {
        setTitle(std::to_string(node_id_) + ": " + label_);
        setStyle(get_node_style(entity_type_));

        // Add output pin (all nodes have one output)
        addOUT<int>("out")->behaviour([]() { return 0; });

        // Add input pins based on entity type's input channels
        for (const auto& channel : input_channels_) {
            auto pin = addIN<int>(channel, 0, filter_any);
            // Allow self-connections (loops from output to own input)
            pin->allowSameNodeConnections(true);
        }
    }

    void draw() override {
        // Display entity type
        ImGui::TextDisabled("%s", get_entity_type_name(entity_type_));

        // Display entity config
        if (!entity_cfg_.empty()) {
            ImGui::TextDisabled("cfg: %s", entity_cfg_.c_str());
        }
    }

    sim::sim_node_id_t getNodeId() const { return node_id_; }

    // Get input pin by channel name
    ImFlow::Pin* getInputPin(const std::string& channel_name) {
        for (auto& pin : getIns()) {
            if (pin->getName() == channel_name) {
                return pin.get();
            }
        }
        return nullptr;
    }

    // Get output pin
    ImFlow::Pin* getOutputPin() {
        auto& outs = getOuts();
        if (!outs.empty()) {
            return outs[0].get();
        }
        return nullptr;
    }

private:
    sim::sim_node_id_t node_id_;
    std::string label_;
    sim::io_entity_type_t entity_type_;
    std::string entity_cfg_;
    std::vector<std::string> input_channels_;
};

// ============================================================================
// network_editor2 implementation
// ============================================================================

network_editor2::network_editor2(loader::global_cfg_t& config)
    : config(config)
    , editor(std::make_unique<ImFlow::ImNodeFlow>("NetworkEditor2"))
{
}

network_editor2::~network_editor2() = default;

void network_editor2::set_dirty_callback(dirty_callback_t callback) {
    dirty_callback = std::move(callback);
}

void network_editor2::mark_dirty() {
    if (dirty_callback) {
        dirty_callback();
    }
}

const sim::sim_node_t* network_editor2::find_node_by_id(sim::sim_node_id_t id) const {
    for (const auto& node : config.sim.nodes) {
        if (node.id == id) {
            return &node;
        }
    }
    return nullptr;
}

std::vector<std::string> network_editor2::get_input_channels(const sim::sim_node_t& node) const {
    // Dataset nodes have no input pins
    if (node.entity_type == sim::io_entity_type_t::mnist_input ||
        node.entity_type == sim::io_entity_type_t::dsprites_input) {
        return {};
    }

    // For neuron groups, get input_channels from the entity config
    if (node.entity_type == sim::io_entity_type_t::ngm2_neuron_group) {
        auto it = config.ngm2.neuron_groups.find(node.entity_cfg);
        if (it != config.ngm2.neuron_groups.end()) {
            return it->second.input_channels;
        }
    }

    return {};
}

bool network_editor2::is_link_valid(sim::sim_node_id_t src_id,
                                     sim::sim_node_id_t dst_id,
                                     const std::string& channel) const {
    // Source node must exist
    const auto* src_node = find_node_by_id(src_id);
    if (!src_node) return false;

    // Destination node must exist
    const auto* dst_node = find_node_by_id(dst_id);
    if (!dst_node) return false;

    // Channel must be non-empty
    if (channel.empty()) return false;

    // Channel must be valid for the destination node
    auto channels = get_input_channels(*dst_node);
    return std::find(channels.begin(), channels.end(), channel) != channels.end();
}


void network_editor2::render(bool* p_open) {
    if (!*p_open) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Network Editor 2 (Read-Only)###NetworkEditor2", p_open)) {

        // Toolbar
        ImGui::Text("Phase 1: Read-only visualization");
        ImGui::SameLine();
        bool do_refresh = ImGui::Button("Refresh");
        ImGui::SameLine();
        ImGui::TextDisabled("Nodes: %zu | Links: %zu",
                           config.sim.nodes.size(), link_states.size());

        // Show validation errors
        if (!duplicate_node_ids_.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                "| WARNING: %zu duplicate node ID(s)!", duplicate_node_ids_.size());
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Duplicate node IDs (skipped):");
                for (auto id : duplicate_node_ids_) {
                    ImGui::BulletText("ID %zu", id);
                }
                ImGui::EndTooltip();
            }
        }

        ImGui::Separator();

        // Only rebuild nodes on first render or when Refresh is clicked
        bool needs_rebuild = do_refresh || editor->getNodesCount() == 0;

        if (needs_rebuild) {
            rebuild_from_config();
        }

        // Render the node editor
        editor->update();
    }
    ImGui::End();
}

void network_editor2::rebuild_from_config() {
    // Save current node positions from ImNodeFlow nodes
    for (auto& [uid, node] : editor->getNodes()) {
        auto* sim_node = dynamic_cast<SimNode*>(node.get());
        if (sim_node) {
            auto it = node_states.find(sim_node->getNodeId());
            if (it != node_states.end()) {
                it->second.position = node->getPos();
                it->second.position_initialized = true;
            }
        }
    }

    // Mark all existing nodes for destruction
    for (auto& [uid, node] : editor->getNodes()) {
        node->destroy();
    }

    // Create/update node states from config, detecting duplicates
    std::unordered_map<sim::sim_node_id_t, EditorNodeState> new_states;
    duplicate_node_ids_.clear();

    for (const auto& node : config.sim.nodes) {
        // Check for duplicate node ID
        if (new_states.contains(node.id)) {
            duplicate_node_ids_.push_back(node.id);
            continue;  // Skip duplicate
        }

        EditorNodeState state;
        state.id = node.id;
        state.label = node.label;
        state.entity_type = node.entity_type;
        state.entity_cfg = node.entity_cfg;

        // Preserve position from old state if available
        auto it = node_states.find(node.id);
        if (it != node_states.end() && it->second.position_initialized) {
            state.position = it->second.position;
            state.position_initialized = true;
        } else {
            // Auto-layout: simple grid placement
            std::size_t idx = new_states.size();
            state.position = ImVec2(100.0f + (idx % 4) * 250.0f,
                                    100.0f + (idx / 4) * 200.0f);
            state.position_initialized = true;
        }

        new_states[node.id] = state;
    }
    node_states = std::move(new_states);

    // Build valid links from config
    link_states.clear();
    for (const auto& dst_node : config.sim.nodes) {
        for (const auto& [src_id, channel] : dst_node.inputs) {
            if (is_link_valid(src_id, dst_node.id, channel)) {
                EditorLinkState link;
                link.src_id = src_id;
                link.dst_id = dst_node.id;
                link.channel_name = channel;
                link_states.push_back(link);
            }
        }
    }

    // Create nodes in ImNodeFlow (only for unique node IDs in node_states)
    std::unordered_map<sim::sim_node_id_t, SimNode*> node_map;
    for (const auto& [node_id, state] : node_states) {
        // Find the config node to get input channels
        const auto* cfg_node = find_node_by_id(node_id);
        if (!cfg_node) continue;

        auto channels = get_input_channels(*cfg_node);

        auto node = editor->addNode<SimNode>(
            state.position,
            node_id,
            state.label,
            state.entity_type,
            state.entity_cfg,
            channels
        );
        node_map[node_id] = node.get();
    }

    // Create links in ImNodeFlow
    for (const auto& link : link_states) {
        auto src_it = node_map.find(link.src_id);
        auto dst_it = node_map.find(link.dst_id);

        if (src_it != node_map.end() && dst_it != node_map.end()) {
            auto* src_node = src_it->second;
            auto* dst_node = dst_it->second;

            auto* out_pin = src_node->getOutputPin();
            auto* in_pin = dst_node->getInputPin(link.channel_name);

            if (out_pin && in_pin) {
                out_pin->createLink(in_pin);
            }
        }
    }
}

} // namespace coast
