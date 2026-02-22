#include "network_editor.h"
#include "ImNodeFlow.h"

#include <algorithm>
#include <limits>
#include <random>

namespace coast {

namespace {

// Connection filter that accepts any connection
auto filter_any = [](ImFlow::Pin*, ImFlow::Pin*) { return true; };

// Node style based on entity type
std::shared_ptr<ImFlow::NodeStyle> get_node_style(sim::io_entity_type_t type) {
    switch (type) {
        case sim::io_entity_type_t::ngm2_neuron_group:
            return ImFlow::NodeStyle::cyan();
        // MNIST-family datasets
        case sim::io_entity_type_t::mnist_input:
        case sim::io_entity_type_t::emnist_input:
        case sim::io_entity_type_t::kmnist_input:
        case sim::io_entity_type_t::color_mnist_input:
            return ImFlow::NodeStyle::green();
        // Visual / 3D datasets
        case sim::io_entity_type_t::dsprites_input:
        case sim::io_entity_type_t::cifar10_input:
        case sim::io_entity_type_t::cifar100_input:
        case sim::io_entity_type_t::chairs3d_input:
            return ImFlow::NodeStyle::brown();
        default:
            return ImFlow::NodeStyle::cyan();
    }
}

// Entity type names for display
const char* get_entity_type_name(sim::io_entity_type_t type) {
    switch (type) {
        case sim::io_entity_type_t::ngm2_neuron_group:  return "NGM2 Group";
        case sim::io_entity_type_t::mnist_input:        return "MNIST Input";
        case sim::io_entity_type_t::emnist_input:       return "EMNIST Input";
        case sim::io_entity_type_t::kmnist_input:       return "KMNIST Input";
        case sim::io_entity_type_t::color_mnist_input:  return "Color MNIST Input";
        case sim::io_entity_type_t::dsprites_input:     return "dSprites Input";
        case sim::io_entity_type_t::cifar10_input:      return "CIFAR-10 Input";
        case sim::io_entity_type_t::cifar100_input:     return "CIFAR-100 Input";
        case sim::io_entity_type_t::chairs3d_input:     return "3D Chairs Input";
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
// network_editor implementation
// ============================================================================

network_editor::network_editor(loader::global_cfg_t& config)
    : config(config)
    , editor(std::make_unique<ImFlow::ImNodeFlow>("NetworkEditor2"))
{
}

network_editor::~network_editor() {
    stop_auto_layout();
}

void network_editor::set_dirty_callback(dirty_callback_t callback) {
    dirty_callback = std::move(callback);
}

void network_editor::set_edit_callbacks(edit_callbacks_t callbacks) {
    edit_callbacks_ = std::move(callbacks);
}

void network_editor::mark_dirty() {
    if (dirty_callback) {
        dirty_callback();
    }
}

const sim::sim_node_t* network_editor::find_node_by_id(sim::sim_node_id_t id) const {
    for (const auto& node : config.sim.nodes) {
        if (node.id == id) {
            return &node;
        }
    }
    return nullptr;
}

std::vector<std::string> network_editor::get_input_channels(const sim::sim_node_t& node) const {
    // Dataset nodes have no input pins
    if (node.entity_type == sim::io_entity_type_t::mnist_input   ||
        node.entity_type == sim::io_entity_type_t::emnist_input  ||
        node.entity_type == sim::io_entity_type_t::kmnist_input  ||
        node.entity_type == sim::io_entity_type_t::color_mnist_input ||
        node.entity_type == sim::io_entity_type_t::dsprites_input ||
        node.entity_type == sim::io_entity_type_t::cifar10_input  ||
        node.entity_type == sim::io_entity_type_t::cifar100_input ||
        node.entity_type == sim::io_entity_type_t::chairs3d_input) {
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

bool network_editor::is_link_valid(sim::sim_node_id_t src_id,
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


void network_editor::render(bool* p_open) {
    if (!*p_open) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Network Editor###NetworkEditor2", p_open)) {

        // Status bar
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

        // Auto Pos button (hold to run force layout)
        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        ImGui::Button("Auto Pos");
        if (ImGui::IsItemActive()) {
            if (!auto_layout_active_) {
                init_auto_layout();
            }
            step_auto_layout();
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "(layouting...)");
        } else if (auto_layout_active_) {
            stop_auto_layout();
        }

        // Reset View button - centers all nodes in viewport
        ImGui::SameLine();
        if (ImGui::Button("Reset View")) {
            reset_view();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Center all nodes in viewport");
        }

        ImGui::Separator();

        // Rebuild on first render or when invalidated by external changes
        bool needs_rebuild = editor->getNodesCount() == 0 || needs_refresh_;
        needs_refresh_ = false;

        if (needs_rebuild) {
            rebuild_from_config();
        }

        // Set up right-click context menu for adding nodes
        editor->rightClickPopUpContent([this, &needs_rebuild](ImFlow::BaseNode* node) {
            if (node == nullptr) {
                // Background right-click - add node menu
                if (ImGui::BeginMenu("Add Node")) {
                    if (ImGui::MenuItem("NGM2 Neuron Group")) {
                        if (edit_callbacks_.add_node) {
                            edit_callbacks_.add_node(sim::io_entity_type_t::ngm2_neuron_group, "new_group");
                            needs_rebuild = true;
                        }
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("MNIST Input")) {
                        if (edit_callbacks_.add_node) {
                            edit_callbacks_.add_node(sim::io_entity_type_t::mnist_input, "mnist");
                            needs_rebuild = true;
                        }
                    }
                    if (ImGui::MenuItem("EMNIST Input")) {
                        if (edit_callbacks_.add_node) {
                            edit_callbacks_.add_node(sim::io_entity_type_t::emnist_input, "emnist");
                            needs_rebuild = true;
                        }
                    }
                    if (ImGui::MenuItem("KMNIST Input")) {
                        if (edit_callbacks_.add_node) {
                            edit_callbacks_.add_node(sim::io_entity_type_t::kmnist_input, "kmnist");
                            needs_rebuild = true;
                        }
                    }
                    if (ImGui::MenuItem("Color MNIST Input")) {
                        if (edit_callbacks_.add_node) {
                            edit_callbacks_.add_node(sim::io_entity_type_t::color_mnist_input, "color_mnist");
                            needs_rebuild = true;
                        }
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("dSprites Input")) {
                        if (edit_callbacks_.add_node) {
                            edit_callbacks_.add_node(sim::io_entity_type_t::dsprites_input, "dsprites");
                            needs_rebuild = true;
                        }
                    }
                    if (ImGui::MenuItem("CIFAR-10 Input")) {
                        if (edit_callbacks_.add_node) {
                            edit_callbacks_.add_node(sim::io_entity_type_t::cifar10_input, "cifar10");
                            needs_rebuild = true;
                        }
                    }
                    if (ImGui::MenuItem("CIFAR-100 Input")) {
                        if (edit_callbacks_.add_node) {
                            edit_callbacks_.add_node(sim::io_entity_type_t::cifar100_input, "cifar100");
                            needs_rebuild = true;
                        }
                    }
                    if (ImGui::MenuItem("3D Chairs Input")) {
                        if (edit_callbacks_.add_node) {
                            edit_callbacks_.add_node(sim::io_entity_type_t::chairs3d_input, "chairs3d");
                            needs_rebuild = true;
                        }
                    }
                    ImGui::EndMenu();
                }
            } else {
                // Node right-click - delete node
                auto* sim_node = dynamic_cast<SimNode*>(node);
                if (sim_node) {
                    if (ImGui::MenuItem("Delete Node")) {
                        if (edit_callbacks_.remove_node) {
                            edit_callbacks_.remove_node(sim_node->getNodeId());
                            needs_rebuild = true;
                        }
                    }
                }
            }
        });

        // Handle Delete key - we intercept it and prevent ImNodeFlow from seeing it
        bool delete_pressed = ImGui::IsKeyPressed(ImGuiKey_Delete, false);
        if (delete_pressed) {
            if (handle_delete_key()) {
                needs_rebuild = true;
            }
        }

        // Temporarily suppress Delete key state so ImNodeFlow doesn't process it
        ImGuiKeyData* delete_key_data = ImGui::GetKeyData(ImGuiKey_Delete);
        ImGuiKeyData saved_delete_state = *delete_key_data;
        if (delete_pressed) {
            delete_key_data->Down = false;
            delete_key_data->DownDuration = -1.0f;
            delete_key_data->DownDurationPrev = -1.0f;
        }

        // Render the node editor (now safe - Delete key is suppressed)
        editor->update();

        // Restore Delete key state
        *delete_key_data = saved_delete_state;

        // Detect and handle newly created links (and trigger rebuild if any)
        if (handle_new_links()) {
            needs_rebuild = true;
        }

        // Rebuild if any edits were made (from context menu or new links)
        if (needs_rebuild) {
            rebuild_from_config();
        }
    }
    ImGui::End();
}

bool network_editor::handle_delete_key() {
    bool any_deleted = false;

    // Collect nodes to delete (can't modify while iterating)
    std::vector<sim::sim_node_id_t> nodes_to_delete;
    for (auto& [uid, node] : editor->getNodes()) {
        if (node->isSelected()) {
            auto* sim_node = dynamic_cast<SimNode*>(node.get());
            if (sim_node) {
                nodes_to_delete.push_back(sim_node->getNodeId());
            }
        }
    }

    // Delete nodes via callback
    for (auto node_id : nodes_to_delete) {
        if (edit_callbacks_.remove_node && edit_callbacks_.remove_node(node_id)) {
            any_deleted = true;
        }
    }

    // Collect links to delete
    struct LinkInfo { sim::sim_node_id_t src; sim::sim_node_id_t dst; std::string channel; };
    std::vector<LinkInfo> links_to_delete;

    for (const auto& link_weak : editor->getLinks()) {
        if (auto link = link_weak.lock()) {
            if (link->isSelected()) {
                auto* out_pin = link->left();
                auto* in_pin = link->right();
                if (out_pin && in_pin) {
                    auto* src_node = dynamic_cast<SimNode*>(out_pin->getParent());
                    auto* dst_node = dynamic_cast<SimNode*>(in_pin->getParent());
                    if (src_node && dst_node) {
                        links_to_delete.push_back({
                            src_node->getNodeId(),
                            dst_node->getNodeId(),
                            in_pin->getName()
                        });
                    }
                }
            }
        }
    }

    // Delete links via callback
    for (const auto& link_info : links_to_delete) {
        if (edit_callbacks_.remove_link &&
            edit_callbacks_.remove_link(link_info.src, link_info.dst, link_info.channel)) {
            any_deleted = true;
        }
    }

    return any_deleted;
}

bool network_editor::handle_new_links() {
    bool any_added = false;

    // Check for links in ImNodeFlow that aren't in our link_states
    // These are newly created by user interaction
    for (const auto& link_weak : editor->getLinks()) {
        if (auto link = link_weak.lock()) {
            auto* out_pin = link->left();
            auto* in_pin = link->right();
            if (!out_pin || !in_pin) continue;

            auto* src_node = dynamic_cast<SimNode*>(out_pin->getParent());
            auto* dst_node = dynamic_cast<SimNode*>(in_pin->getParent());
            if (!src_node || !dst_node) continue;

            sim::sim_node_id_t src_id = src_node->getNodeId();
            sim::sim_node_id_t dst_id = dst_node->getNodeId();
            std::string channel = in_pin->getName();

            // Check if this link exists in our link_states
            bool found = false;
            for (const auto& ls : link_states) {
                if (ls.src_id == src_id && ls.dst_id == dst_id && ls.channel_name == channel) {
                    found = true;
                    break;
                }
            }

            // If not found, it's a new link - delegate to config_editor
            if (!found && edit_callbacks_.add_link) {
                if (edit_callbacks_.add_link(src_id, dst_id, channel)) {
                    any_added = true;
                }
            }
        }
    }

    return any_added;
}

void network_editor::rebuild_from_config() {
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

void network_editor::init_auto_layout() {
    auto_layout_active_ = true;
    sim_to_layout_id_.clear();

    // Create fresh layout graph
    layout_graph_ = std::make_unique<layout::layout_graph>();

    // Random generator for breaking symmetry
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise(-5.0f, 5.0f);

    // Add nodes from ImNodeFlow at their current positions
    for (auto& [uid, node] : editor->getNodes()) {
        auto* sim_node = dynamic_cast<SimNode*>(node.get());
        if (sim_node) {
            ImVec2 pos = node->getPos();
            ImVec2 size = node->getSize();
            // Use half the larger dimension as collision radius
            float radius = std::max(size.x, size.y) * 0.5f;
            // Add small random perturbation to break symmetry
            float x = pos.x + noise(gen);
            float y = pos.y + noise(gen);
            layout::node_id_t layout_id = layout_graph_->add_node(x, y, radius);
            sim_to_layout_id_[sim_node->getNodeId()] = layout_id;
        }
    }

    // Add edges from link_states
    for (const auto& link : link_states) {
        auto src_it = sim_to_layout_id_.find(link.src_id);
        auto dst_it = sim_to_layout_id_.find(link.dst_id);
        if (src_it != sim_to_layout_id_.end() && dst_it != sim_to_layout_id_.end()) {
            layout_graph_->add_edge(src_it->second, dst_it->second, 1.0f, 200.0f);
        }
    }

    // Configure force layout parameters tuned for editor scale
    layout::force_params params;
    params.repulsion = 8000.0f;       // Strong repulsion to keep nodes well separated
    params.attraction = 0.005f;       // Gentle edge pull
    params.damping = 0.85f;           // Smooth settling
    params.max_velocity = 30.0f;      // Smoother animation
    params.gravity = 0.001f;          // Light pull to keep graph centered

    // Set gravity center to current graph center
    auto [cx, cy] = layout_graph_->center();
    params.gravity_center_x = cx;
    params.gravity_center_y = cy;

    force_layout_ = std::make_unique<layout::force_layout>(*layout_graph_);
    force_layout_->set_params(params);
}

void network_editor::step_auto_layout() {
    if (!force_layout_ || !layout_graph_) return;

    // Run one simulation step
    force_layout_->step();

    // Update ImNodeFlow node positions from layout graph
    for (auto& [uid, node] : editor->getNodes()) {
        auto* sim_node = dynamic_cast<SimNode*>(node.get());
        if (!sim_node) continue;

        auto it = sim_to_layout_id_.find(sim_node->getNodeId());
        if (it == sim_to_layout_id_.end()) continue;

        const auto& layout_node = layout_graph_->node(it->second);
        ImVec2 new_pos(layout_node.x, layout_node.y);

        // Update ImNodeFlow position
        node->setPos(new_pos);

        // Also update our shadow state so positions persist after rebuild
        auto state_it = node_states.find(sim_node->getNodeId());
        if (state_it != node_states.end()) {
            state_it->second.position = new_pos;
            state_it->second.position_initialized = true;
        }
    }
}

void network_editor::stop_auto_layout() {
    auto_layout_active_ = false;
    force_layout_.reset();
    layout_graph_.reset();
    sim_to_layout_id_.clear();
}

void network_editor::reset_view() {
    // Reset viewport to default scroll and zoom
    editor->getGrid().setScroll(ImVec2(0.0f, 0.0f));
    editor->getGrid().setZoom(1.0f);

    auto& nodes = editor->getNodes();
    if (nodes.empty()) return;

    // Calculate current bounding box of all nodes
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (auto& [uid, node] : nodes) {
        ImVec2 pos = node->getPos();
        ImVec2 size = node->getSize();
        min_x = std::min(min_x, pos.x);
        min_y = std::min(min_y, pos.y);
        max_x = std::max(max_x, pos.x + size.x);
        max_y = std::max(max_y, pos.y + size.y);
    }

    // Calculate offset to center nodes at (400, 300) - a reasonable viewport position
    float center_x = (min_x + max_x) * 0.5f;
    float center_y = (min_y + max_y) * 0.5f;
    float offset_x = 400.0f - center_x;
    float offset_y = 300.0f - center_y;

    // Move all nodes by offset
    for (auto& [uid, node] : nodes) {
        ImVec2 pos = node->getPos();
        ImVec2 new_pos(pos.x + offset_x, pos.y + offset_y);
        node->setPos(new_pos);

        // Also update shadow state
        auto* sim_node = dynamic_cast<SimNode*>(node.get());
        if (sim_node) {
            auto it = node_states.find(sim_node->getNodeId());
            if (it != node_states.end()) {
                it->second.position = new_pos;
            }
        }
    }
}

} // namespace coast
