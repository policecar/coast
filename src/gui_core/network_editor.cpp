#include "network_editor.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>

#include "imnodes.h"

namespace coast {

namespace {

// Entity type names for combo box
constexpr std::array entity_type_names = { "Neuron Group", "MNIST Input", "dSprites Input" };

} // anonymous namespace

network_editor::network_editor(loader::global_cfg_t& config)
    : config(config)
{
}

void network_editor::set_dirty_callback(dirty_callback_t callback)
{
    dirty_callback = std::move(callback);
}

std::vector<std::string> network_editor::get_input_channels(const sim::sim_node_t& node) const
{
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

int network_editor::find_node_index_by_id(sim::sim_node_id_t id) const
{
    const auto& nodes = config.sim.nodes;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].id == id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int network_editor::find_channel_index(const std::vector<std::string>& channels, const std::string& name) const
{
    for (std::size_t i = 0; i < channels.size(); ++i) {
        if (channels[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

sim::sim_node_id_t network_editor::generate_next_id() const
{
    sim::sim_node_id_t max_id = 0;
    for (const auto& node : config.sim.nodes) {
        if (node.id > max_id) {
            max_id = node.id;
        }
    }
    return max_id + 1;
}

void network_editor::mark_dirty()
{
    if (dirty_callback) {
        dirty_callback();
    }
}

void network_editor::delete_selected_nodes()
{
    auto& nodes = config.sim.nodes;
    int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected <= 0) return;

    std::vector<int> selected_node_ids(static_cast<std::size_t>(num_selected));
    ImNodes::GetSelectedNodes(selected_node_ids.data());

    // Convert ImNodes IDs (which are node.id values) to actual node IDs to delete
    std::vector<sim::sim_node_id_t> ids_to_delete;
    for (int node_id : selected_node_ids) {
        ids_to_delete.push_back(static_cast<sim::sim_node_id_t>(node_id));
    }

    // Remove links referencing deleted nodes
    for (auto& n : nodes) {
        n.inputs.erase(
            std::remove_if(n.inputs.begin(), n.inputs.end(),
                [&ids_to_delete](const auto& p) {
                    return std::find(ids_to_delete.begin(), ids_to_delete.end(), p.first) != ids_to_delete.end();
                }),
            n.inputs.end());
    }

    // Remove the nodes themselves
    nodes.erase(
        std::remove_if(nodes.begin(), nodes.end(),
            [&ids_to_delete](const auto& n) {
                return std::find(ids_to_delete.begin(), ids_to_delete.end(), n.id) != ids_to_delete.end();
            }),
        nodes.end());

    ImNodes::ClearNodeSelection();
    mark_dirty();
}

void network_editor::delete_selected_links()
{
    auto& nodes = config.sim.nodes;
    int num_selected = ImNodes::NumSelectedLinks();
    if (num_selected <= 0) return;

    std::vector<int> selected_link_ids(static_cast<std::size_t>(num_selected));
    ImNodes::GetSelectedLinks(selected_link_ids.data());

    // Build a set of link IDs to delete for fast lookup
    std::set<int> links_to_delete(selected_link_ids.begin(), selected_link_ids.end());

    // Iterate through all links and remove those that are selected
    int current_link_id = 0;
    for (auto& node : nodes) {
        auto input_channels = get_input_channels(node);

        auto it = node.inputs.begin();
        while (it != node.inputs.end()) {
            int src_node_idx = find_node_index_by_id(it->first);
            int channel_idx = find_channel_index(input_channels, it->second);

            if (src_node_idx >= 0 && channel_idx >= 0) {
                if (links_to_delete.count(current_link_id) > 0) {
                    it = node.inputs.erase(it);
                } else {
                    ++it;
                }
                ++current_link_id;
            } else {
                ++it;
            }
        }
    }

    ImNodes::ClearLinkSelection();
    mark_dirty();
}

void network_editor::render(bool* p_open)
{
    if (!*p_open) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Network Editor###NetworkEditor", p_open)) {
        auto& nodes = config.sim.nodes;

        // Toolbar
        if (ImGui::BeginMenuBar()) {
            // This won't show since we didn't request a menu bar
            ImGui::EndMenuBar();
        }

        // Toolbar buttons
        if (ImGui::Button("Add Node...")) {
            ImGui::OpenPopup("AddNodePopup");
        }

        ImGui::SameLine();

        // Delete button - enabled only if nodes or links are selected
        int num_selected_nodes = ImNodes::NumSelectedNodes();
        int num_selected_links = ImNodes::NumSelectedLinks();
        bool can_delete = (num_selected_nodes > 0) || (num_selected_links > 0);

        if (!can_delete) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Delete Selected")) {
            if (num_selected_nodes > 0) {
                delete_selected_nodes();
            }
            if (num_selected_links > 0) {
                delete_selected_links();
            }
        }
        if (!can_delete) {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();
        ImGui::Separator();
        ImGui::SameLine();

        // View controls
        if (ImGui::Button("Auto Layout")) {
            auto_layout();
        }

        ImGui::SameLine();

        if (ImGui::Button("Reset View")) {
            ImNodes::EditorContextResetPanning(ImVec2(0, 0));
        }

        ImGui::SameLine();
        ImGui::TextDisabled("(Pan: middle-drag | Right-click: menu)");

        // Add node popup
        if (ImGui::BeginPopup("AddNodePopup")) {
            if (ImGui::MenuItem("Neuron Group")) {
                sim::sim_node_t new_node;
                new_node.id = generate_next_id();
                new_node.label = "New Group";
                new_node.entity_type = sim::io_entity_type_t::ngm2_neuron_group;
                nodes.push_back(new_node);
                mark_dirty();
            }
            if (ImGui::MenuItem("MNIST Input")) {
                sim::sim_node_t new_node;
                new_node.id = generate_next_id();
                new_node.label = "New MNIST";
                new_node.entity_type = sim::io_entity_type_t::mnist_input;
                nodes.push_back(new_node);
                mark_dirty();
            }
            if (ImGui::MenuItem("dSprites Input")) {
                sim::sim_node_t new_node;
                new_node.id = generate_next_id();
                new_node.label = "New dSprites";
                new_node.entity_type = sim::io_entity_type_t::dsprites_input;
                nodes.push_back(new_node);
                mark_dirty();
            }
            ImGui::EndPopup();
        }

        ImGui::Separator();

        // Auto-layout on first open
        if (needs_layout && !nodes.empty()) {
            auto_layout();
            needs_layout = false;
        }

        ImNodes::BeginNodeEditor();

        // Render all nodes - use node.id as ImNodes node ID for stability
        for (std::size_t n = 0; n < nodes.size(); ++n) {
            auto& node = nodes[n];
            int imnodes_node_id = static_cast<int>(node.id);
            int attr_base = imnodes_node_id * 100;
            auto input_channels = get_input_channels(node);

            ImNodes::BeginNode(imnodes_node_id);

            // Title bar with label
            ImNodes::BeginNodeTitleBar();
            ImGui::Text("ID %d: %s", static_cast<int>(node.id), node.label.c_str());
            ImNodes::EndNodeTitleBar();

            // Editable label
            ImGui::SetNextItemWidth(120);
            // Use a buffer for InputText
            char label_buf[128];
            std::strncpy(label_buf, node.label.c_str(), sizeof(label_buf) - 1);
            label_buf[sizeof(label_buf) - 1] = '\0';
            if (ImGui::InputText("##label", label_buf, sizeof(label_buf))) {
                node.label = label_buf;
                mark_dirty();
            }

            // Entity type combo
            ImGui::SetNextItemWidth(120);
            int type_idx = static_cast<int>(node.entity_type);
            if (ImGui::Combo("##type", &type_idx, entity_type_names.data(),
                             static_cast<int>(entity_type_names.size()))) {
                node.entity_type = static_cast<sim::io_entity_type_t>(type_idx);
                // Clear inputs when type changes - input channels may differ
                node.inputs.clear();
                mark_dirty();
            }

            // Entity config combo - options depend on entity type
            ImGui::SetNextItemWidth(120);
            std::string old_cfg = node.entity_cfg;
            if (ImGui::BeginCombo("##config", node.entity_cfg.empty() ? "(none)" : node.entity_cfg.c_str())) {
                switch (node.entity_type) {
                    case sim::io_entity_type_t::ngm2_neuron_group:
                        for (const auto& [key, val] : config.ngm2.neuron_groups) {
                            if (ImGui::Selectable(key.c_str(), node.entity_cfg == key)) {
                                node.entity_cfg = key;
                            }
                        }
                        break;
                    case sim::io_entity_type_t::mnist_input:
                        for (const auto& [key, val] : config.datasets.mnist) {
                            if (ImGui::Selectable(key.c_str(), node.entity_cfg == key)) {
                                node.entity_cfg = key;
                            }
                        }
                        break;
                    case sim::io_entity_type_t::dsprites_input:
                        for (const auto& [key, val] : config.datasets.dsprites) {
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
            // Clear inputs when config changes - input channels may differ
            if (node.entity_cfg != old_cfg) {
                node.inputs.clear();
                mark_dirty();
            }

            // Input pins - one per available input channel
            for (std::size_t i = 0; i < input_channels.size(); ++i) {
                int input_attr_id = attr_base + 1 + static_cast<int>(i);
                ImNodes::BeginInputAttribute(input_attr_id);
                ImGui::Text("%s", input_channels[i].c_str());
                ImNodes::EndInputAttribute();
            }

            // Single output pin - right-aligned
            ImNodes::BeginOutputAttribute(attr_base);
            float text_width = ImGui::CalcTextSize("out").x;
            float node_width = 120.0f;  // Match the combo box width
            ImGui::Indent(node_width - text_width);
            ImGui::TextUnformatted("out");
            ImGui::Unindent(node_width - text_width);
            ImNodes::EndOutputAttribute();

            ImNodes::EndNode();
        }

        // Render links - use stable IDs based on content
        int link_id = 0;
        for (std::size_t n = 0; n < nodes.size(); ++n) {
            auto& node = nodes[n];
            int dst_node_imnodes_id = static_cast<int>(node.id);
            auto input_channels = get_input_channels(node);

            for (const auto& [src_id, channel_name] : node.inputs) {
                int src_node_idx = find_node_index_by_id(src_id);
                int channel_idx = find_channel_index(input_channels, channel_name);

                if (src_node_idx >= 0 && channel_idx >= 0) {
                    int from_attr = static_cast<int>(src_id) * 100;
                    int to_attr = dst_node_imnodes_id * 100 + 1 + channel_idx;
                    ImNodes::Link(link_id++, from_attr, to_attr);
                }
            }
        }

        ImNodes::EndNodeEditor();

        // Context menu on right-click in editor area
        // Use ChildWindows flag since ImNodes creates a child canvas
        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByPopup) &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            context_menu_pos = ImGui::GetMousePos();
            int hovered_node = -1;
            if (ImNodes::IsNodeHovered(&hovered_node)) {
                context_menu_node = hovered_node;
            } else {
                context_menu_node = -1;
            }
            show_context_menu = true;
        }
        render_context_menu();

        // Handle link creation
        int start_attr, end_attr;
        if (ImNodes::IsLinkCreated(&start_attr, &end_attr)) {
            // start_attr is output (node.id * 100), end_attr is input (node.id * 100 + 1 + channel_idx)
            sim::sim_node_id_t src_node_id = static_cast<sim::sim_node_id_t>(start_attr / 100);
            sim::sim_node_id_t dst_node_id = static_cast<sim::sim_node_id_t>(end_attr / 100);
            int channel_idx = (end_attr % 100) - 1;

            int dst_node_idx = find_node_index_by_id(dst_node_id);
            if (dst_node_idx >= 0 && channel_idx >= 0) {
                auto& dst_node = nodes[static_cast<std::size_t>(dst_node_idx)];
                auto input_channels = get_input_channels(dst_node);

                if (channel_idx < static_cast<int>(input_channels.size())) {
                    std::string channel_name = input_channels[static_cast<std::size_t>(channel_idx)];

                    // Check if this connection already exists
                    bool exists = false;
                    for (const auto& [existing_src, existing_chan] : dst_node.inputs) {
                        if (existing_src == src_node_id && existing_chan == channel_name) {
                            exists = true;
                            break;
                        }
                    }

                    if (!exists) {
                        dst_node.inputs.emplace_back(src_node_id, channel_name);
                        mark_dirty();
                    }
                }
            }
        }

        // Handle link deletion via ImNodes (middle-click or other ImNodes method)
        int destroyed_link_id;
        if (ImNodes::IsLinkDestroyed(&destroyed_link_id)) {
            // Find and remove the link by iterating
            int current_link_id = 0;
            for (auto& node : nodes) {
                auto input_channels = get_input_channels(node);

                for (auto it = node.inputs.begin(); it != node.inputs.end(); ++it) {
                    int src_node_idx = find_node_index_by_id(it->first);
                    int channel_idx = find_channel_index(input_channels, it->second);

                    if (src_node_idx >= 0 && channel_idx >= 0) {
                        if (current_link_id == destroyed_link_id) {
                            node.inputs.erase(it);
                            mark_dirty();
                            goto link_deleted;
                        }
                        ++current_link_id;
                    }
                }
            }
            link_deleted:;
        }

        // Handle Delete key for both nodes and links
        if (ImGui::IsKeyPressed(ImGuiKey_Delete)) {
            if (ImNodes::NumSelectedNodes() > 0) {
                delete_selected_nodes();
            }
            if (ImNodes::NumSelectedLinks() > 0) {
                delete_selected_links();
            }
        }
    }
    ImGui::End();
}

void network_editor::render_context_menu()
{
    if (show_context_menu) {
        ImGui::OpenPopup("NodeContextMenu");
        show_context_menu = false;
    }

    if (ImGui::BeginPopup("NodeContextMenu")) {
        auto& nodes = config.sim.nodes;

        if (context_menu_node < 0) {
            // Background context menu - add new node
            if (ImGui::BeginMenu("Add Node")) {
                if (ImGui::MenuItem("Neuron Group")) {
                    sim::sim_node_t new_node;
                    new_node.id = generate_next_id();
                    new_node.label = "New Group";
                    new_node.entity_type = sim::io_entity_type_t::ngm2_neuron_group;
                    nodes.push_back(new_node);
                    ImNodes::SetNodeScreenSpacePos(
                        static_cast<int>(new_node.id),
                        context_menu_pos);
                    mark_dirty();
                }
                if (ImGui::MenuItem("MNIST Input")) {
                    sim::sim_node_t new_node;
                    new_node.id = generate_next_id();
                    new_node.label = "New MNIST";
                    new_node.entity_type = sim::io_entity_type_t::mnist_input;
                    nodes.push_back(new_node);
                    ImNodes::SetNodeScreenSpacePos(
                        static_cast<int>(new_node.id),
                        context_menu_pos);
                    mark_dirty();
                }
                if (ImGui::MenuItem("dSprites Input")) {
                    sim::sim_node_t new_node;
                    new_node.id = generate_next_id();
                    new_node.label = "New dSprites";
                    new_node.entity_type = sim::io_entity_type_t::dsprites_input;
                    nodes.push_back(new_node);
                    ImNodes::SetNodeScreenSpacePos(
                        static_cast<int>(new_node.id),
                        context_menu_pos);
                    mark_dirty();
                }
                ImGui::EndMenu();
            }

            // Delete selected - enabled only if something is selected
            int num_selected_nodes = ImNodes::NumSelectedNodes();
            int num_selected_links = ImNodes::NumSelectedLinks();
            bool has_selection = (num_selected_nodes > 0) || (num_selected_links > 0);

            if (!has_selection) {
                ImGui::BeginDisabled();
            }
            if (ImGui::MenuItem("Delete Selected")) {
                if (num_selected_nodes > 0) {
                    delete_selected_nodes();
                }
                if (num_selected_links > 0) {
                    delete_selected_links();
                }
            }
            if (!has_selection) {
                ImGui::EndDisabled();
            }
        } else {
            // Node context menu - context_menu_node is the node.id
            if (ImGui::MenuItem("Delete Node")) {
                int node_idx = find_node_index_by_id(static_cast<sim::sim_node_id_t>(context_menu_node));
                if (node_idx >= 0) {
                    auto deleted_id = nodes[static_cast<std::size_t>(node_idx)].id;
                    // Remove links referencing this node
                    for (auto& n : nodes) {
                        n.inputs.erase(
                            std::remove_if(n.inputs.begin(), n.inputs.end(),
                                [deleted_id](const auto& p) { return p.first == deleted_id; }),
                            n.inputs.end());
                    }
                    nodes.erase(nodes.begin() + node_idx);
                    mark_dirty();
                }
            }
        }
        ImGui::EndPopup();
    }
}

void network_editor::auto_layout()
{
    auto& nodes = config.sim.nodes;
    if (nodes.empty()) return;

    // Layout parameters
    constexpr float node_width = 150.0f;
    constexpr float node_height = 120.0f;
    constexpr float horizontal_spacing = 80.0f;
    constexpr float vertical_spacing = 40.0f;
    constexpr float start_x = 50.0f;
    constexpr float start_y = 50.0f;

    // Step 1: Compute layer (depth) for each node using BFS from sources
    // Layer 0 = nodes with no valid inputs (sources)
    std::unordered_map<sim::sim_node_id_t, int> node_layer;

    // Initialize: find source nodes (no inputs or inputs from non-existent nodes)
    for (const auto& node : nodes) {
        bool has_valid_input = false;
        for (const auto& [src_id, channel] : node.inputs) {
            if (find_node_index_by_id(src_id) >= 0) {
                has_valid_input = true;
                break;
            }
        }
        if (!has_valid_input) {
            node_layer[node.id] = 0;
        }
    }

    // Iteratively assign layers: node's layer = max(input layers) + 1
    bool changed = true;
    int max_iterations = static_cast<int>(nodes.size()) + 1;
    while (changed && max_iterations-- > 0) {
        changed = false;
        for (const auto& node : nodes) {
            if (node_layer.count(node.id) > 0) continue;  // Already assigned

            int max_input_layer = -1;
            bool all_inputs_assigned = true;

            for (const auto& [src_id, channel] : node.inputs) {
                if (find_node_index_by_id(src_id) < 0) continue;  // Invalid input

                auto it = node_layer.find(src_id);
                if (it != node_layer.end()) {
                    max_input_layer = std::max(max_input_layer, it->second);
                } else {
                    all_inputs_assigned = false;
                    break;
                }
            }

            if (all_inputs_assigned && max_input_layer >= 0) {
                node_layer[node.id] = max_input_layer + 1;
                changed = true;
            }
        }
    }

    // Handle remaining nodes (in cycles)
    // Find current max layer
    int max_layer = 0;
    for (const auto& [id, layer] : node_layer) {
        max_layer = std::max(max_layer, layer);
    }

    // Collect unassigned nodes
    std::vector<sim::sim_node_id_t> cycle_nodes;
    for (const auto& node : nodes) {
        if (node_layer.count(node.id) == 0) {
            cycle_nodes.push_back(node.id);
        }
    }

    // Iteratively assign cycle nodes based on their assigned inputs
    bool made_progress = true;
    while (made_progress && !cycle_nodes.empty()) {
        made_progress = false;

        for (auto it = cycle_nodes.begin(); it != cycle_nodes.end(); ) {
            sim::sim_node_id_t node_id = *it;
            int node_idx = find_node_index_by_id(node_id);
            if (node_idx < 0) {
                ++it;
                continue;
            }

            const auto& node = nodes[static_cast<std::size_t>(node_idx)];
            int best_layer = -1;

            // Find max layer among assigned inputs
            for (const auto& [src_id, channel] : node.inputs) {
                auto layer_it = node_layer.find(src_id);
                if (layer_it != node_layer.end()) {
                    best_layer = std::max(best_layer, layer_it->second);
                }
            }

            if (best_layer >= 0) {
                // Can assign based on an assigned input
                node_layer[node_id] = best_layer + 1;
                max_layer = std::max(max_layer, best_layer + 1);
                it = cycle_nodes.erase(it);
                made_progress = true;
            } else {
                ++it;
            }
        }
    }

    // Remaining nodes are in isolated cycles - place them sequentially after max_layer
    if (!cycle_nodes.empty()) {
        int cycle_layer = max_layer + 1;
        for (sim::sim_node_id_t node_id : cycle_nodes) {
            node_layer[node_id] = cycle_layer++;
        }
    }

    // Step 2: Group nodes by layer
    std::map<int, std::vector<sim::sim_node_id_t>> layers;
    for (const auto& node : nodes) {
        layers[node_layer[node.id]].push_back(node.id);
    }

    // Step 3: Position nodes
    for (const auto& [layer, layer_nodes] : layers) {
        float x = start_x + static_cast<float>(layer) * (node_width + horizontal_spacing);

        for (std::size_t i = 0; i < layer_nodes.size(); ++i) {
            float y = start_y + static_cast<float>(i) * (node_height + vertical_spacing);
            ImNodes::SetNodeGridSpacePos(static_cast<int>(layer_nodes[i]), ImVec2(x, y));
        }
    }

    // Reset panning to show the layout from the origin
    ImNodes::EditorContextResetPanning(ImVec2(0, 0));
}

} // namespace coast
