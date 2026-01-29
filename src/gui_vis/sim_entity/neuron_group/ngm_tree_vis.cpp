//
// Created by jk on 12.08.25.
//
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#include "inc_on_exit.h"
#include "sim_entity/neuron_group/ngm_tree_vis.h"
#include "vis_registry.h"
#include "sim_env.h"
#include "imgui.h"

namespace coast {

void ngm_tree_vis::get_pixel_data(
    int width, int height,
    uint16_t leaf_idx,
    std::size_t max_idx,
    const ngm2::dendrite_t::synapses_t &synapses,
    std::vector<color_t> &output
)
{
    output.resize(width*height);

    tmp_tree.clear();
    tmp_tree.resize(max_idx+1);

    tmp_tree[leaf_idx] = 1;
    while (leaf_idx > 1) {
        leaf_idx /= 2;
        tmp_tree[leaf_idx] = 1;
    }

    int pix = 0;
    std::size_t syn_cnt = synapses.size();
    for (std::size_t i = 0; i < syn_cnt; ++i)
        if (tmp_tree[synapses.segment_idx[i]]) {
            uint8_t value = static_cast<uint8_t>(std::clamp(synapses.permanence[i] * 255.0f, 0.0f, 255.0f));
            output[pix].r = value;
            output[pix].g = value;
            output[pix].b = value;
            output[pix].a = 255;
            ++pix;
        }
}

gl_texture ngm_tree_vis::get_texture(int width, int height, const color_t *data)
{
    return gl_texture(width, height, data);
}

ngm_tree_vis::~ngm_tree_vis()
{
}

void ngm_tree_vis::create_model(fbgd &vis, const ngm2::neuron_group_t &ng)
{
    nodes.clear();
    neuron_lu.clear();
    branch_start_lu.clear();
    branch_lu.clear();

    // determine size of nodes-array
    std::size_t nodes_size = 0;
    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx ) {
        const ngm2::neuron_t &cur_neuron = ng.get_neuron(neuron_idx);
        neuron_lu.push_back(nodes_size);
        branch_start_lu.push_back(branch_lu.size());
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t &cur_branch = cur_neuron.get_dendrite(branch_idx);
            std::size_t segment_cnt = cur_branch.get_max_segment_idx() + 1;
            branch_lu.push_back(nodes_size);
            nodes_size += segment_cnt;
        }
    }
    nodes.resize(nodes_size);

    // fill nodes
    auto grid_cnt = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    const float grid_spacing = coast::fbgd::get_region_size() / static_cast<float>(grid_cnt+2);
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx ) {
        const ngm2::neuron_t &cur_neuron = ng.get_neuron(neuron_idx);
        float xp = static_cast<float>(neuron_idx % grid_cnt) * grid_spacing + grid_spacing * 1.5f;
        float yp = static_cast<float>(neuron_idx / grid_cnt) * grid_spacing + grid_spacing * 1.5f;

        // create root node
        std::size_t root_node_idx = neuron_lu[neuron_idx];
        nodes[root_node_idx] = {
            neuron_idx,     //std::size_t neuron_idx
            0,              //std::size_t branch_idx
            0,              //uint16_t    segment_idx
            vis.add_node({xp,yp}, true, root_node_idx), //fbgd::node_id_t node_id
            1,               //uint8_t used
            {},
            std::vector<color_t>(),
            0
        };
        // go over all branches and create nodes where necessary
        std::size_t branch_base_idx = branch_start_lu[neuron_idx];
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t &cur_branch = cur_neuron.get_dendrite(branch_idx);
            auto leaf_mask = cur_branch.get_leaf_mask();
            std::size_t max_segment = cur_branch.get_max_segment_idx();
            std::size_t segment_base = branch_lu[branch_base_idx + branch_idx];
            const auto &synapses = cur_branch.get_synapses();
            std::size_t syn_cnt = synapses.size();
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                if (nodes[segment_base + synapses.segment_idx[i]].used == 0) {
                    gl_texture tex;
                    std::vector<color_t> pdata;
                    if (leaf_mask[synapses.segment_idx[i]]) {
                        get_pixel_data(28,28,synapses.segment_idx[i],max_segment,synapses,pdata);
                        tex = get_texture(28,28,pdata.data());
                    }
                    nodes[segment_base + synapses.segment_idx[i]] = {
                        neuron_idx,          //std::size_t neuron_idx
                        branch_idx,          //std::size_t branch_idx
                        synapses.segment_idx[i], //uint16_t    segment_idx
                        vis.add_node({xp + rdis(rgen) * (grid_spacing / 3), yp + rdis(rgen) * (grid_spacing / 3)},false,segment_base + synapses.segment_idx[i]),      //fbgd::node_id_t node_id
                        1,                    //uint8_t used
                        std::move(tex),
                        std::move(pdata),
                        0
                    };
                }
                nodes[segment_base + synapses.segment_idx[i]].synapse_count++;
            }
            // create edges
            const std::size_t max_idx = cur_branch.get_max_segment_idx();
            uint32_t cur_start = (max_idx + 1) / 2;
            while (cur_start > 1) {
                for (std::size_t seg_idx = cur_start; seg_idx < cur_start*2; ++seg_idx) {
                    if (nodes[segment_base + seg_idx].used == 1) {
                        vis.add_edge(nodes[segment_base + seg_idx/2].node_id, nodes[segment_base + seg_idx].node_id);
                        assert(nodes[segment_base + seg_idx/2].used == 1);
                    }
                }
                cur_start /= 2;
            }
            vis.add_edge(nodes[root_node_idx].node_id, nodes[segment_base + 1].node_id);
        }

    }
}

void ngm_tree_vis::update_model(fbgd &vis, const ngm2::neuron_group_t &ng)
{
    for (auto &node : nodes)
        node.synapse_count = 0;

    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx ) {
        const ngm2::neuron_t &cur_neuron = ng.get_neuron(neuron_idx);

        // go over all branches and create nodes where necessary
        std::size_t branch_base_idx = branch_start_lu[neuron_idx];
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t &cur_branch = cur_neuron.get_dendrite(branch_idx);
            auto leaf_mask = cur_branch.get_leaf_mask();
            std::size_t max_segment = cur_branch.get_max_segment_idx();
            std::size_t segment_base = branch_lu[branch_base_idx + branch_idx];
            const auto &synapses = cur_branch.get_synapses();
            std::size_t syn_cnt = synapses.size();
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                auto &cur_node = nodes[segment_base + synapses.segment_idx[i]];
                if (cur_node.pixel_data.empty() == false)
                {
                    if (leaf_mask[synapses.segment_idx[i]] == 0) {
                        cur_node.pixel_data.clear();
                    } else {
                        get_pixel_data(28,28,synapses.segment_idx[i],max_segment,synapses,cur_node.pixel_data);
                        cur_node.texture.update(cur_node.pixel_data);
                    }
                } else if (cur_node.used == 0) {
                    auto &vis_pos = vis.get_node_pos(nodes[segment_base + synapses.segment_idx[i]/2].node_id);
                    float vx = vis_pos[0] + rdis(rgen) * 0.0001f;
                    float vy = vis_pos[1] + rdis(rgen) * 0.0001f;
                    gl_texture tex;
                    std::vector<color_t> pdata;
                    if (leaf_mask[synapses.segment_idx[i]]) {
                        get_pixel_data(28,28,synapses.segment_idx[i],max_segment,synapses,pdata);
                        tex = get_texture(28,28,pdata.data());
                    }
                    cur_node = {
                        neuron_idx,          //std::size_t neuron_idx
                        branch_idx,          //std::size_t branch_idx
                        synapses.segment_idx[i], //uint16_t    segment_idx
                        vis.add_node({vx,vy},false, segment_base + synapses.segment_idx[i]),      //fbgd::node_id_t node_id
                        2,                    //uint8_t used
                        std::move(tex),
                        std::move(pdata),
                        0
                    };
                }
                cur_node.synapse_count++;
            }
            // create edges
            const std::size_t max_idx = cur_branch.get_max_segment_idx();
            uint32_t cur_start = (max_idx + 1) / 2;
            while (cur_start > 1) {
                for (std::size_t seg_idx = cur_start; seg_idx < cur_start*2; ++seg_idx) {
                    if (nodes[segment_base + seg_idx].used == 2) {
                        nodes[segment_base + seg_idx].used = 1;
                        vis.add_edge(nodes[segment_base + seg_idx/2].node_id, nodes[segment_base + seg_idx].node_id);
                        assert(nodes[segment_base + seg_idx/2].used > 0);
                    }
                }
                cur_start /= 2;
            }
        }
    }
}

void ngm_tree_vis::update_representations(const ngm2::neuron_group_t &ng)
{
    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx ) {
        const ngm2::neuron_t &cur_neuron = ng.get_neuron(neuron_idx);
        // go over all branches and create nodes where necessary
        std::size_t branch_base_idx = branch_start_lu[neuron_idx];
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t &cur_branch = cur_neuron.get_dendrite(branch_idx);
            std::size_t max_segment = cur_branch.get_max_segment_idx();
            std::size_t segment_base = branch_lu[branch_base_idx + branch_idx];
            const auto &synapses = cur_branch.get_synapses();
            std::size_t syn_cnt = synapses.size();
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                auto &cur_node = nodes[segment_base + synapses.segment_idx[i]];
                if (cur_node.pixel_data.empty() == false)
                {
                    get_pixel_data(28,28,synapses.segment_idx[i],max_segment,synapses,cur_node.pixel_data);
                    cur_node.texture.update(cur_node.pixel_data);
                }
            }
        }
    }
}

void ngm_tree_vis::free_resources()
{
    for (auto &node : nodes){
        node.texture.release();
        node.pixel_data.clear();
    }
}

void ngm_tree_vis::render_imgui(const char* label,
                           const std::vector<fbgd::node> &vis_nodes,
                           const std::vector<fbgd::edge> &vis_edges,
                           const std::vector<uint32_t> &vis_node_lu)
{
    ImGui::Begin(label);
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();

    // Draw canvas background
    draw_list->AddRectFilled(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
        IM_COL32(30, 30, 30, 255));

    // Scale factors
    const float region_size = fbgd::get_region_size();
    auto transform = [&](float x, float y) -> ImVec2 {
        return ImVec2(
            canvas_pos.x + (x / region_size) * canvas_size.x,
            canvas_pos.y + (y / region_size) * canvas_size.y
        );
    };

    // Draw edges
    for (auto &edge : vis_edges) {
        const auto &vnode1 = vis_nodes[vis_node_lu[edge.from]];
        const auto &vnode2 = vis_nodes[vis_node_lu[edge.to]];
        if (vnode1.node_id > vnode2.node_id)
            continue;
        ImVec2 pos1 = transform(vnode1.pos[0], vnode1.pos[1]);
        ImVec2 pos2 = transform(vnode2.pos[0], vnode2.pos[1]);

        auto node_idx = std::any_cast<std::size_t>(vnode2.payload);
        auto &node2 = nodes[node_idx];
        float sc_ratio = static_cast<float>(node2.synapse_count) / (28.0f * 28.0f);
        float thickness = 2.f + 8.f * sc_ratio;

        draw_list->AddLine(pos1, pos2, IM_COL32(0, 121, 241, 255), thickness);
    }

    // Draw nodes
    for (auto &vis_node : vis_nodes) {
        auto node_idx = std::any_cast<std::size_t>(vis_node.payload);
        auto &node = nodes[node_idx];
        ImVec2 pos = transform(vis_node.pos[0], vis_node.pos[1]);

        if (node.pixel_data.empty() == false && node.texture.is_valid()) {
            float scale = canvas_size.x / 1500.0f;  // Approximate scale
            float w = static_cast<float>(node.texture.get_width()) * scale;
            float h = static_cast<float>(node.texture.get_height()) * scale;
            ImVec2 tex_pos(pos.x - w / 2, pos.y - h / 2);
            draw_list->AddImage(node.texture.as_imgui_texture(),
                tex_pos, ImVec2(tex_pos.x + w, tex_pos.y + h));
        } else {
            float radius = node.segment_idx == 0 ? 6.0f : 3.0f;
            ImU32 color = node.segment_idx == 0 ? IM_COL32(255, 161, 0, 255) : IM_COL32(0, 0, 0, 255);
            draw_list->AddCircleFilled(pos, radius, color);
        }
    }

    ImGui::End();
}

// ============================================================================
// vis_instance adapter for registry integration
// ============================================================================

namespace {

/**
 * Synchronization states for thread-safe visualization updates.
 */
enum class sync_state : int {
    idle = 0,       // Normal operation, no update pending
    requested = 1,  // Sim thread requests update, waiting for GUI
    updating = 2    // GUI is performing update, sim still waiting
};

constexpr auto SYNC_TIMEOUT = std::chrono::milliseconds(100);

/**
 * Adapter that wraps ngm_tree_vis to implement vis_instance interface.
 * This adapter manages its own fbgd instance for layout.
 * Provides thread-safe updates synchronized with the simulation thread.
 */
class ngm_tree_vis_instance : public coast::vis::vis_instance {
public:
    ngm_tree_vis_instance(ngm2::neuron_group_t& ng,
                          sim::env& sim_env,
                          const coast::vis_hints_t& hints,
                          const std::string& hints_key,
                          std::string label)
        : ng_(ng)
        , sim_env_(sim_env)
        , label_(std::move(label))
    {
        // Get update interval from hints using hints_key (entity_cfg)
        auto it = hints.ngm2.neuron_groups.find(hints_key);
        if (it != hints.ngm2.neuron_groups.end()) {
            update_interval_ = static_cast<std::size_t>(it->second.ngm_flat_vis.update_interval);
        } else {
            auto def_it = hints.ngm2.neuron_groups.find("default");
            if (def_it != hints.ngm2.neuron_groups.end()) {
                update_interval_ = static_cast<std::size_t>(def_it->second.ngm_flat_vis.update_interval);
            }
        }

        vis_.create_model(fbgd_, ng_);

        // Register post-swap hook for thread-safe synchronization
        hook_id_ = sim_env_.set_post_swap_hook([this]() {
            on_post_swap();
        });
    }

    ~ngm_tree_vis_instance() override
    {
        // Unregister the hook before destruction
        sim_env_.remove_post_swap_hook(hook_id_);

        // Ensure we're not in the middle of an update
        while (sync_state_.load(std::memory_order_acquire) != static_cast<int>(sync_state::idle)) {
            std::this_thread::yield();
        }
    }

    /**
     * Called from GUI thread every frame.
     * Checks if sim thread requested an update and performs it if so.
     */
    void update() override
    {
        if (!visible_) return;

        // Check if sim thread has requested an update
        int expected = static_cast<int>(sync_state::requested);
        if (sync_state_.compare_exchange_strong(expected,
                                                 static_cast<int>(sync_state::updating),
                                                 std::memory_order_acq_rel)) {
            // Safe to read from neuron_group
            vis_.update_model(fbgd_, ng_);
            fbgd_.update();

            // Signal completion
            sync_state_.store(static_cast<int>(sync_state::idle), std::memory_order_release);
        }
    }

    bool render_imgui() override
    {
        if (!visible_) return true;

        // Call fbgd's draw callback to get latest node/edge data
        bool keep_open = true;

        fbgd_.set_draw_func([this](
            const std::vector<coast::fbgd::node>& nodes,
            const std::vector<coast::fbgd::edge>& edges,
            const std::vector<uint32_t>& node_lu)
        {
            vis_.render_imgui(label_.c_str(), nodes, edges, node_lu);
        });

        fbgd_.draw();

        return keep_open;
    }

    void free_resources() override
    {
        vis_.free_resources();
    }

    [[nodiscard]] bool is_visible() const override { return visible_; }
    void set_visible(bool v) override { visible_ = v; }

private:
    /**
     * Called from simulation thread via post_swap_hook.
     */
    void on_post_swap()
    {
        inc_on_exit inc{step_count_};  // Increment step_count_ on function exit

        if (update_interval_ == 0) return;
        if (!visible_) return;
        if (step_count_ % update_interval_ != 0) return;

        int expected = static_cast<int>(sync_state::idle);
        if (!sync_state_.compare_exchange_strong(expected,
                                                  static_cast<int>(sync_state::requested),
                                                  std::memory_order_acq_rel)) {
            return;
        }

        auto start = std::chrono::steady_clock::now();
        while (sync_state_.load(std::memory_order_acquire) != static_cast<int>(sync_state::idle)) {
            if (std::chrono::steady_clock::now() - start > SYNC_TIMEOUT) {
                expected = static_cast<int>(sync_state::requested);
                if (sync_state_.compare_exchange_strong(expected,
                                                         static_cast<int>(sync_state::idle),
                                                         std::memory_order_acq_rel)) {
                    break;
                }
            }
            std::this_thread::yield();
        }
    }

    ngm2::neuron_group_t& ng_;
    sim::env& sim_env_;
    std::string label_;
    coast::fbgd fbgd_;
    coast::ngm_tree_vis vis_;
    bool visible_ = true;

    // Synchronization state
    std::atomic<int> sync_state_{static_cast<int>(sync_state::idle)};
    std::size_t update_interval_ = 10000;
    std::size_t step_count_ = 0;
    std::size_t hook_id_ = 0;
};

/**
 * Factory function for creating ngm_tree_vis instances.
 */
std::unique_ptr<coast::vis::vis_instance> create_ngm_tree_vis(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    // The entity must be a neuron_group_t
    auto* ng = dynamic_cast<ngm2::neuron_group_t*>(&entity);
    if (!ng) {
        return nullptr;
    }
    return std::make_unique<ngm_tree_vis_instance>(*ng, sim_env, hints, hints_key, label);
}

/**
 * Static registrar - registers at program startup.
 */
struct ngm_tree_vis_registrar {
    ngm_tree_vis_registrar()
    {
        coast::vis::vis_info info {
            "ngm_tree_vis",
            "Neuron Tree",
            "Hierarchical tree view showing dendrite structure with force-directed layout",
            coast::vis::vis_target::sim_entity,
            { sim::io_entity_type_t::ngm2_neuron_group }
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm_tree_vis);
    }
};

static ngm_tree_vis_registrar registrar_instance;

} // anonymous namespace

} // coast
