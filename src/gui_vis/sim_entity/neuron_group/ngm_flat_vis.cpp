//
// Created by jk on 08.09.25.
//

#include "core/gradient.h"
#include "inc_on_exit.h"

#include "sim_entity/neuron_group/ngm_flat_vis.h"
#include "vis_registry.h"
#include "sim_env.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <oneapi/tbb/task_arena.h>

#include "imgui.h"

namespace coast {

ngm_flat_vis::ngm_flat_vis(const ngm2::neuron_group_t &neuron_group, const params_t _params) :
    params(_params),
    ng(neuron_group)
{
    // determine the maximum required size of the pixeldata and texture
    px_width  = 0;
    px_height = 0;
    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t n = 0; n < neuron_cnt; ++n) {
        auto &neuron = ng.get_neuron(n);
        std::size_t dendrite_cnt = neuron.get_dendrite_count();
        uint32_t max_dendrite_height = 0;
        uint32_t max_dendrite_width  = 0;
        uint32_t neuron_width = 0;
        uint32_t neuron_height = 0;
        for (std::size_t d = 0; d < dendrite_cnt; ++d) {
            auto &dendrite = neuron.get_dendrite(d);
            const uint32_t max_rep_cnt = (dendrite.get_max_segment_idx() + 1) / 2;
            params.vis_params.vec_cnt = max_rep_cnt;
            vec_group_vis tmp {params.vis_params};
            const uint32_t dendrite_width = tmp.get_total_width();
            const uint32_t dendrite_height = tmp.get_total_height();
            switch (params.layout) {
                case layout_t::horizontal_per_neuron : neuron_width  += dendrite_width + params.vis_params.margin * 2;  break;
                case layout_t::vertical_per_neuron   : neuron_height += dendrite_height + params.vis_params.margin * 2; break;
            }
            max_dendrite_height = std::max(max_dendrite_height, dendrite_height);
            max_dendrite_width  = std::max(max_dendrite_width,  dendrite_width);
        }
        switch (params.layout) {
            case layout_t::horizontal_per_neuron : {
                px_width = std::max(px_width,neuron_width);
                px_height += max_dendrite_height;
            } break;
            case layout_t::vertical_per_neuron   : {
                px_width  += max_dendrite_width;
                px_height = std::max(px_height,neuron_height);
            }break;
        }
    }
    // create actual pixel data
    pixel_data.resize(px_width * px_height, color_t{255, 255, 0, 255});
    texture = gl_texture(static_cast<int>(px_width), static_cast<int>(px_height), pixel_data.data());
}

void ngm_flat_vis::update()
{
    color_t black{0, 0, 0, 255};
    color_t white{255, 255, 255, 255};
    gradient<2> def_grad(std::array<color_t, 2>{black, white});
    uint32_t group_pos_x = 0;
    uint32_t group_pos_y = 0;
    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t n = 0; n < neuron_cnt; ++n) {
        auto &neuron = ng.get_neuron(n);
        std::size_t dendrite_cnt = neuron.get_dendrite_count();
        uint32_t max_dendrite_height = 0;
        uint32_t max_dendrite_width  = 0;
        for (std::size_t d = 0; d < dendrite_cnt; ++d) {
            auto &dendrite = neuron.get_dendrite(d);
            const uint32_t max_rep_cnt = (dendrite.get_max_segment_idx() + 1) / 2;
            params.vis_params.vec_cnt = max_rep_cnt;
            vec_group_vis vis{params.vis_params};
            const uint32_t rep_cnt = dendrite.get_representation_count();
            if (rep_cnt > max_rep_cnt) {
                std::printf("too many representations: %u vs %u\n", rep_cnt, max_rep_cnt);
                auto mask = dendrite.get_leaf_mask();
                for (auto m : mask)
                    std::printf("%u,", m);
                std::printf("\n");
            }
            for (uint32_t r = 0; r < rep_cnt; ++r)
                vis.update_vec<2>(r, dendrite.get_representation(r), def_grad,
                                  px_pos_t{group_pos_x, group_pos_y},
                                  pixel_data_t{pixel_data, px_width, px_height});
            const uint32_t dendrite_width = vis.get_total_width();
            const uint32_t dendrite_height = vis.get_total_height();
            switch (params.layout) {
                case layout_t::horizontal_per_neuron : group_pos_x += dendrite_width + params.vis_params.margin * 2;  break;
                case layout_t::vertical_per_neuron   : group_pos_y += dendrite_height + params.vis_params.margin * 2; break;
            }
            max_dendrite_height = std::max(max_dendrite_height, dendrite_height);
            max_dendrite_width  = std::max(max_dendrite_width,  dendrite_width);
        }
        switch (params.layout) {
            case layout_t::horizontal_per_neuron : {
                group_pos_x  = 0;
                group_pos_y += max_dendrite_height;
            } break;
            case layout_t::vertical_per_neuron   : {
                group_pos_x += max_dendrite_width;
                group_pos_y  = 0;
            } break;
        }
    }
    texture.update(pixel_data);
}

bool ngm_flat_vis::render_imgui(const char* label, float scale)
{
    bool open = true;
    if (ImGui::Begin(label, &open)) {
        // Zoom controls
        ImGui::Text("Zoom: %.1fx", zoom_level);
        ImGui::SameLine();
        if (ImGui::Button("-")) {
            zoom_level = std::max(0.25f, zoom_level - 0.25f);
        }
        ImGui::SameLine();
        if (ImGui::Button("+")) {
            zoom_level = std::min(4.0f, zoom_level + 0.25f);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            zoom_level = 1.0f;
        }

        ImGui::Separator();

        if (texture.is_valid()) {
            // Calculate scaled size
            float effective_scale = scale * zoom_level;
            ImVec2 size(static_cast<float>(texture.get_width()) * effective_scale,
                        static_cast<float>(texture.get_height()) * effective_scale);

            // Scrollable region with both horizontal and vertical scrollbars
            ImGui::BeginChild("ScrollRegion", ImVec2(0, 0), false,
                              ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::Image(texture.as_imgui_texture(), size);
            ImGui::EndChild();
        }
    }
    ImGui::End();
    return open;
}

void ngm_flat_vis::free_resources()
{
    texture.release();
}

ngm_flat_vis::params_t ngm_flat_vis::get_default(uint32_t rep_width, uint32_t rep_height)
{
    params_t result {};
    result.layout = layout_t::vertical_per_neuron;
    result.vis_params.layout = vec_group_vis::layout_t::vertical;
    result.vis_params.margin = 5;
    result.vis_params.vec_params.elem_height = 1;
    result.vis_params.vec_params.elem_width  = 1;
    result.vis_params.vec_params.rep_width   = rep_width;
    result.vis_params.vec_params.rep_height  = rep_height;
    return result;
}

uint32_t ngm_flat_vis::get_height() const
{
    return px_height;
}

uint32_t ngm_flat_vis::get_width() const
{
    return px_width;
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
 * Adapter that wraps ngm_flat_vis to implement vis_instance interface.
 * Provides thread-safe updates synchronized with the simulation thread.
 */
class ngm_flat_vis_instance : public coast::vis::vis_instance {
public:
    ngm_flat_vis_instance(ngm2::neuron_group_t& ng,
                          sim::env& sim_env,
                          const coast::vis_hints_t& hints,
                          const std::string& hints_key,
                          std::string label)
        : ng_(ng)
        , sim_env_(sim_env)
        , label_(std::move(label))
    {
        // Try to find hints for this specific neuron group using hints_key (entity_cfg)
        auto it = hints.ngm2.neuron_groups.find(hints_key);
        coast::ngm_flat_vis_hint_t hint;
        if (it != hints.ngm2.neuron_groups.end()) {
            hint = it->second.ngm_flat_vis;
        } else {
            // Use default
            auto def_it = hints.ngm2.neuron_groups.find("default");
            if (def_it != hints.ngm2.neuron_groups.end()) {
                hint = def_it->second.ngm_flat_vis;
            }
        }

        update_interval_ = static_cast<std::size_t>(hint.update_interval);

        auto params = coast::ngm_flat_vis::get_default(
            static_cast<uint32_t>(hint.width),
            static_cast<uint32_t>(hint.height)
        );
        vis_ = std::make_unique<coast::ngm_flat_vis>(ng_, params);

        // Register post-swap hook for thread-safe synchronization
        hook_id_ = sim_env_.set_post_swap_hook([this]() {
            on_post_swap();
        });
    }

    ~ngm_flat_vis_instance() override
    {
        // Unregister the hook before destruction
        sim_env_.remove_post_swap_hook(hook_id_);

        // Ensure we're not in the middle of an update
        // Wait for any pending update to complete
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
        if (!vis_ || !visible_) return;

        // Check if sim thread has requested an update
        int expected = static_cast<int>(sync_state::requested);
        if (sync_state_.compare_exchange_strong(expected,
                                                 static_cast<int>(sync_state::updating),
                                                 std::memory_order_acq_rel)) {
            // We successfully transitioned from 'requested' to 'updating'
            // Sim thread is now waiting, safe to read from neuron_group
            vis_->update();

            // Signal completion - transition to 'idle'
            sync_state_.store(static_cast<int>(sync_state::idle), std::memory_order_release);
        }
    }

    bool render_imgui() override
    {
        if (!vis_ || !visible_) return true;
        return vis_->render_imgui(label_.c_str());
    }

    void free_resources() override
    {
        if (vis_) {
            vis_->free_resources();
        }
    }

    [[nodiscard]] bool is_visible() const override { return visible_; }
    void set_visible(bool v) override
    {
        visible_ = v;
        if (vis_) {
            vis_->set_visible(v);
        }
    }

private:
    /**
     * Called from simulation thread via post_swap_hook.
     * Requests an update and waits for GUI thread to complete it.
     */
    void on_post_swap()
    {
        inc_on_exit inc{step_count_};  // Increment step_count_ on function exit

        // Check if it's time for an update
        if (update_interval_ == 0) return;  // 0 means disabled
        if (!visible_) return;  // Don't update if not visible

        if (step_count_ % update_interval_ != 0) return;

        // Request an update
        int expected = static_cast<int>(sync_state::idle);
        if (!sync_state_.compare_exchange_strong(expected,
                                                  static_cast<int>(sync_state::requested),
                                                  std::memory_order_acq_rel)) {
            // State wasn't idle - previous update still in progress, skip
            return;
        }

        // Wait for GUI thread to complete the update
        auto start = std::chrono::steady_clock::now();
        while (sync_state_.load(std::memory_order_acquire) != static_cast<int>(sync_state::idle)) {
            // Check for timeout
            if (std::chrono::steady_clock::now() - start > SYNC_TIMEOUT) {
                // Timeout - try to cancel if still in 'requested' state
                expected = static_cast<int>(sync_state::requested);
                if (sync_state_.compare_exchange_strong(expected,
                                                         static_cast<int>(sync_state::idle),
                                                         std::memory_order_acq_rel)) {
                    // Successfully cancelled - GUI didn't start yet
                    break;
                }
                // CAS failed - GUI has started updating, must wait for completion
            }
            std::this_thread::yield();
        }
    }

    ngm2::neuron_group_t& ng_;
    sim::env& sim_env_;
    std::string label_;
    std::unique_ptr<coast::ngm_flat_vis> vis_;
    bool visible_ = true;

    // Synchronization state
    std::atomic<int> sync_state_{static_cast<int>(sync_state::idle)};
    std::size_t update_interval_ = 10000;
    std::size_t step_count_ = 0;
    std::size_t hook_id_ = 0;
};

/**
 * Factory function for creating ngm_flat_vis instances.
 */
std::unique_ptr<coast::vis::vis_instance> create_ngm_flat_vis(
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
    return std::make_unique<ngm_flat_vis_instance>(*ng, sim_env, hints, hints_key, label);
}

/**
 * Static registrar - registers at program startup.
 */
struct ngm_flat_vis_registrar {
    ngm_flat_vis_registrar()
    {
        coast::vis::vis_info info {
            "ngm_flat_vis",
            "Dendritic Representations",
            "Flat grid view showing neuron representations",
            coast::vis::vis_target::sim_entity,
            { sim::io_entity_type_t::ngm2_neuron_group }
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm_flat_vis);
    }
};

static ngm_flat_vis_registrar registrar_instance;

} // anonymous namespace

} // coast
