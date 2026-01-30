//
// Created by jk on 08.09.25.
//

#include "core/gradient.h"
#include "core/colormap.h"
#include "inc_on_exit.h"

#include "sim_entity/neuron_group/ngm_flat_vis.h"
#include "vis_registry.h"
#include "sim_env.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <oneapi/tbb/task_arena.h>

#include "imgui.h"
#include "imgui_scaling.h"

namespace coast {

namespace {

// Helper to create a 5-stop gradient from a colormap
gradient<5> make_gradient(colormap_t cm)
{
    return gradient<5>(std::array<color_t, 5>{
        apply_colormap(cm, 0.00f),
        apply_colormap(cm, 0.25f),
        apply_colormap(cm, 0.50f),
        apply_colormap(cm, 0.75f),
        apply_colormap(cm, 1.00f)
    });
}

} // anonymous namespace

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
    auto def_grad = make_gradient(colormap);
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
                vis.update_vec<5>(r, dendrite.get_representation(r), def_grad,
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

    // Set initial window size on first appearance
    ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(label, &open)) {
        // Toolbar with zoom display, reset button, and colormap selector
        ImGui::Text("Zoom: %.0f%%", zoom_level * 100.0f);
        ImGui::SameLine();
        if (ImGui::Button("Reset View")) {
            target_zoom = 1.0f;
            target_pan_x = 0.0f;
            target_pan_y = 0.0f;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(scaled(100));
        if (ImGui::BeginCombo("##colormap", colormap_name(colormap))) {
            for (int i = 0; i < static_cast<int>(colormap_t::COUNT); ++i) {
                auto cm = static_cast<colormap_t>(i);
                bool is_selected = (colormap == cm);
                if (ImGui::Selectable(colormap_name(cm), is_selected)) {
                    if (colormap != cm) {
                        colormap = cm;
                        colormap_dirty = true;  // Trigger re-render
                    }
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::Separator();

        if (texture.is_valid()) {
            // Calculate scaled size
            float effective_scale = scale * zoom_level;
            ImVec2 tex_size(static_cast<float>(texture.get_width()) * effective_scale,
                           static_cast<float>(texture.get_height()) * effective_scale);

            // Get available content region
            ImVec2 canvas_size = ImGui::GetContentRegionAvail();
            ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

            // Create an invisible button to capture input over the canvas area
            ImGui::InvisibleButton("canvas", canvas_size,
                                   ImGuiButtonFlags_MouseButtonLeft);
            bool is_hovered = ImGui::IsItemHovered();
            bool is_active = ImGui::IsItemActive();

            ImGuiIO& io = ImGui::GetIO();

            // Smooth interpolation factor (higher = faster response)
            constexpr float lerp_speed = 15.0f;
            float lerp_t = std::min(1.0f, lerp_speed * io.DeltaTime);

            // Handle scroll wheel zoom (only when hovered)
            if (is_hovered && io.MouseWheel != 0.0f) {
                // Get mouse position relative to canvas
                ImVec2 mouse_pos = io.MousePos;
                float mouse_canvas_x = mouse_pos.x - canvas_pos.x;
                float mouse_canvas_y = mouse_pos.y - canvas_pos.y;

                // Calculate position in texture space before zoom (using current values)
                float tex_x = (mouse_canvas_x + pan_x) / (scale * zoom_level);
                float tex_y = (mouse_canvas_y + pan_y) / (scale * zoom_level);

                // Apply zoom to target (exponential for smoother feel)
                float zoom_factor = std::pow(1.15f, io.MouseWheel);
                target_zoom = std::clamp(target_zoom * zoom_factor, 0.1f, 10.0f);

                // Adjust target pan to keep the point under the mouse fixed
                float new_scale = scale * target_zoom;
                target_pan_x = tex_x * new_scale - mouse_canvas_x;
                target_pan_y = tex_y * new_scale - mouse_canvas_y;
            }

            // Handle left-click drag for panning (immediate response feels better)
            if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                target_pan_x -= io.MouseDelta.x;
                target_pan_y -= io.MouseDelta.y;
                // Also update current pan immediately for responsive drag
                pan_x -= io.MouseDelta.x;
                pan_y -= io.MouseDelta.y;
            }

            // Clamp target pan to valid range
            float target_tex_size_x = static_cast<float>(texture.get_width()) * scale * target_zoom;
            float target_tex_size_y = static_cast<float>(texture.get_height()) * scale * target_zoom;
            float max_target_pan_x = std::max(0.0f, target_tex_size_x - canvas_size.x);
            float max_target_pan_y = std::max(0.0f, target_tex_size_y - canvas_size.y);
            target_pan_x = std::clamp(target_pan_x, 0.0f, max_target_pan_x);
            target_pan_y = std::clamp(target_pan_y, 0.0f, max_target_pan_y);

            // Smoothly interpolate current values towards targets
            zoom_level = zoom_level + (target_zoom - zoom_level) * lerp_t;
            pan_x = pan_x + (target_pan_x - pan_x) * lerp_t;
            pan_y = pan_y + (target_pan_y - pan_y) * lerp_t;

            // Recalculate tex_size with interpolated zoom
            tex_size = ImVec2(static_cast<float>(texture.get_width()) * scale * zoom_level,
                              static_cast<float>(texture.get_height()) * scale * zoom_level);

            // Clamp current pan to valid range
            float max_pan_x = std::max(0.0f, tex_size.x - canvas_size.x);
            float max_pan_y = std::max(0.0f, tex_size.y - canvas_size.y);
            pan_x = std::clamp(pan_x, 0.0f, max_pan_x);
            pan_y = std::clamp(pan_y, 0.0f, max_pan_y);

            // Draw the texture using DrawList
            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            // Calculate visible UV coordinates based on pan
            float u0 = pan_x / tex_size.x;
            float v0 = pan_y / tex_size.y;
            float u1 = (pan_x + canvas_size.x) / tex_size.x;
            float v1 = (pan_y + canvas_size.y) / tex_size.y;

            // Clamp UVs and adjust draw rect accordingly
            float draw_x0 = canvas_pos.x;
            float draw_y0 = canvas_pos.y;
            float draw_x1 = canvas_pos.x + canvas_size.x;
            float draw_y1 = canvas_pos.y + canvas_size.y;

            // If zoomed out so image is smaller than canvas, center it
            if (tex_size.x < canvas_size.x) {
                float offset = (canvas_size.x - tex_size.x) * 0.5f;
                draw_x0 = canvas_pos.x + offset;
                draw_x1 = draw_x0 + tex_size.x;
                u0 = 0.0f;
                u1 = 1.0f;
            }
            if (tex_size.y < canvas_size.y) {
                float offset = (canvas_size.y - tex_size.y) * 0.5f;
                draw_y0 = canvas_pos.y + offset;
                draw_y1 = draw_y0 + tex_size.y;
                v0 = 0.0f;
                v1 = 1.0f;
            }

            // Clamp UVs to [0,1]
            u0 = std::clamp(u0, 0.0f, 1.0f);
            v0 = std::clamp(v0, 0.0f, 1.0f);
            u1 = std::clamp(u1, 0.0f, 1.0f);
            v1 = std::clamp(v1, 0.0f, 1.0f);

            draw_list->AddImage(texture.as_imgui_texture(),
                               ImVec2(draw_x0, draw_y0),
                               ImVec2(draw_x1, draw_y1),
                               ImVec2(u0, v0),
                               ImVec2(u1, v1));
        }
    }
    ImGui::End();
    return open;
}

void ngm_flat_vis::free_resources()
{
    texture.release();
}

ngm_flat_vis::params_t ngm_flat_vis::get_default(uint32_t rep_width, uint32_t rep_height, uint32_t num_channels)
{
    params_t result {};
    result.layout = layout_t::vertical_per_neuron;
    result.vis_params.layout = vec_group_vis::layout_t::vertical;
    result.vis_params.margin = 5;
    result.vis_params.vec_params.elem_height = 1;
    result.vis_params.vec_params.elem_width  = 1;
    result.vis_params.vec_params.rep_width   = rep_width;
    result.vis_params.vec_params.rep_height  = rep_height;
    result.vis_params.vec_params.num_channels = num_channels;
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
            static_cast<uint32_t>(hint.height),
            static_cast<uint32_t>(hint.num_channels)
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

        // Check if colormap changed (needs immediate update)
        bool colormap_changed = vis_ && vis_->is_colormap_dirty();
        if (colormap_changed) {
            vis_->clear_colormap_dirty();
        }

        // Update if interval elapsed OR colormap changed
        if (step_count_ % update_interval_ != 0 && !colormap_changed) return;

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
