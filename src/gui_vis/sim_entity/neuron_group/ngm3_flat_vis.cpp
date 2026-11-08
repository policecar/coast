//
// Created by jk on 25.05.26.
//
// Clone of ngm_flat_vis adapted for ngm3::neuron_group_t. Only the
// source-data access in the constructor and update() differs; the
// render_imgui body (zoom / pan / colormap UI) is structurally
// identical.

#include "gui_vis/core/gradient.h"
#include "gui_vis/core/colormap.h"
#include "tools/inc_on_exit.h"

#include "gui_vis/sim_entity/neuron_group/ngm3_flat_vis.h"
#include "gui_vis/vis_registry.h"
#include "sim_core/sim_env.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"   // ngm2.5 reuses this renderer (duck-typed)
#ifdef COAST_WITH_GPU
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"
#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

#include "imgui.h"
#include "gui_core/imgui_scaling.h"

namespace coast {

namespace {

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

// Fetch the permanence span for the (n, d, r) slot of an ngm3 group,
// where d in [0, apical_cnt + proximal_cnt) is the flattened bank
// index: apical dendrites come first, proximal after.
template<class Group>
std::span<const float> ngm3_rep(const Group& ng,
                                std::size_t n, std::size_t d, std::size_t r)
{
    const std::size_t a_cnt = ng.get_apical_count();
    if (d < a_cnt) return ng.get_apical_representation(n, d, r);
    return ng.get_proximal_representation(n, d - a_cnt, r);
}

template<class Group>
std::size_t ngm3_dendrite_count(const Group& ng)
{
    return ng.get_apical_count() + ng.get_proximal_count();
}

// Fetch the segment-id span for the (n, d, r) slot, with the same flattened
// bank indexing as ngm3_rep().
template<class Group>
std::span<const uint8_t> ngm3_segids(const Group& ng,
                                     std::size_t n, std::size_t d, std::size_t r)
{
    const std::size_t a_cnt = ng.get_apical_count();
    if (d < a_cnt) return ng.get_apical_segment_ids(n, d, r);
    return ng.get_proximal_segment_ids(n, d - a_cnt, r);
}

// ngm3 stores rep_cnt = 2^branch_depth representation *slots* per dendrite, but a
// dendritic-tree segment that has not yet split to full depth is stored
// redundantly across several consecutive slots: branching always splits a
// contiguous slot range and writes identical seg-ids/permanences into it. Two
// adjacent slots that share an identical seg-id vector are therefore byte-equal
// copies of the same segment. This predicate flags slot r as such a duplicate of
// r-1 so the vis can skip it and show only the actually present representations
// (the ngm2 flat vis shows only leaf segments for the same reason).
template<class Group>
bool ngm3_rep_is_duplicate(const Group& ng,
                           std::size_t n, std::size_t d, std::size_t r)
{
    if (r == 0) return false;
    const auto cur  = ngm3_segids(ng, n, d, r);
    const auto prev = ngm3_segids(ng, n, d, r - 1);
    // No seg-id info (e.g. GPU mirror not yet read back) → fall back to showing
    // the slot rather than collapsing everything onto rep 0.
    if (cur.empty() || prev.empty() || cur.size() != prev.size())
        return false;
    return std::equal(cur.begin(), cur.end(), prev.begin());
}

} // anonymous namespace

template<class Group>
ngm3_flat_vis<Group>::ngm3_flat_vis(const Group& neuron_group, const params_t _params) :
    params(_params),
    ng(neuron_group)
{
    px_width  = 0;
    px_height = 0;
    const std::size_t neuron_cnt   = ng.get_neuron_count();
    const std::size_t dendrite_cnt = ngm3_dendrite_count(ng);
    // For ngm3 every dendrite of every neuron has the same shape
    // (rep_cnt × input_size). Layout calculation can therefore proceed
    // without reading per-dendrite sizes from the model — the dendrite
    // visual size is identical across the group.
    const uint32_t max_rep_cnt = static_cast<uint32_t>(ng.get_rep_count());
    params.vis_params.vec_cnt = max_rep_cnt;

    for (std::size_t n = 0; n < neuron_cnt; ++n) {
        uint32_t max_dendrite_height = 0;
        uint32_t max_dendrite_width  = 0;
        uint32_t neuron_width  = 0;
        uint32_t neuron_height = 0;
        for (std::size_t d = 0; d < dendrite_cnt; ++d) {
            vec_group_vis tmp{params.vis_params};
            const uint32_t dw = tmp.get_total_width();
            const uint32_t dh = tmp.get_total_height();
            switch (params.layout) {
                case layout_t::horizontal_per_neuron: neuron_width  += dw + params.vis_params.margin * 2; break;
                case layout_t::vertical_per_neuron:   neuron_height += dh + params.vis_params.margin * 2; break;
            }
            max_dendrite_height = std::max(max_dendrite_height, dh);
            max_dendrite_width  = std::max(max_dendrite_width,  dw);
        }
        switch (params.layout) {
            case layout_t::horizontal_per_neuron: {
                px_width   = std::max(px_width, neuron_width);
                px_height += max_dendrite_height;
            } break;
            case layout_t::vertical_per_neuron: {
                px_width  += max_dendrite_width;
                px_height  = std::max(px_height, neuron_height);
            } break;
        }
    }

    if (px_width == 0 || px_height == 0) {
        // Defensive: a degenerate group would otherwise create a
        // zero-size texture. Use a single placeholder pixel.
        px_width  = std::max(px_width,  static_cast<uint32_t>(1));
        px_height = std::max(px_height, static_cast<uint32_t>(1));
    }

    pixel_data.resize(px_width * px_height, color_t{255, 255, 0, 255});
    texture = gl_texture(static_cast<int>(px_width), static_cast<int>(px_height), pixel_data.data());
}

template<class Group>
void ngm3_flat_vis<Group>::update()
{
    // CONSUMER-PULLED refresh (S10): no-op on the CPU group; an on-demand
    // GPU->CPU bank readback on the GPU group. Safe here because the vis_instance
    // runs update() on the GUI thread inside the post-swap barrier, where the sim
    // thread is parked and (for the GPU dispatcher) the device is idle.
    ng.refresh_vis_mirror();

    auto def_grad = make_gradient(colormap);
    uint32_t group_pos_x = 0;
    uint32_t group_pos_y = 0;
    const std::size_t neuron_cnt   = ng.get_neuron_count();
    const std::size_t dendrite_cnt = ngm3_dendrite_count(ng);
    const uint32_t    rep_cnt      = static_cast<uint32_t>(ng.get_rep_count());
    params.vis_params.vec_cnt = rep_cnt;

    for (std::size_t n = 0; n < neuron_cnt; ++n) {
        uint32_t max_dendrite_height = 0;
        uint32_t max_dendrite_width  = 0;
        for (std::size_t d = 0; d < dendrite_cnt; ++d) {
            // vec_cnt stays at rep_cnt (the max) so the per-dendrite block size
            // matches the constructor's worst-case texture layout; we simply pack
            // the present (non-duplicate) representations into the leading slots
            // and leave the trailing slots as background — same as ngm2.
            vec_group_vis vis{params.vis_params};
            uint32_t slot = 0;
            for (uint32_t r = 0; r < rep_cnt; ++r) {
                if (ngm3_rep_is_duplicate(ng, n, d, r)) continue;
                vis.update_vec<5>(slot, ngm3_rep(ng, n, d, r), def_grad,
                                  px_pos_t{group_pos_x, group_pos_y},
                                  pixel_data_t{pixel_data, px_width, px_height});
                ++slot;
            }
            const uint32_t dw = vis.get_total_width();
            const uint32_t dh = vis.get_total_height();
            switch (params.layout) {
                case layout_t::horizontal_per_neuron: group_pos_x += dw + params.vis_params.margin * 2; break;
                case layout_t::vertical_per_neuron:   group_pos_y += dh + params.vis_params.margin * 2; break;
            }
            max_dendrite_height = std::max(max_dendrite_height, dh);
            max_dendrite_width  = std::max(max_dendrite_width,  dw);
        }
        switch (params.layout) {
            case layout_t::horizontal_per_neuron: {
                group_pos_x  = 0;
                group_pos_y += max_dendrite_height;
            } break;
            case layout_t::vertical_per_neuron: {
                group_pos_x += max_dendrite_width;
                group_pos_y  = 0;
            } break;
        }
    }
    texture.update(pixel_data);
}

template<class Group>
bool ngm3_flat_vis<Group>::render_imgui(const char* label, float scale)
{
    bool open = true;
    ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(label, &open)) {
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
                        colormap_dirty = true;
                    }
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::Separator();

        if (texture.is_valid()) {
            float effective_scale = scale * zoom_level;
            ImVec2 tex_size(static_cast<float>(texture.get_width()) * effective_scale,
                            static_cast<float>(texture.get_height()) * effective_scale);

            ImVec2 canvas_size = ImGui::GetContentRegionAvail();
            ImVec2 canvas_pos  = ImGui::GetCursorScreenPos();

            ImGui::InvisibleButton("canvas", canvas_size, ImGuiButtonFlags_MouseButtonLeft);
            bool is_hovered = ImGui::IsItemHovered();
            bool is_active  = ImGui::IsItemActive();

            ImGuiIO& io = ImGui::GetIO();
            constexpr float lerp_speed = 15.0f;
            float lerp_t = std::min(1.0f, lerp_speed * io.DeltaTime);

            if (is_hovered && io.MouseWheel != 0.0f) {
                ImVec2 mouse_pos = io.MousePos;
                float mouse_canvas_x = mouse_pos.x - canvas_pos.x;
                float mouse_canvas_y = mouse_pos.y - canvas_pos.y;
                float tex_x = (mouse_canvas_x + pan_x) / (scale * zoom_level);
                float tex_y = (mouse_canvas_y + pan_y) / (scale * zoom_level);
                float zoom_factor = std::pow(1.15f, io.MouseWheel);
                target_zoom = std::clamp(target_zoom * zoom_factor, 0.1f, 10.0f);
                float new_scale = scale * target_zoom;
                target_pan_x = tex_x * new_scale - mouse_canvas_x;
                target_pan_y = tex_y * new_scale - mouse_canvas_y;
            }

            if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                target_pan_x -= io.MouseDelta.x;
                target_pan_y -= io.MouseDelta.y;
                pan_x -= io.MouseDelta.x;
                pan_y -= io.MouseDelta.y;
            }

            float target_tex_size_x = static_cast<float>(texture.get_width()) * scale * target_zoom;
            float target_tex_size_y = static_cast<float>(texture.get_height()) * scale * target_zoom;
            float max_target_pan_x = std::max(0.0f, target_tex_size_x - canvas_size.x);
            float max_target_pan_y = std::max(0.0f, target_tex_size_y - canvas_size.y);
            target_pan_x = std::clamp(target_pan_x, 0.0f, max_target_pan_x);
            target_pan_y = std::clamp(target_pan_y, 0.0f, max_target_pan_y);

            zoom_level = zoom_level + (target_zoom - zoom_level) * lerp_t;
            pan_x = pan_x + (target_pan_x - pan_x) * lerp_t;
            pan_y = pan_y + (target_pan_y - pan_y) * lerp_t;

            tex_size = ImVec2(static_cast<float>(texture.get_width()) * scale * zoom_level,
                              static_cast<float>(texture.get_height()) * scale * zoom_level);

            float max_pan_x = std::max(0.0f, tex_size.x - canvas_size.x);
            float max_pan_y = std::max(0.0f, tex_size.y - canvas_size.y);
            pan_x = std::clamp(pan_x, 0.0f, max_pan_x);
            pan_y = std::clamp(pan_y, 0.0f, max_pan_y);

            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            float u0 = pan_x / tex_size.x;
            float v0 = pan_y / tex_size.y;
            float u1 = (pan_x + canvas_size.x) / tex_size.x;
            float v1 = (pan_y + canvas_size.y) / tex_size.y;

            float draw_x0 = canvas_pos.x;
            float draw_y0 = canvas_pos.y;
            float draw_x1 = canvas_pos.x + canvas_size.x;
            float draw_y1 = canvas_pos.y + canvas_size.y;

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

template<class Group>
void ngm3_flat_vis<Group>::free_resources()
{
    texture.release();
}

template<class Group>
typename ngm3_flat_vis<Group>::params_t
ngm3_flat_vis<Group>::get_default(uint32_t rep_width, uint32_t rep_height, uint32_t num_channels)
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

template<class Group>
uint32_t ngm3_flat_vis<Group>::get_height() const { return px_height; }
template<class Group>
uint32_t ngm3_flat_vis<Group>::get_width()  const { return px_width;  }

// ============================================================================
// vis_instance adapter for registry integration
// ============================================================================

namespace {

enum class sync_state : int {
    idle      = 0,
    requested = 1,
    updating  = 2
};

constexpr auto SYNC_TIMEOUT = std::chrono::milliseconds(100);

template<class Group>
class ngm3_flat_vis_instance : public coast::vis::vis_instance {
public:
    ngm3_flat_vis_instance(Group& ng,
                           sim::env& sim_env,
                           const coast::vis_hints_t& hints,
                           const std::string& hints_key,
                           std::string label)
        : ng_(ng)
        , sim_env_(sim_env)
        , label_(std::move(label))
    {
        auto it = hints.ngm3.neuron_groups.find(hints_key);
        coast::ngm_flat_vis_hint_t hint;
        if (it != hints.ngm3.neuron_groups.end()) {
            hint = it->second.ngm_flat_vis;
        } else if (auto def_it = hints.ngm3.neuron_groups.find("default");
                   def_it != hints.ngm3.neuron_groups.end())
        {
            hint = def_it->second.ngm_flat_vis;
        }

        update_interval_ = static_cast<std::size_t>(hint.update_interval);

        auto params = coast::ngm3_flat_vis<Group>::get_default(
            static_cast<uint32_t>(hint.width),
            static_cast<uint32_t>(hint.height),
            static_cast<uint32_t>(hint.num_channels)
        );
        vis_ = std::make_unique<coast::ngm3_flat_vis<Group>>(ng_, params);

        hook_id_         = sim_env_.set_post_swap_hook        ([this]() { on_post_swap_signal(); });
        barrier_hook_id_ = sim_env_.set_post_swap_barrier_hook([this]() { on_post_swap_wait();  });
    }

    ~ngm3_flat_vis_instance() override
    {
        // Removal is synchronous (#235): once both calls return, neither hook is
        // running or will run again, so nothing on the sim thread touches `this`.
        // update() runs on this (GUI) thread, so no update is in flight either; the
        // former spin on sync_state_ == idle is not needed (and could hang if a
        // signal fired before its barrier hook was applied).
        sim_env_.remove_post_swap_hook        (hook_id_);
        sim_env_.remove_post_swap_barrier_hook(barrier_hook_id_);
    }

    void update() override
    {
        if (!vis_ || !visible_.load(std::memory_order_acquire)) return;
        int expected = static_cast<int>(sync_state::requested);
        if (sync_state_.compare_exchange_strong(expected,
                                                static_cast<int>(sync_state::updating),
                                                std::memory_order_acq_rel))
        {
            vis_->update();
            sync_state_.store(static_cast<int>(sync_state::idle), std::memory_order_release);
        }
    }

    bool render_imgui() override
    {
        if (!vis_ || !visible_.load(std::memory_order_acquire)) return true;
        return vis_->render_imgui(label_.c_str());
    }

    void free_resources() override
    {
        if (vis_) vis_->free_resources();
    }

    [[nodiscard]] bool is_visible() const override { return visible_.load(std::memory_order_acquire); }
    void set_visible(bool v) override
    {
        visible_.store(v, std::memory_order_release);
        if (vis_) vis_->set_visible(v);
    }

private:
    void on_post_swap_signal()
    {
        inc_on_exit inc{step_count_};
        if (update_interval_ == 0) return;
        if (!visible_.load(std::memory_order_acquire)) return;

        bool colormap_changed = vis_ && vis_->is_colormap_dirty();
        if (colormap_changed) vis_->clear_colormap_dirty();
        if (step_count_ % update_interval_ != 0 && !colormap_changed) return;

        int expected = static_cast<int>(sync_state::idle);
        if (!sync_state_.compare_exchange_strong(expected,
                                                 static_cast<int>(sync_state::requested),
                                                 std::memory_order_acq_rel))
        {
            return;
        }
        pending_wait_ = true;
    }

    void on_post_swap_wait()
    {
        if (!pending_wait_) return;
        pending_wait_ = false;

        auto start = std::chrono::steady_clock::now();
        while (sync_state_.load(std::memory_order_acquire) != static_cast<int>(sync_state::idle)) {
            // Give up early while a hook removal waits for this phase (#235):
            // the remover is usually the GUI thread this loop is waiting for.
            if (std::chrono::steady_clock::now() - start > SYNC_TIMEOUT ||
                sim_env_.hook_removal_waiting()) {
                int expected = static_cast<int>(sync_state::requested);
                if (sync_state_.compare_exchange_strong(expected,
                                                         static_cast<int>(sync_state::idle),
                                                         std::memory_order_acq_rel))
                {
                    break;
                }
            }
            std::this_thread::yield();
        }
    }

    Group&                ng_;
    sim::env&             sim_env_;
    std::string           label_;
    std::unique_ptr<coast::ngm3_flat_vis<Group>> vis_;
    std::atomic<bool>     visible_{true};

    std::atomic<int> sync_state_{static_cast<int>(sync_state::idle)};
    std::size_t update_interval_  = 10000;
    std::size_t step_count_       = 0;
    std::size_t hook_id_          = 0;
    std::size_t barrier_hook_id_  = 0;
    bool        pending_wait_     = false;
};

std::unique_ptr<coast::vis::vis_instance> create_ngm3_flat_vis(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    auto* ng = dynamic_cast<ngm3::neuron_group_t*>(&entity);
    if (!ng) return nullptr;
    return std::make_unique<ngm3_flat_vis_instance<ngm3::neuron_group_t>>(
        *ng, sim_env, hints, hints_key, label);
}

struct ngm3_flat_vis_registrar {
    ngm3_flat_vis_registrar()
    {
        coast::vis::vis_info info {
            "ngm3_flat_vis",
            "NGM3 Dendritic Representations",
            "Flat grid view showing ngm3 neuron representations (apical + proximal banks)",
            coast::vis::vis_target::sim_entity,
            { sim::io_entity_type_t::ngm3_neuron_group }
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm3_flat_vis);
    }
};

static ngm3_flat_vis_registrar ngm3_flat_vis_registrar_instance;

// ngm2.5 reuses this renderer: its group satisfies the same duck-typed accessor
// contract (get_*_representation / get_*_segment_ids / refresh_vis_mirror, which
// materializes the dense mirror from the sparse arena). Vis hints resolve from
// the shared hints.ngm3 tree (no dedicated ngm25 hints tree in v1).
std::unique_ptr<coast::vis::vis_instance> create_ngm25_flat_vis(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    auto* ng = dynamic_cast<ngm25::neuron_group_t*>(&entity);
    if (!ng) return nullptr;
    return std::make_unique<ngm3_flat_vis_instance<ngm25::neuron_group_t>>(
        *ng, sim_env, hints, hints_key, label);
}

struct ngm25_flat_vis_registrar {
    ngm25_flat_vis_registrar()
    {
        coast::vis::vis_info info {
            "ngm25_flat_vis",
            "NGM2.5 Dendritic Representations",
            "Flat grid view showing ngm2.5 neuron representations (apical + proximal banks)",
            coast::vis::vis_target::sim_entity,
            { sim::io_entity_type_t::ngm25_neuron_group }
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm25_flat_vis);
    }
};

static ngm25_flat_vis_registrar ngm25_flat_vis_registrar_instance;

#ifdef COAST_WITH_GPU
// GPU sibling: same flat-vis template instantiated over the GPU group. The only
// difference is the dynamic_cast target and the entity-type key it registers for;
// the templated vis pulls through the same accessor surface (its update() calls
// refresh_vis_mirror() which performs the GPU->CPU readback for this Group).
std::unique_ptr<coast::vis::vis_instance> create_ngm3_flat_vis_gpu(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    auto* ng = dynamic_cast<ngm3::neuron_group_gpu_t*>(&entity);
    if (!ng) return nullptr;
    return std::make_unique<ngm3_flat_vis_instance<ngm3::neuron_group_gpu_t>>(
        *ng, sim_env, hints, hints_key, label);
}

struct ngm3_flat_vis_gpu_registrar {
    ngm3_flat_vis_gpu_registrar()
    {
        coast::vis::vis_info info {
            "ngm3_flat_vis_gpu",
            "NGM3 Dendritic Representations (GPU)",
            "Flat grid view showing ngm3_gpu neuron representations (apical + proximal banks)",
            coast::vis::vis_target::sim_entity,
            { sim::io_entity_type_t::ngm3_neuron_group_gpu }
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm3_flat_vis_gpu);
    }
};

static ngm3_flat_vis_gpu_registrar ngm3_flat_vis_gpu_registrar_instance;

// ngm2.5 GPU: same flat-vis template over the ngm25 GPU group (duck-typed; its
// update() calls refresh_vis_mirror() = the GPU->CPU readback).
std::unique_ptr<coast::vis::vis_instance> create_ngm25_flat_vis_gpu(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    auto* ng = dynamic_cast<ngm25::neuron_group_gpu_t*>(&entity);
    if (!ng) return nullptr;
    return std::make_unique<ngm3_flat_vis_instance<ngm25::neuron_group_gpu_t>>(
        *ng, sim_env, hints, hints_key, label);
}

struct ngm25_flat_vis_gpu_registrar {
    ngm25_flat_vis_gpu_registrar()
    {
        coast::vis::vis_info info {
            "ngm25_flat_vis_gpu",
            "NGM2.5 Dendritic Representations (GPU)",
            "Flat grid view showing ngm2.5_gpu neuron representations (apical + proximal banks)",
            coast::vis::vis_target::sim_entity,
            { sim::io_entity_type_t::ngm25_neuron_group_gpu }
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm25_flat_vis_gpu);
    }
};

static ngm25_flat_vis_gpu_registrar ngm25_flat_vis_gpu_registrar_instance;
#endif // COAST_WITH_GPU

} // anonymous namespace

} // namespace coast
