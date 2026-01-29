//
// Created by jk on 05.09.25.
//
#include <algorithm>
#include <cmath>

#include "inc_on_exit.h"
#include "io_buffer/vec_ring_buffer.h"
#include "vis_registry.h"
#include "io_buffer.h"
#include "sim_env.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <numeric>
#include <thread>

#include "imgui.h"

namespace coast {

// Colormap names for UI
const char* colormap_name(colormap_t cm)
{
    switch (cm) {
        case colormap_t::grayscale: return "Grayscale";
        case colormap_t::green:     return "Green";
        case colormap_t::viridis:   return "Viridis";
        case colormap_t::plasma:    return "Plasma";
        case colormap_t::inferno:   return "Inferno";
        case colormap_t::cool:      return "Cool";
        default:                    return "Unknown";
    }
}

// Interpolate between two colors
static color_t lerp_color(const color_t& a, const color_t& b, float t)
{
    return {
        static_cast<uint8_t>(a.r + (b.r - a.r) * t),
        static_cast<uint8_t>(a.g + (b.g - a.g) * t),
        static_cast<uint8_t>(a.b + (b.b - a.b) * t),
        255
    };
}

// Apply colormap to normalized value [0,1]
color_t apply_colormap(colormap_t cm, float value)
{
    value = std::clamp(value, 0.0f, 1.0f);

    switch (cm) {
        case colormap_t::grayscale: {
            uint8_t v = static_cast<uint8_t>(value * 255.0f);
            return {v, v, v, 255};
        }
        case colormap_t::green: {
            uint8_t v = static_cast<uint8_t>(value * 255.0f);
            return {static_cast<uint8_t>(v / 2), v, static_cast<uint8_t>(v / 4), 255};
        }
        case colormap_t::viridis: {
            // Simplified viridis: dark purple -> teal -> yellow
            static const color_t stops[] = {
                {68, 1, 84, 255},     // 0.0 - dark purple
                {59, 82, 139, 255},   // 0.25 - blue
                {33, 145, 140, 255},  // 0.5 - teal
                {94, 201, 98, 255},   // 0.75 - green
                {253, 231, 37, 255}   // 1.0 - yellow
            };
            float idx = value * 4.0f;
            int i = std::min(static_cast<int>(idx), 3);
            float t = idx - static_cast<float>(i);
            return lerp_color(stops[i], stops[i + 1], t);
        }
        case colormap_t::plasma: {
            // Simplified plasma: dark blue -> purple -> orange -> yellow
            static const color_t stops[] = {
                {13, 8, 135, 255},    // 0.0 - dark blue
                {126, 3, 168, 255},   // 0.25 - purple
                {204, 71, 120, 255},  // 0.5 - pink
                {248, 149, 64, 255},  // 0.75 - orange
                {240, 249, 33, 255}   // 1.0 - yellow
            };
            float idx = value * 4.0f;
            int i = std::min(static_cast<int>(idx), 3);
            float t = idx - static_cast<float>(i);
            return lerp_color(stops[i], stops[i + 1], t);
        }
        case colormap_t::inferno: {
            // Simplified inferno: black -> purple -> red -> yellow
            static const color_t stops[] = {
                {0, 0, 4, 255},       // 0.0 - black
                {87, 16, 110, 255},   // 0.25 - purple
                {188, 55, 84, 255},   // 0.5 - red
                {249, 142, 9, 255},   // 0.75 - orange
                {252, 255, 164, 255}  // 1.0 - light yellow
            };
            float idx = value * 4.0f;
            int i = std::min(static_cast<int>(idx), 3);
            float t = idx - static_cast<float>(i);
            return lerp_color(stops[i], stops[i + 1], t);
        }
        case colormap_t::cool: {
            // Cool: cyan -> magenta
            static const color_t stops[] = {
                {0, 255, 255, 255},   // 0.0 - cyan
                {255, 0, 255, 255}    // 1.0 - magenta
            };
            return lerp_color(stops[0], stops[1], value);
        }
        default:
            return {static_cast<uint8_t>(value * 255.0f), static_cast<uint8_t>(value * 255.0f), static_cast<uint8_t>(value * 255.0f), 255};
    }
}

vec_ring_buffer::vec_ring_buffer(uint16_t _vec_size, uint16_t _ring_size) :
    texture(),
    pixel_data(_vec_size * _ring_size),
    vec_size(_vec_size),
    ring_size(_ring_size),
    ring_pos(0)
{
    std::ranges::fill(pixel_data, color_t{0, 0, 0, 255});
    texture = gl_texture(ring_size, vec_size, pixel_data.data());
}

void vec_ring_buffer::update(const std::span<float> &data, bool auto_norm, float min_val, float max_val)
{
    assert(data.size() == vec_size);
    if (auto_norm) {
        max_val = std::reduce(data.begin(), data.end(), 0.0f, [](float a, float b){ return std::max(a, b); });
        min_val = std::reduce(data.begin(), data.end(), max_val, [](float a, float b){ return std::min(a, b); });
        if (min_val > 0.0f)
            min_val = 0.0f;
    }
    const float range = max_val - min_val;
    const float inv_range = (range > 0.0f) ? 1.0f / range : 1.0f;

    for (std::size_t px_idx = ring_pos, d_idx = 0;
         px_idx < pixel_data.size();
         px_idx += ring_size, ++d_idx)
    {
        float normalized = (data[d_idx] - min_val) * inv_range;
        pixel_data[px_idx] = apply_colormap(colormap, normalized);
    }
    texture.update(pixel_data);
    ring_pos = (ring_pos + 1) % ring_size;
}

bool vec_ring_buffer::render_imgui(const char* label, float scale)
{
    bool open = true;

    // Set initial window size on first appearance
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(label, &open)) {
        // Toolbar with zoom display, reset button, and colormap selector
        ImGui::Text("Zoom: %.0f%%", zoom_level * 100.0f);
        ImGui::SameLine();
        if (ImGui::Button("Reset View")) {
            target_zoom = 2.0f;
            target_pan_x = 0.0f;
            target_pan_y = 0.0f;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        if (ImGui::BeginCombo("##colormap", colormap_name(colormap))) {
            for (int i = 0; i < static_cast<int>(colormap_t::COUNT); ++i) {
                auto cm = static_cast<colormap_t>(i);
                bool is_selected = (colormap == cm);
                if (ImGui::Selectable(colormap_name(cm), is_selected)) {
                    colormap = cm;
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

void vec_ring_buffer::free_resources()
{
    // RAII handles cleanup via gl_texture destructor
    texture.release();
}

// ============================================================================
// vis_instance adapter for registry integration
// ============================================================================

namespace {

constexpr uint16_t DEFAULT_RING_SIZE = 500;  // Number of time steps to show

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
 * Adapter that wraps vec_ring_buffer to implement vis_instance interface.
 * Provides thread-safe updates synchronized with the simulation thread.
 */
class vec_ring_buffer_instance : public coast::vis::vis_instance {
public:
    vec_ring_buffer_instance(sim::io_buffer& buffer,
                             sim::env& sim_env,
                             const coast::vis_hints_t& hints,
                             std::string label)
        : buffer_(buffer)
        , sim_env_(sim_env)
        , label_(std::move(label))
        , vis_(static_cast<uint16_t>(buffer.size()),
               static_cast<uint16_t>(hints.buffer_vis.ring_length > 0 ? hints.buffer_vis.ring_length : DEFAULT_RING_SIZE))
    {
        update_interval_ = static_cast<std::size_t>(hints.buffer_vis.update_interval);

        // Register post-swap hook for thread-safe synchronization
        hook_id_ = sim_env_.set_post_swap_hook([this]() {
            on_post_swap();
        });
    }

    ~vec_ring_buffer_instance() override
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
        if (!visible_) return;

        // Check if sim thread has requested an update
        int expected = static_cast<int>(sync_state::requested);
        if (sync_state_.compare_exchange_strong(expected,
                                                 static_cast<int>(sync_state::updating),
                                                 std::memory_order_acq_rel)) {
            // We successfully transitioned from 'requested' to 'updating'
            // Sim thread is now waiting, safe to read from buffer
            auto data = buffer_.cur_read_buffer();
            vis_.update(std::span<float>(const_cast<float*>(data.data()), data.size()));

            // Signal completion - transition to 'idle'
            sync_state_.store(static_cast<int>(sync_state::idle), std::memory_order_release);
        }
    }

    bool render_imgui() override
    {
        if (!visible_) return true;
        return vis_.render_imgui(label_.c_str());
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

    sim::io_buffer& buffer_;
    sim::env& sim_env_;
    std::string label_;
    coast::vec_ring_buffer vis_;
    bool visible_ = true;

    // Synchronization state
    std::atomic<int> sync_state_{static_cast<int>(sync_state::idle)};
    std::size_t update_interval_ = 100;
    std::size_t step_count_ = 0;
    std::size_t hook_id_ = 0;
};

/**
 * Factory function for creating vec_ring_buffer instances.
 */
std::unique_ptr<coast::vis::vis_instance> create_vec_ring_buffer(
    sim::io_buffer& buffer,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& label)
{
    return std::make_unique<vec_ring_buffer_instance>(buffer, sim_env, hints, label);
}

/**
 * Static registrar - registers at program startup.
 */
struct vec_ring_buffer_registrar {
    vec_ring_buffer_registrar()
    {
        coast::vis::vis_info info {
            "vec_ring_buffer",
            "Time Series",
            "Scrolling time-series view of buffer values",
            coast::vis::vis_target::io_buffer,
            {}  // No entity type restriction for buffer visualizations
        };
        coast::vis::get_vis_registry().register_buffer_vis(info, create_vec_ring_buffer);
    }
};

static vec_ring_buffer_registrar registrar_instance;

} // anonymous namespace

} // coast
