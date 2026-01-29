//
// Created by jk on 05.09.25.
//
#include <algorithm>

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
    for (std::size_t px_idx = ring_pos, d_idx = 0;
         px_idx < pixel_data.size();
         px_idx += ring_size, ++d_idx)
    {
        color_t &cur_pixel = pixel_data[px_idx];
        uint8_t scaled_data = std::clamp<uint16_t>(
            static_cast<uint16_t>((data[d_idx] - min_val) * 255.0 / (max_val - min_val)), 0, 255);
        cur_pixel.r = scaled_data / 2;
        cur_pixel.g = scaled_data;
        cur_pixel.b = scaled_data / 4;
    }
    texture.update(pixel_data);
    ring_pos = (ring_pos + 1) % ring_size;
}

bool vec_ring_buffer::render_imgui(const char* label, float scale)
{
    bool open = true;
    if (ImGui::Begin(label, &open)) {
        if (texture.is_valid()) {
            ImVec2 size(static_cast<float>(texture.get_width()) * scale,
                        static_cast<float>(texture.get_height()) * scale);
            ImGui::Image(texture.as_imgui_texture(), size);
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

constexpr uint16_t DEFAULT_RING_SIZE = 200;  // Number of time steps to show

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
        , vis_(static_cast<uint16_t>(buffer.size()), DEFAULT_RING_SIZE)
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
