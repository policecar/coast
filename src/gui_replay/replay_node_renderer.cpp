//
// Node renderer for buffer visualization in replay dialog
//

#include "gui_replay/replay_node_renderer.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <memory>

namespace coast::replay {

replay_node_renderer::replay_node_renderer() = default;
replay_node_renderer::~replay_node_renderer() = default;

void replay_node_renderer::render(ImDrawList* draw_list,
                                   const layout::node_t& node,
                                   ImVec2 screen_pos,
                                   float zoom,
                                   bool selected,
                                   bool hovered) {
    // Get buffer data from node (stored as shared_ptr)
    const buffer_node_data* data = nullptr;
    if (node.user_data.has_value()) {
        try {
            auto ptr = std::any_cast<std::shared_ptr<buffer_node_data>>(node.user_data);
            data = ptr.get();
        } catch (...) {
            data = nullptr;
        }
    }

    if (!data) {
        // Render placeholder for nodes without data
        float size = node_size_ * zoom;
        draw_list->AddRectFilled(
            ImVec2(screen_pos.x - size * 0.5f, screen_pos.y - size * 0.5f),
            ImVec2(screen_pos.x + size * 0.5f, screen_pos.y + size * 0.5f),
            IM_COL32(60, 60, 80, 200)
        );
        return;
    }

    // Dispatch to appropriate renderer based on mode
    switch (data->config.mode) {
        case vis_mode::image_2d:
            render_image_2d(draw_list, *data, screen_pos, zoom, selected, hovered);
            break;
        case vis_mode::stats_color:
            render_stats_color(draw_list, *data, screen_pos, zoom, selected, hovered);
            break;
        case vis_mode::delta_flow:
            render_delta_flow(draw_list, *data, screen_pos, zoom, selected, hovered);
            break;
    }

    // Draw selection/hover border
    float size = node_size_ * zoom;
    if (selected) {
        draw_list->AddRect(
            ImVec2(screen_pos.x - size * 0.5f - 2, screen_pos.y - size * 0.5f - 2),
            ImVec2(screen_pos.x + size * 0.5f + 2, screen_pos.y + size * 0.5f + 2),
            IM_COL32(255, 200, 80, 255), 0.0f, 0, 3.0f
        );
    } else if (hovered) {
        draw_list->AddRect(
            ImVec2(screen_pos.x - size * 0.5f - 1, screen_pos.y - size * 0.5f - 1),
            ImVec2(screen_pos.x + size * 0.5f + 1, screen_pos.y + size * 0.5f + 1),
            IM_COL32(150, 150, 200, 255), 0.0f, 0, 2.0f
        );
    }

    // Draw label
    if (show_labels_ && zoom > 0.4f) {
        const char* label = data->buffer_name.c_str();
        ImVec2 text_size = ImGui::CalcTextSize(label);
        ImVec2 label_pos = ImVec2(
            screen_pos.x - text_size.x * 0.5f,
            screen_pos.y + size * 0.5f + 4.0f
        );
        draw_list->AddText(label_pos, IM_COL32(200, 200, 200, 255), label);

        // Draw stats below the label
        char stats_str[64];
        std::snprintf(stats_str, sizeof(stats_str), "avg: %.3f", data->current_stats.avg);
        ImVec2 stats_size = ImGui::CalcTextSize(stats_str);
        ImVec2 stats_pos = ImVec2(
            screen_pos.x - stats_size.x * 0.5f,
            label_pos.y + text_size.y + 2.0f
        );
        draw_list->AddText(stats_pos, IM_COL32(150, 150, 150, 200), stats_str);
    }
}

ImVec2 replay_node_renderer::get_size(const layout::node_t& /*node*/, float zoom) {
    float size = node_size_ * zoom;
    return ImVec2(size, size);
}

void replay_node_renderer::update_node_data(layout::node_t& node,
                                             std::span<const float> data,
                                             const frame_stats& stats) {
    buffer_node_data* node_data = nullptr;
    if (node.user_data.has_value()) {
        try {
            auto ptr = std::any_cast<std::shared_ptr<buffer_node_data>>(node.user_data);
            node_data = ptr.get();
        } catch (...) {
            return;
        }
    }

    if (!node_data) {
        return;
    }

    // Store previous data for delta mode
    node_data->previous_data = std::move(node_data->current_data);

    // Copy new data
    node_data->current_data.assign(data.begin(), data.end());
    node_data->current_stats = stats;
    node_data->data_valid = true;

    // Update texture
    update_texture(*node_data);
}

void replay_node_renderer::clear_node_data(layout::node_t& node) {
    buffer_node_data* node_data = nullptr;
    if (node.user_data.has_value()) {
        try {
            auto ptr = std::any_cast<std::shared_ptr<buffer_node_data>>(node.user_data);
            node_data = ptr.get();
        } catch (...) {
            return;
        }
    }

    if (!node_data) {
        return;
    }

    node_data->current_data.clear();
    node_data->previous_data.clear();
    node_data->data_valid = false;
}

void replay_node_renderer::set_global_config(const vis_config& config) {
    global_config_ = config;
}

void replay_node_renderer::apply_global_config_to_nodes(layout::layout_graph& graph) {
    for (auto& node : graph.nodes()) {
        if (!node.user_data.has_value()) continue;

        buffer_node_data* data = nullptr;
        try {
            auto ptr = std::any_cast<std::shared_ptr<buffer_node_data>>(node.user_data);
            data = ptr.get();
        } catch (...) {
            continue;
        }

        if (!data) continue;

        // Apply global settings but preserve node-specific dimensions and channels
        int saved_width = data->config.width;
        int saved_height = data->config.height;
        int saved_channels = data->config.num_channels;

        data->config.mode = global_config_.mode;
        data->config.colormap = global_config_.colormap;
        data->config.auto_normalize = global_config_.auto_normalize;
        data->config.manual_min = global_config_.manual_min;
        data->config.manual_max = global_config_.manual_max;

        // Restore node-specific dimensions and channels
        data->config.width = saved_width;
        data->config.height = saved_height;
        data->config.num_channels = saved_channels;

        // Regenerate texture with new settings
        if (data->data_valid) {
            update_texture(*data);
        }
    }
}

void replay_node_renderer::update_texture(buffer_node_data& data) {
    if (!data.data_valid || data.current_data.empty()) {
        return;
    }

    const auto& config = data.config;
    int w = config.width;
    int h = config.height;
    int channels = config.num_channels;

    // Ensure dimensions fit the data
    std::size_t tex_size = static_cast<std::size_t>(w * h);
    std::size_t expected_data_size = tex_size * static_cast<std::size_t>(channels);

    if (expected_data_size > data.buffer_size) {
        // Adjust dimensions to fit data
        h = static_cast<int>((data.buffer_size / channels + w - 1) / w);
        tex_size = static_cast<std::size_t>(w * h);
    }

    // Create/resize texture if needed
    if (!data.texture.is_valid() ||
        data.texture.get_width() != w ||
        data.texture.get_height() != h) {
        data.texture = gl_texture(w, h);
    }

    // Fill pixel buffer
    pixel_buffer_.resize(tex_size);

    if (channels == 3) {
        // RGB data - use values directly without colormap
        // Data layout is assumed to be interleaved: R0, G0, B0, R1, G1, B1, ...
        // (this is the most common format for image data)

        // Compute normalization for RGB
        float min_val = config.manual_min;
        float max_val = config.manual_max;
        if (config.auto_normalize) {
            min_val = data.current_stats.min_val;
            max_val = data.current_stats.max_val;
        }
        float range = max_val - min_val;
        if (range < 1e-6f) {
            range = 1.0f;
        }
        float inv_range = 1.0f / range;

        for (std::size_t i = 0; i < tex_size; ++i) {
            std::size_t base_idx = i * 3;
            float r = 0.0f, g = 0.0f, b = 0.0f;

            if (base_idx + 2 < data.current_data.size()) {
                r = (data.current_data[base_idx + 0] - min_val) * inv_range;
                g = (data.current_data[base_idx + 1] - min_val) * inv_range;
                b = (data.current_data[base_idx + 2] - min_val) * inv_range;
            }

            pixel_buffer_[i] = color_t{
                static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f),
                static_cast<uint8_t>(std::clamp(g, 0.0f, 1.0f) * 255.0f),
                static_cast<uint8_t>(std::clamp(b, 0.0f, 1.0f) * 255.0f),
                255
            };
        }
    } else {
        // Grayscale data - apply colormap
        float min_val = config.manual_min;
        float max_val = config.manual_max;
        if (config.auto_normalize) {
            min_val = data.current_stats.min_val;
            max_val = data.current_stats.max_val;
        }
        float range = max_val - min_val;
        if (range < 1e-6f) {
            range = 1.0f;
        }
        float inv_range = 1.0f / range;

        for (std::size_t i = 0; i < tex_size; ++i) {
            float value = 0.0f;
            if (i < data.current_data.size()) {
                value = (data.current_data[i] - min_val) * inv_range;
            }
            value = std::clamp(value, 0.0f, 1.0f);
            pixel_buffer_[i] = apply_colormap(config.colormap, value);
        }
    }

    data.texture.update(pixel_buffer_);
}

void replay_node_renderer::render_image_2d(ImDrawList* draw_list,
                                            const buffer_node_data& data,
                                            ImVec2 screen_pos,
                                            float zoom,
                                            [[maybe_unused]] bool selected,
                                            [[maybe_unused]] bool hovered) {
    float size = node_size_ * zoom;
    ImVec2 min_pos = ImVec2(screen_pos.x - size * 0.5f, screen_pos.y - size * 0.5f);
    ImVec2 max_pos = ImVec2(screen_pos.x + size * 0.5f, screen_pos.y + size * 0.5f);

    if (data.texture.is_valid()) {
        draw_list->AddImage(
            data.texture.as_imgui_texture(),
            min_pos, max_pos
        );
    } else {
        // Draw placeholder
        draw_list->AddRectFilled(min_pos, max_pos, IM_COL32(60, 60, 80, 200));
    }
}

void replay_node_renderer::render_stats_color(ImDrawList* draw_list,
                                               const buffer_node_data& data,
                                               ImVec2 screen_pos,
                                               float zoom,
                                               [[maybe_unused]] bool selected,
                                               [[maybe_unused]] bool hovered) {
    float size = node_size_ * zoom;
    ImVec2 min_pos = ImVec2(screen_pos.x - size * 0.5f, screen_pos.y - size * 0.5f);
    ImVec2 max_pos = ImVec2(screen_pos.x + size * 0.5f, screen_pos.y + size * 0.5f);

    // Get stat value
    float value = 0.0f;
    switch (stat_type_) {
        case stat_type::average: value = data.current_stats.avg; break;
        case stat_type::sum: value = data.current_stats.sum; break;
        case stat_type::min: value = data.current_stats.min_val; break;
        case stat_type::max: value = data.current_stats.max_val; break;
        case stat_type::nse: value = data.current_stats.nse; break;
    }

    // Normalize value (assuming 0-1 range, can be adjusted)
    value = std::clamp(value, 0.0f, 1.0f);

    // Apply colormap
    color_t color = apply_colormap(data.config.colormap, value);
    ImU32 im_color = IM_COL32(color.r, color.g, color.b, color.a);

    draw_list->AddRectFilled(min_pos, max_pos, im_color);
}

void replay_node_renderer::render_delta_flow(ImDrawList* draw_list,
                                              const buffer_node_data& data,
                                              ImVec2 screen_pos,
                                              float zoom,
                                              [[maybe_unused]] bool selected,
                                              [[maybe_unused]] bool hovered) {
    float size = node_size_ * zoom;
    ImVec2 min_pos = ImVec2(screen_pos.x - size * 0.5f, screen_pos.y - size * 0.5f);
    ImVec2 max_pos = ImVec2(screen_pos.x + size * 0.5f, screen_pos.y + size * 0.5f);

    // Calculate average delta
    float delta = 0.0f;
    if (!data.previous_data.empty() && data.previous_data.size() == data.current_data.size()) {
        float sum_delta = 0.0f;
        for (std::size_t i = 0; i < data.current_data.size(); ++i) {
            sum_delta += std::abs(data.current_data[i] - data.previous_data[i]);
        }
        delta = sum_delta / static_cast<float>(data.current_data.size());
    }

    // Map delta to color (blue = no change, yellow/red = high change)
    float normalized = std::min(delta * 10.0f, 1.0f);  // Scale factor adjustable

    // Custom blue-to-red gradient for delta visualization
    uint8_t r = static_cast<uint8_t>(normalized * 255);
    uint8_t g = static_cast<uint8_t>((1.0f - std::abs(normalized - 0.5f) * 2.0f) * 150);
    uint8_t b = static_cast<uint8_t>((1.0f - normalized) * 255);

    draw_list->AddRectFilled(min_pos, max_pos, IM_COL32(r, g, b, 220));
}

} // namespace coast::replay
