//
// Created by jk on 05.09.25.
//
#include <algorithm>

#include "vec_ring_buffer.h"

#include <cassert>
#include <numeric>

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

} // coast
