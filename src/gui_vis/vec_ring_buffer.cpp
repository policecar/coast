//
// Created by jk on 05.09.25.
//
#include <algorithm>

#include "vec_ring_buffer.h"

#include <cassert>
#include <numeric>

namespace coast {
vec_ring_buffer::vec_ring_buffer(uint16_t _vec_size, uint16_t _ring_size) :
    texture(),
    pixel_data(_vec_size * _ring_size),
    vec_size(_vec_size),
    ring_size(_ring_size),
    ring_pos(0)
{
    std::ranges::fill(pixel_data,Color {0,0,0,255});
    Image texture_img {
        pixel_data.data(),
        static_cast<int>(ring_size),
        static_cast<int>(vec_size),
        1,
        PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    };
    texture = LoadTextureFromImage(texture_img);
}

void vec_ring_buffer::update(const std::span<float> &data, bool auto_norm, float min_val, float max_val)
{
    assert(data.size() == vec_size);
    if (auto_norm) {
        max_val = std::reduce(data.begin(),data.end(),0.0f,[](float a, float b){ return std::max(a,b);});
        min_val = std::reduce(data.begin(),data.end(),max_val,[](float a, float b){ return std::min(a,b);});
        if (min_val > 0.0f)
            min_val = 0.0f;
    }
    for (std::size_t px_idx = ring_pos, d_idx = 0;
         px_idx < pixel_data.size();
         px_idx += ring_size, ++d_idx)
    {
        Color &cur_pixel    = pixel_data[px_idx];
        uint8_t scaled_data = std::clamp<uint16_t>( static_cast<uint16_t>((data[d_idx] - min_val) * 255.0 / (max_val - min_val)), 0, 255);
        cur_pixel.r = scaled_data / 2;
        cur_pixel.g = scaled_data;
        cur_pixel.b = scaled_data / 4;
    }
    UpdateTexture(texture,pixel_data.data());
    ring_pos = (ring_pos + 1) % ring_size;
}

void vec_ring_buffer::paint(Vector2 pos, float rot, float scale)
{
    DrawTextureEx(texture, pos, rot, scale, WHITE );
}

void vec_ring_buffer::free_resources()
{
    if (IsTextureValid(texture))
        UnloadTexture(texture);
}

} // coast