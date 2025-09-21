//
// Created by jk on 05.09.25.
//

#ifndef VEC_RING_BUFFER_H
#define VEC_RING_BUFFER_H

#include <cstdint>
#include<span>
#include<raylib.h>
#include<vector>

namespace coast {

class vec_ring_buffer {

    Texture2D texture {};
    std::vector<Color> pixel_data {};

    uint16_t vec_size;
    uint16_t ring_size;

    uint16_t ring_pos;

public:
    vec_ring_buffer(uint16_t _vec_size, uint16_t _ring_size);

    void update(const std::span<float> &data, bool auto_norm = true, float min_val = 0.0f, float max_val = 1.0f);
    void paint(Vector2 pos, float rot = 0.0f, float scale = 1.0f);

    void free_resources();
};

} // coast

#endif //VEC_RING_BUFFER_H
