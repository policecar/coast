//
// Created by jk on 05.09.25.
//

#ifndef VEC_RING_BUFFER_H
#define VEC_RING_BUFFER_H

#include <cstdint>
#include <span>
#include <vector>

#include "gui_vis/core/color_types.h"
#include "gui_vis/core/colormap.h"
#include "gui_vis/core/gl_texture.h"

namespace coast {

class vec_ring_buffer {

    gl_texture texture;
    std::vector<color_t> pixel_data;

    uint16_t vec_size;
    uint16_t ring_size;

    uint16_t ring_pos;

    // View state (zoom and pan) - uses target values for smooth interpolation
    float zoom_level = 2.0f;
    float target_zoom = 2.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;
    float target_pan_x = 0.0f;
    float target_pan_y = 0.0f;

    // Colormap selection
    colormap_t colormap = colormap_t::viridis;
    bool colormap_dirty = false;  // Set when colormap changes, cleared after update

public:
    vec_ring_buffer(uint16_t _vec_size, uint16_t _ring_size);

    void update(const std::span<float> &data, bool auto_norm = true, float min_val = 0.0f, float max_val = 1.0f);

    // Render in ImGui window. Returns false if window was closed.
    // update_interval can be modified via slider (range 1-1000)
    bool render_imgui(const char* label, int& update_interval, float scale = 1.0f);

    // Get texture for custom rendering
    [[nodiscard]] const gl_texture& get_texture() const { return texture; }
    [[nodiscard]] uint16_t get_width() const { return ring_size; }
    [[nodiscard]] uint16_t get_height() const { return vec_size; }

    // Colormap dirty flag - set when colormap changes, triggers re-render
    [[nodiscard]] bool is_colormap_dirty() const { return colormap_dirty; }
    void clear_colormap_dirty() { colormap_dirty = false; }

    // No longer needed - RAII handles cleanup
    void free_resources();
};

} // coast

#endif //VEC_RING_BUFFER_H
