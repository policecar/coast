
//
// Created by jk on 08.09.25.
//

#ifndef NGM_FLAT_VIS_H
#define NGM_FLAT_VIS_H

#include "../../primitives/vec_group_vis.h"
#include "hd_ngm2_neuron_group.h"
#include "../../core/color_types.h"
#include "../../core/gl_texture.h"

namespace coast {

class ngm_flat_vis {

public:
    enum class layout_t
    {
        horizontal_per_neuron,
        vertical_per_neuron
    };

    struct params_t
    {
        vec_group_vis::params_t vis_params;
        layout_t                layout;
    };

private:

    params_t params;
    const ngm2::neuron_group_t& ng;
    gl_texture texture;
    std::vector<color_t> pixel_data;
    uint32_t px_width {};
    uint32_t px_height {};

    // View state (zoom and pan)
    float zoom_level = 1.0f;
    float pan_x = 0.0f;  // Pan offset in pixels (scaled coordinates)
    float pan_y = 0.0f;

    // Visibility state (for conditional updates)
    bool visible = false;

public:
    explicit ngm_flat_vis(const ngm2::neuron_group_t& neuron_group, params_t _params);

    void update();

    // Render in ImGui window with scroll and zoom support. Returns false if window was closed.
    bool render_imgui(const char* label, float scale = 1.0f);

    // Visibility state for conditional updates
    void set_visible(bool v) { visible = v; }
    [[nodiscard]] bool is_visible() const { return visible; }

    // Get texture for custom rendering
    [[nodiscard]] const gl_texture& get_texture() const { return texture; }

    void free_resources();

    static params_t get_default(uint32_t rep_width, uint32_t rep_height);

    [[nodiscard]] uint32_t get_height() const;
    [[nodiscard]] uint32_t get_width() const;

};

} // coast

#endif //NGM_FLAT_VIS_H
