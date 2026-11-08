//
// Created by jk on 25.05.26.
//
// ngm3 counterpart of ngm_flat_vis: renders a permanence-grid view of
// a ngm3::neuron_group_t. Same on-screen layout and controls (zoom /
// pan / colormap) as ngm_flat_vis; the only structural difference is
// how the per-(neuron, dendrite, representation) source data is
// pulled — the apical+proximal dendrites of each ngm3 neuron are
// presented as a flat list in declaration order (apical block first,
// then proximal block).

#ifndef NGM3_FLAT_VIS_H
#define NGM3_FLAT_VIS_H

#include "gui_vis/primitives/vec_group_vis.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "gui_vis/core/color_types.h"
#include "gui_vis/core/colormap.h"
#include "gui_vis/core/gl_texture.h"

namespace coast {

// Templated on the source neuron-group type (Group). Both ngm3::neuron_group_t
// (CPU) and ngm3::neuron_group_gpu_t (GPU, S10) satisfy the same duck-typed
// accessor contract: get_neuron_count/get_apical_count/get_proximal_count/
// get_rep_count, get_apical_representation/get_proximal_representation, and
// refresh_vis_mirror() (a no-op on the CPU type; an on-demand GPU->CPU readback
// on the GPU type — IDD §5.7). Member definitions live in the .cpp and are
// instantiated for each Group by the registrars at the bottom of that TU.
template<class Group>
class ngm3_flat_vis {

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
    const Group& ng;
    gl_texture texture;
    std::vector<color_t> pixel_data;
    uint32_t px_width {};
    uint32_t px_height {};

    // View state (zoom and pan) — uses target values for smooth interpolation.
    float zoom_level = 1.0f;
    float target_zoom = 1.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;
    float target_pan_x = 0.0f;
    float target_pan_y = 0.0f;

    bool visible = false;

    colormap_t colormap = colormap_t::viridis;
    bool colormap_dirty = false;

public:
    explicit ngm3_flat_vis(const Group& neuron_group, params_t _params);

    void update();

    bool render_imgui(const char* label, float scale = 1.0f);

    void set_visible(bool v) { visible = v; }
    [[nodiscard]] bool is_visible() const { return visible; }

    [[nodiscard]] bool is_colormap_dirty() const { return colormap_dirty; }
    void clear_colormap_dirty() { colormap_dirty = false; }

    [[nodiscard]] const gl_texture& get_texture() const { return texture; }

    void free_resources();

    static params_t get_default(uint32_t rep_width, uint32_t rep_height, uint32_t num_channels = 1);

    [[nodiscard]] uint32_t get_height() const;
    [[nodiscard]] uint32_t get_width() const;
};

} // namespace coast

#endif // NGM3_FLAT_VIS_H
