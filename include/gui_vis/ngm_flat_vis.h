//
// Created by jk on 08.09.25.
//

#ifndef NGM_FLAT_VIS_H
#define NGM_FLAT_VIS_H

#include "vec_group_vis.h"
#include "hd_ngm2_neuron_group.h"
#include <raylib.h>

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
    Texture2D texture {};
    std::vector<Color> pixel_data {};
    uint32_t px_width {};
    uint32_t px_height {};

public:
    explicit ngm_flat_vis(const ngm2::neuron_group_t& neuron_group, params_t _params);

    void update();

    void paint(Vector2 pos, float scale);

    void free_resources();

    static params_t get_default(uint32_t rep_width, uint32_t rep_height);

    [[nodiscard]] uint32_t get_height() const;
    [[nodiscard]] uint32_t get_width() const;

};

} // coast

#endif //NGM_FLAT_VIS_H
