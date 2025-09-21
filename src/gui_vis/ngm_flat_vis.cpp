//
// Created by jk on 08.09.25.
//

#include "gradient.h"

#include "ngm_flat_vis.h"

#include <oneapi/tbb/task_arena.h>

namespace coast {



ngm_flat_vis::ngm_flat_vis(const ngm2::neuron_group_t &neuron_group, const params_t _params) :
    params(_params),
    ng(neuron_group)
{
    // determine the maximum required size of the pixeldata and texture
    px_width  = 0;
    px_height = 0;
    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t n = 0; n < neuron_cnt; ++n) {
        auto &neuron = ng.get_neuron(n);
        std::size_t dendrite_cnt = neuron.get_dendrite_count();
        uint32_t max_dendrite_height = 0;
        uint32_t max_dendrite_width  = 0;
        uint32_t neuron_width = 0;
        uint32_t neuron_height = 0;
        for (std::size_t d = 0; d < dendrite_cnt; ++d) {
            auto &dendrite = neuron.get_dendrite(d);
            const uint32_t max_rep_cnt = (dendrite.get_max_segment_idx() + 1) / 2;
            params.vis_params.vec_cnt = max_rep_cnt;
            vec_group_vis tmp {params.vis_params};
            const uint32_t dendrite_width = tmp.get_total_width();
            const uint32_t dendrite_height = tmp.get_total_height();
            switch (params.layout) {
                case layout_t::horizontal_per_neuron : neuron_width  += dendrite_width + params.vis_params.margin * 2;  break;
                case layout_t::vertical_per_neuron   : neuron_height += dendrite_height + params.vis_params.margin * 2; break;
            }
            max_dendrite_height = std::max(max_dendrite_height, dendrite_height);
            max_dendrite_width  = std::max(max_dendrite_width,  dendrite_width);
        }
        switch (params.layout) {
            case layout_t::horizontal_per_neuron : {
                px_width = std::max(px_width,neuron_width);
                px_height += max_dendrite_height;
            } break;
            case layout_t::vertical_per_neuron   : {
                px_width  += max_dendrite_width;
                px_height = std::max(px_height,neuron_height);
            }break;
        }
    }
    // create actual pixel data
    pixel_data.resize(px_width * px_height, Color {255,255,0,255});
    Image texture_img {
        pixel_data.data(),
        static_cast<int>(px_width),
        static_cast<int>(px_height),
        1,
        PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    };
    texture = LoadTextureFromImage(texture_img);
}

void ngm_flat_vis::update()
{
    Color black { 0,0,0,255 };
    Color white { 255,255,255,255 };
    gradient<2> def_grad ( std::array<Color,2> {black,white} );
    uint32_t group_pos_x  = 0;
    uint32_t group_pos_y = 0;
    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t n = 0; n < neuron_cnt; ++n) {
        auto &neuron = ng.get_neuron(n);
        std::size_t dendrite_cnt = neuron.get_dendrite_count();
        uint32_t max_dendrite_height = 0;
        uint32_t max_dendrite_width  = 0;
        for (std::size_t d = 0; d < dendrite_cnt; ++d) {
            auto &dendrite = neuron.get_dendrite(d);
            const uint32_t max_rep_cnt = (dendrite.get_max_segment_idx() + 1) / 2;
            params.vis_params.vec_cnt = max_rep_cnt;
            vec_group_vis vis {params.vis_params};
            const uint32_t rep_cnt = dendrite.get_representation_count();
            if (rep_cnt > max_rep_cnt) {
                std::printf("too many representations: %u vs %u\n",rep_cnt,max_rep_cnt);
                auto mask = dendrite.get_leaf_mask();
                for (auto m : mask)
                    std::printf("%u,",m);
                std::printf("\n");
            }
            for (uint32_t r = 0; r < rep_cnt; ++r)
                vis.update_vec<2>(r,dendrite.get_representation(r),def_grad,px_pos_t{group_pos_x,group_pos_y},pixel_data_t{pixel_data,px_width,px_height});
            const uint32_t dendrite_width = vis.get_total_width();
            const uint32_t dendrite_height = vis.get_total_height();
            switch (params.layout) {
                case layout_t::horizontal_per_neuron : group_pos_x += dendrite_width + params.vis_params.margin * 2;  break;
                case layout_t::vertical_per_neuron   : group_pos_y += dendrite_height + params.vis_params.margin * 2; break;
            }
            max_dendrite_height = std::max(max_dendrite_height, dendrite_height);
            max_dendrite_width  = std::max(max_dendrite_width,  dendrite_width);
        }
        switch (params.layout) {
            case layout_t::horizontal_per_neuron : {
                group_pos_x  = 0;
                group_pos_y += max_dendrite_height;
            } break;
            case layout_t::vertical_per_neuron   : {
                group_pos_x += max_dendrite_width;
                group_pos_y  = 0;
            } break;
        }
    }
    UpdateTexture(texture,pixel_data.data());
}

void ngm_flat_vis::paint(Vector2 pos, float scale)
{
    DrawTextureEx(texture, pos, 0.0f, scale, WHITE );
}

void ngm_flat_vis::free_resources()
{
    if (IsTextureValid(texture))
        UnloadTexture(texture);
}

ngm_flat_vis::params_t ngm_flat_vis::get_default(uint32_t rep_width, uint32_t rep_height)
{
    params_t result {};
    result.layout = layout_t::vertical_per_neuron;
    result.vis_params.layout = vec_group_vis::layout_t::vertical;
    result.vis_params.margin = 5;
    result.vis_params.vec_params.elem_height = 1;
    result.vis_params.vec_params.elem_width  = 1;
    result.vis_params.vec_params.rep_width   = rep_width;
    result.vis_params.vec_params.rep_height  = rep_height;
    return result;
}

uint32_t ngm_flat_vis::get_height() const
{
    return px_height;
}

uint32_t ngm_flat_vis::get_width() const
{
    return px_width;
}
} // coast