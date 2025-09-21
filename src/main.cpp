#include <vector>
#include <functional>
#include <random>
#include <string>
#include <cstdlib>
#include <chrono>
#include <print>

#include <raylib.h>

#include "ray_app.h"
#include "fbgd.h"
#include "hd_ngm2.h"
#include "imgui.h"
#include "mnist_io.h"
#include "sim_env.h"
#include "vec_ring_buffer.h"
#include "ngm_flat_vis.h"
#include "hd_ngm2_tools.h"

using namespace coast;
using namespace ngm2;


int main(int argc, char **argv)
{
    /*
    // entropy test
    std::mt19937 rgen {};
    std::uniform_real_distribution<float> dis {0.0f, 0.01f};
    std::uniform_real_distribution<float> dis_high {0.8f, 1.f};

    std::vector<float> tmp(50);
    std::vector<float> tmp2(49);
    for (auto &val : tmp)
        val = dis(rgen);

    float nse = std::sqrt(normalized_shannon_entropy(tmp));

    std::printf("random vector: %f\n",nse);

    for (int i = 0; i < 50; ++i) {
        tmp[i] = dis_high(rgen);
        int k = 0;
        for (int j = 0; j < 49; ++j) {
            if (j == i) ++k;
            tmp2[j] = tmp[k++];
        }
        nse = std::sqrt(normalized_shannon_entropy(tmp));
        float nse2 = std::sqrt(normalized_shannon_entropy(tmp2));
        std::printf("%d high, random vector: %f vs %f \n",i,nse,nse2);
    }


    return 0;
    */

    if (argc < 3) {
        return -1;
    }

    sim::env simulation_environment;

    simulation_environment.emplace_back<sim::mnist_io>(0, 15, std::string(argv[1]), std::string(argv[2]));
    //simulation_environment.emplace_back<sim::mnist_io>(4, 2, std::string(argv[1]), std::string(argv[2]));
    simulation_environment.emplace_back<neuron_group_t>(basic_cng(1, 50, 28*28,  {0}, 1025));
    simulation_environment.emplace_back<neuron_group_t>(basic_cng(2, 50, 100,{1,3}, 2025));
    simulation_environment.emplace_back<neuron_group_t>(basic_cng(3, 50, 100,{1,2}, 3025));

    simulation_environment.init_io_buffers();

    sim::mnist_io &mio = simulation_environment.get_entities<sim::mnist_io>().value()->at(0);
    neuron_group_t &mnist_group = simulation_environment.get_entities<neuron_group_t>().value()->at(0);
    neuron_group_t &post_group = simulation_environment.get_entities<neuron_group_t>().value()->at(1);
    neuron_group_t &post_group2 = simulation_environment.get_entities<neuron_group_t>().value()->at(2);

    ray_app main_app;

    Font deja = LoadFont("../3rd_party/DejaVuSans.ttf");

    vec_ring_buffer vrb1 { static_cast<uint16_t>(simulation_environment.get_io_buffer(1).value()->size()), 750 };
    vec_ring_buffer vrb2 { static_cast<uint16_t>(simulation_environment.get_io_buffer(2).value()->size()), 750 };
    vec_ring_buffer vrb3 { static_cast<uint16_t>(simulation_environment.get_io_buffer(3).value()->size()), 750 };

    ngm_flat_vis::params_t vis_params1 = ngm_flat_vis::get_default(28,28);
    ngm_flat_vis vis1 { mnist_group, vis_params1 };

    ngm_flat_vis::params_t vis_params2 = ngm_flat_vis::get_default(50,2);
    vis_params2.vis_params.vec_params.elem_height = 5;
    ngm_flat_vis vis2 { post_group, vis_params2 };

    ngm_flat_vis::params_t vis_params3 = ngm_flat_vis::get_default(50,2);
    vis_params3.vis_params.vec_params.elem_height = 5;
    ngm_flat_vis vis3 { post_group2, vis_params3 };

    simulation_environment.set_post_process_hook(
        [&]() {
            vrb1.update(simulation_environment.get_io_buffer(1).value()->cur_write_buffer(), false, 0.0f, 1.0f);
            vrb2.update(simulation_environment.get_io_buffer(2).value()->cur_write_buffer(), false, 0.0f, 1.0f);
            vrb3.update(simulation_environment.get_io_buffer(3).value()->cur_write_buffer(), false, 0.0f, 1.0f);
        }
    );

    int process_steps_per_frame = 1000;

    main_app.register_state_func(
        [&] {
            for (int i = 0; i < process_steps_per_frame; ++i) {
                simulation_environment.process();
                simulation_environment.swap_io();
            }
            vis1.update();
            vis2.update();
            vis3.update();
        }
    );

    main_app.register_draw_func(
        [&simulation_environment,&deja] {
            ClearBackground(RAYWHITE);
            DrawFPS(10, 10);
            for (int line = 40; auto &io_ent : simulation_environment.iterate_entities()) {
                DrawTextEx(deja,io_ent.status_str().c_str(),Vector2(10,line),18, 1,BLACK);
                line += 60;
            }
        }
    );

    main_app.register_draw_func(
        [&]() {
            vrb1.paint({5.0f,300.0f},0.0f,2.0f);
            vrb2.paint({5.0f,410.0f},0.0f,2.0f);
            vrb3.paint({5.0f,520.0f},0.0f,2.0f);
            constexpr float vis1_scale = 0.75f;
            vis1.paint({5.0f,630.0f},vis1_scale);
            vis2.paint({5.0f,630.0f + (float)vis1.get_height()*vis1_scale + 10.0f},1.0f);
            vis3.paint({5.0f,630.0f + (float)vis1.get_height()*vis1_scale + 10.0f + (float)vis2.get_height() + 10.0f},1.0f);
        }
    );

    main_app.register_gui_func(
        [&]()
        {
            ImGui::SliderInt("mnist_io_change_interval", &mio.get_change_interval(), 0, 500);
            ImGui::SliderInt("process samples per frame", &process_steps_per_frame, 0, 1000);
            ImGui::SliderFloat("1st local inhibition strength", &mnist_group.get_local_inhibition_strength(), 0.1, 20.0);
            ImGui::SliderFloat("2st local inhibition strength", &post_group.get_local_inhibition_strength(), 0.1, 20.0);
            ImGui::SliderFloat("3st local inhibition strength", &post_group2.get_local_inhibition_strength(), 0.1, 20.0);
        }
    );

    main_app.register_shutdown_func(
        [&]() {
            vrb1.free_resources();
            vrb2.free_resources();
            vrb3.free_resources();
            vis1.free_resources();
            vis2.free_resources();
            vis3.free_resources();
        }
    );

    main_app.run();


    /*
    fbgd graph_vis;
    ngm_vis n_vis;

    graph_vis.set_draw_func([&](
        const std::vector<fbgd::node>& nodes,
        const std::vector<fbgd::edge>& edges,
        const std::vector<uint32_t>&   node_lu
    ) {
       n_vis.paint(nodes,edges,node_lu);
    });

    auto ngm_vec = simulation_environment.get_entities<neuron_group_t>().value();

    auto &ngm_vis_ref = ngm_vec->at(0);
    n_vis.create_model(graph_vis,ngm_vis_ref);

    std::size_t vis_upd_interval = 60;
    std::size_t vis_upd_cnt = 0;

    main_app.register_state_func(
        [&n_vis,&graph_vis,&ngm_vis_ref,&vis_upd_cnt,vis_upd_interval] {

            if ((++vis_upd_cnt %= vis_upd_interval) == 0)
                n_vis.update_model(graph_vis,ngm_vis_ref);

        }
    );

    main_app.register_state_func(
        [&graph_vis] {
            graph_vis.update();
        }
    );



    main_app.register_draw_func(
        [&graph_vis] {
            graph_vis.draw();
        }
    );

    main_app.register_shutdown_func(
        [&n_vis]() {
            n_vis.free_resources();
        }
    );
    */

    return 0;
}