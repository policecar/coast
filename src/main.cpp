#include <vector>
#include <functional>
#include <random>
#include <string>
#include <iostream>

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
    // check if program arguments were provided
    if (argc < 3) {
        std::cout << "Please provide MNIST training images and labes as 1st and 2nd parameter to the program.\n";
        return -1;
    }

    // store program arguments
    const std::string mnist_image_file { argv[1] };
    const std::string mnist_label_file { argv[2] };

    // set up simulation environment
    sim::env simulation_environment;

    /*
     * Set up the different entities that should run in the simulation. Each entity must be derived from
     * io_entity (see sim_core/io_entity.h)
     * The emplace_back function is templatized with the type of the respective entity (e.g., neuron_group_t) and
     * receives the parameters that the respective entity constructor expects.
     * In case of the neuron group it is an extensive parameter structure that describes the neuron group parameterization.
     * To aid in that parameterization the service function basic_cng (see hd_ngm2/hd_ngm2_cfg.h) is used.
     */
    simulation_environment.emplace_back<sim::mnist_io>(0, 15, mnist_image_file, mnist_label_file);
    simulation_environment.emplace_back<neuron_group_t>( basic_cng(1, 50, 28*28,  {0}, 1025) );
    simulation_environment.emplace_back<neuron_group_t>( basic_cng(2, 50, 100,{1,3}, 2025) );
    simulation_environment.emplace_back<neuron_group_t>( basic_cng(3, 50, 100,{1,2}, 3025) );

    /*
     * After setting up all entities, we ask the simulation environment to create and set up the io-buffers that
     * facilitate the exchange between the different simulation entities
     */
    simulation_environment.init_io_buffers();

    /*
     * If we want to access the simulation entities directly, e.g., to couple them to UI-elements for visualization,
     * we can create named references to the different simulation entities. The function "get_entities" is templatized
     * with the respective type of entity we want to get. It returns an optional-value that - if entities of the requested
     * types exist - delivers a vector-like object that can be used to access all entities of the respective type.
     */
    sim::mnist_io  &mio         = simulation_environment.get_entities<sim::mnist_io>().value()->at(0);
    neuron_group_t &mnist_group = simulation_environment.get_entities<neuron_group_t>().value()->at(0);
    neuron_group_t &post_group  = simulation_environment.get_entities<neuron_group_t>().value()->at(1);
    neuron_group_t &post_group2 = simulation_environment.get_entities<neuron_group_t>().value()->at(2);

    /*
     * This object represents the main GUI-application that we use for visualization. It provides a number of hooks
     * to which we can register functions that are called at the respective parts of the GUI-loop.
     */
    ray_app main_app;

    /*
     * setting up visualization of the three io-buffers of this simulation that represent the outputs of the
     * neuron groups.
     */
    vec_ring_buffer vrb1 { static_cast<uint16_t>(simulation_environment.get_io_buffer(1).value()->size()), 750 };
    vec_ring_buffer vrb2 { static_cast<uint16_t>(simulation_environment.get_io_buffer(2).value()->size()), 750 };
    vec_ring_buffer vrb3 { static_cast<uint16_t>(simulation_environment.get_io_buffer(3).value()->size()), 750 };

    /*
     * hooking the update functions of the three io-buffer visualizations to the post-process hook of the simulation
     * process step. We use a lambda that captures vrb1, vrb2, ... by reference ([&])
     */
    simulation_environment.set_post_process_hook(
        [&]() {
            vrb1.update(simulation_environment.get_io_buffer(1).value()->cur_write_buffer(), false, 0.0f, 1.0f);
            vrb2.update(simulation_environment.get_io_buffer(2).value()->cur_write_buffer(), false, 0.0f, 1.0f);
            vrb3.update(simulation_environment.get_io_buffer(3).value()->cur_write_buffer(), false, 0.0f, 1.0f);
        }
    );

    /*
     * setting up visualization of the representations learned by the neuron groups. The get_default function
     * provides a reasonable set of initial parameters and allows to set the semantic dimensions of the representations,
     * e.g., 28 by 28 for representations of mnist inputs
     */
    ngm_flat_vis::params_t vis_params1 = ngm_flat_vis::get_default(28,28);
    ngm_flat_vis vis1 { mnist_group, vis_params1 };

    ngm_flat_vis::params_t vis_params2 = ngm_flat_vis::get_default(50,2);
    vis_params2.vis_params.vec_params.elem_height = 5;
    ngm_flat_vis vis2 { post_group, vis_params2 };

    ngm_flat_vis::params_t vis_params3 = ngm_flat_vis::get_default(50,2);
    vis_params3.vis_params.vec_params.elem_height = 5;
    ngm_flat_vis vis3 { post_group2, vis_params3 };


    // variable to control the number of simulation steps per GUI-frame
    int process_steps_per_frame = 1000;

    /*
     * Registering a state update function to be called during the state-update-phase of the GUI-loop.
     * Local variables are captured by reference. In each GUI-loop iteration we perform "process_steps_per_frame"
     * iterations of the simulation and then update the visualizations of the representations learned by the
     * neuron groups.
     */
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

    // loading a font we need for printing status output
    Font deja = LoadFont("../3rd_party/DejaVuSans.ttf");

    /*
     * Registering a draw function to be called during the draw-phase of the GUI-loop.
     * Local variables are captured by reference. We start drawing by clearing the
     * background, drawing some info about our current FPS and then print the status
     * information of each simulation entity
     */
    main_app.register_draw_func(
        [&] {
            ClearBackground(RAYWHITE);
            DrawFPS(10, 10);
            for (int line = 40; auto &io_ent : simulation_environment.iterate_entities()) {
                DrawTextEx(deja,io_ent.status_str().c_str(),Vector2(10,line),18, 1,BLACK);
                line += 60;
            }
        }
    );

    /*
     * Registering another draw function that is concerned with all the visualization components.
     * Local variables are captured by reference. Vertical offsets currently managed haphazardly, but for now it'll do.
     */
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

    /*
     * Registering a dear imgui function that we can use to draw imgui-dialogs on top of the visualization.
     * Local variables are captured by reference. We Just define a bunch of sliders to play around with some settings
     * during the simulation. The values of the respective variables are directly manipulated through their references.
     * As we do not explicitly create an imgui dialog, the elements will be placed within a default dialog.
     */
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

    /*
     * Registering a function that performs some clean-up operations after the application window is closed.
     */
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

    /*
     * finally it is time to run the GUI application. This call will block until the window is closed.
     */
    main_app.run();

    return 0;
}

