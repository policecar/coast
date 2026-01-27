#include <vector>
#include <iostream>

#include "json_loader.h"
#include "sim_env.h"
#include "sim_builder.h"
#include "sim_builder_interface.h"

#include "sdl_app.h"
#include "gui_manager.h"

// visualization
#include "vec_ring_buffer.h"
#include "ngm_flat_vis.h"


using namespace coast;

int main(int argc, char **argv)
{
    // check if program arguments were provided
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config.json>\n";
        return -1;
    }

    // load configuration
    loader::json_loader jl { argv[1] };

    // set up simulation environment
    sim::env simulation_environment;

    // we use the simulation builder to setup the simulation from the cfg file
    // we need to provide the capabilities to create and initialize entities
    // via corresponding functions before calling "build_local"
    sim::sim_builder sb { simulation_environment };

    sim::register_entity_handlers(sb,jl);

    sb.build_local( jl.get_sim_cfg() );

    /*
     * Set up the different entities that should run in the simulation. Each entity must be derived from
     * io_entity (see sim_core/io_entity.h)
     * The emplace_back function is templatized with the type of the respective entity (e.g., neuron_group_t) and
     * receives the parameters that the respective entity constructor expects.
     * In case of the neuron group it is an extensive parameter structure that describes the neuron group parameterization.
     * To aid in that parameterization the service function build_group_params (see tools/config_loader.h) is used.
     *
     * The setup is driven by the JSON config file: dataset type and parameters are read from cfg.dataset,
     * while neuron group specifications come from cfg.network.groups. Input sizes are resolved dynamically
     * by building a map of each group's output size.
     */


    /*
     * If we want to access the simulation entities directly, e.g., to couple them to UI-elements for visualization,
     * we can create named references to the different simulation entities. The function "get_entities" is templatized
     * with the respective type of entity we want to get. It returns an optional-value that - if entities of the requested
     * types exist - delivers a vector-like object that can be used to access all entities of the respective type.
     */
    auto groups_opt = simulation_environment.get_entities<ngm2::neuron_group_t>();
    if (!groups_opt || groups_opt.value()->empty()) {
        std::cerr << "No neuron groups found in simulation environment.\n";
        return -1;
    }
    auto& groups = *groups_opt.value();

    /*
     * This object represents the main GUI-application that we use for visualization. It provides a number of hooks
     * to which we can register functions that are called at the respective parts of the GUI-loop.
     */
    sdl_app main_app("COAST - Neural Simulation");

    /*
     * Set up ring buffer visualizations that represent the outputs of the neuron groups.
     * We use the group's output ID to get the corresponding IO buffer size.
     */
    std::vector<vec_ring_buffer> vrbs;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        auto buf_opt = simulation_environment.get_io_buffer(groups[i].get_outp_id());
        if (!buf_opt) {
            std::cerr << "IO buffer not found for group " << groups[i].get_outp_id() << "\n";
            return -1;
        }
        vrbs.emplace_back(static_cast<uint16_t>(buf_opt.value()->size()), 750);
    }

    /*
     * hooking the update functions of the io-buffer visualizations to the post-process hook of the simulation
     * process step. We use a lambda that captures vrbs and groups by reference ([&])
     */
    simulation_environment.set_post_process_hook(
        [&]() {
            // IO buffers validated at setup before this hook is registered - safe to dereference
            for (std::size_t i = 0; i < groups.size(); ++i) {
                auto* buf = simulation_environment.get_io_buffer(groups[i].get_outp_id()).value();
                vrbs[i].update(buf->cur_write_buffer(), false, 0.0f, 1.0f);
            }
        }
    );

    /*
     * Setting up visualization of the representations learned by the neuron groups.
     * Visualization dimensions are determined by vis_hints config, keyed by input entity_cfg.
     * Falls back to "default" hints if no exact match is found.
     */
    std::vector<ngm_flat_vis> visualizations;
    visualizations.reserve(groups.size());

    const auto& vis_hints = jl.get_vis_hints_cfg();

    for (std::size_t i = 0; i < groups.size(); ++i) {
        // Determine visualization dimensions based on input types and vis_hints
        int width = 50;
        int height = 1;

        auto inp_ids = groups[i].get_inp_ids();
        if (!inp_ids.empty()) {
            // Check the first input to determine dimensions
            auto* node = sb.get_node(inp_ids[0]);
            if (node) {
                const std::string& cfg_key = node->entity_cfg;
                switch (node->entity_type) {
                    case sim::io_entity_type_t::mnist_input: {
                        auto it = vis_hints.datasets.mnist.find(cfg_key);
                        if (it == vis_hints.datasets.mnist.end()) {
                            it = vis_hints.datasets.mnist.find("default");
                        }
                        if (it != vis_hints.datasets.mnist.end()) {
                            width = it->second.ngm_flat_vis.width;
                            height = it->second.ngm_flat_vis.height;
                        }
                    } break;
                    case sim::io_entity_type_t::dsprites_input: {
                        auto it = vis_hints.datasets.dsprites.find(cfg_key);
                        if (it == vis_hints.datasets.dsprites.end()) {
                            it = vis_hints.datasets.dsprites.find("default");
                        }
                        if (it != vis_hints.datasets.dsprites.end()) {
                            width = it->second.ngm_flat_vis.width;
                            height = it->second.ngm_flat_vis.height;
                        }
                    } break;
                    case sim::io_entity_type_t::ngm2_neuron_group: {
                        auto it = vis_hints.ngm2.neuron_groups.find(cfg_key);
                        if (it == vis_hints.ngm2.neuron_groups.end()) {
                            it = vis_hints.ngm2.neuron_groups.find("default");
                        }
                        if (it != vis_hints.ngm2.neuron_groups.end()) {
                            width = it->second.ngm_flat_vis.width;
                            height = it->second.ngm_flat_vis.height;
                        }
                    } break;
                    default:
                        break;
                }
            }
        }

        ngm_flat_vis::params_t params = ngm_flat_vis::get_default(width, height);
        if (height == 1) {
            // For flat representations, increase element height for visibility
            params.vis_params.vec_params.elem_height = 5;
        }
        visualizations.emplace_back(groups[i], params);
    }


    /*
     * Set up GUI manager for menu bar and dialogs.
     */
    gui_manager gui(main_app, simulation_environment, groups, vrbs, visualizations, jl.get_config());

    /*
     * Registering a state update function to be called during the state-update-phase of the GUI-loop.
     * In each GUI-loop iteration we perform "steps_per_frame" iterations of the simulation
     * and then update the visualizations of the representations learned by the neuron groups.
     */
    main_app.register_state_func(
        [&] {
            if (gui.is_simulation_running()) {
                for (int i = 0; i < gui.get_steps_per_frame(); ++i) {
                    simulation_environment.process();
                    simulation_environment.swap_io();
                    gui.update_activity_history();
                }
                for (auto& vis : visualizations) {
                    if (vis.is_visible()) {
                        vis.update();
                    }
                }
            }
        }
    );

    /*
     * Registering GUI callback for menu bar and dialogs.
     */
    main_app.register_gui_func(
        [&] {
            gui.render_menu_bar();
            gui.render_dialogs();
            gui.render_status_bar();
        }
    );

    /*
     * Registering a function that performs some clean-up operations after the application window is closed.
     */
    main_app.register_shutdown_func(
        [&]() {
            for (auto& vrb : vrbs) {
                vrb.free_resources();
            }
            for (auto& vis : visualizations) {
                vis.free_resources();
            }
        }
    );

    /*
     * finally it is time to run the GUI application. This call will block until the window is closed.
     */
    main_app.run();

    return 0;
}
