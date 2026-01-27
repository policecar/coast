#include "sim_instance.h"
#include "sim_builder_interface.h"

#include <iostream>
#include <stdexcept>

namespace coast {

sim_instance::sim_instance(const std::string& config_path)
    : config_path(config_path)
{
    // Load configuration
    json_loader = std::make_unique<loader::json_loader>(config_path);

    // Set up simulation environment
    sim_env = std::make_unique<sim::env>();

    // Create simulation builder and register entity handlers
    sim_builder = std::make_unique<sim::sim_builder>(*sim_env);
    sim::register_entity_handlers(*sim_builder, *json_loader);

    // Build the simulation from config
    sim_builder->build_local(json_loader->get_sim_cfg());

    // Get reference to neuron groups
    auto groups_opt = sim_env->get_entities<ngm2::neuron_group_t>();
    if (!groups_opt || groups_opt.value()->empty()) {
        throw std::runtime_error("No neuron groups found in simulation environment");
    }
    groups = groups_opt.value();

    // Set up visualizations
    setup_ring_buffers();
    setup_post_process_hook();
    setup_visualizations();
}

sim_instance::~sim_instance()
{
    // Remove the post-process hook before destroying sim_env
    if (sim_env && post_process_hook_id != 0) {
        sim_env->remove_post_process_hook(post_process_hook_id);
    }
}

sim_instance::sim_instance(sim_instance&& other) noexcept
    : config_path(std::move(other.config_path))
    , json_loader(std::move(other.json_loader))
    , sim_env(std::move(other.sim_env))
    , sim_builder(std::move(other.sim_builder))
    , groups(other.groups)
    , ring_buffers(std::move(other.ring_buffers))
    , visualizations(std::move(other.visualizations))
    , post_process_hook_id(other.post_process_hook_id)
{
    other.groups = nullptr;
    other.post_process_hook_id = 0;
}

sim_instance& sim_instance::operator=(sim_instance&& other) noexcept
{
    if (this != &other) {
        // Clean up existing resources
        if (sim_env && post_process_hook_id != 0) {
            sim_env->remove_post_process_hook(post_process_hook_id);
        }

        config_path = std::move(other.config_path);
        json_loader = std::move(other.json_loader);
        sim_env = std::move(other.sim_env);
        sim_builder = std::move(other.sim_builder);
        groups = other.groups;
        ring_buffers = std::move(other.ring_buffers);
        visualizations = std::move(other.visualizations);
        post_process_hook_id = other.post_process_hook_id;

        other.groups = nullptr;
        other.post_process_hook_id = 0;
    }
    return *this;
}

void sim_instance::free_resources()
{
    for (auto& vrb : ring_buffers) {
        vrb.free_resources();
    }
    for (auto& vis : visualizations) {
        vis.free_resources();
    }
}

void sim_instance::setup_ring_buffers()
{
    ring_buffers.clear();
    ring_buffers.reserve(groups->size());

    for (std::size_t i = 0; i < groups->size(); ++i) {
        auto buf_opt = sim_env->get_io_buffer((*groups)[i].get_outp_id());
        if (!buf_opt) {
            throw std::runtime_error(
                "IO buffer not found for group " + std::to_string((*groups)[i].get_outp_id())
            );
        }
        ring_buffers.emplace_back(static_cast<uint16_t>(buf_opt.value()->size()), 750);
    }
}

void sim_instance::setup_post_process_hook()
{
    // Capture pointers for the lambda (safe as long as this instance exists)
    auto* env_ptr = sim_env.get();
    auto* groups_ptr = groups;
    auto* ring_buffers_ptr = &ring_buffers;

    post_process_hook_id = sim_env->set_post_process_hook(
        [env_ptr, groups_ptr, ring_buffers_ptr]() {
            for (std::size_t i = 0; i < groups_ptr->size(); ++i) {
                auto* buf = env_ptr->get_io_buffer((*groups_ptr)[i].get_outp_id()).value();
                (*ring_buffers_ptr)[i].update(buf->cur_write_buffer(), false, 0.0f, 1.0f);
            }
        }
    );
}

void sim_instance::setup_visualizations()
{
    visualizations.clear();
    visualizations.reserve(groups->size());

    const auto& vis_hints = json_loader->get_vis_hints_cfg();

    for (std::size_t i = 0; i < groups->size(); ++i) {
        // Determine visualization dimensions based on input types and vis_hints
        int width = 50;
        int height = 1;

        auto inp_ids = (*groups)[i].get_inp_ids();
        if (!inp_ids.empty()) {
            // Check the first input to determine dimensions
            auto* node = sim_builder->get_node(inp_ids[0]);
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
        visualizations.emplace_back((*groups)[i], params);
    }
}

} // namespace coast
