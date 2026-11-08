#include "tools/dataset_loader.h"

#include "json_cfg/json_loader.h"
#include "sim_core/sim_env.h"
#include "tools/sim_builder.h"
#include "tools/sim_builder_interface.h"

#include <algorithm>
#include <set>

namespace coast {

dataset_loader::dataset_loader(const std::string& config_path, hdf5::executor& h5_exec)
    : config_path_(config_path)
{
    json_loader_ = std::make_unique<loader::json_loader>(config_path);
    env_         = std::make_unique<sim::env>();
    builder_     = std::make_unique<sim::sim_builder>(*env_);

    // Register every handler — harmless, because only the handlers whose nodes
    // survive the filter below are ever invoked (build_local dispatches per node).
    sim::register_entity_handlers(*builder_, *json_loader_, h5_exec);

    // Filter the config down to CPU dataset nodes. Dropping neuron groups means no
    // ngm config is needed; dropping GPU datasets means no gpu_context is installed
    // (and they are not matrix-resolvable anyway). A node_set is kept only if every
    // node it references survived, otherwise build_local would throw on a missing
    // template node.
    const sim::sim_cfg_t& full = json_loader_->get_sim_cfg();
    sim::sim_cfg_t filtered;
    std::set<sim::sim_node_id_t> kept_ids;

    for (const auto& node : full.nodes) {
        if (sim::is_dataset_entity_type(node.entity_type) &&
            !sim::is_gpu_entity_type(node.entity_type)) {
            filtered.nodes.push_back(node);
            kept_ids.insert(node.id);
        }
    }
    for (const auto& ns : full.node_sets) {
        const bool all_kept = !ns.nodes.empty() &&
            std::all_of(ns.nodes.begin(), ns.nodes.end(),
                        [&](sim::sim_node_id_t id) { return kept_ids.contains(id); });
        if (all_kept) filtered.node_sets.push_back(ns);
    }

    builder_->build_local(filtered);
}

dataset_loader::~dataset_loader() = default;

} // namespace coast
