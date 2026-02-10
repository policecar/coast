//
// JSON config file loader
//

#ifndef JSON_LOADER_H
#define JSON_LOADER_H

#include <nlohmann/json.hpp>
#include "hd_ngm2_cfg.h"
#include "sim_builder_cfg.h"
#include "dataset_cfg.h"
#include "core/vis_hints.h"

using json = nlohmann::ordered_json;

namespace loader
{

struct global_cfg_t
{
    dataset::cfg_store datasets;
    ngm2::cfg_store    ngm2;
    sim::sim_cfg_t     sim;
    coast::vis_hints_t vis_hints;

    // Logging and snapshot settings (optional in JSON, loaded separately)
    std::string snapshot_dir = "snapshots";  // Directory for auto-save snapshots
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(global_cfg_t,
    datasets,
    ngm2,
    sim,
    vis_hints
)

class json_loader
{
    global_cfg_t global_cfg;

public:
    explicit json_loader(const std::string &filename);

    const dataset::cfg_store& get_datasets_cfg()  const { return global_cfg.datasets;  }
    const ngm2::cfg_store&    get_ngm2_cfg()      const { return global_cfg.ngm2;      }
    const sim::sim_cfg_t&     get_sim_cfg()       const { return global_cfg.sim;       }
    const coast::vis_hints_t  get_vis_hints_cfg() const { return global_cfg.vis_hints; }

    // Mutable access for configuration editing
    loader::global_cfg_t& get_config() { return global_cfg; }
    const loader::global_cfg_t& get_config() const { return global_cfg; }
};

}

#endif // JSON_LOADER_H
