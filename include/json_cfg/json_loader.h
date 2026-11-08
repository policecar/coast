//
// JSON config file loader
//

#ifndef JSON_LOADER_H
#define JSON_LOADER_H

#include <nlohmann/json.hpp>
#include "hd_ngm2/hd_ngm2_cfg.h"
#include "hd_ngm3/hd_ngm3_cfg.h"
#include "hd_ngm25/hd_ngm25_cfg.h"
#include "tools/sim_builder_cfg.h"
#include "io/dataset_cfg.h"
#include "retina/retina_cfg.h"
#include "core/vis_hints.h"
#include "observe/observe_cfg.h"

using json = nlohmann::ordered_json;

namespace loader
{

struct global_cfg_t
{
    dataset::cfg_store     datasets;
    retina::cfg_store      retina;
    ngm2::cfg_store        ngm2;
    ngm3::cfg_store        ngm3;
    ngm25::cfg_store       ngm25;
    sim::sim_cfg_t         sim;
    coast::vis_hints_t     vis_hints;
    observe::observe_cfg_t observe;

    // Snapshot directory LEAF. Empty ("" = "defer to the app default") means the
    // app_config's paths.snapshot_dir leaf applies, resolved under config_base
    // (coast::paths::resolve_snapshot_output_dir). A non-empty value wins; an
    // absolute value bypasses config_base entirely.
    std::string snapshot_dir;

    // A frozen run: every learning entity keeps its learning off (the neuron groups,
    // via cfg_store::frozen; SPEC.md M.7.6).
    bool frozen = false;

    // The run's identity (core/run_id.h, SPEC.md S.5.14): written into each
    // checkpoint, and a dataset continues its sequence from a checkpoint only when it
    // was written by the same run_id. "auto" asks for a fresh id.
    std::string run_id = "auto";
};

// _WITH_DEFAULT: missing keys fall back to the C++ default-initialised value.
// This lets older configs (no observe / snapshot_dir / ngm3) still load cleanly.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(global_cfg_t,
    datasets,
    retina,
    ngm2,
    ngm3,
    ngm25,
    sim,
    vis_hints,
    observe,
    snapshot_dir,
    frozen,
    run_id
)

class json_loader
{
    global_cfg_t global_cfg;

public:
    explicit json_loader(const std::string &filename);

    const dataset::cfg_store& get_datasets_cfg()  const { return global_cfg.datasets;  }
    const retina::cfg_store&  get_retina_cfg()    const { return global_cfg.retina;    }
    const ngm2::cfg_store&    get_ngm2_cfg()      const { return global_cfg.ngm2;      }
    const ngm3::cfg_store&    get_ngm3_cfg()      const { return global_cfg.ngm3;      }
    const ngm25::cfg_store&   get_ngm25_cfg()     const { return global_cfg.ngm25;     }
    const sim::sim_cfg_t&     get_sim_cfg()       const { return global_cfg.sim;       }
    const coast::vis_hints_t  get_vis_hints_cfg() const { return global_cfg.vis_hints; }

    // Mutable access for configuration editing
    loader::global_cfg_t& get_config() { return global_cfg; }
    const loader::global_cfg_t& get_config() const { return global_cfg; }
};

}

#endif // JSON_LOADER_H
