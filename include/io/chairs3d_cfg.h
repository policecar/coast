//
// 3D Chairs dataset configuration
//

#ifndef CHAIRS3D_CFG_H
#define CHAIRS3D_CFG_H

#include <string>

#include "core/seed.h"
#include "io/dataset_run.h"

namespace chairs3d {

struct chairs3d_param_t {
    std::string            data_dir        = "";
    int                    image_width     = 64;   // Resize target
    int                    image_height    = 64;
    std::size_t            change_interval = 15;
    std::pair<double,double> noise_range   = {0.0, 0.05};
    seed_t                 random_seed     = {};
    // Not configured: set by sim::register_entity_handlers (dataset_run.h).
    dataset::run_binding_t run;
};

}

#endif //CHAIRS3D_CFG_H
