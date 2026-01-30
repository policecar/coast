//
// 3D Chairs dataset configuration
//

#ifndef CHAIRS3D_CFG_H
#define CHAIRS3D_CFG_H

#include <string>

namespace chairs3d {

struct chairs3d_param_t {
    std::string            data_dir        = "";
    int                    image_width     = 64;   // Resize target
    int                    image_height    = 64;
    std::size_t            change_interval = 15;
    std::pair<double,double> noise_range   = {0.0, 0.05};
    int                    random_seed     = 0;
};

}

#endif //CHAIRS3D_CFG_H
