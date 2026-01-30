//
// KMNIST dataset configuration
//

#ifndef KMNIST_CFG_H
#define KMNIST_CFG_H

#include <string>

namespace kmnist {

struct kmnist_param_t {
    std::string            image_file      = "";
    std::string            label_file      = "";
    std::size_t            change_interval = 15;
    std::pair<double,double> noise_range   = {0.0, 0.05};
    int                    random_seed     = 0;
};

}

#endif //KMNIST_CFG_H
