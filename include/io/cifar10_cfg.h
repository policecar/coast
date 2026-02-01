//
// CIFAR-10 dataset configuration
//

#ifndef CIFAR10_CFG_H
#define CIFAR10_CFG_H

#include <string>

namespace cifar10 {

struct cifar10_param_t {
    std::string            data_dir        = "";
    std::size_t            change_interval = 15;
    std::pair<double,double> noise_range   = {0.0, 0.05};
    int                    random_seed     = 0;
    bool                   use_test_set    = false;
};

}

#endif //CIFAR10_CFG_H
