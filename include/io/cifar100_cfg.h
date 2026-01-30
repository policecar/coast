//
// CIFAR-100 dataset configuration
//

#ifndef CIFAR100_CFG_H
#define CIFAR100_CFG_H

#include <string>

namespace cifar100 {

struct cifar100_param_t {
    std::string            data_dir          = "";
    std::size_t            change_interval   = 15;
    std::pair<double,double> noise_range     = {0.0, 0.05};
    int                    random_seed       = 0;
    bool                   use_test_set      = false;
    bool                   use_coarse_labels = false;  // Use 20 superclasses instead of 100 fine classes
};

}

#endif //CIFAR100_CFG_H
