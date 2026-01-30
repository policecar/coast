//
// Created by jk on 25.01.26.
//

#ifndef DATASET_CFG_H
#define DATASET_CFG_H

#include <unordered_map>

#include "mnist_cfg.h"
#include "dsprites_cfg.h"
#include "emnist_cfg.h"
#include "kmnist_cfg.h"
#include "cifar10_cfg.h"
#include "cifar100_cfg.h"
#include "chairs3d_cfg.h"

namespace dataset {

struct cfg_store {

    std::unordered_map<std::string,mnist::mnist_param_t>       mnist;
    std::unordered_map<std::string,dsprites::dsprites_param_t> dsprites;
    std::unordered_map<std::string,emnist::emnist_param_t>     emnist;
    std::unordered_map<std::string,kmnist::kmnist_param_t>     kmnist;
    std::unordered_map<std::string,cifar10::cifar10_param_t>   cifar10;
    std::unordered_map<std::string,cifar100::cifar100_param_t> cifar100;
    std::unordered_map<std::string,chairs3d::chairs3d_param_t> chairs3d;

    cfg_store()
    {
        mnist["default"];
        dsprites["default"];
        emnist["default"];
        kmnist["default"];
        cifar10["default"];
        cifar100["default"];
        chairs3d["default"];
    }

};


}

#endif //DATASET_CFG_H
