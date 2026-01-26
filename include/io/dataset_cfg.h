//
// Created by jk on 25.01.26.
//

#ifndef DATASET_CFG_H
#define DATASET_CFG_H

#include <unordered_map>

#include "mnist_cfg.h"
#include "dsprites_cfg.h"

namespace dataset {

struct cfg_store {

    std::unordered_map<std::string,mnist::mnist_param_t>       mnist;
    std::unordered_map<std::string,dsprites::dsprites_param_t> dsprites;

    cfg_store()
    {
        mnist["default"];
        dsprites["default"];
    }

};


}

#endif //DATASET_CFG_H
