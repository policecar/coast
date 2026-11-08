//
// Created by jk on 25.01.26.
//

#ifndef DSPRITES_CFG_H
#define DSPRITES_CFG_H

#include <string>
#include "io/sequence_params.h"

namespace dsprites {

struct dsprites_param_t : dataset::sequence_params_t {
    std::string hdf5_file = "";
};

}


#endif //DSPRITES_CFG_H
