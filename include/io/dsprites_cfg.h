//
// Created by jk on 25.01.26.
//

#ifndef DSPRITES_CFG_H
#define DSPRITES_CFG_H

#include <string>

namespace dsprites {

struct dsprites_param_t {
    std::string            hdf5_file         = "";
    std::size_t            change_interval   = 15;
    std::pair<double,double> noise_range     = {0.0, 0.05};
    int                    random_seed       = 0;
    bool                   randomize_readout = false;
};

}


#endif //DSPRITES_CFG_H
