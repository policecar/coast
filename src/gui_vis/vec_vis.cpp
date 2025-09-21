//
// Created by jk on 08.09.25.
//

#include "vec_vis.h"

namespace coast {

vec_vis::vec_vis(const param_t &_params) :
    params(_params)
{}

const vec_vis::param_t & vec_vis::get_params() const
{
    return params;
}

} // coast