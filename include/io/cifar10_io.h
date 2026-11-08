//
// CIFAR-10 dataset IO entity
//

#ifndef CIFAR10_IO_H
#define CIFAR10_IO_H

#include "io/cifar_io_base.h"
#include "io/cifar10_cfg.h"
#include "core/entity_element_macros.h"

namespace cifar10 {

class cifar10_io : public dataset::cifar_io_base {
public:
    ELEMENT_TYPE(cifar10_io, "cifar10_input")

    explicit cifar10_io(const cifar10_param_t& params, sim::output_id_t output_id);
};

} // namespace cifar10

#endif // CIFAR10_IO_H
