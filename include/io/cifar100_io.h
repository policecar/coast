//
// CIFAR-100 dataset IO entity
//

#ifndef CIFAR100_IO_H
#define CIFAR100_IO_H

#include "io/cifar_io_base.h"
#include "io/cifar100_cfg.h"
#include "core/entity_element_macros.h"

namespace cifar100 {

class cifar100_io : public dataset::cifar_io_base {
public:
    ELEMENT_TYPE(cifar100_io, "cifar100_input")

    explicit cifar100_io(const cifar100_param_t& params, sim::output_id_t output_id);
};

} // namespace cifar100

#endif // CIFAR100_IO_H
