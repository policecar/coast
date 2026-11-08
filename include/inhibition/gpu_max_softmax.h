//
// Created for the COAST swappable-inhibition feature.
//
// GPU max_softmax: the kernel-injection counterpart of max_softmax_t. A
// self-contained WGSL port of max_softmax_apply with its own {neuron_cnt, beta}
// uniform.
//

#ifndef COAST_INHIBITION_GPU_MAX_SOFTMAX_H
#define COAST_INHIBITION_GPU_MAX_SOFTMAX_H

#include "inhibition/gpu_inhibition.h"

namespace inhibition {

class gpu_max_softmax_t : public gpu_inhibitor_t {
    float                 beta_;
    coast::gpu::buffer_id uniform_ = coast::gpu::invalid_buffer;
    coast::gpu::kernel_id kernel_  = coast::gpu::invalid_kernel;
public:
    // Throws std::invalid_argument unless beta is finite and >= 0 (as max_softmax_t).
    explicit gpu_max_softmax_t(const inhibition_params_t& p);  // beta = p.strength

    void emit(coast::gpu::gpu_device& dev,
              const gpu_host_handles& h,
              coast::gpu::step_plan&  sp) override;
};

}

#endif // COAST_INHIBITION_GPU_MAX_SOFTMAX_H
