//
// Created for the COAST swappable-inhibition feature.
//
// GPU counterpart of inhibitor_t: a kernel-emitter. The host (a GPU neuron
// group) describes the buffers it owns; emit() registers the emitter's own
// kernel + uniform on the device and appends one compute op binding those host
// buffers into the host's step_plan ("kernel injection"). The injected WGSL is
// self-contained (its own uniform layout), so it never couples to the host
// entity's uniform struct.
//

#ifndef COAST_INHIBITION_GPU_H
#define COAST_INHIBITION_GPU_H

#include <array>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "core/gpu/gpu_device.h"
#include "inhibition/inhibition_cfg.h"

namespace logging {
class state_writer;
class state_reader;
}

namespace inhibition {

// Host-owned handles the emitter binds. Strength is NOT here — it belongs to the
// emitter (set from the resolved inhibition config), so a named inhibition with
// a strength differing from the group's local_inhibition_strength is honored.
struct gpu_host_handles {
    coast::gpu::buffer_id       act_in;      // raw per-neuron activities (read)
    coast::gpu::buffer_id       out_page;    // inhibited output write page (WGSL declares read_write; host binds it as the write page)
    std::uint32_t               neuron_cnt;
    std::array<std::uint32_t,3> dispatch;    // workgroup counts
};

class gpu_inhibitor_t {
public:
    virtual ~gpu_inhibitor_t() = default;

    // The parameters this inhibitor runs with, as for inhibitor_t::params().
    [[nodiscard]] const inhibition_params_t& params() const { return params_; }

    // Register this emitter's kernel + uniform on `dev` and append its compute
    // op (binding h's buffers) to `sp`. Called once, from the host's
    // build_step_plan(). Preconditions: h.act_in / h.out_page are registered;
    // h.neuron_cnt == the group's neuron count.
    virtual void emit(coast::gpu::gpu_device& dev,
                      const gpu_host_handles& h,
                      coast::gpu::step_plan&  sp) = 0;

    // Persistence hooks, mirroring the CPU inhibitor_t base. Default no-op: the
    // GPU inhibitor kind (max_softmax) is stateless
    // (make_gpu_inhibitor rejects the stateful random_index). Present so the GPU neuron group can
    // write/read a symmetric "inhibition" checkpoint subgroup — matching the CPU
    // schema exactly (CPU<->GPU checkpoint interchange) and future-proofing a
    // stateful GPU inhibitor.
    virtual void save_state(logging::state_writer&) const {}
    virtual void load_state(logging::state_reader&, [[maybe_unused]] std::size_t neuron_count) {}

protected:
    explicit gpu_inhibitor_t(const inhibition_params_t& p) : params_(p) {}

    inhibition_params_t params_;
};

std::unique_ptr<gpu_inhibitor_t> make_gpu_inhibitor(const inhibition_params_t& params);

// Resolve-and-construct (backward-compatible default path; see make_inhibitor).
std::unique_ptr<gpu_inhibitor_t> make_gpu_inhibitor(
    const std::unordered_map<std::string, inhibition_params_t>& dict,
    const std::string& name,
    double fallback_strength);

}

#endif // COAST_INHIBITION_GPU_H
