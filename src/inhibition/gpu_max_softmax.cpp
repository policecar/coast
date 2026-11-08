//
// Created for the COAST swappable-inhibition feature.
//

#include "inhibition/gpu_max_softmax.h"

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace inhibition {

namespace {

// Self-contained WGSL port of max_softmax_apply (src/inhibition/max_softmax.cpp):
// out_i = max * exp(-beta * (max - v_i)), max reduced from 0.0. Like the CPU,
// the input passes through unchanged unless max is a normal float; the range
// test also rejects NaN, since every comparison with NaN is false.
constexpr std::string_view k_wgsl = R"WGSL(
const WG: u32 = 256u;
const FLT_MIN: f32 = 1.17549435e-38;
const FLT_MAX: f32 = 3.40282347e38;

struct InhibParams {
    neuron_cnt: u32,
    beta:       f32,
    _pad0:      u32,
    _pad1:      u32,
};

@group(0) @binding(0) var<storage, read>       i_act: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_out: array<f32>;
@group(0) @binding(2) var<uniform>             ip:    InhibParams;

var<workgroup> i_max: array<f32, WG>;

@compute @workgroup_size(WG)
fn max_softmax_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let n = ip.neuron_cnt;

    var mx = 0.0;
    for (var i = t; i < n; i = i + WG) { mx = max(mx, i_act[i]); }
    i_max[t] = mx;
    workgroupBarrier();

    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { i_max[t] = max(i_max[t], i_max[t + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    // Every invocation reads the reduced value after the last barrier.
    let max_val = i_max[0];

    if (!(max_val >= FLT_MIN && max_val <= FLT_MAX)) {
        for (var i = t; i < n; i = i + WG) { i_out[i] = i_act[i]; }
        return;
    }
    for (var i = t; i < n; i = i + WG) {
        i_out[i] = max_val * exp(-ip.beta * (max_val - i_act[i]));
    }
}
)WGSL";

// 16-byte uniform matching InhibParams (std140: u32,f32,u32,u32).
struct uni_t {
    std::uint32_t neuron_cnt;
    float         beta;
    std::uint32_t pad0;
    std::uint32_t pad1;
};
static_assert(sizeof(uni_t) == 16, "InhibParams uniform must be 16 bytes");

} // namespace

gpu_max_softmax_t::gpu_max_softmax_t(const inhibition_params_t& p)
    : gpu_inhibitor_t(p), beta_(static_cast<float>(p.strength))
{
    if (!std::isfinite(beta_) || beta_ < 0.0f)
        throw std::invalid_argument("gpu max_softmax: beta (strength) must be finite and >= 0, got " +
                                    std::to_string(beta_));
}

void gpu_max_softmax_t::emit(coast::gpu::gpu_device& dev,
                             const gpu_host_handles& h,
                             coast::gpu::step_plan&  sp)
{
    using namespace coast::gpu;

    uniform_ = dev.register_buffer(topology::uniform, sizeof(uni_t), false);
    const uni_t u{ h.neuron_cnt, beta_, 0u, 0u };
    dev.upload(uniform_, std::as_bytes(std::span<const uni_t>(&u, 1)));

    kernel_ = dev.register_kernel(k_wgsl, "max_softmax_main");

    plan_op op{};
    op.kind   = op_kind::compute;
    op.kernel = kernel_;
    plan_dispatch d{};
    d.bindings = {
        { h.act_in,   access::read,  0 },
        { h.out_page, access::write, 1 },
        { uniform_,   access::read,  2 },
    };
    d.dispatch = h.dispatch;
    op.dispatches.push_back(std::move(d));
    sp.ops.push_back(std::move(op));
}

}
