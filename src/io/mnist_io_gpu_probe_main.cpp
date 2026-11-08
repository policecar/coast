// Isolation probe for the mnist_io_gpu dataset entity (S5 / develop.s5).
//
// Following the S1-S4 pattern: a standalone unsandboxed exe that exercises the
// new GPU mechanism in isolation from JSON / sim_builder before it is wired into
// configs. It hand-builds a real sim::env with one mnist_io_gpu (id 0) plus a
// trivial CPU sink (id 1) that reads id 0 — the sink's presence makes the
// topology pre-pass select io_buffer_gpu_to_cpu for buffer 0, so the noised image
// the kernel produces is read back to the host and we can assert on it.
//
// Checks (all on the host after N steps):
//   1. PER-PIXEL CORRECTNESS — the strongest check: gpu_out[i] must equal
//      clamp(cpu_norm_image[idx][i] + noise_i, 0, 1) with noise_i in [lo, hi].
//      i.e. cpu_img[i] <= gpu_out[i] <= min(cpu_img[i] + hi, 1) (+ fp slack).
//      This simultaneously verifies: read-image-by-INDEX (right idx), noise in
//      range, and the clamp. No NaN/Inf anywhere.
//   2. HOST SCHEDULE — cur_idx advanced as expected (change_interval == 1).
//
// Run UNSANDBOXED — the sandbox blocks /dev/dri so Vulkan enumerates 0 adapters.
// Must run from the repo root (loads ./data/mnist/...).

#include "core/gpu/gpu_device.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"
#include "io/mnist_io_gpu.h"
#include "io/mnist_cfg.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <span>
#include <vector>

namespace {

constexpr int kSteps = 10;
constexpr sim::output_id_t kIdGpu  = 0;
constexpr sim::output_id_t kIdSink = 1;

// Trivial CPU consumer of the GPU output — exists only to force the gpu_to_cpu
// io_buffer variant on buffer 0 so the kernel output is CPU-visible.
struct cpu_sink final : sim::io_entity {
    sim::io_buffer* in_  = nullptr;
    sim::io_buffer* out_ = nullptr;
    std::array<std::size_t,1> inp_ids_{ kIdGpu };

    void set_output_buffer(sim::io_buffer& b) override { out_ = &b; }
    void set_input_buffer(sim::output_id_t, sim::io_buffer& b) override { in_ = &b; }
    sim::step_result process() override { /* nothing — the seam itself is the point */ return {}; }
    [[nodiscard]] sim::output_id_t get_outp_id()  const override { return kIdSink; }
    [[nodiscard]] std::size_t      get_outp_size() const override { return 1; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return inp_ids_; }
};

bool finite_in_unit(std::span<const float> v) {
    for (float x : v) {
        if (!std::isfinite(x) || x < -1e-6f || x > 1.0f + 1e-6f) return false;
    }
    return true;
}

} // namespace

int main() {
    try {
        mnist::mnist_param_t params;
        params.image_file      = "./data/mnist/train-images-idx3-ubyte";
        params.label_file      = "./data/mnist/train-labels-idx1-ubyte";
        params.change_interval = 1;
        params.noise_range     = {0.0, 0.05};
        params.random_seed     = seed_t{ 0 };
        params.shuffle         = false;  // probe asserts the sequential schedule
                                         // (cur_idx 0,1,2,...); shuffle default is
                                         // now on, so pin it off here.

        sim::env env;
        env.emplace_back<mnist::mnist_io_gpu>(params, kIdGpu);
        env.emplace_back<cpu_sink>();
        env.enable_gpu();                 // enable BEFORE init_io_buffers
        env.init_io_buffers();            // -> io_buffer_gpu_to_cpu for buffer 0
        std::printf("adapter: %s\n\n", env.gpu_adapter_name().c_str());

        for (int s = 0; s < kSteps; ++s) {
            sim::throw_on_step_fault(env.process(), "env::process");
            sim::throw_on_step_fault(env.swap_io(), "env::swap_io");
        }

        // Recover the entity for cur_idx + probe readback.
        auto ents = env.get_entities<mnist::mnist_io_gpu>();
        auto& e = (*ents.value())[0];

        bool ok = true;

        // The last produced step used idx = cur_idx - 1 (process() writes the
        // uniform with the current idx, then advances).
        const std::size_t used_idx = e.current_index() - 1;
        auto cpu_img = e.get_norm_image(used_idx);

        auto gpu_out = (*env.get_io_buffer(kIdGpu))->cur_read_buffer();

        // 1. per-pixel correctness + finiteness
        bool pix_ok = (gpu_out.size() == cpu_img.size()) && finite_in_unit(gpu_out);
        if (pix_ok) {
            for (std::size_t i = 0; i < gpu_out.size(); ++i) {
                const float lo = cpu_img[i] - 1e-5f;
                const float hi = std::min(cpu_img[i] + 0.05f, 1.0f) + 1e-5f;
                if (gpu_out[i] < lo || gpu_out[i] > hi) {
                    std::printf("    pixel %zu: gpu=%.6f cpu=%.6f (allowed [%.6f,%.6f])\n",
                                i, gpu_out[i], cpu_img[i], lo, hi);
                    pix_ok = false;
                    break;
                }
            }
        }
        std::printf("  [per-pixel]  %s: gpu_out == clamp(cpu_img[%zu] + noise[0,0.05])\n",
                    pix_ok ? "PASS" : "FAIL", used_idx);
        ok = ok && pix_ok;

        // 2. host schedule advanced
        const bool sched_ok = (e.current_index() == static_cast<std::size_t>(kSteps));
        std::printf("  [schedule]   %s: cur_idx == %d (advanced one image per step)\n",
                    sched_ok ? "PASS" : "FAIL", kSteps);
        ok = ok && sched_ok;

        std::printf("\n[coast-mnist-io-gpu-probe] %s\n", ok ? "ALL PASS" : "FAILURES");
        return ok ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[coast-mnist-io-gpu-probe] FAILED: %s\n", ex.what());
        return 1;
    }
}
