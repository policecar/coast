// S5 spike-validate measurements for mnist_io_gpu (develop.s5.spike-validate).
//
// Re-confirms the two framework spikes in situ (unified-memory dataset upload;
// batched-submission host cost) and measures the two readback costs that feed the
// D7-alpha-3 probe-shape decision at checkpoint beta:
//
//   M1  Dataset upload bandwidth (unified-memory): time the full ~188 MB MNIST
//       wgpuQueueWriteBuffer vs a plain host memcpy of the same bytes. On the
//       Strix Halo LPDDR5x both hit DRAM bandwidth — the "copy" is not a PCIe
//       transfer.
//   M2  Cost A — phase-2 round-trip floor: submit+wait_idle of a trivial encoder.
//       The dispatcher does two of these per step (process + swap_io), so 2x this
//       is the fixed per-step GPU-sync floor regardless of kernel/buffer count.
//   M3  Cost B — probe-output readback latency vs size (12 B actual probe, then
//       1 KB / 1 MB / 10 MB): copy->submit->map->pump->memcpy round-trip. Tells us
//       how large a per-entity probe buffer can grow before per-step readback
//       (alpha-3) costs more than kernel-side gating (alpha-1).
//   M4  Real per-step cost of the live entity (batched-submission in situ): build
//       a real env (mnist_io_gpu + CPU sink), run N steps, report ms/step. Should
//       track ~2x M2 (two fixed submits) — flat in the number of kernels recorded.
//
// Diagnostic only. Run UNSANDBOXED from the repo root (needs /dev/dri + ./data).

#include "gpu_context.h"
#include "wgpu_util.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"
#include "io/mnist_io_gpu.h"
#include "io/mnist_cfg.h"

#include <webgpu/webgpu.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

WGPUBuffer make_buf(WGPUDevice dev, uint64_t bytes, WGPUBufferUsage usage) {
    WGPUBufferDescriptor d = {};
    d.usage = usage;
    d.size  = bytes;
    return wgpuDeviceCreateBuffer(dev, &d);
}

void submit_wait(coast::gpu::gpu_context& g, WGPUCommandEncoder enc) {
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(g.queue, 1, &cmd);
    g.wait_idle();
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
}

struct map_state { bool done = false; bool ok = false; };
void on_map(WGPUMapAsyncStatus s, WGPUStringView, void* u, void*) {
    auto* m = static_cast<map_state*>(u);
    m->ok = (s == WGPUMapAsyncStatus_Success); m->done = true;
}

// One Cost-B round-trip: copy gpu->staging, submit, map, read. Returns nothing;
// caller times the loop.
void readback_once(coast::gpu::gpu_context& g, WGPUBuffer src, WGPUBuffer staging,
                   uint64_t bytes, std::vector<unsigned char>& dst) {
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, staging, 0, bytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(g.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    map_state m;
    WGPUBufferMapCallbackInfo mc = {};
    mc.mode = WGPUCallbackMode_AllowProcessEvents;
    mc.callback = on_map; mc.userdata1 = &m;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, bytes, mc);
    coast::gpu::pump_until(g.instance, m.done);
    const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, bytes);
    std::memcpy(dst.data(), mapped, bytes);
    wgpuBufferUnmap(staging);
}

constexpr sim::output_id_t kIdGpu = 0, kIdSink = 1;
struct cpu_sink final : sim::io_entity {
    sim::io_buffer* in_ = nullptr; sim::io_buffer* out_ = nullptr;
    std::array<std::size_t,1> ids_{ kIdGpu };
    void set_output_buffer(sim::io_buffer& b) override { out_ = &b; }
    void set_input_buffer(sim::output_id_t, sim::io_buffer& b) override { in_ = &b; }
    sim::step_result process() override { return {}; }
    [[nodiscard]] sim::output_id_t get_outp_id() const override { return kIdSink; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 1; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return ids_; }
};

} // namespace

int main() {
    try {
        // ---- M1/M2/M3 on a standalone context ------------------------------
        auto ctx = std::make_unique<coast::gpu::gpu_context>();
        std::printf("adapter: %s\n\n", ctx->adapter_name.c_str());
        auto& g = *ctx;

        // M1 — dataset upload vs host memcpy.
        {
            const uint64_t bytes = static_cast<uint64_t>(60000) * 784 * sizeof(float);
            std::vector<float> host(bytes / sizeof(float), 0.5f);
            WGPUBuffer ds = make_buf(g.device, bytes,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));

            auto t0 = clk::now();
            wgpuQueueWriteBuffer(g.queue, ds, 0, host.data(), bytes);
            WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(g.device, nullptr);
            submit_wait(g, e);                  // ensure the write completed
            double up_ms = ms_since(t0);

            std::vector<float> sink(host.size());
            auto t1 = clk::now();
            std::memcpy(sink.data(), host.data(), bytes);
            double mc_ms = ms_since(t1);

            const double gb = static_cast<double>(bytes) / 1e9;
            std::printf("M1 upload  : %.1f MB  queueWrite=%.2f ms (%.1f GB/s)  "
                        "host memcpy=%.2f ms (%.1f GB/s)\n",
                        gb * 1000.0, up_ms, gb / (up_ms / 1e3), mc_ms, gb / (mc_ms / 1e3));
            wgpuBufferRelease(ds);
        }

        // M2 — Cost A: submit+wait_idle round-trip floor (trivial encoder).
        {
            constexpr int K = 500;
            WGPUBuffer a = make_buf(g.device, 256,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst));
            WGPUBuffer b = make_buf(g.device, 256,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst));
            // warmup
            for (int i = 0; i < 10; ++i) {
                WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(g.device, nullptr);
                wgpuCommandEncoderCopyBufferToBuffer(e, a, 0, b, 0, 256);
                submit_wait(g, e);
            }
            auto t0 = clk::now();
            for (int i = 0; i < K; ++i) {
                WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(g.device, nullptr);
                wgpuCommandEncoderCopyBufferToBuffer(e, a, 0, b, 0, 256);
                submit_wait(g, e);
            }
            double per = ms_since(t0) / K;
            std::printf("M2 Cost A  : submit+wait_idle round-trip = %.4f ms  "
                        "(2x/step floor = %.4f ms)\n", per, 2 * per);
            wgpuBufferRelease(a); wgpuBufferRelease(b);
        }

        // M3 — Cost B: readback latency vs size.
        {
            constexpr int K = 200;
            const uint64_t sizes[] = { 12, 1024, 1ull << 20, 10ull << 20 };
            std::printf("M3 Cost B  : readback (copy+map+memcpy) round-trip\n");
            for (uint64_t bytes : sizes) {
                WGPUBuffer src = make_buf(g.device, bytes,
                    static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc));
                WGPUBuffer staging = make_buf(g.device, bytes,
                    static_cast<WGPUBufferUsage>(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst));
                std::vector<unsigned char> dst(bytes);
                for (int i = 0; i < 10; ++i) readback_once(g, src, staging, bytes, dst);  // warmup
                auto t0 = clk::now();
                for (int i = 0; i < K; ++i) readback_once(g, src, staging, bytes, dst);
                double per = ms_since(t0) / K;
                const double gbps = (bytes >= (1u<<20)) ? (static_cast<double>(bytes)/1e9)/(per/1e3) : 0.0;
                if (gbps > 0.0)
                    std::printf("             %8llu B : %.4f ms  (%.1f GB/s)\n",
                                static_cast<unsigned long long>(bytes), per, gbps);
                else
                    std::printf("             %8llu B : %.4f ms\n",
                                static_cast<unsigned long long>(bytes), per);
                wgpuBufferRelease(src); wgpuBufferRelease(staging);
            }
        }
        ctx.reset();

        // ---- M4 — real per-step cost of the live entity --------------------
        {
            mnist::mnist_param_t p;
            p.image_file = "./data/mnist/train-images-idx3-ubyte";
            p.label_file = "./data/mnist/train-labels-idx1-ubyte";
            p.change_interval = 1; p.noise_range = {0.0, 0.05}; p.random_seed = seed_t{ 0 };

            sim::env env;
            env.emplace_back<mnist::mnist_io_gpu>(p, kIdGpu);
            env.emplace_back<cpu_sink>();
            env.enable_gpu();              // env builds its own device internally
            env.init_io_buffers();

            constexpr int N = 1000;
            for (int i = 0; i < 20; ++i) {   // warmup
                sim::throw_on_step_fault(env.process(), "env::process");
                sim::throw_on_step_fault(env.swap_io(), "env::swap_io");
            }
            auto t0 = clk::now();
            for (int i = 0; i < N; ++i) {
                sim::throw_on_step_fault(env.process(), "env::process");
                sim::throw_on_step_fault(env.swap_io(), "env::swap_io");
            }
            double per = ms_since(t0) / N;
            std::printf("\nM4 per-step: %.4f ms/step over %d steps (2 kernels + GPU->CPU readback, "
                        "2 submits)\n", per, N);
        }

        std::printf("\n[coast-mnist-io-gpu-spike] done\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[coast-mnist-io-gpu-spike] FAILED: %s\n", ex.what());
        return 1;
    }
}
