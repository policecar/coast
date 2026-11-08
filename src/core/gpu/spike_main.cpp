// Framework spike measurements (S1, checkpoint-alpha groundwork).
//
// Spike 1 — unified-memory write path: how fast can the CPU get data into a GPU
// storage buffer on Strix Halo's LPDDR5x? Compares (a) wgpuQueueWriteBuffer
// (the standard staged upload) and (b) a mappedAtCreation buffer (memcpy into a
// directly-mapped range) against (baseline) a plain host->host memcpy. If the
// GPU write paths approach memcpy bandwidth, the "copy" is effectively free on
// unified memory — no PCIe-style transfer.
//
// Spike 2 — batched submission host cost: records N buffer-copy commands into a
// single encoder and submits once, for growing N. Measures host-side record vs
// finish vs submit time separately. If submit cost stays ~flat as N grows, a
// per-step submit is O(1) wrt kernel count (MH7). Uses copies, not compute
// pipelines (those are S2) — this is groundwork; real per-step cost is S5.
//
// Diagnostic only; not part of the production build path. Run unsandboxed.

#include "gpu_context.h"

#include <webgpu/webgpu.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

namespace {

using wall_clock = std::chrono::steady_clock;
using fsec       = std::chrono::duration<double>;

double now_to(const wall_clock::time_point& t0) {
    return fsec(wall_clock::now() - t0).count();
}

WGPUBuffer make_buffer(WGPUDevice dev, uint64_t size, WGPUBufferUsage usage,
                       bool mapped_at_creation) {
    WGPUBufferDescriptor d = {};
    d.usage            = usage;
    d.size             = size;
    d.mappedAtCreation = mapped_at_creation ? 1u : 0u;
    return wgpuDeviceCreateBuffer(dev, &d);
}

double gib_per_s(double bytes, double secs) {
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / secs;
}

// ---------------------------------------------------------------- Spike 1 ----
void spike_unified_memory(coast::gpu::gpu_context& ctx) {
    constexpr uint64_t kSize = 200ull * 1024 * 1024;  // 200 MiB
    constexpr int      kReps = 5;
    std::printf("\n=== Spike 1: unified-memory write path (%llu MiB) ===\n",
                static_cast<unsigned long long>(kSize / (1024ull * 1024)));

    std::vector<std::uint8_t> host(kSize, 0x5a);
    std::vector<std::uint8_t> host2(kSize);

    // Baseline: host->host memcpy (CPU memory bandwidth reference).
    {
        double best = 1e9;
        for (int i = 0; i < kReps; ++i) {
            auto t0 = wall_clock::now();
            std::memcpy(host2.data(), host.data(), kSize);
            best = std::min(best, now_to(t0));
        }
        std::printf("  baseline host->host memcpy : %.2f GiB/s\n", gib_per_s(kSize, best));
    }

    // (a) wgpuQueueWriteBuffer + flush.
    {
        WGPUBuffer buf = make_buffer(ctx.device, kSize,
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst), false);
        double best = 1e9;
        for (int i = 0; i < kReps; ++i) {
            auto t0 = wall_clock::now();
            wgpuQueueWriteBuffer(ctx.queue, buf, 0, host.data(), kSize);
            ctx.wait_idle();
            best = std::min(best, now_to(t0));
        }
        wgpuBufferRelease(buf);
        std::printf("  (a) wgpuQueueWriteBuffer   : %.2f GiB/s\n", gib_per_s(kSize, best));
    }

    // (b) mappedAtCreation: memcpy into the directly-mapped range, then unmap.
    {
        double best = 1e9;
        for (int i = 0; i < kReps; ++i) {
            WGPUBuffer buf = make_buffer(ctx.device, kSize,
                static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc), true);
            auto t0 = wall_clock::now();
            void* mapped = wgpuBufferGetMappedRange(buf, 0, kSize);
            std::memcpy(mapped, host.data(), kSize);
            wgpuBufferUnmap(buf);
            best = std::min(best, now_to(t0));
            wgpuBufferRelease(buf);
        }
        std::printf("  (b) mappedAtCreation memcpy: %.2f GiB/s\n", gib_per_s(kSize, best));
    }
}

// ---------------------------------------------------------------- Spike 2 ----
void spike_batched_submission(coast::gpu::gpu_context& ctx) {
    constexpr uint64_t kBufSize  = 4096;     // small src/dst
    constexpr uint64_t kCopySize = 256;      // bytes per recorded copy
    constexpr int      kReps     = 20;
    const int counts[] = {1, 10, 100, 1000, 5000};

    std::printf("\n=== Spike 2: batched submission host cost (one submit per batch) ===\n");
    std::printf("  %6s | %12s | %12s | %12s | %14s\n",
                "N", "record us", "finish us", "submit us", "submit us/cmd");

    WGPUBuffer src = make_buffer(ctx.device, kBufSize,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc), false);
    WGPUBuffer dst = make_buffer(ctx.device, kBufSize,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst), false);

    for (int n : counts) {
        double rec_best = 1e9, fin_best = 1e9, sub_best = 1e9;
        for (int r = 0; r < kReps; ++r) {
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);

            auto t0 = wall_clock::now();
            for (int k = 0; k < n; ++k) {
                wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, dst, 0, kCopySize);
            }
            double rec = now_to(t0);

            auto t1 = wall_clock::now();
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
            double fin = now_to(t1);

            auto t2 = wall_clock::now();
            wgpuQueueSubmit(ctx.queue, 1, &cmd);
            double sub = now_to(t2);

            ctx.wait_idle();
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);

            rec_best = std::min(rec_best, rec);
            fin_best = std::min(fin_best, fin);
            sub_best = std::min(sub_best, sub);
        }
        std::printf("  %6d | %12.2f | %12.2f | %12.2f | %14.4f\n",
                    n, rec_best * 1e6, fin_best * 1e6, sub_best * 1e6,
                    (sub_best * 1e6) / n);
    }

    wgpuBufferRelease(src);
    wgpuBufferRelease(dst);
}

}  // namespace

int main() {
    try {
        coast::gpu::gpu_context ctx;
        std::printf("adapter: %s\n", ctx.adapter_name.c_str());
        spike_unified_memory(ctx);
        spike_batched_submission(ctx);
        ctx.check_or_throw();
        std::printf("\n[coast-gpu-spike] done, no faults.\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[coast-gpu-spike] FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
