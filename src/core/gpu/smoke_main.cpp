// Minimal wgpu bootstrap smoke test (S1, checkpoint-alpha evidence for A-II1).
//
// Constructs a gpu_context (instance -> adapter -> device -> queue on the
// Vulkan backend / Metal on Apple), prints the adapter identity and the key
// device limits we care about for ngm3/mnist, surfaces any startup validation
// error, then exercises clean reverse-order teardown via the destructor.
//
// Exit 0 on success, 1 on any failure (exception carrying the verbatim wgpu
// diagnostic).

#include "gpu_context.h"

#include <cstdio>
#include <exception>

namespace {

void print_limits(const WGPULimits& l) {
    std::printf("device limits:\n");
    std::printf("  maxStorageBufferBindingSize     : %llu bytes (%.1f MiB)\n",
                static_cast<unsigned long long>(l.maxStorageBufferBindingSize),
                static_cast<double>(l.maxStorageBufferBindingSize) / (1024.0 * 1024.0));
    std::printf("  maxBufferSize                   : %llu bytes (%.1f MiB)\n",
                static_cast<unsigned long long>(l.maxBufferSize),
                static_cast<double>(l.maxBufferSize) / (1024.0 * 1024.0));
    std::printf("  maxStorageBuffersPerShaderStage : %u\n", l.maxStorageBuffersPerShaderStage);
    std::printf("  maxBindGroups                   : %u\n", l.maxBindGroups);
    std::printf("  maxBindingsPerBindGroup         : %u\n", l.maxBindingsPerBindGroup);
    std::printf("  maxComputeInvocationsPerWorkgroup: %u\n", l.maxComputeInvocationsPerWorkgroup);
    std::printf("  maxComputeWorkgroupSizeX/Y/Z    : %u / %u / %u\n",
                l.maxComputeWorkgroupSizeX, l.maxComputeWorkgroupSizeY, l.maxComputeWorkgroupSizeZ);
    std::printf("  maxComputeWorkgroupStorageSize  : %u bytes\n", l.maxComputeWorkgroupStorageSize);
    std::printf("  maxComputeWorkgroupsPerDimension: %u\n", l.maxComputeWorkgroupsPerDimension);
}

}  // namespace

int main() {
    try {
        std::printf("[coast-gpu-smoke] constructing gpu_context...\n");
        coast::gpu::gpu_context ctx;

        std::printf("adapter: %s\n", ctx.adapter_name.c_str());
        print_limits(ctx.limits);

        // Surface any uncaptured validation error recorded during bring-up.
        ctx.check_or_throw();

        // No work queued; should return immediately. Proves the poll path works.
        ctx.wait_idle();

        std::printf("[coast-gpu-smoke] OK — tearing down (reverse order)...\n");
        // ctx destructor releases queue -> device -> adapter -> instance here.
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[coast-gpu-smoke] FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("[coast-gpu-smoke] teardown complete, no faults.\n");
    return 0;
}
