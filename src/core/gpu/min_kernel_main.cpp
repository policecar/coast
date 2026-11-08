// Minimal end-to-end compute-kernel verification (S2 / develop.s2.min-kernel-test).
//
// Exercises the whole S2 toolchain on real hardware:
//   Test A — add-1 kernel: load WGSL -> pipeline -> bind group -> dispatch ->
//            submit -> read back; verify every element was incremented by 1.0.
//            This is the module->pipeline->dispatch->readback path every later
//            GPU step reuses.
//   Test B — shared_prng concatenation: prepend shared_prng_wgsl() to a kernel
//            that calls pcg_uniform(); verify it compiles AND every output lands
//            in [0, 1). Validates the concatenation-include mechanism.
//   Test C — error fidelity: load deliberately-broken WGSL; require a thrown
//            std::runtime_error and print its what() so the line/column
//            diagnostic can be eyeballed. This also settles the open question of
//            whether the compilation-info callback is driven by the event pump.
//
// Diagnostic only; not part of the production build path. Run UNSANDBOXED — the
// sandbox blocks /dev/dri so Vulkan enumerates zero adapters.

#include "gpu_context.h"
#include "core/gpu/shared_prng.h"
#include "wgpu_util.h"
#include "wgsl_module.h"

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kCount = 256;                  // elements
constexpr uint64_t kBytes = kCount * sizeof(float);

WGPUBuffer make_buffer(WGPUDevice dev, uint64_t size, WGPUBufferUsage usage) {
    WGPUBufferDescriptor d = {};
    d.usage = usage;
    d.size  = size;
    return wgpuDeviceCreateBuffer(dev, &d);
}

// Map-read callback result holder.
struct map_result {
    bool done = false;
    bool ok   = false;
};

void on_map(WGPUMapAsyncStatus status, WGPUStringView /*msg*/, void* ud1, void* /*ud2*/) {
    auto* r = static_cast<map_result*>(ud1);
    r->ok   = (status == WGPUMapAsyncStatus_Success);
    r->done = true;
}

// Run one storage-buffer kernel over kCount floats seeded with `seed`, then read
// the result back into `out`. Owns no module — caller passes the compiled one.
void run_storage_kernel(coast::gpu::gpu_context& ctx, WGPUShaderModule mod,
                        const std::vector<float>& seed, std::vector<float>& out) {
    WGPUBuffer data = make_buffer(ctx.device, kBytes,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage |
                                     WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst));
    WGPUBuffer staging = make_buffer(ctx.device, kBytes,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst));

    wgpuQueueWriteBuffer(ctx.queue, data, 0, seed.data(), kBytes);

    WGPUComputePipelineDescriptor pd = {};
    pd.compute.module     = mod;
    pd.compute.entryPoint = coast::gpu::sv_from("main");
    WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(ctx.device, &pd);

    WGPUBindGroupEntry entry = {};
    entry.binding = 0;
    entry.buffer  = data;
    entry.offset  = 0;
    entry.size    = kBytes;
    // GetBindGroupLayout returns an owned reference — release it after the bind
    // group captures it (the bind group holds its own internal reference).
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipe, 0);
    WGPUBindGroupDescriptor bgd = {};
    bgd.layout     = layout;
    bgd.entryCount = 1;
    bgd.entries    = &entry;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(ctx.device, &bgd);
    wgpuBindGroupLayoutRelease(layout);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (kCount + 63) / 64, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuCommandEncoderCopyBufferToBuffer(enc, data, 0, staging, 0, kBytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(ctx.queue, 1, &cmd);
    ctx.wait_idle();

    map_result mr;
    WGPUBufferMapCallbackInfo mc = {};
    mc.mode      = WGPUCallbackMode_AllowProcessEvents;
    mc.callback  = on_map;
    mc.userdata1 = &mr;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, kBytes, mc);
    coast::gpu::pump_until(ctx.instance, mr.done);
    if (!mr.ok) throw std::runtime_error("run_storage_kernel: buffer map failed");

    const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, kBytes);
    out.resize(kCount);
    std::memcpy(out.data(), mapped, kBytes);
    wgpuBufferUnmap(staging);

    wgpuComputePassEncoderRelease(pass);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuBindGroupRelease(bg);
    wgpuComputePipelineRelease(pipe);
    wgpuBufferRelease(staging);
    wgpuBufferRelease(data);
}

// ---------------------------------------------------------------- Test A ----
bool test_add_one(coast::gpu::gpu_context& ctx) {
    const char* wgsl = R"(
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&data)) { data[i] = data[i] + 1.0; }
}
)";
    WGPUShaderModule mod = coast::gpu::load_wgsl_module(ctx.instance, ctx.device, wgsl, "add_one");

    std::vector<float> seed(kCount), out;
    for (uint32_t i = 0; i < kCount; ++i) seed[i] = static_cast<float>(i);
    run_storage_kernel(ctx, mod, seed, out);
    wgpuShaderModuleRelease(mod);

    for (uint32_t i = 0; i < kCount; ++i) {
        if (out[i] != seed[i] + 1.0f) {
            std::printf("  [A] FAIL: element %u = %f, expected %f\n", i, out[i], seed[i] + 1.0f);
            return false;
        }
    }
    std::printf("  [A] add-1 kernel: PASS (every element += 1.0)\n");
    return true;
}

// ---------------------------------------------------------------- Test B ----
bool test_prng_concat(coast::gpu::gpu_context& ctx) {
    const char* kernel = R"(
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&data)) { data[i] = pcg_uniform(i ^ 0x9e3779b9u); }
}
)";
    std::string src = std::string(coast::gpu::shared_prng_wgsl()) + kernel;
    WGPUShaderModule mod = coast::gpu::load_wgsl_module(ctx.instance, ctx.device, src, "prng_concat");

    std::vector<float> seed(kCount, 0.0f), out;
    run_storage_kernel(ctx, mod, seed, out);
    wgpuShaderModuleRelease(mod);

    for (uint32_t i = 0; i < kCount; ++i) {
        if (!(out[i] >= 0.0f && out[i] < 1.0f)) {
            std::printf("  [B] FAIL: pcg_uniform[%u] = %f, outside [0,1)\n", i, out[i]);
            return false;
        }
    }
    std::printf("  [B] shared_prng concat: PASS (compiled; all outputs in [0,1))\n");
    return true;
}

// ---------------------------------------------------------------- Test C ----
bool test_error_fidelity(coast::gpu::gpu_context& ctx) {
    // 'flaot' typo + missing entry-point attributes -> WGSL validation errors.
    const char* bad = R"(
@group(0) @binding(0) var<storage, read_write> data: array<flaot>;
fn main() { data[0] = 1.0; }
)";
    try {
        WGPUShaderModule mod =
            coast::gpu::load_wgsl_module(ctx.instance, ctx.device, bad, "bad_wgsl");
        wgpuShaderModuleRelease(mod);
        std::printf("  [C] FAIL: bad WGSL did NOT throw\n");
        return false;
    } catch (const std::exception& e) {
        std::printf("  [C] error fidelity: PASS (threw). Diagnostic was:\n");
        std::printf("------------------------------------------------------------\n");
        std::printf("%s\n", e.what());
        std::printf("------------------------------------------------------------\n");
        return true;
    }
}

}  // namespace

int main() {
    try {
        coast::gpu::gpu_context ctx;
        std::printf("adapter: %s\n\n", ctx.adapter_name.c_str());

        bool ok = true;
        ok &= test_add_one(ctx);
        ok &= test_prng_concat(ctx);
        ok &= test_error_fidelity(ctx);

        ctx.check_or_throw();
        std::printf("\n[coast-gpu-min-kernel] %s\n", ok ? "ALL PASS" : "FAILURES");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[coast-gpu-min-kernel] FAILED: %s\n", e.what());
        return 1;
    }
}
