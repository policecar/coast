#ifndef COAST_CORE_GPU_GPU_CONTEXT_H
#define COAST_CORE_GPU_GPU_CONTEXT_H

#include <webgpu/webgpu.h>

#include <atomic>
#include <mutex>
#include <string>

namespace coast::gpu {

// Owns the wgpu lifecycle: instance -> adapter -> device -> queue, plus the
// queried device limits and the async-fault surface (device-lost and
// uncaptured-error callbacks). A single instance is held as
// std::unique_ptr<gpu_context> on sim::env, lazily constructed in build_env
// when any GPU entity is configured; it stays null in CPU-only runs.
//
// Threading: constructed and destroyed on the sim thread. The fault callbacks
// may fire on an arbitrary wgpu-internal thread at an arbitrary time, so they
// only flip an atomic flag and stash the message under a mutex — never call
// back into the wgpu API (the header forbids re-entrant calls from them).
// check_or_throw() surfaces a recorded fault on the sim thread at a phase
// boundary.
struct gpu_context {
    WGPUInstance instance = nullptr;
    WGPUAdapter  adapter  = nullptr;
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;
    WGPULimits   limits   = {};   // device limits, queried after device creation

    // Human-readable adapter identity (device name + backend), from
    // wgpuAdapterGetInfo. Useful for the smoke executable and diagnostics.
    std::string adapter_name;

    // Async fault surface. faulted_ is set true by either callback; the message
    // is stored under fault_msg_mutex_. Read/cleared by check_or_throw().
    std::atomic<bool> faulted_{false};
    std::mutex        fault_msg_mutex_;
    std::string       fault_message_;

    // Full bring-up. Selects the Vulkan backend (Metal on Apple), requests the
    // adapter's reported limits as the device's required limits (maximal
    // headroom, always satisfiable), and wires the device-lost +
    // uncaptured-error callbacks into the device descriptor. Throws
    // std::runtime_error carrying the verbatim wgpu diagnostic on any failure.
    gpu_context();

    // Releases queue -> device -> adapter -> instance, in reverse creation
    // order. Never throws.
    ~gpu_context();

    // If a fault was recorded by a callback, throw std::runtime_error with the
    // stored message and clear the flag. Called at phase boundaries (start of
    // env::process() / env::swap_io()) on the sim thread.
    void check_or_throw();

    // Block until all submitted GPU work has completed. Default build
    // (-DCOAST_GPU_FAST_WAIT, on) uses a non-sleeping OnSubmittedWorkDone +
    // ProcessEvents pump; -DCOAST_GPU_FAST_WAIT=OFF falls back to the blocking
    // wgpuDevicePoll(device, /*wait=*/true, nullptr). See the impl note and
    // benchmarks/m3_gpu_timing/findings_m3_followup.md.
    void wait_idle();

    gpu_context(const gpu_context&)            = delete;
    gpu_context& operator=(const gpu_context&) = delete;
    gpu_context(gpu_context&&)                 = delete;
    gpu_context& operator=(gpu_context&&)      = delete;
};

}  // namespace coast::gpu

#endif  // COAST_CORE_GPU_GPU_CONTEXT_H
