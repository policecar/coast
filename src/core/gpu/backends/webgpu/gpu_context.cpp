#include "gpu_context.h"

#include "wgpu_util.h"

#include <webgpu/wgpu.h>   // wgpu-native extensions: wgpuDevicePoll

#include <stdexcept>
#include <string>

namespace coast::gpu {

namespace {

// Backend is fixed per platform: Strix Halo / Linux -> Vulkan, Apple -> Metal.
#ifdef __APPLE__
constexpr WGPUBackendType kBackend = WGPUBackendType_Metal;
#else
constexpr WGPUBackendType kBackend = WGPUBackendType_Vulkan;
#endif

// ---- async request result holders (filled by the callbacks below) ----------

struct adapter_result {
    bool                     done   = false;
    WGPURequestAdapterStatus status = WGPURequestAdapterStatus_Error;
    WGPUAdapter              adapter = nullptr;
    std::string              message;
};

struct device_result {
    bool                    done   = false;
    WGPURequestDeviceStatus status = WGPURequestDeviceStatus_Error;
    WGPUDevice              device = nullptr;
    std::string             message;
};

void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                WGPUStringView message, void* ud1, void* /*ud2*/) {
    auto* r    = static_cast<adapter_result*>(ud1);
    r->status  = status;
    r->adapter = adapter;
    r->message = sv_to_string(message);
    r->done    = true;
}

void on_device(WGPURequestDeviceStatus status, WGPUDevice device,
               WGPUStringView message, void* ud1, void* /*ud2*/) {
    auto* r    = static_cast<device_result*>(ud1);
    r->status  = status;
    r->device  = device;
    r->message = sv_to_string(message);
    r->done    = true;
}

}  // namespace

// Fault callbacks. They may fire on any thread at any time, so they only record
// state — no calls back into the wgpu API. They are file-scope (not in the
// anonymous namespace) only because they reference gpu_context's members; kept
// as free functions to match the C callback ABI.
namespace {

void on_uncaptured_error(WGPUDevice const* /*device*/, WGPUErrorType /*type*/,
                         WGPUStringView message, void* ud1, void* /*ud2*/) {
    auto* ctx = static_cast<gpu_context*>(ud1);
    if (ctx == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(ctx->fault_msg_mutex_);
        ctx->fault_message_ = "uncaptured error: " + sv_to_string(message);
    }
    ctx->faulted_.store(true, std::memory_order_release);
}

void on_device_lost(WGPUDevice const* /*device*/, WGPUDeviceLostReason reason,
                    WGPUStringView message, void* ud1, void* /*ud2*/) {
    // Destroyed / CallbackCancelled are the normal teardown path — not faults.
    if (reason == WGPUDeviceLostReason_Destroyed ||
        reason == WGPUDeviceLostReason_CallbackCancelled) {
        return;
    }
    auto* ctx = static_cast<gpu_context*>(ud1);
    if (ctx == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(ctx->fault_msg_mutex_);
        ctx->fault_message_ = "device lost: " + sv_to_string(message);
    }
    ctx->faulted_.store(true, std::memory_order_release);
}

}  // namespace

gpu_context::gpu_context() {
    // 1. Instance.
    instance = wgpuCreateInstance(nullptr);
    if (instance == nullptr) {
        throw std::runtime_error("gpu_context: wgpuCreateInstance returned null");
    }

    // 2. Adapter (explicit backend, high-performance preference).
    WGPURequestAdapterOptions adapter_opts = {};
    adapter_opts.backendType     = kBackend;
    adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;

    adapter_result ar;
    WGPURequestAdapterCallbackInfo acb = {};
    acb.mode      = WGPUCallbackMode_AllowProcessEvents;
    acb.callback  = on_adapter;
    acb.userdata1 = &ar;
    wgpuInstanceRequestAdapter(instance, &adapter_opts, acb);
    pump_until(instance, ar.done);
    if (ar.status != WGPURequestAdapterStatus_Success || ar.adapter == nullptr) {
        throw std::runtime_error("gpu_context: adapter request failed: " + ar.message);
    }
    adapter = ar.adapter;

    // 3. Adapter identity for diagnostics.
    WGPUAdapterInfo info = {};
    if (wgpuAdapterGetInfo(adapter, &info) == WGPUStatus_Success) {
        std::string dev = sv_to_string(info.device);
        adapter_name = dev.empty() ? std::string("unknown") : dev;
        if (kBackend == WGPUBackendType_Vulkan) adapter_name += " [Vulkan]";
        else if (kBackend == WGPUBackendType_Metal) adapter_name += " [Metal]";
        wgpuAdapterInfoFreeMembers(info);
    }

    // 4. Required limits = the adapter's reported limits. This is "raised"
    //    (well above the 128 MB / 8-storage-buffer WebGPU defaults) yet always
    //    satisfiable, since we never ask for more than the adapter advertises.
    WGPULimits required_limits = {};
    if (wgpuAdapterGetLimits(adapter, &required_limits) != WGPUStatus_Success) {
        throw std::runtime_error("gpu_context: wgpuAdapterGetLimits failed");
    }
    required_limits.nextInChain = nullptr;

    // 5. Device, with both fault callbacks wired into the descriptor (v29 has no
    //    post-hoc setter — this is the canonical place to install them).
    WGPUDeviceDescriptor dev_desc = {};
    dev_desc.label          = sv_from("coast_device");
    dev_desc.requiredLimits = &required_limits;
    dev_desc.deviceLostCallbackInfo.mode      = WGPUCallbackMode_AllowSpontaneous;
    dev_desc.deviceLostCallbackInfo.callback  = on_device_lost;
    dev_desc.deviceLostCallbackInfo.userdata1 = this;
    dev_desc.uncapturedErrorCallbackInfo.callback  = on_uncaptured_error;
    dev_desc.uncapturedErrorCallbackInfo.userdata1 = this;

    device_result dr;
    WGPURequestDeviceCallbackInfo dcb = {};
    dcb.mode      = WGPUCallbackMode_AllowProcessEvents;
    dcb.callback  = on_device;
    dcb.userdata1 = &dr;
    wgpuAdapterRequestDevice(adapter, &dev_desc, dcb);
    pump_until(instance, dr.done);
    if (dr.status != WGPURequestDeviceStatus_Success || dr.device == nullptr) {
        throw std::runtime_error("gpu_context: device request failed: " + dr.message);
    }
    device = dr.device;

    // 6. Queue + queried device limits.
    queue = wgpuDeviceGetQueue(device);
    if (queue == nullptr) {
        throw std::runtime_error("gpu_context: wgpuDeviceGetQueue returned null");
    }
    if (wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success) {
        throw std::runtime_error("gpu_context: wgpuDeviceGetLimits failed");
    }
}

gpu_context::~gpu_context() {
    if (queue != nullptr)    wgpuQueueRelease(queue);
    if (device != nullptr)   wgpuDeviceRelease(device);
    if (adapter != nullptr)  wgpuAdapterRelease(adapter);
    if (instance != nullptr) wgpuInstanceRelease(instance);
}

void gpu_context::check_or_throw() {
    if (!faulted_.load(std::memory_order_acquire)) return;
    std::string msg;
    {
        std::lock_guard<std::mutex> lk(fault_msg_mutex_);
        msg = fault_message_;
    }
    faulted_.store(false, std::memory_order_release);
    throw std::runtime_error("gpu_context: " + msg);
}

#ifdef COAST_GPU_FAST_WAIT
namespace {
// One-shot completion flag for the OnSubmittedWorkDone fast-wait path.
struct work_done_t { bool done = false; };
void on_work_done(WGPUQueueWorkDoneStatus /*status*/, WGPUStringView /*msg*/,
                  void* ud1, void* /*ud2*/) {
    static_cast<work_done_t*>(ud1)->done = true;
}
}  // namespace
#endif

void gpu_context::wait_idle() {
#ifdef COAST_GPU_FAST_WAIT
    // Fast wait (compile-time default, -DCOAST_GPU_FAST_WAIT). At the pinned
    // wgpu-native v29 (wgpu-hal 29.0.1) the Metal host wait behind
    // wgpuDevicePoll(wait=true) is a thread::sleep(1ms) poll loop over
    // MTLCommandBuffer.status() — the event-driven Condvar wait exists only on wgpu
    // trunk, after v29 (benchmarks/m3_gpu_timing/findings_m3_followup.md). Drive
    // completion through OnSubmittedWorkDone + a non-sleeping ProcessEvents pump
    // instead: ~86x faster per wait on M3/Metal (~3.2-3.7x SPS in-sim), numerically
    // inert. On RADV/Vulkan devicePoll(wait) is already cheap (~22-127us), so this
    // path is ~neutral there. Retire and bump the wgpu-native pin once a release
    // past v29 ships the upstream wait.
    work_done_t wd;
    WGPUQueueWorkDoneCallbackInfo cb = {};
    cb.mode      = WGPUCallbackMode_AllowProcessEvents;
    cb.callback  = on_work_done;
    cb.userdata1 = &wd;
    wgpuQueueOnSubmittedWorkDone(queue, cb);
    pump_until(instance, wd.done);
#else
    wgpuDevicePoll(device, /*wait=*/true, nullptr);
#endif
}

}  // namespace coast::gpu
