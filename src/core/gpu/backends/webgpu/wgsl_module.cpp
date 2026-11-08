#include "wgsl_module.h"

#include "wgpu_util.h"

#include <stdexcept>
#include <string>

namespace coast::gpu {

namespace {

// Result holder for the async wgpuDevicePopErrorScope callback.
//
// wgpu-native does NOT implement wgpuShaderModuleGetCompilationInfo (it panics:
// src/unimplemented.rs "not implemented"). Validation errors — including the
// full naga WGSL diagnostic with line/column and a source snippet — are instead
// delivered through a validation error scope wrapped around the create call. The
// scope also keeps the error from reaching gpu_context's global uncaptured-error
// surface, so a compile failure here does not later trip check_or_throw().
struct pop_result {
    bool          done = false;
    WGPUErrorType type = WGPUErrorType_NoError;
    std::string   message;
};

void on_pop(WGPUPopErrorScopeStatus status, WGPUErrorType type,
            WGPUStringView message, void* ud1, void* /*ud2*/) {
    auto* r   = static_cast<pop_result*>(ud1);
    r->type   = (status == WGPUPopErrorScopeStatus_Success) ? type : WGPUErrorType_NoError;
    r->message = sv_to_string(message);
    r->done   = true;
}

}  // namespace

WGPUShaderModule load_wgsl_module(WGPUInstance instance, WGPUDevice device,
                                  std::string_view source, std::string_view label) {
    // Build the WGSL-source-chained shader module descriptor.
    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code        = sv_from(source);

    WGPUShaderModuleDescriptor desc = {};
    desc.nextInChain = &wgsl.chain;
    desc.label       = sv_from(label);

    // Wrap creation in a validation error scope to capture the compile diagnostic.
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device, &desc);

    pop_result pr;
    WGPUPopErrorScopeCallbackInfo cb = {};
    cb.mode      = WGPUCallbackMode_AllowProcessEvents;
    cb.callback  = on_pop;
    cb.userdata1 = &pr;
    wgpuDevicePopErrorScope(device, cb);
    pump_until(instance, pr.done);

    // A captured validation error is fatal: surface it verbatim (the naga
    // diagnostic carries line/column/expected). Release the (possibly error-)
    // module first.
    if (pr.type != WGPUErrorType_NoError) {
        if (mod != nullptr) wgpuShaderModuleRelease(mod);
        throw std::runtime_error("WGSL compile error [" + std::string(label) + "]:\n" +
                                 pr.message);
    }
    if (mod == nullptr) {
        throw std::runtime_error(
            "load_wgsl_module: wgpuDeviceCreateShaderModule returned null for '" +
            std::string(label) + "' with no reported error");
    }

    return mod;
}

}  // namespace coast::gpu
