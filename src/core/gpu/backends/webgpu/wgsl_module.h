#ifndef COAST_CORE_GPU_WGSL_MODULE_H
#define COAST_CORE_GPU_WGSL_MODULE_H

#include <webgpu/webgpu.h>

#include <string_view>

namespace coast::gpu {

// Compile a WGSL source string into a WGPUShaderModule on `device`. `instance`
// is needed to pump the async compilation-info callback to completion; `label`
// names the module in wgpu validation output and in our diagnostics.
//
// Error fidelity is the deliverable. After creating the module we synchronously
// fetch wgpuShaderModuleGetCompilationInfo and inspect every message. If ANY
// message is Error severity, the module is released and a std::runtime_error is
// thrown whose what() reproduces the full wgpu diagnostic verbatim — one line
// per message, formatted "<label>:<line>:<col>: <severity>: <text>". This is the
// day-to-day kernel-author iteration loop, so the message must carry the
// line/column wgpu reports. Warning/info messages are non-fatal and dropped.
//
// Returns a module the caller owns and must release with wgpuShaderModuleRelease.
// Throws std::runtime_error on a null module handle or any compile error.
WGPUShaderModule load_wgsl_module(WGPUInstance instance, WGPUDevice device,
                                  std::string_view source,
                                  std::string_view label = "wgsl");

}  // namespace coast::gpu

#endif  // COAST_CORE_GPU_WGSL_MODULE_H
