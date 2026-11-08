#ifndef COAST_CORE_GPU_WGPU_UTIL_H
#define COAST_CORE_GPU_WGPU_UTIL_H

#include <webgpu/webgpu.h>

#include <stdexcept>
#include <string>
#include <string_view>

// Small inline helpers shared by the WebGPU backend translation units
// (gpu_context, wgsl_module, backend_webgpu, the diagnostic executables). Pulls
// the full <webgpu/webgpu.h>, so it is a backend-private header — it lives under
// backends/webgpu/ and is never included from agnostic core code.
namespace coast::gpu {

// wgpu strings are (data, length) views and are NOT guaranteed null-terminated;
// length may be the WGPU_STRLEN sentinel meaning "data is null-terminated".
inline std::string sv_to_string(WGPUStringView v) {
    if (v.data == nullptr) return {};
    if (v.length == WGPU_STRLEN) return std::string(v.data);
    return std::string(v.data, v.length);
}

// View a std::string_view (which includes string literals) as a WGPUStringView.
// The viewed storage must outlive every wgpu call that reads the view.
inline WGPUStringView sv_from(std::string_view s) {
    return WGPUStringView{s.data(), s.size()};
}

// Drive the wgpu event loop until `done` flips true. wgpu-native typically fires
// AllowProcessEvents callbacks synchronously inside the originating request, so
// the loop body often runs zero times; the iteration cap is a hang guard, never
// an expected path.
inline void pump_until(WGPUInstance instance, const bool& done) {
    constexpr int kMaxSpins = 1'000'000;
    int spins = 0;
    while (!done) {
        wgpuInstanceProcessEvents(instance);
        if (++spins > kMaxSpins) {
            throw std::runtime_error(
                "coast::gpu: timed out waiting for an async wgpu request to complete");
        }
    }
}

}  // namespace coast::gpu

#endif  // COAST_CORE_GPU_WGPU_UTIL_H
