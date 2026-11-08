#include "core/gpu/shared_prng.h"

namespace coast::gpu {

namespace {

// Embed the WGSL snippet at build time. #embed is a C23 preprocessor directive
// supported in GCC 15's C++ modes (IDD design decision P6-6) — within the
// project's GCC-15 baseline. The path resolves relative to this source file
// (shared_prng.wgsl sits alongside this .cpp), exactly like #include "...".
// If a collaborator's toolchain lacks #embed, the documented fallback is a
// CMake codegen step that turns the .wgsl into a byte array.
//
// unsigned char so any byte (e.g. UTF-8 in comments) embeds without a narrowing
// conversion; length is sizeof, so no trailing-NUL trick is needed.
constexpr unsigned char kSharedPrngWgsl[] = {
#embed "shared_prng.wgsl"
};

}  // namespace

std::string_view shared_prng_wgsl() {
    return std::string_view(reinterpret_cast<const char*>(kSharedPrngWgsl),
                            sizeof(kSharedPrngWgsl));
}

}  // namespace coast::gpu
