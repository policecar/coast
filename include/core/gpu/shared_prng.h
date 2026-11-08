#ifndef COAST_CORE_GPU_SHARED_PRNG_H
#define COAST_CORE_GPU_SHARED_PRNG_H

#include <string_view>

namespace coast::gpu {

// The PCG-1D32 hash-PRNG WGSL snippet (the contents of shared_prng.wgsl),
// embedded into the binary at build time. Prepend it to an entity's WGSL source
// before calling load_wgsl_module() so the kernel can call pcg_hash() /
// pcg_uniform(). The snippet defines functions only (no bindings, no entry
// points), so concatenation never clashes with the entity's own declarations.
//
//   auto module = load_wgsl_module(inst, dev,
//       std::string(shared_prng_wgsl()) + entity_src, "my_entity");
std::string_view shared_prng_wgsl();

}  // namespace coast::gpu

#endif  // COAST_CORE_GPU_SHARED_PRNG_H
