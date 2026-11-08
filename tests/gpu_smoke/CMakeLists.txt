# tests/gpu_smoke/CMakeLists.txt — register the standalone GPU validator
# harnesses as ctest cases.
#
# These executables are defined in the root CMakeLists for the webgpu backend.
# Each one hand-builds a minimal GPU pipeline, asserts its own invariants
# (printing explicit "PASS" lines), and returns non-zero on failure — so they
# are meaningful pass/fail tests, not just demos. Before this they ran only by
# hand; here they become CI smoke coverage of the whole GPU path wherever a
# device is present.
#
# Excluded on purpose: coast_gpu_spike / coast_mnist_io_gpu_spike (performance
# measurement harnesses — they print numbers, not pass/fail).
#
# All cases are labeled "gpu":
#   - run only these         : ctest -L gpu
#   - exclude (CPU/sanitizer): ctest -LE gpu
# For any backend other than webgpu the harness targets do not exist, so this
# whole file is skipped and no GPU smoke tests are registered.

if(NOT COAST_GPU_BACKEND STREQUAL "webgpu")
    return()
endif()

# Each harness loads ./data/mnist for the dataset-backed cases, so run from the
# repo root. 180 s ceiling: the ngm3 probe drives ~5200 steps (~seconds on a
# real GPU) but headroom covers a slow/contended adapter.
function(coast_add_gpu_smoke name target)
    add_test(NAME ${name} COMMAND ${target} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(${name} PROPERTIES LABELS "gpu" TIMEOUT 180)
endfunction()

coast_add_gpu_smoke(gpu.smoke.context        coast_gpu_smoke)          # device init + limits
coast_add_gpu_smoke(gpu.smoke.wgsl_toolchain coast_gpu_min_kernel)     # WGSL compile + bad-shader error path
coast_add_gpu_smoke(gpu.smoke.dispatcher     coast_dispatcher_probe)   # device-driven step dispatch
coast_add_gpu_smoke(gpu.smoke.mnist_io       coast_mnist_io_gpu_probe) # per-pixel correctness + schedule
coast_add_gpu_smoke(gpu.smoke.ngm3           coast_ngm3_gpu_probe)     # mnist_io->ngm3 sanity + learning/branch

# gpu.smoke.ngm3 is TBB-heavy on the host side: measured 22 s with 2x nproc busy
# loops and 40-48 s with 4x, against 180 s — a ceiling a heavier load would hit.
# A TIMEOUT must hold under heavy load (CLAUDE.md "Testing"), so this one gets
# ctest's default 1500 s instead (>= 30x the 4x-load time).
set_tests_properties(gpu.smoke.ngm3 PROPERTIES TIMEOUT 1500)
