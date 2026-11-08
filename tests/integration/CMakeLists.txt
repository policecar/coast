include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/coast_add_integration_test.cmake)

# Helpers library, linked into every A-tier integration test.
add_library(coast_integration_helpers STATIC
    integration_helpers.h
    integration_helpers.cpp
)
target_include_directories(coast_integration_helpers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(coast_integration_helpers PUBLIC
    doctest::doctest
    coast_lib
    coast_test_support
    coast_compare_lib
)
target_compile_features(coast_integration_helpers PUBLIC cxx_std_23)

# A-tier tests.
coast_add_integration_test(
    NAME        mnist_seed0_baseline
    CONFIG      configs/mnist.json
    STEPS       100000
    SEED_OFFSET 0
)
coast_add_integration_test(
    NAME        mnist_seed1_baseline
    CONFIG      configs/mnist.json
    STEPS       100000
    SEED_OFFSET 1
)
coast_add_integration_test(
    NAME        mnist_seed2_baseline
    CONFIG      configs/mnist.json
    STEPS       100000
    SEED_OFFSET 2
)

# --- S12: ngm3 heterogeneous A-tier tests (one per io_buffer variant, D6) ----
# Shared 2-node mnist->ngm3 pipeline; configs differ only in the two entity
# device types. The CPU baseline is strict (deterministic); the three GPU
# topologies use aggregate compare (Shape-A statistical parity, MH3) since GPU
# parallel-reduction order is not bit-reproducible. Tolerances are pinned by
# the s12.tolerance-pin step and attached via TOLERANCES once measured.
coast_add_integration_test(
    NAME        mnist_ngm3_philox
    CONFIG      configs/mnist_ngm3_philox.json
    STEPS       10000
    SEED_OFFSET 0
)
coast_add_integration_test(
    NAME         mnist_ngm3_cpu_to_gpu
    CONFIG       configs/mnist_ngm3_cpu_to_gpu.json
    STEPS        10000
    SEED_OFFSET  0
    COMPARE_MODE aggregate
    GPU
)
coast_add_integration_test(
    NAME         mnist_ngm3_gpu_to_cpu
    CONFIG       configs/mnist_ngm3_gpu_to_cpu.json
    STEPS        10000
    SEED_OFFSET  0
    COMPARE_MODE aggregate
    GPU
)
coast_add_integration_test(
    NAME         mnist_ngm3_heterogeneous
    CONFIG       configs/mnist_ngm3_heterogeneous.json
    STEPS        10000
    SEED_OFFSET  0
    COMPARE_MODE aggregate
    GPU
)

# --- ngm2.5 CPU A-tier golden (review must-fix #2) --------------------------
# Full-pipeline regression guard for the ngm2.5 model on the mnist->ngm2.5
# pipeline. The CPU model is deterministic run-to-run (RNG parity + self-
# consistency exactness), so strict compare applies. The GPU config
# (mnist_ngm25_gpu.json) is intentionally not registered here: like the ngm3 GPU
# topologies it would need aggregate parity + a real adapter; deferred.
coast_add_integration_test(
    NAME        mnist_ngm25
    CONFIG      configs/mnist_ngm25.json
    STEPS       10000
    SEED_OFFSET 0
)

# s12.tolerance-pin: Shape-A aggregate noise-floor probe (standalone exe, no
# golden/ctest registration — a measurement tool feeding checkpoint gamma).
add_executable(ngm3_noise_floor ngm3_noise_floor/main.cpp)
target_link_libraries(ngm3_noise_floor PRIVATE coast_lib)
target_compile_features(ngm3_noise_floor PRIVATE cxx_std_23)

# Aggregate-parity probe for any neuron group (ngm2, ngm25, ngm3; CPU or GPU),
# standalone exe, no ctest registration: per-seed output and learned-state
# statistics for a two-sample comparison (SPEC.md M.0.6 b, M.7.20). See the
# file header for usage.
add_executable(ngm_parity_probe ngm_parity_probe/main.cpp)
target_link_libraries(ngm_parity_probe PRIVATE coast_lib)
target_compile_features(ngm_parity_probe PRIVATE cxx_std_23)

# Frozen (inference-only) inertness: self-comparing model-state equality
# (state(t) == state(t+N) minus rng streams), so no goldens/ctest golden dirs.
add_executable(integration_frozen_inert frozen_inert/test_frozen_inert.cpp)
target_link_libraries(integration_frozen_inert PRIVATE coast_integration_helpers)
target_compile_features(integration_frozen_inert PRIVATE cxx_std_23)
add_test(NAME integration.frozen_inert COMMAND integration_frozen_inert
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Seed composition (issue #167, core/seed.h): offsets compose modulo 2^32 and the
# reported effective seed is the seed the RNG uses, checked by comparing runs.
# In-process, self-comparing, no goldens.
add_executable(integration_seed_composition seed_composition/test_seed_composition.cpp)
target_link_libraries(integration_seed_composition PRIVATE coast_integration_helpers)
target_compile_features(integration_seed_composition PRIVATE cxx_std_23)
add_test(NAME integration.seed_composition COMMAND integration_seed_composition
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Cross-implementation equivalence (SPEC.md M.0.6): ngm2, ngm25 and ngm3-CPU
# driven from one injected initial state with randomness neutralised, compared
# step by step under a common logical key. In-process, dataset-free, no goldens.
add_executable(integration_impl_equivalence impl_equivalence/test_impl_equivalence.cpp)
target_link_libraries(integration_impl_equivalence PRIVATE doctest::doctest coast_lib)
target_compile_features(integration_impl_equivalence PRIVATE cxx_std_23)
add_test(NAME integration.impl_equivalence COMMAND integration_impl_equivalence
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# GPU variant (ngm3 GPU group): needs a real adapter, so it is registered only
# under the webgpu backend, carries the _gpu suffix and the "gpu" label
# (sandboxed runs exclude it via -E '_gpu' or -LE gpu).
if(COAST_GPU_BACKEND STREQUAL "webgpu")
    add_executable(integration_frozen_inert_gpu frozen_inert/test_frozen_inert_gpu.cpp)
    target_link_libraries(integration_frozen_inert_gpu PRIVATE coast_integration_helpers)
    target_compile_features(integration_frozen_inert_gpu PRIVATE cxx_std_23)
    add_test(NAME integration.frozen_inert_gpu COMMAND integration_frozen_inert_gpu
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(integration.frozen_inert_gpu PROPERTIES LABELS "gpu")

    # GPU variant of the equivalence test: the same source with COAST_EQUIV_GPU,
    # which adds ngm25-GPU and ngm3-GPU rigs to every case they can run.
    add_executable(integration_impl_equivalence_gpu impl_equivalence/test_impl_equivalence.cpp)
    target_link_libraries(integration_impl_equivalence_gpu PRIVATE doctest::doctest coast_lib)
    target_compile_features(integration_impl_equivalence_gpu PRIVATE cxx_std_23)
    target_compile_definitions(integration_impl_equivalence_gpu PRIVATE COAST_EQUIV_GPU=1)
    add_test(NAME integration.impl_equivalence_gpu COMMAND integration_impl_equivalence_gpu
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(integration.impl_equivalence_gpu PROPERTIES LABELS "gpu")
endif()

# After all integration tests are declared, build the update-goldens machinery.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/coast_update_goldens.cmake)
