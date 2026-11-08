+++
title = "Reference"
weight = 50
+++

# File Reference

Quick lookup for every header and source file in the project. Organized by module.

## core/

| File | Contains |
|------|----------|
| `include/core/entity_element.h` | `entity_element` base class — hierarchy traversal, probe fast-path, active_probe_count atomic |
| `include/core/entity_element_macros.h` | `ELEMENT_TYPE`, `ELEMENT_CHILDREN_*`, `ELEMENT_PROBES`, `PROBE` macros |
| `include/core/element_type.h` | `element_type` — type-safe identifier wrapping `type_info` + display name |
| `include/core/structure_registry.h` | `structure_registry`, `structure_entry` — O(1) element-to-path lookup, tree traversal |
| `include/core/seed.h` | `seed_t`, `seed_offset_t` — strong 32-bit seed types; modulo-2³² composition; input rule [0, 2³² − 1] for JSON and `--seed-offset` (nlohmann hooks) |
| `include/core/probe_name_registry.h` | `probe_name_registry` — global static `type_index → set<string>` for probe name discovery |
| `include/core/vis_hints.h` | `vis_hints_t`, `ngm_flat_vis_hint_t`, `ngm_tree_vis_hint_t`, dataset hints, buffer_vis hints. A plain config POD despite the name — parsed headlessly and written into the HDF5 buffer-log topology, so it lives in `core/`, not `gui_vis/` |
| `include/core/demand_published.h` | `demand_published<T>` — single-slot demand-driven publisher (producer recomputes only after the consumer takes the last value). Plain atomics/mutex, GUI-free |
| `include/core/rng/fast_rng.h` | Type aliases: `fast_rng_t` = `rng_t<mt19937_simd>`, `standard_rng_t` |
| `include/core/rng/mt19937_simd.h` | `mt19937_simd` — SIMD Mersenne Twister (AVX-512/AVX2/NEON/scalar), 624-element state, bulk temper |
| `include/core/rng/rng_wrapper.h` | `rng_t<Generator>` — thin generic wrapper, conforms to UniformRandomBitGenerator |
| `include/core/run_id.h` | `run_id` of a run: `new_run_id`, `resolve_run_id` ("auto" → fresh id), used by coast and coastd |
| `include/core/app_config.h` | `coast::paths::app_config` — application-level config (where outputs go, where the app finds its assets), distinct from the per-experiment config; every field defaulted, so the file is optional. `load_app_config()`, `write_app_config_file()`, `resolve_snapshot_output_dir()` |
| `include/core/path_resolver.h` | `coast::paths` — `expand_config_base()` (`<config_dir>` expands only as a whole path component), `resolve_config_output()`, `leaf_escapes_config_base()` |
| `include/core/require.h` | `COAST_REQUIRE(cond, kind)` — always-compiled hot-path contract check; returns `std::unexpected` with a `step_error` instead of compiling out like `assert()` |
| `include/core/step_error.h` | `core::step_error` — hot-path error payload of the `std::expected` error channel |
| `include/core/local_time.h` | `core::format_local_time()` — thread-safe (`localtime_r`) `strftime` formatting; `k_ctime_format`, `k_file_stamp_format`; `snapshot_path()`, the one snapshot file name (headless checkpoints, GUI auto-save) |
| `include/core/rng/mt11213b_simd.h` | `mt11213b_simd` — SIMD MT11213b (351-word state), drop-in alternative to `mt19937_simd` via `fast_rng.h` |
| `include/core/rng/philox_4x32.h` | `philox_4x32` — counter-based Philox-4x32-10 PRNG (~44-byte state, no twist) |
| `include/core/rng/philox_4x32_avx512.h` | `philox_4x32_avx512` — SIMD-buffered Philox-4x32-10, 16 instances in parallel under AVX-512 |
| `include/core/rng/xor_table_prng.h` | `xor_table_prng` — five prime-length (73…97) read-only tables combined by XOR |
| `src/core/entity_element.cpp` | `probe_slow_path()` — hash map lookup on the sim thread |
| `src/core/structure_registry.cpp` | `build()`, `build_recursive()`, path computation, element_map population |
| `src/core/probe_name_registry.cpp` | Mutex-guarded register/get operations on the global registry |
| `src/core/app_config.cpp` | `load_app_config()` / `write_app_config_file()`; a `config_base` misuse is reported once, at load |
| `src/core/path_resolver.cpp` | Token replacement behind `expand_config_base()` / `resolve_config_output()` |
| `src/core/run_id.cpp` | `new_run_id()` (32 hex digits from `std::random_device`), `resolve_run_id()` |

## core/gpu/  *(COAST_GPU_BACKEND: none|webgpu|null)*

| File | Contains |
|------|----------|
| `include/core/gpu/gpu_device.h` | Backend-agnostic `gpu_device` facade — opaque `buffer_id`/`kernel_id`/`plan_id` handles + a `step_plan` POD; drives `begin_step`/`enqueue`/`submit_step`/`rendezvous_step`, `upload`, `advance_generation`; no WGPU types |
| `src/core/gpu/backends/webgpu/gpu_context.h` | (backend-private) `gpu_context` — instance/adapter/device/queue + limits + atomic fault state; `check_or_throw()`, `wait_idle()`; non-copyable; constructed and owned by the backend inside `make_gpu_device()` |
| `src/core/gpu/backends/webgpu/wgpu_util.h` | (backend-private) `pump_until()` (drive async wgpu callbacks to completion) + small create helpers |
| `src/core/gpu/backends/webgpu/wgsl_module.h` | (backend-private) `load_wgsl_module()` — WGSL string → `WGPUShaderModule`; throws verbatim naga diagnostic on compile error |
| `include/core/gpu/shared_prng.h` | `shared_prng_wgsl()` — `string_view` over the embedded PRNG snippet |
| `src/core/gpu/backends/webgpu/backend_webgpu.cpp` | One of the three backend TUs that see `<webgpu/webgpu.h>` (the build gives the SDK include root to these alone); implements the `gpu_device` facade and owns the `gpu_context` — all encoder/submit machinery lives here behind the opaque handles |
| `src/core/gpu/backends/webgpu/gpu_context.cpp` | wgpu bring-up (Vulkan/Metal backend select), fault callbacks, teardown |
| `src/core/gpu/backends/webgpu/wgsl_module.cpp` | Validation-error-scoped module creation |
| `src/core/gpu/shared_prng.{wgsl,cpp}` | PCG-1D32 stateless hash PRNG; `#embed` of the `.wgsl` |
| `src/core/gpu/backends/null/backend_null.cpp` | The `null` backend: a host-only, dependency-free `gpu_device` (no GPU SDK) |
| `src/core/gpu/smoke_main.cpp` | `coast_gpu_smoke` executable — wgpu bootstrap smoke test (`gpu.smoke.context`) |
| `src/core/gpu/min_kernel_main.cpp` | `coast_gpu_min_kernel` executable — end-to-end compute-kernel verification (`gpu.smoke.wgsl_toolchain`) |
| `src/core/gpu/spike_main.cpp` | `coast_gpu_spike` executable — framework spike measurements (unified-memory write path, submission cost) |

## sim_core/

| File | Contains |
|------|----------|
| `include/sim_core/sim_env.h` | `sim::env` — entity container, IO buffer container, hook system (4 hook points), `process()`, `swap_io()`. Also `entity_vec`, `typed_entity_vec<T>` type erasure |
| `include/sim_core/io_entity.h` | `io_entity` interface — `process()`, `get_outp_id/size()`, `get_inp_ids()`, `set_output_buffer/set_input_buffer()`, `save/load_state()`. Types: `output_id_t`, `output_channel_t` |
| `include/sim_core/io_buffer.h` | Abstract `io_buffer` base (virtual swap/size/cur_*_buffer, state-IO `snapshot`/`restore` read-page pair; backend-agnostic GPU identity `gpu_register`/`gpu_page_id`/`gpu_topology`/`gpu_backed` plus `read_back_sync`/`device_to_cpu_readback` — now carries opaque `buffer_id`/`topology` from `core/gpu/gpu_device.h`, no WGPU types) + concrete `io_buffer_cpu` |
| `include/sim_core/io_buffer_gpu.h` *(WGPU)* | `io_buffer_cpu_to_gpu`, `io_buffer_gpu_to_cpu`, `io_buffer_gpu` — the three cross-device variants |
| `include/sim_core/gpu_io_entity.h` | `gpu_io_entity : io_entity` — fully agnostic (includes `core/gpu/gpu_device.h`, no WGPU types); builds a static `step_plan` once via `build_step_plan(gpu_device&)`, exposes `plan()` (plan_id), `graph_epoch()` (constant 0 this cycle), `read_probes()` |
| `include/sim_core/sim_thread_controller.h` | `sim_thread_controller` — `jthread`, `sim_mode` enum (paused/throttled/unlimited), atomics for mode/sps/step_count, `do_step()`, `thread_main()`, `run_quiesced()` (blocking task on the sim thread at the post-swap boundary) |
| `include/sim_core/sim_instance.h` | `sim_instance` — RAII container wrapping `json_loader`, `sim::env`, `sim_builder`, `sim_thread_controller`, `logging_manager`. Constructed from a config path. (Moved here from `gui_core/` during the testing-infrastructure cycle so unit tests can build envs without pulling in GUI deps.) |
| `src/sim_core/sim_env.cpp` | `emplace_back()`, `iterate_entities()`, `init_io_buffers()` (variant topology pre-pass + wiring), `process()` (agnostic `gpu_device` facade: begin_step → per-entity enqueue(plan) → submit_step (no wait) → dispatch_cpu_entities (CPU∥GPU overlap) → rendezvous_step (the one mandatory per-step coherence wait) → read_probes) / `swap_io()` (gpu_device::upload cpu_to_gpu → advance_generation ping-pong flip → io_buffer::device_to_cpu_readback gpu_to_cpu → swap_buffer), `process_pending_hooks()` |
| `src/sim_core/io_buffer.cpp` | `io_buffer_cpu` — pure index-flip `swap_buffer()`, CPU span accessors |
| `src/sim_core/io_buffer_gpu.cpp` *(WGPU)* | The three GPU `io_buffer` variants (cpu_to_gpu / gpu_to_cpu / gpu) selected by topology, now backend-agnostic — their physical GPU pages live in the WebGPU backend, resolved via opaque `buffer_id`; override `device_to_cpu_readback`/`read_back_sync`/`snapshot`/`restore` as needed |
| `src/sim_core/sim_thread_controller.cpp` | `thread_main()` — main loop with mode switch, SPS measurement (EMA/500ms), `do_step()`, `drain_quiesced()` at the quiescent boundary |
| `src/sim_core/sim_instance.cpp` | Simulation construction, move semantics |
| `include/sim_core/run_manifest.h` | `run_manifest_inputs_t`, `exit_reason_t`, `savestate_output_t`, `node_seed_t`, `write_run_manifest()` — the `coast_runtime.json` writer (schema_version "1") |
| `include/sim_core/coast_build_info.h.in` | `configure_file()` template for the generated `coast_build_info.h`: git SHA + dirty flag + build type + compiler + target CPU, captured at configure time |
| `src/sim_core/run_manifest.cpp` | Manifest serialization (`nlohmann::ordered_json`), the active `datasets` and `retina` mirrors, hostname |
| `src/sim_core/dispatcher_probe_main.cpp` | `coast_dispatcher_probe` executable — isolation probe for the env two-phase encoder dispatcher (`gpu.smoke.dispatcher`) |

## hd_ngm2/

| File | Contains |
|------|----------|
| `include/hd_ngm2/hd_ngm2_neuron_group.h` | `neuron_group_t` — implements `io_entity`, holds `vector<neuron_t>`, `process()` with competitive learning |
| `include/hd_ngm2/hd_ngm2_neuron.h` | `neuron_t` — `entity_element`, holds `vector<dendrite_t>`, `get_response()`, `adapt()` (gated by the group's tracked learning window) |
| `include/hd_ngm2/hd_ngm2_dendrite.h` | `dendrite_t` — `entity_element`, `synapses_t` SOA, segment tree, `get_response()`, `adapt_synapses()`, `adapt_branches()` |
| `include/hd_ngm2/hd_ngm2_cfg.h` | Config structs: `dendrite_params_t`, `neuron_params_t`, `neuron_group_params_t` (incl. `frozen`, set from the run's), `cfg_store` (carries the run's `frozen`; `group_or_default` hands it out and refuses a per-group value), `dendrite_type_t` enum, `learning_window_t` (type of the tracked windows) |
| `include/hd_ngm2/hd_ngm2_tools.h` | Utilities: `hoyer_noise()`, `noise_attenuation()`, `sigmoid()`, `softmax()`, `normalize()` |
| `src/hd_ngm2/hd_ngm2_neuron_group.cpp` | `process()` implementation — response, inhibition, stochastic winner, primary + common adaptation |
| `src/hd_ngm2/hd_ngm2_neuron.cpp` | `get_response()` — apical×proximal+noise; `adapt()` — learning window filter (window passed in by the group), delegate to dendrites |
| `src/hd_ngm2/hd_ngm2_dendrite.cpp` | `get_response()` — permanence gating, tree propagation, noise-gate attenuation (1 − Hoyer, constexpr windows); `adapt_synapses()` — Hebbian update; `adapt_branches()` — mismatch-driven tree growth |
| `include/hd_ngm2/hd_ngm2_tools.h` | `input_stats_t` + `compute_input_stats()` (Refactor A; shared with ngm3), `channel_constants()` (the per-channel step constants ngm3 and ngm25 expand per input position), `hoyer_noise`, the noise-gate windows + `noise_attenuation`, `sigmoid`, `softmax`, `normalize` |
| `include/hd_ngm2/hd_ngm2.h` | Umbrella include: tools, dendrite, neuron, neuron group, cfg |

## hd_ngm3/

| File | Contains |
|------|----------|
| `include/hd_ngm3/hd_ngm3_neuron_group.h` | `neuron_group_t` — CPU flat-SoA model; per-bank permanences/segment_ids/adapt_history/mismatch/rep_act vectors, per-neuron scratch, `bank_inputs_t` routing, per-neuron `fast_rng_t` |
| `include/hd_ngm3/hd_ngm3_cfg.h` | ngm3 params (incl. `frozen`); `bank_t` / `input_route_t` are `ngm/bank.h`'s |
| `include/hd_ngm3/hd_ngm3_host.h` | Host code the CPU and GPU group share: `validate_params()` (GPU limits as `group_caps_t`), `bank_inputs_t` (a bank's routed channels: `width()`, `bind()`, `layout()`), `route_inputs()`; the checkpoint parts of the shared schema: `write_checkpoint_shape()` / `check_checkpoint_shape()`, `stage_bank()` (read + `check_bank_layout`), `read_neuron_mx_act()`, `corrupt_checkpoint()` |
| `include/hd_ngm3/hd_ngm3_cold_init.h` | `cold_init()` — the cold RNG contract shared by the CPU and GPU group: master stream from the seed, per-neuron stream seeds, then the apical and proximal rep-0 Poisson permanences, in that order |
| `include/hd_ngm3/hd_ngm3_layout.h` | `check_bank_layout` — the value law (permanence in [0,1], finite adapt history ≥ 0, finite mismatch) and the replica layout laws (range, run alignment, full runs, bitwise replica coherence) checked on a bank's dense arrays |
| `include/hd_ngm3/hd_ngm3_gpu_neuron_group.h` *(WGPU)* | `neuron_group_gpu_t : gpu_io_entity` — GPU buffer inventory, vis mirrors, readback helpers |
| `src/hd_ngm3/hd_ngm3_neuron_group.cpp` | Full CPU algorithm: gather, stats, response (TBB), combine, inhibition, winner, adapt (segment-tree fold), branch |
| `src/hd_ngm3/hd_ngm3_cold_init.cpp` | `cold_init()`; the CPU group's `init_storage` and the GPU group's `build_step_plan` both start from it, so their initial permanences are identical by construction |
| `src/hd_ngm3/hd_ngm3_layout.cpp` | The layout check, run by the CPU and GPU `load_state` before anything is assigned or uploaded |
| `src/hd_ngm3/hd_ngm3_gpu_neuron_group.cpp` | GPU host orchestration: uniforms, `record()` kernel sequence, `read_probes()`, `refresh_vis_mirror()`, readback |
| `src/hd_ngm3/hd_ngm3_gpu.wgsl` | Compute kernels: stats, fused response+combine, inhibition, winner, adapt, branch |
| `src/hd_ngm3/hd_ngm3_host.cpp` | The shared host code: `validate_params()`, `bank_inputs_t`, `route_inputs()`, checkpoint shape write/check, `stage_bank()` |
| `src/hd_ngm3/hd_ngm3_gpu_probe_main.cpp` | `coast_ngm3_gpu_probe` executable — isolation probe for the ngm3 GPU group: `mnist_io_gpu → ngm3_gpu → cpu_sink` (`gpu.smoke.ngm3`) |

## hd_ngm25/

| File | Contains |
|------|----------|
| `include/hd_ngm25/hd_ngm25_cfg.h` | ngm2.5 params (flat schema), `penalty_policy_t`, `config_record()`; `bank_t` / `input_route_t` are `ngm/bank.h`'s |
| `include/hd_ngm25/hd_ngm25_arena.h` | `synapse_arena_t` (per-group CSR synapse arena, SoA), `group_shape_t`, `check_arena_consistency()` (invariants 1–5), `build_arena()` (D5 cold init), `repack_with_splits()` |
| `include/hd_ngm25/hd_ngm25_prng.h` | Stateless `pcg_hash` / `syn_key` draws and the penalty-policy seam |
| `include/hd_ngm25/hd_ngm25_response.h`, `_adapt.h`, `_branch.h` | The step kernels as `ngm25::detail` free functions (map R1–R3, R5–R7, R8) |
| `include/hd_ngm25/hd_ngm25_host.h` | Host code the CPU and GPU group share: `validate_params()` (GPU limits as `group_caps_t`), `route_inputs()`, `dense_mirror_t` (the ngm3-shaped vis mirror), and the checkpoint schema — `write_checkpoint()` and `stage_checkpoint()`, which reads and checks a whole checkpoint without touching the group |
| `include/hd_ngm25/hd_ngm25_neuron_group.h` | `neuron_group_t` — the CPU entity, a thin caller of the kernels |
| `include/hd_ngm25/hd_ngm25_gpu_neuron_group.h` *(WGPU)* | `neuron_group_gpu_t : gpu_io_entity` — host arena as source of truth, mirrored to device buffers |
| `src/hd_ngm25/hd_ngm25_host.cpp` | The shared host code; `load_state` of both groups = `stage_checkpoint`, the inhibitor's load, then a commit of moves, so a refused checkpoint leaves the group unchanged |
| `src/hd_ngm25/hd_ngm25_arena.cpp` | Arena consistency checker and initial arena construction with Poisson permanence init (`ngm/cold_init.h`) |
| `src/hd_ngm25/hd_ngm25_response.cpp` | R1 `build_step_ctx`, R2 `response_dendrite`, R3 `combine_neuron`, and the `response_all` driver |
| `src/hd_ngm25/hd_ngm25_adapt.cpp` | R5 winner + two adapt passes, R6 adapt envelope, R7 dendrite adapt sweep |
| `src/hd_ngm25/hd_ngm25_branch.cpp` | R8 branch decide predicate; execution delegates to `repack_with_splits` |
| `src/hd_ngm25/hd_ngm25_neuron_group.cpp` | The CPU `io_entity`: `process()` assembles the R1–R8 step skeleton over the kernels |
| `src/hd_ngm25/hd_ngm25_gpu_neuron_group.cpp` *(WGPU)* | The GPU group: host `synapse_arena_t` as source of truth, mirrored into GPU storage buffers; kernels run through the `gpu_device` facade |
| `src/hd_ngm25/hd_ngm25_gpu.wgsl` *(WGPU)* | ngm2.5 GPU kernels; `syn_key` is a line-for-line transcription of `hd_ngm25_prng.h`, so CPU and GPU draw identical values |
| `src/hd_ngm25/hd_ngm25_gpu_probe_main.cpp` *(WGPU)* | `coast_ngm25_gpu_probe` executable — dataset-free isolation probe for the ngm2.5 GPU group |
| `src/hd_ngm25/hd_ngm25_gpu_validate_main.cpp` *(WGPU)* | `coast_ngm25_gpu_validate` executable — CPU/GPU `pcg_hash(syn_key(…))` parity and CPU-vs-GPU forward-output agreement |
| `src/hd_ngm25/hd_ngm25_gpu_premise_main.cpp` *(WGPU)* | `coast_ngm25_gpu_premise` executable — throwaway premise-check micro-benchmark (sparse arena vs the dense ngm3-GPU floor) |

## ngm/  *(model logic shared by ngm2, ngm25 and ngm3)*

| File | Contains |
|------|----------|
| `include/ngm/bank.h` | `ngm::bank_t`, `ngm::input_route_t` — bank identity and input routing, one definition for ngm3 and ngm25 (both re-export them) |
| `include/ngm/common_config_record.h` | `ngm::common_config_record<P>()` — the configuration-record field list ngm3 and ngm25 share (ngm25 adds `penalty_policy`) |
| `include/ngm/cold_init.h` | `ngm::cold_permanence_t` — the cold permanence draw every model uses: Poisson(int(100·τ))/100 clamped to [0,1], over the caller's engine (each model keeps its own engine and draw order) |
| `include/ngm/window_tracking.h` | `ngm::window_tracking` — activity learning window tracking (SPEC M.4.2a): constants, the four edges, the update, checkpoint save/read, and the WGSL prelude the GPU groups prepend |

## inhibition/

| File | Contains |
|------|----------|
| `include/inhibition/inhibition.h` / `src/inhibition/inhibition.cpp` | `inhibitor_t : core::entity_element` — the polymorphic inhibition interface a neuron group owns; `make_inhibitor()` |
| `include/inhibition/inhibition_cfg.h` | `inhibition_type_t` (`random_index`, `max_softmax`), `random_index_variant_t`, `inhibition_params_t`, `resolve_params()` |
| `include/inhibition/max_softmax.h` / `src/inhibition/max_softmax.cpp` | `max_softmax_t` — softmax over the group output, rescaled so the winner keeps its activity: `out_i = max(v)·exp(−β·(max(v) − v_i))` |
| `include/inhibition/random_index.h` / `src/inhibition/random_index.cpp` | `random_index_t` — random recurrent lateral inhibition from a seeded N×N permanence matrix (CPU only) |
| `include/inhibition/gpu_inhibition.h` / `src/inhibition/gpu_inhibition.cpp` | `gpu_inhibitor_t`, `gpu_host_handles`, `make_gpu_inhibitor()` — kernel injection into the host group's `step_plan`; rejects kinds without a GPU port |
| `include/inhibition/gpu_max_softmax.h` / `src/inhibition/gpu_max_softmax.cpp` | `gpu_max_softmax_t` — self-contained WGSL port of `max_softmax_apply` |

## io/

| File | Contains |
|------|----------|
| `include/io/sequence_params.h` | `dataset::sequence_params_t` — shared base for all seven sequenced loaders (change_interval, noise_range, random_seed, shuffle, index_manifest) and the run binding |
| `include/io/dataset_run.h` | `dataset::run_binding_t` (a loader's identity record); `save_run_binding`, `restores_sequence` — the S.5.14 restore rule |
| `include/io/sample_sequencer.h` | `sample_sequencer` — epoch/index tracking, optional index manifest; `load_index_manifest()`, `ensure_manifest_readable()`, `build_sequencer()`, `restore_sequenced_state()` |
| `include/io/idx_io_base.h` | `dataset::idx_io_base` — shared base for IDX-format loaders (mnist, emnist, kmnist, color_mnist); holds output_buf, id, seq_, rgen, dis, seq_params_; implements set_output_buffer, get_outp_id, get_outp_size, save_state, load_state (restore decided by dataset::restores_sequence) |
| `include/io/cifar_io_base.h` | `dataset::cifar_io_base` — shared base for CIFAR loaders (cifar10, cifar100); holds images, image_count, seq_, rng, seq_params_; implements process(), status_str(), the dataset accessors, load_batch_file() and rebuild_sequencer() |
| `include/io/dataset_cfg.h` | `dataset::cfg_store` (one parameter map per dataset family) and `for_each_family`, whose keys are `entity_types`' dataset families |
| `include/io/mnist_cfg.h` | `mnist_param_t : dataset::sequence_params_t` — image/label paths; change_interval, noise_range, seed, shuffle inherited |
| `include/io/mnist_io.h` | `dataset::idx_gray_io<Tag> : idx_io_base` — one implementation for MNIST, EMNIST and KMNIST; aliases `mnist::mnist_io`, `emnist::emnist_io`, `kmnist::kmnist_io` |
| `include/io/mnist_io_gpu.h` *(WGPU)* | `mnist_io_gpu : gpu_io_entity, mnist_db` — full dataset on GPU, 1 kernel (read+noise); standalone (does not share idx_io_base) |
| `src/io/mnist_io_gpu.{cpp,wgsl}` *(WGPU)* | GPU MNIST host orchestration + the two compute kernels |
| `include/io/emnist_cfg.h` | `emnist_param_t` — Extended MNIST config (loader: `emnist::emnist_io` in `mnist_io.h`) |
| `include/io/kmnist_cfg.h` | `kmnist_param_t` — Kuzushiji MNIST config (loader: `kmnist::kmnist_io` in `mnist_io.h`) |
| `include/io/color_mnist_cfg.h` / `color_mnist_io.h` | `color_mnist_io : idx_io_base` — 28x28x3 RGB; rejects non-28x28 image files; overrides get_outp_size + save/load for current_color |
| `include/io/cifar10_cfg.h` / `cifar10_io.h` | `cifar10_io : cifar_io_base` — CIFAR-10; constructor only (5-batch training load or test set) |
| `include/io/cifar100_cfg.h` / `cifar100_io.h` | `cifar100_io : cifar_io_base` — CIFAR-100; constructor only (one file, coarse + fine label bytes) |
| `include/io/dsprites_cfg.h` / `dsprites_io.h` | dSprites — 64x64x1 from HDF5; standalone (no shared base) |
| `include/io/chairs3d_cfg.h` / `chairs3d_io.h` | 3D Chairs — configurable image dimensions, loaded via stb_image; standalone (no sample_sequencer) |
| `src/io/sample_sequencer.cpp` | `sample_sequencer`, `load_index_manifest`, `ensure_manifest_readable`, `build_sequencer`, `restore_sequenced_state` implementations |
| `src/io/mnist_db.cpp` | IDX binary format reader (big-endian), `get_norm_image()`; reads unsigned-byte (0x08) data only; throws on another data type, a truncated header, an overflowing size, data that does not match the header, or a dataset with no image data |
| `src/io/idx_io_base.cpp` | IDX-family base: ctor, set_output_buffer, get_outp_id, get_outp_size, save_state, load_state |
| `src/io/cifar_io_base.cpp` | CIFAR-family base: ctor, load_batch_file (throws on short/oversized files), process, status_str, rebuild_sequencer, set_output_buffer, get_outp_id, get_outp_size, save_state, load_state |
| `src/io/mnist_io.cpp` | `idx_gray_io` process() and status_str(); explicit instantiations for MNIST/EMNIST/KMNIST |
| `src/io/cifar10_io.cpp` / `cifar100_io.cpp` | CIFAR constructors: batch file list and label byte count |
| `src/io/dsprites_io.cpp` | HDF5 loading with RAII handles; the image count from the file goes through `element_count` / `require_stored` / `read_all` |
| `include/io/mnist_db.h` | `idx_file`, `mnist_db` — IDX reader; accepts unsigned-byte (0x08) data only |
| `src/io/dataset_run.cpp` | `save_run_binding()`, `restores_sequence()` — the dataset identity record in a checkpoint |
| `src/io/color_mnist_io.cpp` | Color MNIST: fixed 28x28x3 output, one random colour per presentation |
| `src/io/chairs3d_io.cpp` | 3D Chairs: recursive PNG scan (sorted), stb_image load and resize |
| `src/io/mnist_io_gpu_probe_main.cpp` *(WGPU)* | `coast_mnist_io_gpu_probe` executable — isolation probe for `mnist_io_gpu` (`gpu.smoke.mnist_io`) |
| `src/io/mnist_io_gpu_spike_main.cpp` *(WGPU)* | `coast_mnist_io_gpu_spike` executable — dataset-upload bandwidth and readback-cost measurements |

## retina/

| File | Contains |
|------|----------|
| `include/retina/retina_cfg.h` | `retina_param_t` (sizes, polarity, centre/surround sigmas, surround weight k, gain, optional divisive normalisation, noise), `polarity_t`, `cfg_store` |
| `include/retina/retina.h` / `src/retina/retina.cpp` | `validate()`, `contrast_filter_t` (separable centre-surround filter, optional RGB→luminance), `polarize()` |
| `include/retina/retina_io.h` / `src/retina/retina_io.cpp` | `retina_io` — one polarity of the response plus a noise floor; one input, size-checked in `init_io()` |

## json_cfg/

| File | Contains |
|------|----------|
| `include/json_cfg/json_loader.h` | `json_loader` class, `global_cfg_t` struct |
| `include/json_cfg/dataset_json.h` | `dataset::cfg_store`; each family's params serialise via `NLOHMANN_DEFINE_DERIVED_TYPE_*` over the `sequence_params_t` base macro (dsprites hand-written for its deprecation warning; chairs3d flat) |
| `include/json_cfg/ngm_bank_json.h` | JSON adapters for `ngm::bank_t` (strict enum) and `ngm::input_route_t`, shared by the ngm3 and ngm25 schema |
| `include/json_cfg/ngm2_json.h` | `ngm2::cfg_store`, nlohmann macros for dendrite/neuron/group params |
| `include/json_cfg/sim_json.h` | `sim_cfg_t`, `sim_node_t`, `sim_node_set_t`, `sim_node_delta_t`, entity type enum serialization |
| `include/json_cfg/retina_json.h` | `retina_param_t` macros; `polarity_t` via `COAST_JSON_STRICT_ENUM` (unknown string throws) |
| `include/json_cfg/strict_enum.h` | `COAST_JSON_STRICT_ENUM` — strict enum ↔ JSON mapping used by every config enum (unknown value throws); `COAST_JSON_STRICT_ENUM_RANGE` takes the names from a table (`io_entity_type_t`) |
| `include/json_cfg/vis_hints_json.h` | nlohmann macros for `vis_hints_t` and all hint sub-structs (the structs themselves live in `include/core/vis_hints.h`) |
| `src/json_cfg/json_loader.cpp` | File I/O, `global_cfg = json;` deserialization, snapshot_dir handling |
| `include/json_cfg/ngm3_json.h` | nlohmann adapters for the ngm3 params and `cfg_store` |
| `include/json_cfg/ngm25_json.h` | nlohmann adapters for the ngm2.5 flat schema; the `"ngm25"` config key binds to `ngm25::cfg_store` |
| `include/json_cfg/inhibition_json.h` | JSON adapters for the inhibition config; included by `ngm2_json.h`, and through it by `ngm3_json.h` |

## tools/

| File | Contains |
|------|----------|
| `include/tools/sim_builder.h` | `sim_builder` — create/init function registries (`handles(type)`), node map |
| `include/tools/sim_builder_cfg.h` | `sim_node_t`, `sim_node_set_t`, `sim_node_delta_t`, `io_entity_type_t` enum and `entity_types`, the one table of entity types (name, display name, kind, GPU, dataset family) from which `is_gpu_entity_type()` / `is_dataset_entity_type()` / `is_neuron_group_entity_type()`, `entity_type_display_name()` and the JSON names derive |
| `include/tools/sim_builder_interface.h` | `register_entity_handlers(sim_builder&, json_loader&, hdf5::executor&, seed_offset_t)` — one line per entity type over composable helpers (`register_group` / `register_params` to create, `register_init_io` / `register_init_structure` to wire); `unit.tools.entity_registration` checks every enumerator is handled |
| `include/tools/dataset_loader.h` | `dataset_loader` — builds *only* a config's CPU dataset entities (no neuron groups / threads / GPU); owns them in its own `sim::env` |
| `include/tools/metric.h` | header-only `compute_metric()` (cosine similarity, Minkowski-`p`), `metric_kind`, `metric_params` |
| `include/tools/metric_registry.h` | `metric_def` + `metric_registry()` — enumerable metric metadata (name, fixed domain, param descriptor) over `metric.h`, for building metric dropdowns without hard-coding `metric_kind` per caller |
| `include/tools/inc_on_exit.h` | `inc_on_exit` RAII counter incrementer |
| `src/tools/sim_builder.cpp` | `build_local()` — 5-phase construction algorithm |
| `src/tools/dataset_loader.cpp` | dataset-only config filter (CPU dataset nodes + compatible node_sets) + build |
| `include/tools/distortion.h` | Header-only, GUI-free signal distortions (Gaussian noise, dropout) for the Metric Explorer's distortion mode |
| `include/tools/ensemble_code.h` | Header-only synthetic "ensemble code" generator for the Metric Explorer: sparse codes from M prototypes |
| `include/tools/histogram.h` | Header-only `histogram_bins()`, `order_by_key()` for the Metric Explorer |

## probe/

| File | Contains |
|------|----------|
| `include/probe/spsc_queue.h` | `spsc_queue<T>` — lock-free ring buffer (default 4096), `try_push/pop`, `drain` |
| `include/probe/probe_instance.h` | `probe_instance` — per-probe state, SPSC queue, decimation, activate/deactivate. `probe_sample` struct. Global `set/get_current_step()` |
| `include/probe/entity_probe_set.h` | `entity_probe_set` — TBB concurrent_hash_map, `record()` hot path, `discover/activate/deactivate` |
| `include/probe/probe_observer.h` | `probe_observer` abstract interface — `push()`, `render()`, `name()` |
| `include/probe/observers/scalar_observer.h` | Current value + running min/max/mean/count |
| `include/probe/observers/time_series_observer.h` | ImPlot line chart, circular buffer (1000), optional EMA |
| `include/probe/observers/histogram_observer.h` | ImPlot bar chart, auto-ranging (first 100 samples) or fixed range, 50 bins |
| `include/probe/observers/csv_file_observer.h` | Direct CSV file output, flushes every 100 samples |
| `src/probe/probe_instance.cpp` | `record()`, `update_observers()`, activate/deactivate with atomic management |
| `src/probe/entity_probe_set.cpp` | TBB hash map operations, hash key computation |
| `src/probe/observers/scalar_observer.cpp` | `push()`, `render()`, `reset()` of the scalar observer |
| `src/probe/observers/time_series_observer.cpp` | Circular-buffer plot, EMA toggle and recalculation |
| `src/probe/observers/histogram_observer.cpp` | Range clamping, binning of pending values, bar plot |
| `src/probe/observers/csv_file_observer.cpp` | CSV header, step/value rows, periodic flush |

## observe/

| File | Contains |
|------|----------|
| `include/observe/observe_accessor.h` | `observe_accessor` — `observe()` overloads (scalar/1-D/N-D span), `group_active()`, `trigger()` |
| `include/observe/observe_dtype.h` | `dtype_traits<T>` — every scalar `hdf5::native_type` maps, and `std::array<T, N>` |
| `include/observe/observe_trigger.h` | `trigger_mode_t` (START/STOP/SINGLE_CAPTURE), `combine_mode_t`, `step_range_t` |
| `include/observe/observe_site.h` | `observe_site_t` — HDF5 handles, shape lock, rows written |
| `include/observe/observe_group.h` | `observe_group_t` — ranges, combine, `prog_active`/`prog_single_pending`/`active_this_step` atomics, site map |
| `include/observe/observe_cfg.h` | `observe_cfg_t` typed config structs + `from_json` parsing |
| `include/observe/observe_registry.h` | `observe_registry` singleton — attach/detach, post-process pass, `resolve_site`, trigger evaluation |
| `include/observe/observe_writer.h` | `observe_writer_t` — per-group HDF5 writer submitting to `hdf5::executor` |
| `include/observe/observe_chunk.h` | `pick_chunk_rows()` — byte-budget HDF5 chunk-geometry policy (pure, HDF5-free) |
| `src/observe/observe_accessor.cpp` | accessor → registry `emit()` glue |
| `src/observe/observe_config.cpp` | `apply_config()` — seed registry groups from parsed config |
| `src/observe/observe_registry.cpp` | post-process observation pass, lazy site resolution, trigger evaluation |
| `src/observe/observe_writer.cpp` | file/dataset lazy-create, row append, flush + close; continuing a run's file on a resume (`observe_run_t`) |

## diag/  *(diagnostics — distinct from logging/ below)*

| File | Contains |
|------|----------|
| `include/diag/level.h` | `coast::diag::level` enum + `level_name` / case-insensitive `level_from` — split out so `cli/options.h` needn't pull the whole facility for one enum. Header-only |
| `include/diag/diag.h` | `coast::diag` — `COAST_TRACE/DEBUG/INFO/WARN/ERROR` macros (std::format), runtime `set_level`/`level_enabled`, sinks (`set_sink`/`set_stream_sink`/`set_log_file`, mutex-serialized, reentrancy-guarded). Includes `level.h`. Header-only |
| `include/diag/progress.h` | `coast::diag::progress` — structured run-progress contract; `format_progress()` (emitter) + `parse_progress()` (consumer) share one line grammar; `emit_progress()` (flushed, thread-safe). Header-only |

## logging/

| File | Contains |
|------|----------|
| `include/logging/logging.h` | Convenience include for the logging subsystem |
| `include/logging/logging_common.h` | `log_result`, `log_stats`, `compression_config`, `chunk_config` |
| `include/logging/hdf5_utils.h` | RAII wrappers: `hdf5::file`, `group`, `dataspace`, `plist`, `dataset`, `id_guard` (raw ids). Free functions: `write/read_attr`, `link/attr_exists`, `append_rows` (grow an extensible dataset and write rows at an index — the buffer logger's and the observe writer's one append path). Checked reads of files COAST did not write: `require_rank`, `element_count`, `require_stored`, `read_all`, `read_array`, `read_fixed_strings`, `read_string_attr` |
| `include/logging/logging_manager.h` | `logging_manager` — owns buffer_logger, state_logger; `extract_state`/`try_extract_state`/`serialize_state`/`load_state` (quiesced split; `try_extract_state` refuses a halted run and never throws; a refused runtime load returns a failed result, never throws) + `load_state_startup` (build_env exemption) |
| `include/logging/buffer_logger.h` | `buffer_logger` — post_swap hook, frame buffering, chunked HDF5 writes, topology extraction |
| `include/logging/state_logger.h` | `state_logger` — full-state persistence: static `extract_state`/`serialize_state`/`parse_state`/`apply_state` split + monolithic `save_state` (test convenience) / `load_state` (startup exemption); `hdf5_state_writer/reader` |
| `include/logging/state_checks.h` | `check_shape` / `shape_field` — the hard check a group's `load_state` runs first: a checkpoint of another shape throws, naming each differing or missing field |
| `include/logging/config_record.h` | Checkpoint configuration record (M.7.8): `config_record`, `write_config_record`, `diff_config_record` / `check_config_record`, `report_unrecorded_configs`; channel record (M.7.5): `channel_layout`, `write_channel_record`, `check_channel_record` (refuses a differing layout), `report_unrecorded_channels` |
| `include/logging/state_io.h` | `state_writer`/`state_reader` abstract interfaces, `state_serializable` |
| `include/logging/mem_state_io.h` | `mem_group` in-memory state tree, `mem_state_writer`/`mem_state_reader`, tree↔HDF5 adapters (`replay`, `parse_hdf5_to_tree` — also the whole-file check both load paths run first) |
| `src/logging/buffer_logger.cpp` | `capture_frame()`, `write_frames()`, HDF5 schema creation, topology extraction |
| `src/logging/state_logger.cpp` | `hdf5_state_writer`/`hdf5_state_reader`, the extract/serialize/parse/apply implementations, monolithic `save/load_state()` |
| `src/logging/state_checks.cpp` | `check_shape()` — reads each structural field via `has_field` / `read_uint64` and throws on any mismatch |
| `src/logging/mem_state_io.cpp` | Tree navigation, HDF5 replay (tree → `hdf5_state_writer`), generic HDF5 → tree parse |
| `src/logging/hdf5_parse.h` | Internal: the HDF5 → tree group parse (and, without a tree, check) shared by `parse_hdf5_to_tree` and `hdf5_state_reader::read_scalars` |
| `src/logging/hdf5_utils.cpp` | Free function implementations for attribute read/write |
| `src/logging/logging_manager.cpp` | Lifecycle management, combined stats |
| `include/logging/rng_state_io.h` | `write_rng()` / `read_rng()` — persist `fast_rng_t` state so a resumed run is bit-identical; an absent group keeps the seeded stream |
| `src/logging/config_record.cpp` | `write_config_record()` / `check_config_record()`, the channel record, `report_unrecorded_configs()` |

## hdf5/

| File | Contains |
|------|----------|
| `include/hdf5/executor.h` | `hdf5::executor` — single-thread funnel for all libhdf5 calls (library built Threadsafety: OFF); owns the one worker thread allowed to enter libhdf5 |
| `include/hdf5/handle.h` | `hdf5::handle<CloseFn>` — move-only RAII over `hid_t` with off-thread destruction (destructor posts the `H5*close` call to the executor) |
| `include/hdf5/native_type.h` | `hdf5::native_type<T>()` — the one C++ type → `H5T_NATIVE_*` table (state writer, state-tree reader, observe dtypes) |
| `src/hdf5/executor.cpp` | The executor's worker thread, `submit()`, shutdown (post-stop jobs are dropped) |

## h5browse/

| File | Contains |
|------|----------|
| `include/h5browse/scalar_type.h` | The 10 writer dtypes (f32/f64/i8-i64/u8-u64) + compile-time dispatch, HDF5-free |
| `include/h5browse/scan_stats.h` | `scan_stats` — merge-associative Welford-style count/NaN/Inf/min/max/mean/M2 accumulator |
| `include/h5browse/base_histogram.h` / `src/h5browse/base_histogram.cpp` | `base_histogram` — fixed-bin histogram, power-of-two range-doubling, exact `rebin`/`merge` |
| `include/h5browse/row_filter.h` / `src/h5browse/row_filter.cpp` | `row_filter` — AND-only structured column-condition filter; doubles as an IR and a provenance-string serializer |
| `include/h5browse/stats_kernel.h` / `src/h5browse/stats_kernel.cpp` | Hot per-chunk accumulation loops, own TU (perf-pinned), `extern template` over the 10 dtypes |
| `include/h5browse/h5b_error.h` / `src/h5browse/h5b_error.cpp` | `h5b::error` / `h5b::result<T>` value-based error channel; `H5Ewalk` triage-message harvesting |
| `include/h5browse/file_model.h` / `src/h5browse/file_model.cpp` | `open_and_map()` — family detection (observe/buffer_log), semantic tree, `dataset_desc` |
| `include/h5browse/chunk_cache.h` / `src/h5browse/chunk_cache.cpp` | `chunk_cache` — byte-budgeted (512 MB) LRU cache of `shared_ptr<const row_block>` |
| `include/h5browse/dataset_reader.h` / `src/h5browse/dataset_reader.cpp` | Shape-matched executor reads (15.9× vs. flat memspace); depth-2 read pipeline |
| `include/h5browse/pairwise.h` / `src/h5browse/pairwise.cpp` | Pairwise request types + TBB tiled-block kernel (sampled and full-O(N²) share one code path) |
| `include/h5browse/scan_engine.h` / `src/h5browse/scan_engine.cpp` | `scan_engine` — scan thread, two-lane command queues, `latest_value` snapshot mailboxes, log ring |
| `include/h5browse/console_api.h` / `src/h5browse/console_api.cpp` | The file surface the interpreter parses: `file_console` (own read-only handle + LRU cache), `dataset_handle`, `row_view`, plus the free `open()` → `shared_ptr<file_console>` (many files at once), `set_executor`, and `var_name_for`. **A parse boundary** — every declaration costs every session start. Errors are `std::runtime_error` (typeinfo already in libstdc++, so it crosses the JIT boundary) |
| `include/sim_console/sim_console_api.h` / `src/sim_console/sim_console_api.cpp` | The `coast::simc` sim-control façade: `config()` → editable `run_config` (JSON-Pointer `set`/`get_*`, `run_dir`/`seed` sugar, `dump`); `run()` writes `config.json` into a run dir and returns a `sim_run`; `sim_run::step()` (blocking) / `step_async()` (non-blocking) spawn a child `coast -n` with state-chaining + per-step timestamped logs; `save_state`/`load_state`/`last_log`/`running`. Async runs feed a process-wide registry drawn by `render_runs_window()` (the "Simulation Runs" table, driven each frame beside `coast::plot::render_all()`); `poll_runs`/`wait_all_runs`/`active_run_count`/`clear_finished_runs`. **Also a parse boundary** — `nlohmann/json` + `<mutex>` held behind the pimpl; edited by JSON Pointer. `COAST_SIMC_BINARY` overrides the spawned binary |

## run_config/

| File | Contains |
|------|----------|
| `include/run_config/run_config.h` / `src/run_config/run_config.cpp` | `run_config` — an editable simulation config (load a base JSON, `set()`/`get_*()` by JSON pointer, `dump()`), shared by `coast::simc` and `coast::coastd` |

## campaign/

| File | Contains |
|------|----------|
| `include/campaign/campaign_api.h` / `src/campaign/campaign_api.cpp` | `coast::campaign` — meta-loader over a directory of runs: `open()` scans the configs, discovers the swept parameters by diffing them (`axis`), and filters `run` handles into a `selection`; HDF5 is opened on demand. See [Campaign meta-loader]({{% relref "/architecture/modules/campaign" %}}) |

## console/  *(COAST_WITH_CONSOLE, auto-detected: needs system LLVM/Clang dev libs)*

| File | Contains |
|------|----------|
| `include/console/repl.h` / `src/console/repl.cpp` | `console::repl` — the `clang::Interpreter` wrapper behind an LLVM-free pimpl. Built once, file-independent (its preamble pulls in `console_api.h`, so `open()` is in scope). `exec` (raw, unguarded — COAST-controlled code only) / `exec_guarded` (host-side catch, runs unwrapped so declarations persist; the path for **all** user input incl. `load()`), `split_into_units` + `scan_comment_state` (`#include` isolation, skipped for lines not wholly outside a `/* */`), `Undo(1)` only on Execute failure, `.L`/`.x`/`.I`/`.load`, and the retaining JIT code arena |
| `include/console/session.h` / `src/console/session.cpp` | `console::session` — the console thread, the non-blocking command queue, and `fd_capture` (stdout/stderr dup + background reader thread) producing `output_line{input\|output\|error\|info}`. Builds the interpreter once (`start`), assigns/dedupes file-handle variables (`open_file` → var name), tracks `open_files()`, `reset` rebuilds + re-opens, registers `open()`'s executor |
| `include/console/console_panel.h` / `src/console/console_panel.cpp` | `console::console_panel` — the ImGui notebook panel implementing `coast::console_iface`: re-runnable input cells, Ctrl+Enter, meta-command dispatch, the `.?` cheatsheet, interpreter pre-warm on first show, multi-file toolbar |
| `include/console/quote_cpp.h` | `quote_cpp()` — a double-quoted C++ string literal safe to splice into interpreter source (escapes quotes, backslashes, control bytes) |

## gui_plot/

Lives in `coast_gui` and depends on **nothing** from console/LLVM — the coupling is runtime-only (a macro `#include`s these headers), which is why every header here except `plot_tools.h` is ImGui-free.

| File | Contains |
|------|----------|
| `include/gui_plot/plot_data.h` | Plain DTOs: `range`, `zone`, `outer_ring`, `grid_result`, and `h5_save_data` — the GL/ImGui-free snapshot that can be moved to an executor worker |
| `include/gui_plot/figure.h` / `src/gui_plot/figure.cpp` | `coast::plot::figure` — base class, the `region` POD (a plain ImVec2 replacement so the header needs no ImGui), `painter` (`std::function<void(region)>` capturing `shared_from_this()`), `plot_axis`; seams `paint_body` (pure) / `paint_controls` / `fill_snapshot` / `is_saveable` |
| `include/gui_plot/plot_manager.h` / `src/gui_plot/plot_manager.cpp` | `plot_manager` — heap singleton owning the registry (which **owns** shown figures), save queues, executor + gui_host, uid. Reaping = closed **and** `use_count()==1`; `deinit()` sets a process-wide tombstone so `instance()` throws rather than silently re-creating a host-less manager |
| `include/gui_plot/plot_detail.h` | Module-internal glue: `canvas_save_data`, `make_figure_from_h5`, `detail::export_scale` and sizing helpers |
| `include/gui_plot/plot_h5.h` / `src/gui_plot/plot_h5.cpp` | HDF5 (de)serialization — the only TU pulling libhdf5. `/plot` (`coast_plot_version=1`), `/canvas` (`coast_canvas_version=1`) + `cell_<r>_<c>` subgroups; `H5Aexists`/`H5Lexists` probing so auto-ranged loads don't dump the HDF5 error stack |
| `src/gui_plot/plot_load.cpp` | The one kind-string → concrete-type map; `recount_outliers()` regenerating 1D histogram tallies from raw ys |
| `src/gui_plot/plot_capture.cpp` | Off-screen FBO PNG export (`stb_image_write`) + the manager's capture methods and `flush_captures`. **Double `build_frame()`** — a throwaway priming frame, because ImPlot AutoFits the frame *after* data submission |
| `include/gui_plot/plot_tools.h` / `src/gui_plot/plot_tools.cpp` | **The only ImGui-including header.** `bin_points`, `reshape_field`, `draw_colormap_grid`, `draw_candles`; theme layer (`theme_scope` RAII, built-in `dark`/`bright`, editor, `coast_plot_themes.json` persistence) |
| `include/gui_plot/histogram.h` / `src/gui_plot/histogram.cpp` | 1D histogram — fixed-at-construction domain, under/overflow tallies; saves **raw ys** so tallies recompute on load |
| `include/gui_plot/line.h` / `src/gui_plot/line.cpp` | Line plot — `update(y)` and `update(x, y)` |
| `include/gui_plot/histogram2d.h` / `src/gui_plot/histogram2d.cpp` | 2D histogram — Sturges auto-binning, log density, candle mode, palette; persists its `outer` ring because a saved grid cannot recompute it |
| `include/gui_plot/pixels.h` / `src/gui_plot/pixels.cpp` | Fixed w×h field, row-major, row 0 = top; palette + axis toggles |
| `include/gui_plot/image.h` / `src/gui_plot/image.cpp` | PNG/RGBA display via `gl_texture`; the only **non-saveable** kind |
| `include/gui_plot/canvas.h` / `src/gui_plot/canvas.cpp` | rows×cols grid hosting other figures (`place()`); hosted figures must not also be `show()`n |
| `include/gui_plot/console_plot.h` / `src/gui_plot/console_plot.cpp` | Umbrella header a macro includes (all kinds + free functions); the `.cpp` is thin forwarders to `plot_manager::instance()` |

## gui_core/

| File | Contains |
|------|----------|
| `include/gui_core/gui_manager.h` | `gui_manager` — menu bar (System/Simulation/Introspection/**Tools**/Analysis), dialogs, browsers, editors, the persistent sim-independent `metric_explorer`, `free_resources()`, auto-save, snapshot resume; status-dialog strings published via `demand_published` from a post_swap hook |
| `include/gui_core/entity_browser.h` | `entity_browser` — sortable table, detail panel, vis checkbox management |
| `include/gui_core/buffer_browser.h` | `buffer_browser` — buffer table with live stats (sim-thread post_swap publish), vis management |
| `include/gui_core/dataset_browser.h` | `dataset_browser` — view-only consumer of `dataset_access` + `thumbnail_grid`: dataset selector + grid |
| `include/gui_core/dataset_access.h` | `dataset_handle`, `enumerate_datasets(sim::env&, const sim::sim_builder&)` — the one dataset→accessor switch; `convert_sample_to_rgba()` |
| `include/gui_core/thumbnail_grid.h` | `thumbnail_grid` — reusable virtualized thumbnail grid, lazy texture cache; view-only or selectable |
| `include/gui_core/config_editor.h` | `config_editor` — full global_cfg_t editing, dirty tracking, save/restore |
| `include/gui_core/network_editor.h` | `network_editor` — ImNodeFlow graph editor for simulation topology |
| `include/gui_core/logging_window.h` | `logging_window` — buffer logging and state controls, auto-checkpoint |
| `include/gui_core/imgui_scaling.h` | `scaled(float)` — HiDPI scaling helper |
| `src/gui_core/sdl_app.cpp` | SDL2 + OpenGL init, ImGui init, main event loop, shutdown |
| `src/gui_core/gui_manager.cpp` | Menu rendering (Tools/Analysis always-on), control dialog, status bar, set_sim_instance(), `free_resources()`, auto-save |
| `src/gui_core/entity_browser.cpp` | Table rendering, sorting, filtering, vis management |
| `src/gui_core/buffer_browser.cpp` | Buffer table, sim-thread stats sweep (`compute_buffer_stats`) + demand-driven publish, vis management |
| `src/gui_core/dataset_browser.cpp` | View over `dataset_access` + `thumbnail_grid` |
| `src/gui_core/dataset_access.cpp` | `enumerate_datasets()` switch (incl. `color_mnist`), `convert_sample_to_rgba()` |
| `src/gui_core/thumbnail_grid.cpp` | Virtualized grid render, lazy texture cache, click-to-toggle selection |
| `src/gui_core/config_editor.cpp` | 2300+ lines: section renderers for all config types |
| `src/gui_core/network_editor.cpp` | ImNodeFlow integration, shadow state, edit callbacks |
| `src/gui_core/logging_window.cpp` | File browser integration, auto-checkpoint implementation |
| `include/gui_core/sdl_app.h` | `sdl_app`, `sdl_app_paths` — the SDL/ImGui app shell; paths are passed in resolved, so it stays free of the app-config type |
| `include/gui_core/console_iface.h` | `console_iface` — abstract handle to the embedded C++ console; the implementation lives in `coast_console`, the only LLVM-linked target, and is injected through `run_gui` |

## gui_tools/

| File | Contains |
|------|----------|
| `include/gui_tools/dataset_source.h` | `dataset_source` interface + `imported_source` (wraps `dataset_loader`) / `synthetic_source` (ensemble codes); stable `(source_id, local_idx)` axis refs; `datasets()` → ready-named handles |
| `include/gui_tools/matrix_canvas.h` | `matrix_canvas` — zoomable colored-grid renderer; `inputs` struct (values, vmin/vmax, palette, row/col thumbs); owns zoom/pan, `reset_view()` |
| `include/gui_tools/metric_explorer.h` | `metric_explorer` — "Tools > Metric Explorer" panel; pure-import data model (owns `dataset_loader` sources), axes, metric/palette, sample picker |
| `include/gui_tools/h5_browser_window.h` | `h5_browser_window` — "Analysis > HDF5 Browser" panel; thin renderer over `h5b::scan_engine`, consumes snapshots + enqueues commands, never blocks |
| `src/gui_tools/dataset_source.cpp` | `imported_source`/`synthetic_source` impls: config-backed enumeration (stem-prefixed names) + in-memory synthetic-code handle |
| `src/gui_tools/matrix_canvas.cpp` | Cursor-centred zoom, per-cell fill, scaled hide-below-threshold text, pinned+scaling headers, legend |
| `src/gui_tools/metric_explorer.cpp` | Import/remove/clear sources, matrix compute (`compute_metric`), seeded/click sample picker |
| `src/gui_tools/h5_browser_window.cpp` | Tree+search, stats table, virtualized table (`ImGuiListClipper`), graph, histogram, pairwise, and log-panel views |
| `include/gui_tools/campaign_panel.h` / `src/gui_tools/campaign_panel.cpp` | `campaign_panel` (Analysis menu) — thin ImGui front over `coast::campaign`: scan a directory of runs, filter by the discovered parameters, open a run in the HDF5 browser or hand the selection to the console |

## gui_vis/

| File | Contains |
|------|----------|
| `include/gui_vis/vis_registry.h` | `vis_registry` singleton, `vis_info`, `vis_instance` base, factory types, registration API |
| `include/gui_vis/core/color_types.h` | `color_t` RGBA struct, color constants |
| `include/gui_vis/core/colormap.h` | `colormap_t` enum mirroring ImPlot's continuous colormaps (viridis, plasma, hot, cool, jet, spectral, greys, …), `apply_colormap()`, `colormap_from_name()` |
| `include/gui_vis/core/gl_texture.h` | `gl_texture` RAII wrapper — `update()`, `as_imgui_texture()`, `release()` |
| `include/gui_vis/core/gradient.h` | `gradient<N>` — N-stop linear color interpolation |
| `include/gui_vis/primitives/vec_vis.h` | `vec_vis` — float vector → 2D pixel grid |
| `include/gui_vis/primitives/vec_group_vis.h` | `vec_group_vis` — layout manager for multiple vec_vis |
| `include/gui_vis/io_buffer/vec_ring_buffer.h` | `vec_ring_buffer` — scrolling time-series texture |
| `include/gui_vis/sim_entity/neuron_group/ngm_flat_vis.h` | `ngm_flat_vis` — "Dendritic Representations" grid view |
| `src/gui_vis/vis_registry.cpp` | Registry singleton, registration, factory invocation |
| `src/gui_vis/core/colormap.cpp` | Piecewise 5-stop interpolation for viridis/plasma/inferno |
| `src/gui_vis/core/gl_texture.cpp` | OpenGL texture creation, update, release |
| `src/gui_vis/io_buffer/vec_ring_buffer.cpp` | Ring buffer rendering, 3-state atomic FSM adapter |
| `src/gui_vis/sim_entity/neuron_group/ngm_flat_vis.cpp` | 545 lines: texture grid, zoom/pan canvas, colormap selector |
| `include/gui_vis/sim_entity/neuron_group/ngm_tree_vis_v2.h` | `ngm_tree_vis_v2` — ngm2 "Neuron Tree" on the `gui_layout` graph, force layout and canvas |
| `include/gui_vis/sim_entity/neuron_group/ngm3_flat_vis.h` | `ngm3_flat_vis<Group>` — flat permanence-grid view for any group with the ngm3 accessor contract (ngm3, ngm25, CPU and GPU) |
| `include/gui_vis/sim_entity/neuron_group/ngm3_tree_vis.h` | `ngm3_tree_vis` — tree view (member templates over the group type), one tree per neuron, apical/proximal edges colour-coded; leaf textures read from one rep row of the permanence matrix |
| `src/gui_vis/primitives/vec_vis.cpp` | `vec_vis` construction and params |
| `src/gui_vis/primitives/vec_group_vis.cpp` | `vec_group_vis` layout: total width/height |
| `src/gui_vis/sim_entity/neuron_group/ngm_tree_vis_v2.cpp` | Tree model, per-neuron repulsion, select/hover leaf textures; registers `ngm_tree_vis` for ngm2 |
| `src/gui_vis/sim_entity/neuron_group/ngm3_flat_vis.cpp` | Registers `ngm3_flat_vis` and `ngm25_flat_vis`, plus `ngm3_flat_vis_gpu` / `ngm25_flat_vis_gpu` under `COAST_WITH_GPU` |
| `src/gui_vis/sim_entity/neuron_group/ngm3_tree_vis.cpp` | Registers `ngm3_tree_vis` and `ngm25_tree_vis`, plus `ngm3_tree_vis_gpu` / `ngm25_tree_vis_gpu` under `COAST_WITH_GPU` |

## gui_probe/

| File | Contains |
|------|----------|
| `include/gui_probe/probe_browser.h` | `probe_browser` — tree view, observer window management, CSV integration |
| `include/gui_probe/observer_panel.h` | `observer_panel` — single-window view of all active observers |
| `src/gui_probe/probe_browser.cpp` | Tree rendering, observer popup, window lifecycle |
| `src/gui_probe/observer_panel.cpp` | Collapsible headers, observer remove buttons |

## gui_observe/

| File | Contains |
|------|----------|
| `include/gui_observe/observe_panel.h` | `observe_panel` — Observer Capture window header |
| `src/gui_observe/observe_panel.cpp` | State badge (reads `active_this_step`), range editing, combine selector, trigger buttons, add-group |

## gui_layout/

| File | Contains |
|------|----------|
| `include/gui_layout/layout_graph.h` | `layout_graph` — graph data structure, `node_t`, `edge_t`, O(1) lookup |
| `include/gui_layout/force_layout.h` | `force_layout` — Barnes-Hut quadtree, O(n log n) repulsion, `settle()` |
| `include/gui_layout/render/node_renderer.h` | `node_renderer` abstract base + `debug_node_renderer` |
| `include/gui_layout/render/graph_canvas.h` | `graph_canvas` — ImGui widget, zoom/pan/select/drag |
| `src/gui_layout/layout_graph.cpp` | Node/edge management, bounds computation |
| `src/gui_layout/force_layout.cpp` | Quadtree build, force computation, velocity integration |
| `src/gui_layout/render/node_renderer.cpp` | Debug renderer: circles with colors |
| `src/gui_layout/render/graph_canvas.cpp` | Canvas rendering, interaction handling, coordinate conversion |

## gui_replay/

| File | Contains |
|------|----------|
| `include/gui_replay/buffer_log_reader.h` | `buffer_log_reader` — HDF5 frame reader, topology parser, lazy step/time loading |
| `include/gui_replay/buffer_replay_dialog.h` | `buffer_replay_dialog` — main dialog, playback controls, graph construction |
| `include/gui_replay/replay_node_renderer.h` | `replay_node_renderer` — 3 vis modes (image_2d, stats_color, delta_flow), `buffer_node_data` |
| `src/gui_replay/buffer_log_reader.cpp` | HDF5 hyperslab reading, metadata parsing |
| `src/gui_replay/buffer_replay_dialog.cpp` | Full replay UI, playback timer, graph layout |
| `src/gui_replay/replay_node_renderer.cpp` | Texture generation, colormap application, delta computation |

## cli/

| File | Contains |
|------|----------|
| `include/cli/options.h` | `coast::cli::options` — every CLI field (config_path, headless, headless_flag_given, steps_mode + max_steps, seed_offset, buffer_log_path, etc.). Run length is the `run_length` enum (`unset`/`finite`/`unlimited`), not a sentinel count |
| `include/cli/parse.h` | `coast::cli::parse(int, char**)` — throws `coast::cli::parse_error` on bad args; `print_usage(ostream&, const char*)` |
| `src/cli/parse.cpp` | argv loop, per-flag dispatch, implication rules (`-n` ⇒ `--headless` ⇒ `--run`) |

## app/

| File | Contains |
|------|----------|
| `include/app/build_env.h` | `coast::app::build_env(opts, h5_exec) → unique_ptr<sim_instance>`. Loads config, applies seed_offset, resolves `--resume`/`--load-state`; `state_load_error` for a state that cannot be loaded (the `--resume` remedy appended when the snapshot was found, not named) |
| `include/app/run_headless.h` | `coast::app::run_headless(sim_instance&, opts) → int` |
| `include/app/run_gui.h` | `coast::app::run_gui(unique_ptr<sim_instance>, opts, h5_exec) → int` |
| `src/app/build_env.cpp` | `find_latest_snapshot()` for `--resume`; startup state load via `load_state_startup` (pre-stepping exemption) |
| `src/app/run_headless.cpp` | The headless driver: buffer-logging start/stop, step limit, the SIGTERM latch (`--save-on-term`), the supervisory poll loop (adaptive polling, progress, probe observers), end classification (signal / completion / fault), final state save (extract via `run_quiesced`, write future awaited before the manifest) |
| `include/app/headless_checkpoint.h` | `headless_checkpointer` — `--checkpoint N`: `on_post_swap()` (the sim-thread hook: exact-boundary snapshot, async write), `write_final()` (save-on-term through `run_quiesced`, deduplicated against the last periodic write), `collect()` (reaps finished writes — during the run, so a failure is reported when it happens, and waiting before the manifest) |
| `include/app/run_provenance.h` | `run_provenance` — the `--emit-run-manifest` inputs collected at start, `record_savestate()`, `set_exit_reason()`; writes the manifest once in its destructor. node seeds are labelled by their built node (so `node_set` copies carry their derived label) |
| `src/app/run_gui.cpp` | `sdl_app` + `gui_manager` setup, state-update callback for menu-driven config loads |
| `src/app/headless_checkpoint.cpp` | `headless_checkpointer` — write at the post-swap boundary, `write_final()`, `collect()` |
| `src/app/run_provenance.cpp` | `run_provenance` — the manifest inputs, one `node_seed_t` per seeded entity read from the built entity, `record_savestate()` |

## tools/coast_compare/

| File | Contains |
|------|----------|
| `tools/coast_compare/compare.h` | `tolerance_set`, `field_result`, `golden_match`, `compare_outcome`, `compare_against_goldens()`, `render_report()`, `exit_code_of()` |
| `tools/coast_compare/compare.cpp` | JSON walker (nested objects, inline arrays, binary-ref sidecars), multi-version match algorithm, severity-sorted report renderer, `aggregate` mode (`reduce_stats` mean/max/std, `emit_aggregate`) |
| `tools/coast_compare/main.cpp` | Argv parsing, dispatch into the lib |

## coastd/  *(the headless-job daemon — a separate executable)*

Its own target tree, added via `add_subdirectory(coastd)`. Headers are written daemon-rooted (`#include "daemon/types.h"`); the inner directory is `daemon/`, not `coastd/`, to avoid confusion with `coast_lib`'s `include/core/`. The dependency is one-way: `coast_lib`'s `include/` carries no coastd headers. See the [coastd section]({{% relref "/coastd" %}}).

| File | Contains |
|------|----------|
| `coastd/include/daemon/types.h` | The shared vocabulary: `job`, `reality`, `action` variant, `daemon_config`, `bucket`/`job_status` enums, `clamp_edge_bounds()` |
| `coastd/include/daemon/reconcile.h` · `src/reconcile.cpp` | **The pure state machine.** `reconcile(jobs, reality, cfg) → actions` + `backoff_ms()` |
| `coastd/include/daemon/core.h` · `src/core.cpp` | The single-writer loop, the action dispatcher + in-flight dedup sets, the `std::future` RPC seam, `boot()`/`recover_at_boot()` |
| `coastd/include/daemon/job_store.h` · `src/job_store.cpp` | Git-versioned per-file store: `load_all`, `put`/`try_put`, `move_bucket`, async coalescing `commit`, `sweep_stale_tmp` |
| `coastd/include/daemon/job_json.h` · `src/job_json.cpp` | Job ↔ JSON, `schema_version` tolerance, status tokens |
| `coastd/include/daemon/reality_probe.h` · `src/reality_probe.cpp` | `rescan()` — the fresh, uncached world each pass |
| `coastd/include/daemon/builder.h` · `src/builder.cpp` | Worker tasks: resolve (fetch + `rev-parse`), build (worktree + submodules + cmake), gc; `.coastd-ready` sentinel |
| `coastd/include/daemon/supervisor.h` · `src/supervisor.cpp` | One-shot child ownership: spawn, ~10 Hz tail poll, SIGTERM-grace escalation, `waitpid` reap → `child_exited` |
| `coastd/include/daemon/child_cmd.h` · `src/child_cmd.cpp` | The `coast`↔`coastd` contract: `prepare_run_config` (two files), `plan_launch` (fresh vs resume), `newest_checkpoint`, `read_outcome` |
| `coastd/include/daemon/child_proc.h` · `src/child_proc.cpp` | `posix_spawn` wrapper, signal delivery, start-token capture |
| `coastd/include/daemon/line_tailer.h` | Buffer, split on `\n`, retain the partial line — feeds `diag/progress.h`'s `parse_progress` |
| `coastd/include/daemon/proc_util.h` · `src/proc_util.cpp` | `is_same_process(pid, start_token)` — the recycle-safe pid duplicate-guard; `set_cloexec` |
| `coastd/include/daemon/worker_pool.h` · `src/worker_pool.cpp` | Bounded workers, task → `worker_result` event |
| `coastd/include/daemon/event_queue.h` · `events.h` | MPSC blocking `pop_until`; the `client_command`/`worker_result`/`child_exited` variant |
| `coastd/include/daemon/protocol.h` | **The wire contract** (header-only, `coast_lib`-free): JSONL envelope, `job_summary`/`machine_summary` DTOs, `parse_line`, `json_depth_ok` |
| `coastd/include/daemon/comms_server.h` · `src/comms_server.cpp` | TCP edge: accept loop, thread-per-connection, command handlers, config path-safety scan, edge bounds |
| `coastd/include/daemon/discovery_server.h` · `src/discovery_server.cpp` | UDP Model-P probe responder |
| `coastd/include/daemon/machine_info.h` · `src/machine_info.cpp` | `make_machine_summary` — the static self-description |
| `coastd/include/daemon/hash_inputs.h` · `src/hash_inputs.cpp` | `hash_inputs` — SHA-256 of files under `dataset_base`, with the containment rules and caps of the `hash_inputs` command |
| `coastd/include/daemon/sha256.h` · `src/sha256.cpp` | vendored SHA-256 (FIPS 180-4), streaming; no crypto dependency |
| `coastd/include/daemon/atomic_write.h` · `src/atomic_write.cpp` | `tmp → fsync → rename → fsync(dir)` durable write |
| `coastd/include/client/coastd_client.h` · `src/client/coastd_client.cpp` | The thin client: `connection` round-trips, `host_config` fixed list, `discover()`, `load_config_file()` |
| `coastd/include/daemon/config_load.h` · `src/config_load.cpp` | `load_config_file()` — `coastd.json` layered over the defaults (CLI overrides applied after) |
| `coastd/include/daemon/log_sweep.h` · `src/log_sweep.cpp` | `sweep_logs()` — age-based log housekeeping (compress old, delete ancient), `append_capped()` |
| `coastd/include/daemon/net_defaults.h` | Default TCP/UDP ports (51900/51901), dependency-free |
| `coastd/include/daemon/sock_compat.h` | Portable SIGPIPE suppression for socket writes, `write_all()` |
| `coastd/include/daemon/version.h` | The daemon's version string (`--version`, job provenance, machine_info) |
| `coastd/src/main.cpp` | CLI + JSON config layering, single-instance `flock`, `--install-service`, ordered shutdown |
| `coastd/service/coastd.service` · `com.coast.coastd.plist` · `README.md` | systemd user unit (stock `KillMode`) and the **unverified** launchd template |
| `include/coastd_console/coastd_console_api.h` · `src/coastd_console/` | `coast::coastd` console façade — compiled `job_list`/`host_list` handles for the JIT boundary |
| `include/gui_tools/coastd_panel.h` · `src/gui_tools/coastd_panel.cpp` | The GUI queue/submit/control panel (compile-verified only) |

## tests/

The test tree is opt-in via `cmake -B build-tests -DBUILD_TESTING=ON`. See [Testing]({{% relref "/testing" %}}) for the conceptual model.

| File | Contains |
|------|----------|
| `tests/CMakeLists.txt` | Test root — adds subdirectories, defines `coast_add_integration_test()` helper |
| `tests/{unit,integration,integration_b,integration_ngm3,integration_observe,integration_headless,gpu_smoke,smoke,support,third_party}/CMakeLists.txt`, `tests/unit/*/CMakeLists.txt` | Per-directory test registration; `gpu_smoke/` registers the GPU probe executables as `gpu.smoke.*` |
| `tests/third_party/doctest/doctest.h` | Vendored doctest 2.4.11 — single-header test framework |
| `tests/third_party/doctest/LICENSE.txt` | doctest's licence |
| `tests/integration/integration_helpers.{h,cpp}` | `run_io_buffer_fingerprint_test`, `open_blob`, `compare`, `run_for` — A-tier shared scaffolding |
| `tests/support/blob_writer.{h,cpp}` | `coast::test::blob_writer` — JSON + binary-sidecar writer |
| `tests/support/subprocess.h` | POSIX `fork+exec+poll` wrapper for B-tier; returns `{exit_code, stdout_text, stderr_text}` |
| `tests/support/fault_gpu_device.h` | `coast::test::fault_gpu_device` — in-memory `gpu_device` that fails chosen readbacks; GPU-entity fault tests without an adapter (#234) |
| `tests/support/subprocess.cpp` | `fork+exec` implementation behind `subprocess.h` (argv/envp arrays, inherited environment plus extras) |
| `tests/support/await_progress.h` | `await_progress` — wait for an outcome by progress, not by a timeout |
| `tests/support/sim_sync.h` | Logical-time synchronisation with the sim thread (no sleeps as synchronisation) |
| `tests/support/diag_capture.h` | Captures diag output for a scope; restores the default sink and level on destruction |
| `tests/support/h5_root_links.h` | Names of every link under an HDF5 file's root |
| `tests/sanitizer_suppressions/tsan.supp` | ThreadSanitizer suppressions (see `SANITIZERS.md`) |
| `tests/unit/cli/` | `coast::cli::parse` — argv permutations, error paths |
| `tests/unit/json_cfg/` | `json_loader` against fixture configs |
| `tests/unit/probe/` | `probe_instance` decimation, `spsc_queue` semantics |
| `tests/unit/coast_compare/` | The comparator's own unit tests |
| `tests/unit/test_support/` | `blob_writer` round-trip + binary-sidecar dereferencing |
| `tests/unit/io/` | `sample_sequencer` semantics + index-manifest behavior; `test_mnist_io_index_manifest` seam test (manifest order through a real loader); `test_dsprites_io` — an image set the file declares but does not store is refused (#159) |
| `tests/unit/hd_ngm3/` | `frozen_inert` — compute_step-level state-inertness of a frozen ngm3 group; `gpu_readback_faults` *(GPU backend)* — every save readback is required, a failing checkpoint leaves the last good file intact, a failed vis refresh keeps the last mirror |
| `tests/unit/hd_ngm25/test_gpu_readback_faults.cpp` *(GPU backend)* | Failed readbacks on the ngm25 GPU group: save throws, a failed growth read commits nothing (no repack, no upload, the sync retries), a failed vis refresh keeps the last mirror |
| `tests/unit/sim_core/` | `run_manifest` — field contract incl. the `datasets` and `retina` mirrors |
| `tests/unit/retina/` | `test_retina` — parameter validation (sigma bounds, noise floor), filter response (k = 0 / 1, centre-surround sign, luminance, per-channel), divisive normalisation (per-plane RMS, contrast scaling, blank plane, off path bit-identical), ON/OFF split, the entity's input binding and noise floor, the checkpoint configuration record, JSON round-trip |
| `tests/unit/console/` *(COAST_WITH_CONSOLE)* | `test_console_repl` — interpreter lifecycle (`open()` over real data) + the recovery regressions (a parse error preserves earlier declarations; a fresh `#include` in a failing block does not corrupt later cells). `test_console_session` — no-file operation, reset (rebuild + re-open), several files open at once, `.I`, `.x`, declaration persistence. Gated by `if(TARGET coast_console)`; both link `-rdynamic` |
| `tests/unit/h5browse/test_console_api.cpp` | The `open()` file façade — links `coast_lib` only, no LLVM |
| `tests/integration_b/test_b_tier.cpp` (`b_tier.simc_facade`) | The `coast::simc` sim-control façade end-to-end — spawns the built `coast` via `COAST_SIMC_BINARY` |
| `tests/unit/coastd/` | The daemon suite (`unit.coastd.*`): `test_reconcile` (pure state machine), `test_job_store`/`test_job_json` (durable store, quarantine), `test_child_cmd`/`test_line_tailer`/`test_proc_util`/`test_child_proc`/`test_supervisor` (command derivation, deadlines, kill escalation), `test_builder` (real worktrees against this repo), `test_core_lifecycle`/`test_child_exit`/`test_recovery` (full flows, boot orphan detection, crash-loop cap), `test_comms`/`test_console` (drive the **real** `coastd` binary), `test_concurrency`. `run_real_gate.sh` — opt-in end-to-end bit-identity gate vs a direct `coast` run |
| `tests/unit/logging/test_hdf5_shape_guards.cpp` | Crafted checkpoints (#150): both load paths refuse the same malformed files, a refusal leaves no HDF5 handle open, `mem_state_reader` refuses a field of another rank; a dataset declaring more than the file stores is refused (#159), a fully written compressed one loads |
| `tests/unit/gui_replay/` *(COAST_WITH_GUI)* | `test_buffer_log_reader` — crafted buffer logs: malformed or unstored (#159) topology/metadata/data refuses at `open()`, malformed vis hints fall back to defaults, a name filling its slot |
| `tests/unit/gui_plot/` | `test_console_plot` — H5 round-trips, registry create/update/clear, canvas hosting, concurrent update-vs-reader. `test_histogram_ranges` — `bin_points` ranging semantics, the outer-ring contract, writer/reader agreement. `test_plot_manager_deinit` — **own executable**: `deinit()`'s tombstone is process-wide and would poison cases that call `instance()` |
| `tests/unit/app/` | `test_app_config` — app_config defaults, key-wise overlay, strict failure for a named bad file; `test_path_resolver` — the `config_base` policy of `coast::paths` |
| `tests/unit/campaign/` | `test_campaign` — the `coast::campaign` meta-loader over scratch run directories: axis discovery, queries, incl. a manifest from coast's own writer |
| `tests/unit/core/` | `test_demand_published` — the single-slot publisher contract; `test_seed` — the seed rule of `core/seed.h`; `test_step_error` — `COAST_REQUIRE` / `step_error` |
| `tests/unit/diag/` | `test_diag` — sinks, runtime level, log file, no-deadlock reentrancy; `test_progress` — the wire format shared by `format_progress` and `parse_progress` |
| `tests/unit/h5browse/` | `base_histogram`, `chunk_cache`, `dataset_reader`, `file_model` (against real writer output), `malformed` (zero-crash contract), `pairwise`, `row_filter`, `scalar_type`, `scan_engine`, `scan_stats`; `bench_kernel_throughput` — manual benchmark, not a ctest |
| `tests/unit/hd_ngm2/` | `config_record`, `config_validation`, `max_depth` (deepest accepted tree), `noise_gate`, `observe_order`, `pick_winner`, `state_shape` (checkpoint refusal), `window_tracking` (`ngm/window_tracking.h`, shared by all three models) |
| `tests/unit/hd_ngm25/` | The ngm2.5 increments: `arena_consistency`, `repack_splits`, `response`, `adapt`, `branch`, `persistence` (exact resume), `vis_mirror`; plus `config_record`, `config_validation`, `max_depth`, `process_guard` |
| `tests/unit/hdf5/` | `test_executor` — error handling and ordering of `hdf5::executor`; `test_append_rows` — `hdf5::append_rows` |
| `tests/unit/inhibition/` | `inhibition_cfg` (`resolve_params`, `make_inhibitor`, config records), `max_softmax`, `random_index`, `gpu_max_softmax` (CPU↔GPU parity) |
| `tests/unit/logging/` | `buffer_logger_lifecycle`, `buffer_logger_write_errors`, `checkpoint_version`, `config_record`, `mem_state_io`, `rng_state_io`, `state_checks` |
| `tests/unit/observe/` | `config_parse`, `dtype_lock`, `trigger`, `nd_observe`, `chunk_geom`, `writer_roundtrip`; `bench_writer` — micro-benchmark, not a ctest |
| `tests/unit/tools/` | `metric`, `metric_registry`, `distortion`, `ensemble_code`, `histogram`, `sim_builder` (repeated node id), `entity_types` / `entity_registration` |
| `tests/unit/scripts/test_update_golden.sh` | Drives `scripts/update_golden.sh` with a fake capture executable |
| `tests/integration/cmake/` | `coast_add_integration_test` macro implementation, golden-update script wiring |
| `tests/integration/mnist_seed{0,1,2}_baseline/` | One A-tier directory per (config, seed) pair — `main.cpp` + `goldens/` subdir |
| `tests/integration/mnist_ngm3_{philox,cpu_to_gpu,gpu_to_cpu,heterogeneous}/` | The four ngm3 A-tier tests — one per `io_buffer` variant (philox strict baseline, the rest aggregate) |
| `tests/integration/frozen_inert/` | Self-comparing frozen-inertness tests (CPU + GPU): `state(t) == state(t+N)`, no retained goldens; shared helpers in `frozen_inert_helpers.h` |
| `tests/integration/impl_equivalence/` | Cross-implementation differential test (ngm2 / ngm25 / ngm3 CPU, SPEC.md M.0.6): one injected initial state, randomness neutralised, outputs and logical state compared step by step |
| `tests/integration/mnist_ngm25/` | The ngm2.5 A-tier fingerprint test (`configs/mnist_ngm25.json`) — `main.cpp` + `goldens/` |
| `tests/integration/seed_composition/` | Seed composition through the full sim path: effective seed = config seed + `--seed-offset` + node offset, modulo 2³²; self-comparing, no goldens |
| `tests/integration/ngm3_noise_floor/` | Aggregate noise-floor probe across RNG seed offsets — standalone executable, not a ctest |
| `tests/integration/ngm_parity_probe/` | Aggregate-parity measurement for any neuron group (SPEC.md M.0.6 b, M.7.20) — standalone executable, not a ctest |
| `tests/integration_ngm3/` | In-process ngm3/ngm25 tests: `synthetic_smoke`, `branch_timing`, state round-trip/resume/read-page restore regressions, and the GPU cases (`gpu_state_roundtrip`, `gpu_layout_check`, `gpu_forced_rebuild`, `branch_gate_after_load_gpu`, `ngm25_xdev_checkpoint`) |
| `tests/integration_observe/` | `observe_capture_test` — the observe pipeline end to end through a real run; `observe_contract_test` + `observe_contract.json` — the CPU observe contract offline consumers read |
| `tests/integration_headless/run_headless_feed_test.cpp` | A headless run failing by an exception ends its progress feed on exactly one `phase=failed` line |
| `tests/integration_b/test_b_tier.cpp` | All B-tier subprocess cases share this binary |
| `tests/integration_b/observe_dtype_mismatch_main.cpp`, `observe_emit_dtype_mismatch_main.cpp` | Death-test helpers: the observe dtype/shape lock aborts on the `resolve_site()` and on the `emit()` cache-hit path |
| `tests/smoke/test_doctest_link.cpp` | Link sanity; per-component GUI smoke deferred |
| `tests/how_to_run.md` | Operator-facing: build, ctest, filters, golden capture, sanitizer commands |
| `tests/how_to_write.md` | Author-facing: per-tier how-to, conventions, decision tree |
| `scripts/update_golden.sh` | N=5 rotation script invoked by the `coast_update_goldens` CMake target |
| `scripts/migrate_max_branch_level.py` | Renames the pre-#206 tree-depth names (`dendrite_branch_depth`, `max_branch_depth`) to `max_branch_level` in config JSON and in ngm3 checkpoints, which coast otherwise refuses. Dry run by default; `--apply` writes, keeping `<file>.pre-max_branch_level` backups (needs `h5py` for `.h5`) |

## Top-Level

| File | Contains |
|------|----------|
| `src/main.cpp` | ~160 lines (a third of them comments): `coast::cli::parse` → `coast::app::build_env` → dispatch to `run_headless` / `run_gui`. Also the single place that knows a build is headless-only — it gates the GUI entry point and applies the run-length rule to an unmarked run |
| `CMakeLists.txt` | All build targets, SIMD flags, dependencies, the `coast_lib` (headless core) and `coast_gui` (SDL2/OpenGL/ImGui) static libraries, `coast` exe (whole-archive-linked), `COAST_WITH_GUI` option, `BUILD_TESTING` option, `COAST_SANITIZER` option, `coast_tidy` target |
| `CMakePresets.json` | Configure/build presets: `macos`, `macos-tests` (pinned Homebrew LLVM toolchain, Darwin-gated), `headless`, `headless-tests` (`COAST_WITH_GUI=OFF`) |
| `.clang-tidy` | Conservative tidy rule subset (bugprone-* + performance-* + readability-redundant-* + modernize-use-nullptr) |
| `SANITIZERS.md` | How to invoke ASan/UBSan/TSan; FP-drift caveat; TSan triage notes |
| `docs/macos_validation.md` | Apple Metal validation procedure — 6 incremental GPU probes + A-tier suite, expected platform differences, result template |
