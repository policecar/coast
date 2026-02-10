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
| `include/core/probe_name_registry.h` | `probe_name_registry` — global static `type_index → set<string>` for probe name discovery |
| `include/core/rng/fast_rng.h` | Type aliases: `fast_rng_t` = `rng_t<mt19937_simd>`, `standard_rng_t` |
| `include/core/rng/mt19937_simd.h` | `mt19937_simd` — SIMD Mersenne Twister (AVX-512/AVX2/NEON/scalar), 624-element state, bulk temper |
| `include/core/rng/rng_wrapper.h` | `rng_t<Generator>` — thin generic wrapper, conforms to UniformRandomBitGenerator |
| `src/core/entity_element.cpp` | `probe_slow_path()` — hash map lookup on the sim thread |
| `src/core/structure_registry.cpp` | `build()`, `build_recursive()`, path computation, element_map population |
| `src/core/probe_name_registry.cpp` | Mutex-guarded register/get operations on the global registry |

## sim_core/

| File | Contains |
|------|----------|
| `include/sim_core/sim_env.h` | `sim::env` — entity container, IO buffer container, hook system (4 hook points), `process()`, `swap_io()`. Also `entity_vec`, `typed_entity_vec<T>` type erasure |
| `include/sim_core/io_entity.h` | `io_entity` interface — `process()`, `get_outp_id/size()`, `get_inp_ids()`, `set_outp_func/inp_func()`, `save/load_state()`. Types: `output_id_t`, `output_channel_t` |
| `include/sim_core/sim_thread_controller.h` | `sim_thread_controller` — `jthread`, `sim_mode` enum (paused/throttled/unlimited), atomics for mode/sps/step_count, `do_step()`, `thread_main()` |
| `src/sim_core/sim_env.cpp` | `emplace_back()`, `iterate_entities()`, `init_io_buffers()` (2-pass wiring), `process_pending_hooks()`, hook registration |
| `src/sim_core/sim_thread_controller.cpp` | `thread_main()` — main loop with mode switch, SPS measurement (EMA/500ms), `do_step()` |

## hd_ngm2/

| File | Contains |
|------|----------|
| `include/hd_ngm2/hd_ngm2_neuron_group.h` | `neuron_group_t` — implements `io_entity`, holds `vector<neuron_t>`, `process()` with competitive learning |
| `include/hd_ngm2/hd_ngm2_neuron.h` | `neuron_t` — `entity_element`, holds `vector<dendrite_t>`, `get_response()`, `adapt()`, learning window |
| `include/hd_ngm2/hd_ngm2_dendrite.h` | `dendrite_t` — `entity_element`, `synapses_t` SOA, segment tree, `get_response()`, `adapt_synapses()`, `adapt_branches()` |
| `include/hd_ngm2/hd_ngm2_cfg.h` | Config structs: `dendrite_params_t`, `neuron_params_t`, `neuron_group_params_t`, `cfg_store`, `dendrite_type_t` enum, `learning_window_t` |
| `include/hd_ngm2/hd_ngm2_tools.h` | Utilities: `local_inhibition()`, `normalized_shannon_entropy()`, `sigmoid()`, `softmax()`, `normalize()` |
| `src/hd_ngm2/hd_ngm2_neuron_group.cpp` | `process()` implementation — response, inhibition, stochastic winner, primary + common adaptation |
| `src/hd_ngm2/hd_ngm2_neuron.cpp` | `get_response()` — apical×proximal+noise; `adapt()` — learning window filter, delegate to dendrites |
| `src/hd_ngm2/hd_ngm2_dendrite.cpp` | `get_response()` — permanence gating, tree propagation, NSE attenuation; `adapt_synapses()` — Hebbian update; `adapt_branches()` — mismatch-driven tree growth |

## io/

| File | Contains |
|------|----------|
| `include/io/dataset_cfg.h` | Common dataset types |
| `include/io/mnist_cfg.h` | `mnist_param_t` — image/label paths, change_interval, noise_range, seed |
| `include/io/mnist_io.h` | `mnist_io` — implements `io_entity`, inherits `mnist_db` for IDX file reading |
| `include/io/emnist_cfg.h` / `emnist_io.h` | Extended MNIST (same format as MNIST) |
| `include/io/kmnist_cfg.h` / `kmnist_io.h` | Kuzushiji MNIST (same format as MNIST) |
| `include/io/cifar10_cfg.h` / `cifar10_io.h` | CIFAR-10 — 32x32x3 RGB, binary batch format, planar→interleaved conversion |
| `include/io/cifar100_cfg.h` / `cifar100_io.h` | CIFAR-100 — optional coarse/fine labels |
| `include/io/dsprites_cfg.h` / `dsprites_io.h` | dSprites — 64x64x1 from HDF5, optional randomized readout |
| `include/io/chairs3d_cfg.h` / `chairs3d_io.h` | 3D Chairs — configurable image dimensions, loaded via stb_image |
| `include/io/color_mnist_cfg.h` / `color_mnist_io.h` | Colorized MNIST — 28x28x3 |
| `src/io/mnist_db.cpp` | IDX binary format reader (big-endian), `get_norm_image()` |
| `src/io/mnist_io.cpp` | MNIST `process()` implementation |
| `src/io/cifar10_io.cpp` | CIFAR-10 binary loading, RGB conversion |
| `src/io/dsprites_io.cpp` | HDF5 loading with RAII handles |

## json_cfg/

| File | Contains |
|------|----------|
| `include/json_cfg/json_loader.h` | `json_loader` class, `global_cfg_t` struct |
| `include/json_cfg/dataset_json.h` | `dataset::cfg_store`, nlohmann macros for all dataset param types |
| `include/json_cfg/ngm2_json.h` | `ngm2::cfg_store`, nlohmann macros for dendrite/neuron/group params |
| `include/json_cfg/sim_json.h` | `sim_cfg_t`, `sim_node_t`, `sim_node_set_t`, `sim_node_delta_t`, entity type enum serialization |
| `include/json_cfg/gui_vis_json.h` | `vis_hints_t` and all hint sub-structs, nlohmann macros |
| `src/json_cfg/json_loader.cpp` | File I/O, `global_cfg = json;` deserialization, snapshot_dir handling |

## tools/

| File | Contains |
|------|----------|
| `include/tools/sim_builder.h` | `sim_builder` — create/init function registries, node map |
| `include/tools/sim_builder_cfg.h` | `sim_node_t`, `sim_node_set_t`, `sim_node_delta_t`, `io_entity_type_t` enum |
| `include/tools/sim_builder_interface.h` | `register_entity_handlers()` — factory registration for all entity types |
| `include/tools/inc_on_exit.h` | `inc_on_exit` RAII counter incrementer |
| `src/tools/sim_builder.cpp` | `build_local()` — 5-phase construction algorithm |

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

## logging/

| File | Contains |
|------|----------|
| `include/logging/logging.h` | Convenience include for the logging subsystem |
| `include/logging/logging_common.h` | `log_result`, `log_stats`, `compression_config`, `chunk_config` |
| `include/logging/hdf5_utils.h` | RAII wrappers: `hdf5::file`, `group`, `dataspace`, `plist`, `dataset`. Free functions: `write/read_attr`, `link/attr_exists` |
| `include/logging/logging_manager.h` | `logging_manager` — owns buffer_logger, probe_logger, state_logger |
| `include/logging/buffer_logger.h` | `buffer_logger` — post_swap hook, frame buffering, chunked HDF5 writes, topology extraction |
| `include/logging/probe_logger.h` | `probe_logger` — file observer management, HDF5/CSV output, buffered writes |
| `include/logging/state_logger.h` | `state_logger` — save/load full simulation state |
| `include/logging/state_io.h` | `state_writer`/`state_reader` abstract interfaces, `state_serializable` |
| `include/logging/file_observer.h` | `file_observer` — `probe_observer` that delegates to `probe_logger` |
| `src/logging/buffer_logger.cpp` | `capture_frame()`, `write_frames()`, HDF5 schema creation, topology extraction |
| `src/logging/probe_logger.cpp` | `record_value()`, `flush_hdf5/csv()`, `add/remove_probe()` |
| `src/logging/state_logger.cpp` | `hdf5_state_writer`/`hdf5_state_reader`, `save/load_state()` |
| `src/logging/hdf5_utils.cpp` | Free function implementations for attribute read/write |
| `src/logging/logging_manager.cpp` | Lifecycle management, combined stats |
| `src/logging/file_observer.cpp` | `push()` delegates to `probe_logger::record_value()` |

## gui_core/

| File | Contains |
|------|----------|
| `include/gui_core/gui_manager.h` | `gui_manager` — menu bar, dialogs, browsers, editors, auto-save, snapshot resume |
| `include/gui_core/sim_instance.h` | `sim_instance` — wraps json_loader + env + builder + thread_controller + logging_manager |
| `include/gui_core/entity_browser.h` | `entity_browser` — sortable table, detail panel, vis checkbox management |
| `include/gui_core/buffer_browser.h` | `buffer_browser` — buffer table with live stats, vis management |
| `include/gui_core/dataset_browser.h` | `dataset_browser` — virtualized thumbnail grid, dataset selector |
| `include/gui_core/config_editor.h` | `config_editor` — full global_cfg_t editing, dirty tracking, save/restore |
| `include/gui_core/network_editor.h` | `network_editor` — ImNodeFlow graph editor for simulation topology |
| `include/gui_core/logging_window.h` | `logging_window` — buffer/probe/state logging controls, auto-checkpoint |
| `include/gui_core/imgui_scaling.h` | `scaled(float)` — HiDPI scaling helper |
| `src/gui_core/sdl_app.cpp` | SDL2 + OpenGL init, ImGui init, main event loop, shutdown |
| `src/gui_core/gui_manager.cpp` | Menu rendering, control dialog, status bar, set_sim_instance(), auto-save |
| `src/gui_core/sim_instance.cpp` | Simulation construction, move semantics |
| `src/gui_core/entity_browser.cpp` | Table rendering, sorting, filtering, vis management |
| `src/gui_core/buffer_browser.cpp` | Buffer table, live stats, vis management |
| `src/gui_core/dataset_browser.cpp` | Thumbnail rendering, virtualized scrolling, sample conversion |
| `src/gui_core/config_editor.cpp` | 2300+ lines: section renderers for all config types |
| `src/gui_core/network_editor.cpp` | ImNodeFlow integration, shadow state, edit callbacks |
| `src/gui_core/logging_window.cpp` | File browser integration, auto-checkpoint implementation |

## gui_vis/

| File | Contains |
|------|----------|
| `include/gui_vis/vis_registry.h` | `vis_registry` singleton, `vis_info`, `vis_instance` base, factory types, registration API |
| `include/gui_vis/core/color_types.h` | `color_t` RGBA struct, color constants |
| `include/gui_vis/core/colormap.h` | `colormap_t` enum (grayscale, viridis, plasma, etc.), `apply_colormap()` |
| `include/gui_vis/core/gl_texture.h` | `gl_texture` RAII wrapper — `update()`, `as_imgui_texture()`, `release()` |
| `include/gui_vis/core/gradient.h` | `gradient<N>` — N-stop linear color interpolation |
| `include/gui_vis/core/vis_hints.h` | `vis_hints_t`, `ngm_flat_vis_hint_t`, `ngm_tree_vis_hint_t`, dataset hints, buffer_vis hints |
| `include/gui_vis/primitives/vec_vis.h` | `vec_vis` — float vector → 2D pixel grid |
| `include/gui_vis/primitives/vec_group_vis.h` | `vec_group_vis` — layout manager for multiple vec_vis |
| `include/gui_vis/io_buffer/vec_ring_buffer.h` | `vec_ring_buffer` — scrolling time-series texture |
| `include/gui_vis/layout/fbgd.h` | `fbgd` — grid-based force-directed layout for ngm_tree_vis |
| `include/gui_vis/sim_entity/neuron_group/ngm_flat_vis.h` | `ngm_flat_vis` — "Dendritic Representations" grid view |
| `include/gui_vis/sim_entity/neuron_group/ngm_tree_vis.h` | `ngm_tree_vis` — "Neuron Tree" force-directed tree |
| `src/gui_vis/vis_registry.cpp` | Registry singleton, registration, factory invocation |
| `src/gui_vis/core/colormap.cpp` | Piecewise 5-stop interpolation for viridis/plasma/inferno |
| `src/gui_vis/core/gl_texture.cpp` | OpenGL texture creation, update, release |
| `src/gui_vis/io_buffer/vec_ring_buffer.cpp` | Ring buffer rendering, 3-state atomic FSM adapter |
| `src/gui_vis/layout/fbgd.cpp` | Grid-based force simulation, draw callback |
| `src/gui_vis/sim_entity/neuron_group/ngm_flat_vis.cpp` | 545 lines: texture grid, zoom/pan canvas, colormap selector |
| `src/gui_vis/sim_entity/neuron_group/ngm_tree_vis.cpp` | 1001 lines: tree model, force layout, adaptive radii, ImDrawList rendering |

## gui_probe/

| File | Contains |
|------|----------|
| `include/gui_probe/probe_browser.h` | `probe_browser` — tree view, observer window management, CSV integration |
| `include/gui_probe/observer_panel.h` | `observer_panel` — single-window view of all active observers |
| `src/gui_probe/probe_browser.cpp` | Tree rendering, observer popup, window lifecycle |
| `src/gui_probe/observer_panel.cpp` | Collapsible headers, observer remove buttons |

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

## Top-Level

| File | Contains |
|------|----------|
| `src/main.cpp` | CLI parsing (`cli_options`), `run_headless()`, GUI mode setup, callback registration |
| `CMakeLists.txt` | All build targets, SIMD flags, dependencies, module organization |
