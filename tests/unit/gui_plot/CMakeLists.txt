# gui_plot unit tests — data/registry layer of the ImPlot console bridge.
# Drawing itself still needs a live GUI context and is verified by hand, but logic
# kept OUT of the painters (e.g. canvas_view's transform) is testable here.

# The canvas's pan/zoom transform: anchored zoom, clamping, world/screen mapping.
add_executable(test_canvas_view test_canvas_view.cpp)
target_link_libraries(test_canvas_view PRIVATE doctest::doctest coast_gui)
add_test(NAME unit.gui_plot.canvas_view COMMAND test_canvas_view)

# canvas::paint_body driven against a null-backend ImGui context: grid layout,
# zoom wiring, culling. Needs imgui_internal.h for the resulting window rects.
add_executable(test_canvas_layout test_canvas_layout.cpp)
target_link_libraries(test_canvas_layout PRIVATE doctest::doctest coast_gui)
target_include_directories(test_canvas_layout PRIVATE ${CMAKE_SOURCE_DIR}/3rd_party/imgui)
add_test(NAME unit.gui_plot.canvas_layout COMMAND test_canvas_layout)

add_executable(test_console_plot test_console_plot.cpp)
target_link_libraries(test_console_plot PRIVATE doctest::doctest coast_gui)
add_test(NAME unit.gui_plot.console_plot COMMAND test_console_plot)

# Fixed histogram domains: bin_points' ranging/outer-ring semantics + the
# save/load round-trip of the domains.
add_executable(test_histogram_ranges test_histogram_ranges.cpp)
target_link_libraries(test_histogram_ranges PRIVATE doctest::doctest coast_gui)
add_test(NAME unit.gui_plot.histogram_ranges COMMAND test_histogram_ranges)

# The render cache: invalidation on update()/load and the candle stats it holds.
add_executable(test_plot_cache test_plot_cache.cpp)
target_link_libraries(test_plot_cache PRIVATE doctest::doctest coast_gui)
add_test(NAME unit.gui_plot.plot_cache COMMAND test_plot_cache)

# The plot_manager deinit() tombstone is process-global, so it lives in its own
# executable — it cannot share a process with the cases above.
add_executable(test_plot_manager_deinit test_plot_manager_deinit.cpp)
target_link_libraries(test_plot_manager_deinit PRIVATE doctest::doctest coast_gui)
add_test(NAME unit.gui_plot.plot_manager_deinit COMMAND test_plot_manager_deinit)

# draw_candles must frame its axes on the figure's declared domain (like the
# heatmap path does), not on whichever columns happen to hold data.
add_executable(test_candle_axes test_candle_axes.cpp)
target_link_libraries(test_candle_axes PRIVATE doctest::doctest coast_gui)
target_include_directories(test_candle_axes PRIVATE ${CMAKE_SOURCE_DIR}/3rd_party/implot
                                                   ${CMAKE_SOURCE_DIR}/3rd_party/imgui)
add_test(NAME unit.gui_plot.candle_axes COMMAND test_candle_axes)
