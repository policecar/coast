cmake_minimum_required(VERSION 3.15)
project(coast)

# Export compile_commands.json so clangd knows include paths and compiler flags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(TBB REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# nlohmann/json - header-only, use submodule directly

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)

# Architecture-specific optimizations (applied globally)
if (APPLE)
<<<<<<< HEAD
    add_compile_options(-march=native -Wall -Wpedantic -Wextra -Wno-psabi -stdlib=libstdc++)
else()
    add_compile_options(-march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl -Wall -Wpedantic -Wextra -Wno-psabi)
=======
    add_compile_options(-march=native -stdlib=libstdc++)
else()
    add_compile_options(-march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl)
>>>>>>> 9bdb24b (build: suppress warnings for 3rd-party libraries)
endif()

# ImPlot library (3rd-party, warnings disabled)
add_library(implot STATIC
    3rd_party/implot/implot.cpp
    3rd_party/implot/implot_items.cpp
)
target_include_directories(implot PUBLIC 3rd_party/implot 3rd_party/imgui)
target_compile_options(implot PRIVATE -w)

# ImNodeFlow library (3rd-party, warnings disabled)
add_library(ImNodeFlow STATIC
    3rd_party/ImNodeFlow/src/ImNodeFlow.cpp
)
target_include_directories(ImNodeFlow PUBLIC
    3rd_party/ImNodeFlow/include
    3rd_party/ImNodeFlow/src
    3rd_party/imgui
)
target_compile_definitions(ImNodeFlow PUBLIC IMGUI_DEFINE_MATH_OPERATORS)
target_compile_options(ImNodeFlow PRIVATE -w)

add_executable(coast
        # Main
        src/main.cpp

        # SDL2 application framework
        src/gui_core/sdl_app.cpp
        include/gui_core/sdl_app.h
        # GUI manager (menu bar, dialogs)
        src/gui_core/gui_manager.cpp
        include/gui_core/gui_manager.h
        # Configuration editor
        src/gui_core/config_editor.cpp
        include/gui_core/config_editor.h
        # Network editor (ImNodeFlow-based)
        src/gui_core/network_editor.cpp
        include/gui_core/network_editor.h
        # Simulation instance (RAII wrapper)
        src/gui_core/sim_instance.cpp
        include/gui_core/sim_instance.h
        # Browser windows for introspection
        src/gui_core/entity_browser.cpp
        include/gui_core/entity_browser.h
        src/gui_core/buffer_browser.cpp
        include/gui_core/buffer_browser.h
        src/gui_core/logging_window.cpp
        include/gui_core/logging_window.h

        # Dear ImGui core
        3rd_party/imgui/imconfig.h
        3rd_party/imgui/imgui.cpp
        3rd_party/imgui/imgui.h
        3rd_party/imgui/imgui_demo.cpp
        3rd_party/imgui/imgui_draw.cpp
        3rd_party/imgui/imgui_internal.h
        3rd_party/imgui/imgui_tables.cpp
        3rd_party/imgui/imgui_widgets.cpp
        3rd_party/imgui/imstb_rectpack.h
        3rd_party/imgui/imstb_textedit.h
        3rd_party/imgui/imstb_truetype.h

        # Dear ImGui backends for SDL2 + OpenGL3
        3rd_party/imgui/backends/imgui_impl_sdl2.cpp
        3rd_party/imgui/backends/imgui_impl_sdl2.h
        3rd_party/imgui/backends/imgui_impl_opengl3.cpp
        3rd_party/imgui/backends/imgui_impl_opengl3.h
        3rd_party/imgui/backends/imgui_impl_opengl3_loader.h

        # Visualization core
        src/gui_vis/core/gl_texture.cpp
        include/gui_vis/core/gl_texture.h
        include/gui_vis/core/color_types.h
        include/gui_vis/core/colormap.h
        src/gui_vis/core/colormap.cpp
        include/gui_vis/core/gradient.h
        include/gui_vis/core/vis_hints.h

        # Visualization primitives
        src/gui_vis/primitives/vec_vis.cpp
        include/gui_vis/primitives/vec_vis.h
        src/gui_vis/primitives/vec_group_vis.cpp
        include/gui_vis/primitives/vec_group_vis.h

        # IO buffer visualizations
        src/gui_vis/io_buffer/vec_ring_buffer.cpp
        include/gui_vis/io_buffer/vec_ring_buffer.h

        # Visualization layout
        src/gui_vis/layout/fbgd.cpp
        include/gui_vis/layout/fbgd.h

        # Sim entity visualizations
        src/gui_vis/sim_entity/neuron_group/ngm_flat_vis.cpp
        include/gui_vis/sim_entity/neuron_group/ngm_flat_vis.h
        src/gui_vis/sim_entity/neuron_group/ngm_tree_vis.cpp
        include/gui_vis/sim_entity/neuron_group/ngm_tree_vis.h

        # Visualization registry
        src/gui_vis/vis_registry.cpp
        include/gui_vis/vis_registry.h

        # Dataset I/O
        include/io/mnist_db.h
        src/io/mnist_db.cpp
        src/io/mnist_io.cpp
        include/io/mnist_io.h
        include/io/dsprites_io.h
        src/io/dsprites_io.cpp
        include/io/mnist_cfg.h
        include/io/dataset_cfg.h
        include/io/dsprites_cfg.h

        # HD-NGM2 neural model
        include/hd_ngm2/hd_ngm2_dendrite.h
        src/hd_ngm2/hd_ngm2_dendrite.cpp
        include/hd_ngm2/hd_ngm2_neuron.h
        src/hd_ngm2/hd_ngm2_neuron.cpp
        include/hd_ngm2/hd_ngm2_tools.h
        include/hd_ngm2/hd_ngm2_neuron_group.h
        src/hd_ngm2/hd_ngm2_neuron_group.cpp
        include/hd_ngm2/hd_ngm2.h
        include/hd_ngm2/hd_ngm2_cfg.h

        # Simulation core
        include/sim_core/io_entity.h
        src/sim_core/sim_env.cpp
        include/sim_core/sim_env.h
        src/sim_core/io_buffer.cpp
        include/sim_core/io_buffer.h
        src/sim_core/sim_thread_controller.cpp
        include/sim_core/sim_thread_controller.h

        # JSON configuration
        include/json_cfg/json_loader.h
        src/json_cfg/json_loader.cpp
        include/json_cfg/ngm2_json.h
        include/json_cfg/sim_json.h
        include/json_cfg/dataset_json.h

        # Tools
        include/tools/sim_builder_cfg.h
        include/tools/sim_builder.h
        src/tools/sim_builder.cpp
        include/tools/sim_builder_interface.h
        include/json_cfg/gui_vis_json.h
)

# 3rd-party includes (marked as SYSTEM to suppress warnings)
target_include_directories(coast SYSTEM PRIVATE
        3rd_party/imgui
        3rd_party/imgui/backends
        3rd_party/implot
        3rd_party/ImNodeFlow/include
        3rd_party/ImNodeFlow/src
        3rd_party/imgui-filebrowser
        3rd_party/json/include
        ${SDL2_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
)

# Project includes (warnings enabled)
target_include_directories(coast PRIVATE
        include/gui_core
        include/gui_vis
        include/hd_ngm2
        include/sim_core
        include/tools
        include/io
        include/json_cfg
)

# Enable warnings for our own code
target_compile_options(coast PRIVATE -Wall -Wpedantic -Wextra)

if(APPLE)
    target_include_directories(coast PRIVATE /opt/homebrew/include/c++/15)
endif()

target_link_libraries(coast PRIVATE
    TBB::tbb
    SDL2::SDL2
    OpenGL::GL
    ${HDF5_C_LIBRARIES}
    implot
    ImNodeFlow
)
target_compile_definitions(coast PRIVATE IMGUI_DEFINE_MATH_OPERATORS)

if (APPLE)
    target_link_libraries(coast PRIVATE "-framework IOKit")
    target_link_libraries(coast PRIVATE "-framework Cocoa")
endif()
