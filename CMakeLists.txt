cmake_minimum_required(VERSION 3.15)
project(coast)

# Dependencies
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(TBB REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# nlohmann/json - header-only, use submodule directly

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)

if (APPLE)
    add_compile_options(-march=native -Wall -Wpedantic -Wextra -stdlib=libstdc++)
else()
    add_compile_options(-march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl -Wall -Wpedantic -Wextra)
endif()

# ImPlot library
add_library(implot STATIC
    3rd_party/implot/implot.cpp
    3rd_party/implot/implot_items.cpp
)
target_include_directories(implot PUBLIC 3rd_party/implot 3rd_party/imgui)

# ImNodes library
add_library(imnodes STATIC
    3rd_party/imnodes/imnodes.cpp
)
target_include_directories(imnodes PUBLIC 3rd_party/imnodes 3rd_party/imgui)

# ImNodeFlow library
add_library(ImNodeFlow STATIC
    3rd_party/ImNodeFlow/src/ImNodeFlow.cpp
)
target_include_directories(ImNodeFlow PUBLIC
    3rd_party/ImNodeFlow/include
    3rd_party/ImNodeFlow/src
    3rd_party/imgui
)
target_compile_definitions(ImNodeFlow PUBLIC IMGUI_DEFINE_MATH_OPERATORS)

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
        # Network editor
        src/gui_core/network_editor.cpp
        include/gui_core/network_editor.h
        # Network editor 2 (ImNodeFlow-based)
        src/gui_core/network_editor2.cpp
        include/gui_core/network_editor2.h
        # Simulation instance (RAII wrapper)
        src/gui_core/sim_instance.cpp
        include/gui_core/sim_instance.h

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

        # OpenGL texture wrapper
        src/gui_vis/gl_texture.cpp
        include/gui_vis/gl_texture.h
        include/gui_vis/color_types.h

        # Visualization components
        src/gui_vis/fbgd.cpp
        include/gui_vis/fbgd.h
        src/gui_vis/ngm_vis.cpp
        include/gui_vis/ngm_vis.h
        src/gui_vis/vec_ring_buffer.cpp
        include/gui_vis/vec_ring_buffer.h
        src/gui_vis/vec_vis.cpp
        include/gui_vis/vec_vis.h
        include/gui_vis/gradient.h
        src/gui_vis/vec_group_vis.cpp
        include/gui_vis/vec_group_vis.h
        src/gui_vis/ngm_flat_vis.cpp
        include/gui_vis/ngm_flat_vis.h

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
        include/gui_vis/gui_vis_cfg.h
        include/json_cfg/gui_vis_json.h
)

target_include_directories(coast PRIVATE
        3rd_party/imgui
        3rd_party/imgui/backends
        3rd_party/implot
        3rd_party/imnodes
        3rd_party/ImNodeFlow/include
        3rd_party/ImNodeFlow/src
        3rd_party/imgui-filebrowser
        include/gui_core
        include/gui_vis
        include/hd_ngm2
        include/sim_core
        include/tools
        include/io
        include/json_cfg
        ${SDL2_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
        3rd_party/json/include
)

if(APPLE)
    target_include_directories(coast PRIVATE /opt/homebrew/include/c++/15)
endif()

target_link_libraries(coast PRIVATE
    TBB::tbb
    SDL2::SDL2
    OpenGL::GL
    ${HDF5_C_LIBRARIES}
    implot
    imnodes
    ImNodeFlow
)
target_compile_definitions(coast PRIVATE IMGUI_DEFINE_MATH_OPERATORS)

if (APPLE)
    target_link_libraries(coast PRIVATE "-framework IOKit")
    target_link_libraries(coast PRIVATE "-framework Cocoa")
endif()
