cmake_minimum_required(VERSION 3.15)
project(coast)

find_package(raylib 3.0 REQUIRED)
find_package(TBB REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)

if (APPLE)
    add_compile_options(-march=native -ffast-math -Wall -Wpedantic -Wextra -stdlib=libstdc++)  # libstdc++ for gcc, libc++ for clang
else()
    add_compile_options(-march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl -ffast-math -Wall -Wpedantic -Wextra)
endif()

add_executable(coast src/main.cpp
        src/gui_core/ray_app.cpp
        include/gui_core/ray_app.h
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
        3rd_party/rlImGui/imgui_impl_raylib.h
        3rd_party/rlImGui/rlImGui.cpp
        3rd_party/rlImGui/rlImGui.h
        3rd_party/rlImGui/rlImGuiColors.h
        src/gui_vis/fbgd.cpp
        include/gui_vis/fbgd.h
        src/gui_vis/ngm_vis.cpp
        include/gui_vis/ngm_vis.h
        include/tools/mnist_db.h
        src/tools/mnist_db.cpp
        include/hd_ngm2/hd_ngm2_dendrite.h
        src/hd_ngm2/hd_ngm2_dendrite.cpp
        include/hd_ngm2/hd_ngm2_neuron.h
        src/hd_ngm2/hd_ngm2_neuron.cpp
        include/hd_ngm2/hd_ngm2_tools.h
        include/hd_ngm2/hd_ngm2_neuron_group.h
        src/hd_ngm2/hd_ngm2_neuron_group.cpp
        include/hd_ngm2/hd_ngm2.h
        include/sim_core/io_entity.h
        include/hd_ngm2/hd_ngm2_cfg.h
        src/sim_core/sim_env.cpp
        include/sim_core/sim_env.h
        src/sim_core/io_buffer.cpp
        include/sim_core/io_buffer.h
        src/tools/mnist_io.cpp
        include/tools/mnist_io.h
        src/gui_vis/vec_ring_buffer.cpp
        include/gui_vis/vec_ring_buffer.h
        src/gui_vis/vec_vis.cpp
        include/gui_vis/vec_vis.h
        include/gui_vis/gradient.h
        src/gui_vis/vec_group_vis.cpp
        include/gui_vis/vec_group_vis.h
        src/gui_vis/ngm_flat_vis.cpp
        include/gui_vis/ngm_flat_vis.h
)

target_include_directories(coast PRIVATE
        3rd_party/imgui
        3rd_party/rlImGui
        include/gui_core
        include/gui_vis
        include/hd_ngm2
        include/sim_core
        include/tools
if(APPLE)
        /opt/homebrew/include/c++/15
endif()
)

target_link_libraries(coast PRIVATE TBB::tbb raylib)

if (APPLE)
    target_link_libraries(coast PRIVATE "-framework IOKit")
    target_link_libraries(coast PRIVATE "-framework Cocoa")
    target_link_libraries(coast PRIVATE "-framework OpenGL")
endif()
