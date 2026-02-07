//
// Node renderer implementations
//

#include "gui_layout/render/node_renderer.h"

#include <cstdio>

namespace coast::layout::render {

void debug_node_renderer::render(ImDrawList* draw_list,
                                  const node_t& node,
                                  ImVec2 screen_pos,
                                  float zoom,
                                  bool selected,
                                  bool hovered) {
    float radius = node.radius * zoom;

    // Draw filled circle
    ImU32 fill = fill_color;
    if (hovered) {
        fill = hover_color;
    }
    draw_list->AddCircleFilled(screen_pos, radius, fill);

    // Draw border
    ImU32 border = border_color;
    float thickness = 2.0f;
    if (selected) {
        border = selected_color;
        thickness = 3.0f;
    } else if (hovered) {
        border = hover_color;
        thickness = 2.5f;
    }
    draw_list->AddCircle(screen_pos, radius, border, 0, thickness);

    // Draw label
    if (show_labels && zoom > 0.5f) {
        char label[32];
        std::snprintf(label, sizeof(label), "%u", node.id);
        ImVec2 text_size = ImGui::CalcTextSize(label);
        ImVec2 text_pos = ImVec2(screen_pos.x - text_size.x * 0.5f,
                                  screen_pos.y - text_size.y * 0.5f);
        draw_list->AddText(text_pos, text_color, label);
    }
}

ImVec2 debug_node_renderer::get_size(const node_t& node, float zoom) {
    float size = node.radius * zoom * 2.0f;
    return ImVec2(size, size);
}

} // namespace coast::layout::render
