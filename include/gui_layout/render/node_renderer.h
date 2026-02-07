//
// Abstract base class for node rendering in graph visualization
//

#ifndef COAST_NODE_RENDERER_H
#define COAST_NODE_RENDERER_H

#include "gui_layout/layout_graph.h"
#include "imgui.h"

namespace coast::layout::render {

/// Abstract base class for rendering nodes in a graph visualization
class node_renderer {
public:
    virtual ~node_renderer() = default;

    /// Render a single node at screen position
    /// @param draw_list ImGui draw list for rendering
    /// @param node The node to render
    /// @param screen_pos Center position in screen coordinates
    /// @param zoom Current zoom level
    /// @param selected Whether this node is selected
    /// @param hovered Whether this node is hovered
    virtual void render(ImDrawList* draw_list,
                        const node_t& node,
                        ImVec2 screen_pos,
                        float zoom,
                        bool selected,
                        bool hovered) = 0;

    /// Get the visual size of a node for hit testing
    /// @param node The node
    /// @param zoom Current zoom level
    /// @return Size in screen coordinates
    virtual ImVec2 get_size(const node_t& node, float zoom) = 0;
};

/// Simple debug renderer that draws circles with optional labels
class debug_node_renderer : public node_renderer {
public:
    void render(ImDrawList* draw_list,
                const node_t& node,
                ImVec2 screen_pos,
                float zoom,
                bool selected,
                bool hovered) override;

    ImVec2 get_size(const node_t& node, float zoom) override;

    // Configuration
    ImU32 fill_color = IM_COL32(60, 60, 80, 200);
    ImU32 border_color = IM_COL32(100, 100, 140, 255);
    ImU32 selected_color = IM_COL32(255, 200, 80, 255);
    ImU32 hover_color = IM_COL32(150, 150, 200, 255);
    ImU32 text_color = IM_COL32(255, 255, 255, 255);
    bool show_labels = true;
};

} // namespace coast::layout::render

#endif // COAST_NODE_RENDERER_H
