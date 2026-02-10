//
// ImGui canvas widget for rendering graphs with zoom/pan/interaction
//

#include "gui_layout/render/graph_canvas.h"

#include <algorithm>
#include <cmath>

namespace coast::layout::render {

graph_canvas::graph_canvas() {
    target_pan_ = pan_;
    target_zoom_ = zoom_;
}

void graph_canvas::set_graph(layout_graph* graph) {
    // Only reset selection state when graph actually changes
    if (graph_ != graph) {
        graph_ = graph;
        selected_ = std::nullopt;
        hovered_ = std::nullopt;
        dragging_ = std::nullopt;
    }
}

void graph_canvas::set_node_renderer(node_renderer* renderer) {
    renderer_ = renderer;
}

bool graph_canvas::render(const char* label, ImVec2 size) {
    bool modified = false;

    // Get available region
    ImVec2 available = ImGui::GetContentRegionAvail();
    if (size.x <= 0) size.x = available.x;
    if (size.y <= 0) size.y = available.y;

    // Create child window for the canvas
    ImGui::BeginChild(label, size, ImGuiChildFlags_Border,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    canvas_pos_ = ImGui::GetCursorScreenPos();
    canvas_size_ = ImGui::GetContentRegionAvail();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Background
    draw_list->AddRectFilled(canvas_pos_,
                              ImVec2(canvas_pos_.x + canvas_size_.x,
                                     canvas_pos_.y + canvas_size_.y),
                              IM_COL32(30, 30, 35, 255));

    // Draw grid
    float grid_step = 50.0f * zoom_;
    if (grid_step > 10.0f) {
        ImU32 grid_color = IM_COL32(50, 50, 55, 100);
        ImVec2 grid_offset = ImVec2(
            std::fmod(pan_.x * zoom_, grid_step),
            std::fmod(pan_.y * zoom_, grid_step)
        );

        for (float x = grid_offset.x; x < canvas_size_.x; x += grid_step) {
            draw_list->AddLine(
                ImVec2(canvas_pos_.x + x, canvas_pos_.y),
                ImVec2(canvas_pos_.x + x, canvas_pos_.y + canvas_size_.y),
                grid_color
            );
        }
        for (float y = grid_offset.y; y < canvas_size_.y; y += grid_step) {
            draw_list->AddLine(
                ImVec2(canvas_pos_.x, canvas_pos_.y + y),
                ImVec2(canvas_pos_.x + canvas_size_.x, canvas_pos_.y + y),
                grid_color
            );
        }
    }

    // Set up clipping
    draw_list->PushClipRect(canvas_pos_,
                             ImVec2(canvas_pos_.x + canvas_size_.x,
                                    canvas_pos_.y + canvas_size_.y),
                             true);

    // Update animations
    update_animation();

    // Handle interactions
    ImGui::SetCursorScreenPos(canvas_pos_);
    ImGui::InvisibleButton("canvas_interaction", canvas_size_,
                            ImGuiButtonFlags_MouseButtonLeft |
                            ImGuiButtonFlags_MouseButtonMiddle |
                            ImGuiButtonFlags_MouseButtonRight);
    bool canvas_hovered = ImGui::IsItemHovered();
    handle_interaction(canvas_hovered);

    // Render
    if (graph_) {
        render_edges(draw_list);
        render_nodes(draw_list);
    }

    draw_list->PopClipRect();

    // Check if any node was dragged
    if (dragging_.has_value()) {
        modified = true;
    }

    ImGui::EndChild();

    return modified;
}

void graph_canvas::center_on_graph() {
    if (!graph_ || graph_->node_count() == 0) {
        return;
    }

    auto [cx, cy] = graph_->center();
    target_pan_.x = canvas_size_.x * 0.5f / zoom_ - cx;
    target_pan_.y = canvas_size_.y * 0.5f / zoom_ - cy;
}

void graph_canvas::fit_graph_in_view() {
    if (!graph_ || graph_->node_count() == 0) {
        return;
    }

    auto [min_x, min_y, max_x, max_y] = graph_->bounds();
    float width = max_x - min_x;
    float height = max_y - min_y;

    if (width <= 0 || height <= 0) {
        return;
    }

    // Add padding
    float padding = 50.0f;
    float available_w = canvas_size_.x - padding * 2.0f;
    float available_h = canvas_size_.y - padding * 2.0f;

    // Calculate zoom to fit
    float zoom_x = available_w / width;
    float zoom_y = available_h / height;
    target_zoom_ = std::clamp(std::min(zoom_x, zoom_y), min_zoom_, max_zoom_);

    // Center
    float cx = (min_x + max_x) * 0.5f;
    float cy = (min_y + max_y) * 0.5f;
    target_pan_.x = canvas_size_.x * 0.5f / target_zoom_ - cx;
    target_pan_.y = canvas_size_.y * 0.5f / target_zoom_ - cy;
}

void graph_canvas::set_zoom(float zoom) {
    target_zoom_ = std::clamp(zoom, min_zoom_, max_zoom_);
    if (!smooth_animation_) {
        zoom_ = target_zoom_;
    }
}

void graph_canvas::set_pan(ImVec2 pan) {
    target_pan_ = pan;
    if (!smooth_animation_) {
        pan_ = target_pan_;
    }
}

void graph_canvas::clear_selection() {
    selected_ = std::nullopt;
}

void graph_canvas::select_node(node_id_t id) {
    if (graph_ && graph_->has_node(id)) {
        selected_ = id;
    }
}

void graph_canvas::set_edge_style(const edge_style& style) {
    edge_style_ = style;
}

void graph_canvas::set_edge_renderer(edge_render_callback callback) {
    custom_edge_renderer_ = std::move(callback);
}

ImVec2 graph_canvas::world_to_screen(float x, float y) const {
    return ImVec2(
        canvas_pos_.x + (x + pan_.x) * zoom_,
        canvas_pos_.y + (y + pan_.y) * zoom_
    );
}

ImVec2 graph_canvas::screen_to_world(ImVec2 screen) const {
    float inv_zoom = 1.0f / zoom_;
    return ImVec2(
        (screen.x - canvas_pos_.x) * inv_zoom - pan_.x,
        (screen.y - canvas_pos_.y) * inv_zoom - pan_.y
    );
}

void graph_canvas::handle_interaction(bool hovered) {
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 mouse_pos = io.MousePos;

    // Hit test for node under cursor
    hovered_ = std::nullopt;
    if (hovered && graph_ && renderer_) {
        hovered_ = hit_test(mouse_pos);
    }

    // Mouse wheel zoom
    if (zoom_enabled_ && hovered && io.MouseWheel != 0) {
        // Zoom towards mouse position
        ImVec2 world_before = screen_to_world(mouse_pos);

        float zoom_delta = io.MouseWheel * zoom_speed_;
        target_zoom_ = std::clamp(target_zoom_ * (1.0f + zoom_delta), min_zoom_, max_zoom_);

        if (!smooth_animation_) {
            zoom_ = target_zoom_;
            // Adjust pan to keep mouse position fixed in world space
            target_pan_.x = (mouse_pos.x - canvas_pos_.x) / zoom_ - world_before.x;
            target_pan_.y = (mouse_pos.y - canvas_pos_.y) / zoom_ - world_before.y;
            pan_ = target_pan_;
        }
    }

    // Node selection and dragging
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hovered) {
        if (hovered_.has_value()) {
            selected_ = hovered_;
            if (dragging_enabled_) {
                dragging_ = hovered_;
                drag_start_pos_ = screen_to_world(mouse_pos);
            }
        } else {
            selected_ = std::nullopt;
        }
    }

    // Node dragging
    if (dragging_.has_value() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImVec2 world_pos = screen_to_world(mouse_pos);
        if (graph_ && graph_->has_node(*dragging_)) {
            auto& n = graph_->node(*dragging_);
            n.x = world_pos.x;
            n.y = world_pos.y;
            n.vx = 0.0f;
            n.vy = 0.0f;
        }
    }

    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        dragging_ = std::nullopt;
    }

    // Panning with middle mouse or right mouse
    if (pan_enabled_) {
        bool start_pan = (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) ||
                          (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !hovered_.has_value()));

        if (start_pan && hovered) {
            panning_ = true;
            pan_start_mouse_ = mouse_pos;
            pan_start_offset_ = pan_;
        }

        if (panning_ && (ImGui::IsMouseDown(ImGuiMouseButton_Middle) ||
                         ImGui::IsMouseDown(ImGuiMouseButton_Right))) {
            ImVec2 delta = ImVec2(mouse_pos.x - pan_start_mouse_.x,
                                   mouse_pos.y - pan_start_mouse_.y);
            float inv_zoom = 1.0f / zoom_;
            target_pan_.x = pan_start_offset_.x + delta.x * inv_zoom;
            target_pan_.y = pan_start_offset_.y + delta.y * inv_zoom;
            if (!smooth_animation_) {
                pan_ = target_pan_;
            }
        }

        if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle) ||
            ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
            panning_ = false;
        }
    }
}

void graph_canvas::update_animation() {
    if (smooth_animation_) {
        const float lerp_speed = 0.15f;

        // Smoothly interpolate zoom
        zoom_ += (target_zoom_ - zoom_) * lerp_speed;
        if (std::abs(target_zoom_ - zoom_) < 0.001f) {
            zoom_ = target_zoom_;
        }

        // Smoothly interpolate pan
        pan_.x += (target_pan_.x - pan_.x) * lerp_speed;
        pan_.y += (target_pan_.y - pan_.y) * lerp_speed;
        if (std::abs(target_pan_.x - pan_.x) < 0.01f) {
            pan_.x = target_pan_.x;
        }
        if (std::abs(target_pan_.y - pan_.y) < 0.01f) {
            pan_.y = target_pan_.y;
        }
    } else {
        zoom_ = target_zoom_;
        pan_ = target_pan_;
    }
}

void graph_canvas::render_edges(ImDrawList* draw_list) {
    if (!graph_) return;

    for (const auto& edge : graph_->edges()) {
        if (!graph_->has_node(edge.from) || !graph_->has_node(edge.to)) {
            continue;
        }

        const auto& n1 = graph_->node(edge.from);
        const auto& n2 = graph_->node(edge.to);

        ImVec2 from_screen = world_to_screen(n1.x, n1.y);
        ImVec2 to_screen = world_to_screen(n2.x, n2.y);

        if (custom_edge_renderer_) {
            custom_edge_renderer_(draw_list, edge, from_screen, to_screen, zoom_);
        } else {
            draw_list->AddLine(from_screen, to_screen, edge_style_.color, edge_style_.thickness);

            // Draw arrow if directed
            if (edge_style_.show_direction) {
                ImVec2 dir = ImVec2(to_screen.x - from_screen.x, to_screen.y - from_screen.y);
                float length = std::sqrt(dir.x * dir.x + dir.y * dir.y);
                if (length > 0) {
                    float inv_len = 1.0f / length;
                    dir.x *= inv_len;
                    dir.y *= inv_len;

                    // Position arrow at target node edge
                    float target_radius = n2.radius * zoom_;
                    ImVec2 arrow_tip = ImVec2(
                        to_screen.x - dir.x * target_radius,
                        to_screen.y - dir.y * target_radius
                    );

                    float arrow_size = edge_style_.arrow_size * zoom_;
                    ImVec2 perp = ImVec2(-dir.y, dir.x);
                    ImVec2 p1 = ImVec2(arrow_tip.x - dir.x * arrow_size + perp.x * arrow_size * 0.5f,
                                        arrow_tip.y - dir.y * arrow_size + perp.y * arrow_size * 0.5f);
                    ImVec2 p2 = ImVec2(arrow_tip.x - dir.x * arrow_size - perp.x * arrow_size * 0.5f,
                                        arrow_tip.y - dir.y * arrow_size - perp.y * arrow_size * 0.5f);

                    draw_list->AddTriangleFilled(arrow_tip, p1, p2, edge_style_.color);
                }
            }
        }
    }
}

void graph_canvas::render_nodes(ImDrawList* draw_list) {
    if (!graph_ || !renderer_) return;

    for (const auto& node : graph_->nodes()) {
        ImVec2 screen_pos = world_to_screen(node.x, node.y);

        bool is_selected = selected_.has_value() && *selected_ == node.id;
        bool is_hovered = hovered_.has_value() && *hovered_ == node.id;

        renderer_->render(draw_list, node, screen_pos, zoom_, is_selected, is_hovered);
    }
}

std::optional<node_id_t> graph_canvas::hit_test(ImVec2 screen_pos) {
    if (!graph_ || !renderer_) {
        return std::nullopt;
    }

    // Test nodes in reverse order (topmost first)
    auto nodes = graph_->nodes();
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        const auto& node = *it;
        ImVec2 node_screen = world_to_screen(node.x, node.y);
        ImVec2 size = renderer_->get_size(node, zoom_);

        float half_w = size.x * 0.5f;
        float half_h = size.y * 0.5f;

        if (screen_pos.x >= node_screen.x - half_w &&
            screen_pos.x <= node_screen.x + half_w &&
            screen_pos.y >= node_screen.y - half_h &&
            screen_pos.y <= node_screen.y + half_h) {
            return node.id;
        }
    }

    return std::nullopt;
}

} // namespace coast::layout::render
