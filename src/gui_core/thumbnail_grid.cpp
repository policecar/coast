#include "gui_core/thumbnail_grid.h"

#include "imgui.h"

#include <algorithm>

namespace coast {

void thumbnail_grid::free_resources()
{
    for (auto& thumb : thumb_cache) {
        thumb.texture.release();
    }
    thumb_cache.clear();
}

void thumbnail_grid::update_thumbnail(const dataset_handle& ds,
                                      std::size_t cache_idx, std::size_t sample_idx)
{
    if (cache_idx >= thumb_cache.size()) return;

    auto& thumb = thumb_cache[cache_idx];

    auto sample = ds.get_sample(sample_idx);
    convert_sample_to_rgba(sample, ds.width, ds.height, ds.channels, THUMB_SIZE, thumb_pixels);

    if (!thumb.texture.is_valid()) {
        thumb.texture = gl_texture(THUMB_SIZE, THUMB_SIZE, thumb_pixels.data());
    } else {
        thumb.texture.update(thumb_pixels);
    }

    thumb.sample_idx = sample_idx;
    thumb.valid = true;
}

void thumbnail_grid::render(const dataset_handle& ds, std::set<std::size_t>* sel)
{
    if (!ds.valid() || ds.sample_count == 0) return;

    const float cell_size = static_cast<float>(THUMB_SIZE + THUMB_PADDING);
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const int cols = std::max(1, static_cast<int>(avail.x / cell_size));
    const int total_rows = static_cast<int>((ds.sample_count + cols - 1) / cols);

    ImGui::BeginChild("ThumbnailScroll", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);

    const float scroll_y = ImGui::GetScrollY();
    const float visible_height = ImGui::GetWindowHeight();

    const int first_visible_row = std::max(0, static_cast<int>(scroll_y / cell_size) - VISIBLE_ROWS_BUFFER);
    const int last_visible_row = std::min(total_rows - 1,
        static_cast<int>((scroll_y + visible_height) / cell_size) + VISIBLE_ROWS_BUFFER);
    const int visible_rows = last_visible_row - first_visible_row + 1;
    const int visible_cells = visible_rows * cols;

    if (thumb_cache.size() < static_cast<std::size_t>(visible_cells)) {
        thumb_cache.resize(visible_cells);
    }

    // Spacer for the rows scrolled off the top, so the scrollbar extent is correct.
    if (first_visible_row > 0) {
        ImGui::Dummy(ImVec2(avail.x, first_visible_row * cell_size));
    }

    int cache_idx = 0;
    for (int row = first_visible_row; row <= last_visible_row; ++row) {
        for (int col = 0; col < cols; ++col) {
            const std::size_t sample_idx = static_cast<std::size_t>(row) * cols + col;
            if (sample_idx >= ds.sample_count) break;

            if (cache_idx < static_cast<int>(thumb_cache.size())) {
                auto& thumb = thumb_cache[cache_idx];
                if (!thumb.valid || thumb.sample_idx != sample_idx) {
                    update_thumbnail(ds, cache_idx, sample_idx);
                }

                if (thumb.texture.is_valid()) {
                    if (col > 0) ImGui::SameLine();
                    ImGui::PushID(static_cast<int>(sample_idx));

                    const ImVec2 thumb_dim(THUMB_SIZE, THUMB_SIZE);
                    if (sel) {
                        // Selectable: a borderless image button toggles membership.
                        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
                        if (ImGui::ImageButton("##t", thumb.texture.as_imgui_texture(), thumb_dim)) {
                            if (sel->count(sample_idx)) sel->erase(sample_idx);
                            else                        sel->insert(sample_idx);
                        }
                        ImGui::PopStyleVar();
                        if (sel->count(sample_idx)) {
                            ImGui::GetWindowDrawList()->AddRect(
                                ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
                                IM_COL32(255, 200, 0, 255), 0.0f, 0, 3.0f);
                        }
                    } else {
                        // View-only: plain image (preserves the browser's look).
                        ImGui::Image(thumb.texture.as_imgui_texture(), thumb_dim);
                    }

                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Sample #%zu", sample_idx);
                    }
                    ImGui::PopID();
                }
            }
            ++cache_idx;
        }
    }

    // Spacer for the rows below the visible window.
    const int rows_after = total_rows - last_visible_row - 1;
    if (rows_after > 0) {
        ImGui::Dummy(ImVec2(avail.x, rows_after * cell_size));
    }

    ImGui::EndChild();
}

} // namespace coast
