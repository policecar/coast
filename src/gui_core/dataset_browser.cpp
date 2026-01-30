#include "dataset_browser.h"
#include "sim_instance.h"
#include "sim_builder.h"
#include "imgui.h"

// Dataset headers for type casting
#include "mnist_io.h"
#include "emnist_io.h"
#include "kmnist_io.h"
#include "dsprites_io.h"
#include "cifar10_io.h"
#include "cifar100_io.h"
#include "chairs3d_io.h"
#include "color_mnist_io.h"

#include <algorithm>
#include <cmath>

namespace coast {

namespace {

const char* entity_type_name(sim::io_entity_type_t type)
{
    switch (type) {
        case sim::io_entity_type_t::mnist_input:       return "MNIST";
        case sim::io_entity_type_t::emnist_input:      return "EMNIST";
        case sim::io_entity_type_t::kmnist_input:      return "KMNIST";
        case sim::io_entity_type_t::dsprites_input:    return "dSprites";
        case sim::io_entity_type_t::cifar10_input:     return "CIFAR-10";
        case sim::io_entity_type_t::cifar100_input:    return "CIFAR-100";
        case sim::io_entity_type_t::chairs3d_input:    return "3D Chairs";
        case sim::io_entity_type_t::color_mnist_input: return "Color MNIST";
        default:                                       return nullptr;
    }
}

bool is_dataset_type(sim::io_entity_type_t type)
{
    return entity_type_name(type) != nullptr;
}

} // anonymous namespace

dataset_browser::dataset_browser(sim_instance& sim)
    : sim(sim)
{
    rebuild_dataset_list();
}

dataset_browser::~dataset_browser()
{
    free_resources();
}

void dataset_browser::render()
{
    if (!is_visible) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Dataset Browser", &is_visible)) {
        // Dataset selector
        if (dataset_names.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "No datasets loaded in current simulation.");
        } else {
            ImGui::SetNextItemWidth(200);
            if (ImGui::BeginCombo("Dataset", selected_dataset_idx >= 0 ? dataset_names[selected_dataset_idx].c_str() : "Select...")) {
                for (int i = 0; i < static_cast<int>(dataset_names.size()); ++i) {
                    bool is_selected = (selected_dataset_idx == i);
                    if (ImGui::Selectable(dataset_names[i].c_str(), is_selected)) {
                        select_dataset(i);
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            if (selected_dataset_idx >= 0) {
                // Status line
                ImGui::SameLine();
                ImGui::Text("| %dx%d", img_width, img_height);
                ImGui::SameLine();
                ImGui::Text("| %d ch", img_channels);
                ImGui::SameLine();
                ImGui::Text("| %zu samples", sample_count);

                ImGui::Separator();

                // Thumbnail grid
                render_thumbnail_grid();
            }
        }
    }
    ImGui::End();
}

void dataset_browser::free_resources()
{
    for (auto& thumb : thumb_cache) {
        thumb.texture.release();
    }
    thumb_cache.clear();
}

void dataset_browser::rebuild_dataset_list()
{
    dataset_names.clear();
    dataset_entity_ids.clear();

    auto& sim_env = sim.get_sim_env();
    const auto& builder = sim.get_builder();

    for (auto& entity : sim_env.iterate_entities()) {
        const auto id = entity.get_outp_id();
        const auto* node = builder.get_node(id);

        if (node && is_dataset_type(node->entity_type)) {
            std::string name = node->label;
            if (name.empty()) {
                name = entity_type_name(node->entity_type);
            }
            dataset_names.push_back(name);
            dataset_entity_ids.push_back(id);
        }
    }

    // Auto-select first dataset if available
    if (!dataset_names.empty() && selected_dataset_idx < 0) {
        select_dataset(0);
    }
}

void dataset_browser::select_dataset(int idx)
{
    if (idx < 0 || idx >= static_cast<int>(dataset_names.size())) {
        selected_dataset_idx = -1;
        get_sample_func = nullptr;
        return;
    }

    selected_dataset_idx = idx;
    std::size_t entity_id = dataset_entity_ids[idx];

    auto& sim_env = sim.get_sim_env();
    const auto& builder = sim.get_builder();
    const auto* node = builder.get_node(entity_id);

    if (!node) return;

    sim::io_entity* entity_ptr = nullptr;
    for (auto& entity : sim_env.iterate_entities()) {
        if (entity.get_outp_id() == entity_id) {
            entity_ptr = &entity;
            break;
        }
    }

    if (!entity_ptr) return;

    // Get image info and set up sample getter based on entity type
    get_sample_func = nullptr;
    img_width = 0;
    img_height = 0;
    img_channels = 0;
    sample_count = 0;

    switch (node->entity_type) {
        case sim::io_entity_type_t::mnist_input: {
            auto* ds = dynamic_cast<mnist::mnist_io*>(entity_ptr);
            if (ds) {
                auto [w, h, c] = ds->get_image_info();
                img_width = w; img_height = h; img_channels = c;
                sample_count = ds->sample_count();
                get_sample_func = [ds](std::size_t idx) { return ds->get_sample(idx); };
            }
        } break;
        case sim::io_entity_type_t::emnist_input: {
            auto* ds = dynamic_cast<emnist::emnist_io*>(entity_ptr);
            if (ds) {
                auto [w, h, c] = ds->get_image_info();
                img_width = w; img_height = h; img_channels = c;
                sample_count = ds->sample_count();
                get_sample_func = [ds](std::size_t idx) { return ds->get_sample(idx); };
            }
        } break;
        case sim::io_entity_type_t::kmnist_input: {
            auto* ds = dynamic_cast<kmnist::kmnist_io*>(entity_ptr);
            if (ds) {
                auto [w, h, c] = ds->get_image_info();
                img_width = w; img_height = h; img_channels = c;
                sample_count = ds->sample_count();
                get_sample_func = [ds](std::size_t idx) { return ds->get_sample(idx); };
            }
        } break;
        case sim::io_entity_type_t::dsprites_input: {
            auto* ds = dynamic_cast<dsprites::dsprites_io*>(entity_ptr);
            if (ds) {
                auto [w, h, c] = ds->get_image_info();
                img_width = w; img_height = h; img_channels = c;
                sample_count = ds->sample_count();
                get_sample_func = [ds](std::size_t idx) { return ds->get_sample(idx); };
            }
        } break;
        case sim::io_entity_type_t::cifar10_input: {
            auto* ds = dynamic_cast<cifar10::cifar10_io*>(entity_ptr);
            if (ds) {
                auto [w, h, c] = ds->get_image_info();
                img_width = w; img_height = h; img_channels = c;
                sample_count = ds->sample_count();
                get_sample_func = [ds](std::size_t idx) { return ds->get_sample(idx); };
            }
        } break;
        case sim::io_entity_type_t::cifar100_input: {
            auto* ds = dynamic_cast<cifar100::cifar100_io*>(entity_ptr);
            if (ds) {
                auto [w, h, c] = ds->get_image_info();
                img_width = w; img_height = h; img_channels = c;
                sample_count = ds->sample_count();
                get_sample_func = [ds](std::size_t idx) { return ds->get_sample(idx); };
            }
        } break;
        case sim::io_entity_type_t::chairs3d_input: {
            auto* ds = dynamic_cast<chairs3d::chairs3d_io*>(entity_ptr);
            if (ds) {
                auto [w, h, c] = ds->get_image_info();
                img_width = w; img_height = h; img_channels = c;
                sample_count = ds->sample_count();
                get_sample_func = [ds](std::size_t idx) { return ds->get_sample(idx); };
            }
        } break;
        case sim::io_entity_type_t::color_mnist_input: {
            auto* ds = dynamic_cast<color_mnist::color_mnist_io*>(entity_ptr);
            if (ds) {
                // Show underlying grayscale images (color is applied at runtime)
                img_width = 28; img_height = 28; img_channels = 1;
                sample_count = ds->sample_count();
                get_sample_func = [ds](std::size_t idx) { return ds->get_norm_image(idx); };
            }
        } break;
        default:
            break;
    }

    // Clear thumbnail cache when switching datasets
    free_resources();
    thumb_pixels.resize(THUMB_SIZE * THUMB_SIZE);
}

void dataset_browser::render_thumbnail_grid()
{
    if (!get_sample_func || sample_count == 0) return;

    const float cell_size = static_cast<float>(THUMB_SIZE + THUMB_PADDING);
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const int cols = std::max(1, static_cast<int>(avail.x / cell_size));
    const int total_rows = static_cast<int>((sample_count + cols - 1) / cols);

    // Calculate visible range based on scroll position
    ImGui::BeginChild("ThumbnailScroll", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);

    const float scroll_y = ImGui::GetScrollY();
    const float visible_height = ImGui::GetWindowHeight();

    const int first_visible_row = std::max(0, static_cast<int>(scroll_y / cell_size) - VISIBLE_ROWS_BUFFER);
    const int last_visible_row = std::min(total_rows - 1, static_cast<int>((scroll_y + visible_height) / cell_size) + VISIBLE_ROWS_BUFFER);
    const int visible_rows = last_visible_row - first_visible_row + 1;
    const int visible_cells = visible_rows * cols;

    // Ensure we have enough cached thumbnails
    if (thumb_cache.size() < static_cast<std::size_t>(visible_cells)) {
        thumb_cache.resize(visible_cells);
    }

    // Set up dummy spacing for rows before visible area
    if (first_visible_row > 0) {
        ImGui::Dummy(ImVec2(avail.x, first_visible_row * cell_size));
    }

    // Render visible rows
    int cache_idx = 0;
    for (int row = first_visible_row; row <= last_visible_row; ++row) {
        for (int col = 0; col < cols; ++col) {
            const std::size_t sample_idx = static_cast<std::size_t>(row * cols + col);
            if (sample_idx >= sample_count) break;

            // Update thumbnail if needed
            if (cache_idx < static_cast<int>(thumb_cache.size())) {
                auto& thumb = thumb_cache[cache_idx];
                if (!thumb.valid || thumb.sample_idx != sample_idx) {
                    update_thumbnail(cache_idx, sample_idx);
                }

                // Render thumbnail
                if (thumb.texture.is_valid()) {
                    if (col > 0) ImGui::SameLine();
                    ImGui::Image(thumb.texture.as_imgui_texture(),
                                ImVec2(THUMB_SIZE, THUMB_SIZE));

                    // Tooltip with sample index
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Sample #%zu", sample_idx);
                    }
                }
            }
            ++cache_idx;
        }
    }

    // Set up dummy spacing for rows after visible area
    const int rows_after = total_rows - last_visible_row - 1;
    if (rows_after > 0) {
        ImGui::Dummy(ImVec2(avail.x, rows_after * cell_size));
    }

    ImGui::EndChild();
}

void dataset_browser::update_thumbnail(std::size_t cache_idx, std::size_t sample_idx)
{
    if (cache_idx >= thumb_cache.size()) return;

    auto& thumb = thumb_cache[cache_idx];

    // Get sample data
    auto sample = get_sample_func(sample_idx);
    convert_sample_to_rgba(sample, thumb_pixels);

    // Create or update texture
    if (!thumb.texture.is_valid()) {
        thumb.texture = gl_texture(THUMB_SIZE, THUMB_SIZE, thumb_pixels.data());
    } else {
        thumb.texture.update(thumb_pixels);
    }

    thumb.sample_idx = sample_idx;
    thumb.valid = true;
}

void dataset_browser::convert_sample_to_rgba(std::span<const float> sample, std::vector<color_t>& out)
{
    out.resize(THUMB_SIZE * THUMB_SIZE);

    // Calculate scaling factors for resampling
    const float scale_x = static_cast<float>(img_width) / THUMB_SIZE;
    const float scale_y = static_cast<float>(img_height) / THUMB_SIZE;

    for (int ty = 0; ty < THUMB_SIZE; ++ty) {
        for (int tx = 0; tx < THUMB_SIZE; ++tx) {
            // Nearest neighbor sampling
            const int sx = std::min(static_cast<int>(tx * scale_x), img_width - 1);
            const int sy = std::min(static_cast<int>(ty * scale_y), img_height - 1);
            const std::size_t src_pixel = sy * img_width + sx;
            const std::size_t dst_pixel = ty * THUMB_SIZE + tx;

            if (img_channels == 1) {
                // Grayscale
                const std::size_t src_idx = src_pixel;
                if (src_idx < sample.size()) {
                    uint8_t val = static_cast<uint8_t>(std::clamp(sample[src_idx] * 255.0f, 0.0f, 255.0f));
                    out[dst_pixel] = color_t{val, val, val, 255};
                } else {
                    out[dst_pixel] = color_t{255, 0, 255, 255};  // Magenta for error
                }
            } else if (img_channels == 3) {
                // RGB
                const std::size_t src_idx = src_pixel * 3;
                if (src_idx + 2 < sample.size()) {
                    uint8_t r = static_cast<uint8_t>(std::clamp(sample[src_idx + 0] * 255.0f, 0.0f, 255.0f));
                    uint8_t g = static_cast<uint8_t>(std::clamp(sample[src_idx + 1] * 255.0f, 0.0f, 255.0f));
                    uint8_t b = static_cast<uint8_t>(std::clamp(sample[src_idx + 2] * 255.0f, 0.0f, 255.0f));
                    out[dst_pixel] = color_t{r, g, b, 255};
                } else {
                    out[dst_pixel] = color_t{255, 0, 255, 255};  // Magenta for error
                }
            }
        }
    }
}

} // namespace coast
