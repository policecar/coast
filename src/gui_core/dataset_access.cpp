#include "gui_core/dataset_access.h"

#include "sim_core/sim_env.h"
#include "tools/sim_builder.h"

// Dataset headers for type casting
#include "io/mnist_io.h"
#include "io/dsprites_io.h"
#include "io/cifar10_io.h"
#include "io/cifar100_io.h"
#include "io/chairs3d_io.h"
#include "io/color_mnist_io.h"

#include <algorithm>
#include <cstdint>

namespace coast {

namespace {

// Fill geometry + accessor from a loader of concrete type T. The dynamic_cast
// checks the entity really is a T; the captured pointer is valid for the
// sim_instance's lifetime.
template <typename T>
void resolve_as(sim::io_entity* entity_ptr, dataset_handle& h)
{
    auto* ds = dynamic_cast<T*>(entity_ptr);
    if (!ds) return;
    auto [w, hh, c] = ds->get_image_info();
    h.width = w; h.height = hh; h.channels = c;
    h.sample_count = ds->sample_count();
    h.get_sample = [ds](std::size_t idx) { return ds->get_sample(idx); };
}

void resolve_geometry(sim::io_entity_type_t type, sim::io_entity* entity_ptr,
                      dataset_handle& h)
{
    using T = sim::io_entity_type_t;
    switch (type) {
        case T::mnist_input:    resolve_as<mnist::mnist_io>(entity_ptr, h);       break;
        case T::emnist_input:   resolve_as<emnist::emnist_io>(entity_ptr, h);     break;
        case T::kmnist_input:   resolve_as<kmnist::kmnist_io>(entity_ptr, h);     break;
        case T::dsprites_input: resolve_as<dsprites::dsprites_io>(entity_ptr, h); break;
        case T::cifar10_input:  resolve_as<cifar10::cifar10_io>(entity_ptr, h);   break;
        case T::cifar100_input: resolve_as<cifar100::cifar100_io>(entity_ptr, h); break;
        case T::chairs3d_input: resolve_as<chairs3d::chairs3d_io>(entity_ptr, h); break;
        case sim::io_entity_type_t::color_mnist_input: {
            auto* ds = dynamic_cast<color_mnist::color_mnist_io*>(entity_ptr);
            if (ds) {
                // Show underlying grayscale images (color is applied at runtime).
                h.width = 28; h.height = 28; h.channels = 1;
                h.sample_count = ds->sample_count();
                h.get_sample = [ds](std::size_t idx) { return ds->get_norm_image(idx); };
            }
        } break;
        case sim::io_entity_type_t::mnist_input_gpu:
            // GPU MNIST (mnist_io_gpu) exposes samples via mnist_db, not the CPU
            // mnist::mnist_io resolved above — not surfaced in the CPU explorer yet.
        case sim::io_entity_type_t::ngm2_neuron_group:
        case sim::io_entity_type_t::ngm3_neuron_group:
        case sim::io_entity_type_t::ngm3_neuron_group_gpu:
        case sim::io_entity_type_t::ngm25_neuron_group:
        case sim::io_entity_type_t::ngm25_neuron_group_gpu:
        case sim::io_entity_type_t::retina:
        case sim::io_entity_type_t::ENTITY_COUNT:
            // Not resolvable here: no CPU geometry/accessor. Dropped by valid().
            break;
    }
}

} // anonymous namespace

std::vector<dataset_handle> enumerate_datasets(sim::env& env, const sim::sim_builder& builder)
{
    std::vector<dataset_handle> handles;

    for (auto& entity : env.iterate_entities()) {
        const auto id = entity.get_outp_id();
        const auto* node = builder.get_node(id);
        if (!node || !sim::is_dataset_entity_type(node->entity_type)) continue;

        dataset_handle h;
        h.entity_id = id;
        h.name = node->label.empty() ? sim::entity_type_display_name(node->entity_type) : node->label;
        resolve_geometry(node->entity_type, &entity, h);

        // Only surface handles that actually resolved an accessor (a dynamic_cast
        // miss leaves get_sample empty -> the dataset is simply absent, not fatal).
        if (h.valid()) handles.push_back(std::move(h));
    }

    return handles;
}

void convert_sample_to_rgba(std::span<const float> sample,
                            int w, int h, int channels, int thumb_size,
                            std::vector<color_t>& out)
{
    out.resize(static_cast<std::size_t>(thumb_size) * thumb_size);

    // Nearest-neighbour resample from the source image to the thumbnail grid.
    const float scale_x = static_cast<float>(w) / thumb_size;
    const float scale_y = static_cast<float>(h) / thumb_size;

    for (int ty = 0; ty < thumb_size; ++ty) {
        for (int tx = 0; tx < thumb_size; ++tx) {
            const int sx = std::min(static_cast<int>(tx * scale_x), w - 1);
            const int sy = std::min(static_cast<int>(ty * scale_y), h - 1);
            const std::size_t src_pixel = static_cast<std::size_t>(sy) * w + sx;
            const std::size_t dst_pixel = static_cast<std::size_t>(ty) * thumb_size + tx;

            if (channels == 1) {
                const std::size_t src_idx = src_pixel;
                if (src_idx < sample.size()) {
                    uint8_t val = static_cast<uint8_t>(std::clamp(sample[src_idx] * 255.0f, 0.0f, 255.0f));
                    out[dst_pixel] = color_t{val, val, val, 255};
                } else {
                    out[dst_pixel] = color_t{255, 0, 255, 255};  // Magenta for error
                }
            } else if (channels == 3) {
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
