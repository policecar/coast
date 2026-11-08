//
// Created by jk on 6/29/25.
//

#ifndef MNIST_DB_H
#define MNIST_DB_H

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <span>
#include <utility>
#include <iostream>

namespace mnist {

// IDX file reader. Only unsigned-byte data (type 0x08) is accepted: it is the
// only type any COAST loader consumes, so the other IDX types are rejected
// instead of being read into buffers nothing uses.
class idx_file {

    std::vector<uint8_t>     u8_data;
    std::vector<std::size_t> dimensions;
    std::size_t              element_size;
    std::size_t              num_elements;

public:
    explicit idx_file(const std::string &filename);

    [[nodiscard]] std::size_t get_num_elements()   const { return num_elements;      }
    [[nodiscard]] std::size_t get_num_dimensions() const { return dimensions.size(); }
    [[nodiscard]] std::size_t get_element_size()   const { return element_size;      }

    [[nodiscard]] const std::vector<std::size_t>& get_dimensions() const { return dimensions; }

    [[nodiscard]] const std::vector<uint8_t>& get_u8() const { return u8_data; }

    // Element `idx`; an index past the end is clamped to the last element.
    [[nodiscard]] std::span<const uint8_t> get_element(std::size_t idx) const;
};

class mnist_db {
    idx_file img_data;
    idx_file label_data;

    std::vector<float> norm_img_data;
public:
    explicit mnist_db(const std::string &img_file, const std::string &label_file);

    void print_info() const;

    [[nodiscard]] std::span<const uint8_t> get_image(std::size_t idx) const;
    [[nodiscard]] std::span<const float> get_norm_image(std::size_t idx) const;
    // Whole normalized dataset as one contiguous [image_cnt * image_size] span.
    // Used by the GPU entity to upload the full dataset in a single
    // wgpuQueueWriteBuffer at construction (mnist_io_gpu, S5).
    [[nodiscard]] std::span<const float> get_all_norm_images() const;
    [[nodiscard]] uint8_t get_label(std::size_t idx) const;
    [[nodiscard]] std::size_t get_image_cnt() const;
    [[nodiscard]] std::size_t get_image_size() const;
};

} // mdb

#endif //MNIST_DB_H
