//
// Created by jk on 6/29/25.
//

#include <bit>

#include "io/mnist_db.h"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <limits>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>

namespace mnist {

idx_file::idx_file(const std::string &filename) :
    u8_data(), dimensions(), element_size(), num_elements()
{
    // Closed on every exit, including a bad_alloc while reading.
    const std::unique_ptr<std::FILE, int (*)(std::FILE*)> owner(
        std::fopen(filename.c_str(), "rb"), &std::fclose);
    std::FILE* const file = owner.get();
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    uint32_t tmp;
    if (std::fread(&tmp, sizeof(tmp), 1, file) != 1) {
        throw std::runtime_error("Could not read file: " + filename);
    }
    const uint32_t magic = std::byteswap(tmp);
    if ((magic >> 16) != 0) {
        throw std::runtime_error("Invalid magic number in file: " + filename);
    }
    const uint8_t data_type = (magic >> 8) & 0xff;
    const uint8_t dim_count = (magic >> 0) & 0xff;
    if (data_type != 0x08) {
        throw std::runtime_error("Unsupported IDX data type in file: " + filename +
                                 " (only unsigned byte, 0x08, is supported)");
    }
    if (dim_count == 0) {
        throw std::runtime_error("Invalid dimensions in file: " + filename);
    }
    for (uint8_t i = 0; i < dim_count; i++) {
        if (std::fread(&tmp, sizeof(tmp), 1, file) != 1) {
            throw std::runtime_error("Truncated header in file: " + filename);
        }
        dimensions.push_back(std::byteswap(tmp));
    }
    // Fold in std::size_t with an overflow check: the dimensions come from the file.
    element_size = 1;
    for (const std::size_t d : dimensions | std::views::drop(1)) {
        if (d != 0 && element_size > std::numeric_limits<std::size_t>::max() / d) {
            throw std::runtime_error("Element size overflows in file: " + filename);
        }
        element_size *= d;
    }
    num_elements = dimensions[0];
    if (num_elements != 0 &&
        element_size > std::numeric_limits<std::size_t>::max() / num_elements) {
        throw std::runtime_error("Data size overflows in file: " + filename);
    }
    // The header must describe exactly the data that follows it: a short file
    // (interrupted download) or trailing bytes both mean the file is not what
    // the header claims. Compared against the file size before allocating, so
    // a header that declares terabytes fails here rather than in the allocator.
    const std::size_t declared   = num_elements * element_size;
    const std::size_t header     = sizeof(uint32_t) * (1 + std::size_t{dim_count});
    const std::size_t file_bytes = std::filesystem::file_size(filename);
    if (file_bytes - header != declared) {   // header bytes were read, so file_bytes >= header
        throw std::runtime_error("Unexpected data size in file: " + filename +
                                 " (header declares " + std::to_string(declared) +
                                 " bytes, file holds " + std::to_string(file_bytes - header) + ")");
    }
    u8_data.resize(declared);
    if (std::fread(u8_data.data(), 1, declared, file) != declared) {
        throw std::runtime_error("Could not read data from file: " + filename);
    }
}

std::span<const uint8_t> idx_file::get_element(std::size_t idx) const
{
    if (idx >= num_elements) idx = num_elements - 1;
    return { u8_data.data() + idx * element_size, element_size };
}

mnist_db::mnist_db(const std::string &img_file, const std::string &label_file) :
    img_data(img_file),
    label_data(label_file),
    norm_img_data( img_data.get_u8().size() )
{
    // idx_file guarantees the data matches the header; an empty dataset is
    // still well-formed IDX but cannot be sampled (every get_norm_image would
    // be out of range), so reject it here.
    if (img_data.get_num_elements() == 0 || img_data.get_element_size() == 0) {
        throw std::runtime_error("Image file holds no image data: " + img_file);
    }
    if (img_data.get_num_elements() != label_data.get_num_elements()) {
        throw std::runtime_error("Image file and label file have different number of elements");
    }

    for (auto [fval,ival] : std::views::zip(norm_img_data,img_data.get_u8())) {
        // addition of epsilon is a ROOT thing and just needed for easier visualization
        //fval = static_cast<float>(ival) / (255.0f + std::numeric_limits<float>::epsilon()*2);
        fval = static_cast<float>(ival) / 255.0f;
    }

}

void mnist_db::print_info() const
{
    std::cout << "Image data:\n\tResolution: ";
    for (auto val : img_data.get_dimensions()) {
        std::cout << val << " ";
    }
    std::cout << "\n\tNumber of elements: " << img_data.get_num_elements() << std::endl;
    std::cout << "\n\tSize of element: " << img_data.get_element_size() << std::endl;
    std::cout << "Label data:\n\tResolution: ";
    for (auto val : label_data.get_dimensions()) {
        std::cout << val << " ";
    }
    std::cout << "\n\tNumber of elements: " << label_data.get_num_elements() << std::endl;
    std::cout << "\n\tSize of element: " << label_data.get_element_size() << std::endl;
}

std::span<const uint8_t> mnist_db::get_image(std::size_t idx) const
{
    return img_data.get_element(idx);
}

std::span<const float> mnist_db::get_norm_image(std::size_t idx) const
{
    // Real runtime bounds check: a bare assert() is compiled out under -DNDEBUG
    // (Release), so an out-of-range index would read past the buffer (segfault /
    // corrupt output) instead of failing cleanly.
    const std::size_t elem = img_data.get_element_size();
    if ((idx + 1) * elem > norm_img_data.size()) {
        throw std::out_of_range(
            "mnist_db::get_norm_image: image index " + std::to_string(idx) +
            " is out of range (dataset has " + std::to_string(get_image_cnt()) +
            " images)");
    }
    return { norm_img_data.data() + idx * elem, elem };
}

std::span<const float> mnist_db::get_all_norm_images() const
{
    return { norm_img_data.data(), norm_img_data.size() };
}

uint8_t mnist_db::get_label(std::size_t idx) const
{
    return label_data.get_u8().at(idx);
}

std::size_t mnist_db::get_image_cnt() const
{
    return img_data.get_num_elements();
}

std::size_t mnist_db::get_image_size() const
{
    return img_data.get_element_size();
}
} // mdb