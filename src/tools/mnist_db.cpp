//
// Created by jk on 6/29/25.
//

#include <bit>

#include "mnist_db.h"

#include <cassert>
#include <iostream>
#include <numeric>
#include <limits>

namespace mdb {

uint8_t idx_file::bswap(uint8_t x)
{
    return std::byteswap(x);
}

int16_t idx_file::bswab(int16_t x)
{
    return std::byteswap(x);
}

int32_t idx_file::bswab(int32_t x)
{
    return std::byteswap(x);
}

idx_file::idx_file(const std::string &filename) :
    u8_data(), i8_data(), i16_data(), i32_data(), f32_data(), f64_data(),
    dimensions(),element_size(),num_elements()
{
//    std::fstream ifs(filename,std::ios::binary | std::ios::in);
    std::FILE *file = std::fopen(filename.c_str(),"rb");
    if (!file) {
        std::cerr << "Could not open file " << filename << std::endl;
        return;
    }
    uint32_t tmp;
    std::size_t elements_read = std::fread(&tmp, sizeof(tmp), 1, file);
    if (elements_read != 1) {
        std::cerr << "Could not read file " << filename << std::endl;
        std::cerr << "Read " << elements_read << " elements from " << filename << std::endl;
    }
    uint32_t magic = std::byteswap(tmp);
    if ((magic >> 16) != 0) {
        std::cerr << "Invalid magic number" << std::endl;
        return;
    }
    uint8_t data_type = (magic >> 8) & 0xff;
    uint8_t dim_count = (magic >> 0) & 0xff;
    if (dim_count == 0) {
        std::cerr << "Invalid dimensions" << std::endl;
        return;
    }
    for (uint8_t i = 0; i < dim_count; i++) {
        //    ifs >> tmp;
        //ifs.read((char *)&tmp, sizeof(tmp));
        std::fread(&tmp, sizeof(tmp), 1, file);
        dimensions.push_back(std::byteswap(tmp));
    }
    element_size = std::accumulate(dimensions.begin()+1, dimensions.end(), 1, std::multiplies<>());
    num_elements = dimensions[0];
    switch (data_type) {
        case 0x08 : {
            read_data(file, u8_data);
        } break;
        case 0x09 : {
            read_data(file, i8_data);
        } break;
        case 0x0B : {
            read_data(file, i16_data);
        } break;
        case 0x0C : {
            read_data(file, i32_data);
        } break;
        case 0x0D : {
            read_data(file, f32_data);
        } break;
        case 0x0E : {
            read_data(file, f64_data);
        } break;
        default:    std::cerr << "Invalid data type" << std::endl;
    }
    std::fclose(file);
}

mnist_db::mnist_db(const std::string &img_file, const std::string &label_file) :
    img_data(img_file),
    label_data(label_file),
    norm_img_data( img_data.get_u8().size() )
{
    if (img_data.get_u8().size() != img_data.get_num_elements() * img_data.get_element_size()) {
        std::cerr << "unexpected data in img_file" << std::endl;
        std::cerr << img_data.get_u8().size() << std::endl;
        std::cerr << img_data.get_num_elements() << std::endl;
        std::cerr << img_data.get_element_size() << std::endl;
        return;
    }
    if (img_data.get_num_elements() != label_data.get_num_elements()) {
        std::cerr << "img_file and label_file have different number of elements" << std::endl;
        return;
    }

    for (auto &val : norm_img_data) {
        std::size_t idx = &val - norm_img_data.data();
        // addition of epsilon is a ROOT thing and just needed for easier visualization
        val = std::numeric_limits<float>::epsilon() +
              static_cast<float>(img_data.get_u8().at(idx)) /
                  (255.0f + std::numeric_limits<float>::epsilon()*2);
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
    return img_data.get_element<uint8_t>(idx);
}

std::span<const float> mnist_db::get_norm_image(std::size_t idx) const
{
    if((idx + 1) * img_data.get_element_size() > norm_img_data.size()) {
        std::cout << "idx: " << idx << std::endl;
        std::cout << "elem_size: " << img_data.get_element_size() << std::endl;
        std::cout << "data.size: " << norm_img_data.size() << std::endl;
        assert(false);
    }

    return { norm_img_data.data() + idx * img_data.get_element_size(), img_data.get_element_size()};
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