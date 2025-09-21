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

namespace mdb {

class idx_file {

    std::vector<uint8_t>    u8_data;
    std::vector<int8_t>     i8_data;
    std::vector<int16_t>    i16_data;
    std::vector<int32_t>    i32_data;
    std::vector<float>      f32_data;
    std::vector<double>     f64_data;

    std::vector<std::size_t> dimensions;
    std::size_t              element_size;
    std::size_t              num_elements;

    uint8_t bswap(uint8_t x);
    int16_t bswab(int16_t x);
    int32_t bswab(int32_t x);

    template<typename T>
    void read_data(std::FILE *file, std::vector<T> &data);
public:
    explicit idx_file(const std::string &filename);

    [[nodiscard]] std::size_t get_num_elements()   const { return num_elements;      }
    [[nodiscard]] std::size_t get_num_dimensions() const { return dimensions.size(); }
    [[nodiscard]] std::size_t get_element_size()   const { return element_size;      }

    [[nodiscard]] const std::vector<std::size_t>& get_dimensions() const { return dimensions; }

    [[nodiscard]] const std::vector<uint8_t>& get_u8()  const { return u8_data;  }
    [[nodiscard]] const std::vector<int8_t>&  get_i8()  const { return i8_data;  }
    [[nodiscard]] const std::vector<int16_t>& get_i16() const { return i16_data; }
    [[nodiscard]] const std::vector<int32_t>& get_i32() const { return i32_data; }
    [[nodiscard]] const std::vector<float>&   get_f32() const { return f32_data; }
    [[nodiscard]] const std::vector<double>&  get_f64() const { return f64_data; }

    template<typename T>
    [[nodiscard]] std::span<const T> get_element(std::size_t idx) const;
};

template<typename T>
void idx_file::read_data(std::FILE *file, std::vector<T> &data)
{
    T tmp;
    while (true) {
        int elements_read = std::fread(&tmp,sizeof(T),1,file);
        if (elements_read != 1) break;
        if constexpr (std::is_integral_v<T>) {
            data.push_back(bswap(tmp));
        } else {
            data.push_back(tmp);
        }
    }
}

template<typename T>
std::span<const T> idx_file::get_element(std::size_t idx) const
{
    if (idx >= num_elements) idx = num_elements - 1;
    if constexpr (std::is_same_v<T, uint8_t>) {
        return std::span(&u8_data[idx * element_size], element_size);
    }
    if constexpr (std::is_same_v<T, int8_t>) {
        return std::span(&i8_data[idx * element_size], element_size);
    }
    if constexpr (std::is_same_v<T, int16_t>) {
        return std::span(&i16_data[idx * element_size], element_size);
    }
    if constexpr (std::is_same_v<T, int32_t>) {
        return std::span(&i32_data[idx * element_size], element_size);
    }
    if constexpr (std::is_same_v<T, float>) {
        return std::span(&f32_data[idx * element_size], element_size);
    }
    if constexpr (std::is_same_v<T, double>) {
        return std::span(&f64_data[idx * element_size], element_size);
    }

}

class mnist_db {
    idx_file img_data;
    idx_file label_data;

    std::vector<float> norm_img_data;
public:
    explicit mnist_db(const std::string &img_file, const std::string &label_file);

    void print_info() const;

    [[nodiscard]] std::span<const uint8_t> get_image(std::size_t idx) const;
    [[nodiscard]] std::span<const float> get_norm_image(std::size_t idx) const;
    [[nodiscard]] uint8_t get_label(std::size_t idx) const;
    [[nodiscard]] std::size_t get_image_cnt() const;
    [[nodiscard]] std::size_t get_image_size() const;
};

} // mdb

#endif //MNIST_DB_H
