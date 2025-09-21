//
// Created by jk on 04.09.25.
//

#ifndef IO_BUFFER_H
#define IO_BUFFER_H

#include <array>
#include <vector>
#include <cstdint>
#include <functional>
#include <span>

namespace sim {

class io_buffer {

public:
    struct stats
    {
        float sum;
        float avg;
        float min_val;
        float max_val;
        float nse;
    };

private:

    std::array<std::vector<float>,2> buffer;
    stats read_buffer_stats;
    uint8_t write_idx;
    uint8_t read_idx;

    void update_stats();

public:

    explicit io_buffer(std::size_t size);

    void swap_buffer()
    {
        read_idx  =  write_idx;
        write_idx = (write_idx + 1) & 1;
        update_stats();
    }

    std::function<std::span<float>()> outp_buffer_func()
    {
        return [this]() -> std::span<float> {
            return { buffer[write_idx].begin(), buffer[write_idx].end() };
        };
    }

    using inp_buf_t = std::tuple<std::span<const float>,sim::io_buffer::stats>;

    std::function<inp_buf_t()> inp_buffer_func()
    {
        return [this]() -> inp_buf_t {
            return {
                std::span<const float> { buffer[read_idx].begin(), buffer[read_idx].end() },
                read_buffer_stats
            };
        };
    }

    [[nodiscard]] std::size_t size() const { return buffer[0].size(); }

    [[nodiscard]] std::span<float> cur_write_buffer();

    [[nodiscard]] std::span<const float> cur_read_buffer();

};

} // sim

#endif //IO_BUFFER_H
