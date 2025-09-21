//
// Created by jk on 04.09.25.
//

#include <numeric>
#include <algorithm>
#include "hd_ngm2_tools.h"
#include "io_buffer.h"

namespace sim {
void io_buffer::update_stats()
{
    std::span<const float> read_buf = cur_read_buffer();
    read_buffer_stats.sum = std::reduce(read_buf.begin(),read_buf.end());
    read_buffer_stats.avg = read_buffer_stats.sum / static_cast<float>(read_buf.size());
    read_buffer_stats.min_val = std::reduce(
        read_buf.begin(), read_buf.end(),
        std::numeric_limits<float>::max(),
        [](const float a, const float b)
        {
            return std::min(a,b);
        }
    );
    read_buffer_stats.max_val = std::reduce(
        read_buf.begin(), read_buf.end(),
        0.0f,
        [](const float a, const float b)
        {
            return std::max(a,b);
        }
    );
    read_buffer_stats.nse = ngm2::normalized_shannon_entropy(read_buf);
}

io_buffer::io_buffer(std::size_t size) :
    buffer{ std::vector<float>(size), std::vector<float>(size) },
    read_buffer_stats(),
    write_idx(0),
    read_idx(1)
{}

std::span<float> io_buffer::cur_write_buffer()
{
    return { buffer[write_idx].begin(), buffer[write_idx].end() };
}

std::span<const float> io_buffer::cur_read_buffer()
{
    return { buffer[read_idx].begin(), buffer[read_idx].end() };
}

} // sim