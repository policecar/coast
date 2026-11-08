//
// Created by jk on 04.09.25.
//

#include "sim_core/io_buffer.h"

#include <algorithm>

namespace sim {

io_buffer_cpu::io_buffer_cpu(std::size_t size) :
    buffer{ std::vector<float>(size), std::vector<float>(size) },
    write_idx(0),
    read_idx(1)
{}

std::span<float> io_buffer_cpu::cur_write_buffer()
{
    return { buffer[write_idx].begin(), buffer[write_idx].end() };
}

std::span<const float> io_buffer_cpu::cur_read_buffer()
{
    return { buffer[read_idx].begin(), buffer[read_idx].end() };
}

bool io_buffer_cpu::snapshot(std::span<float> dst)
{
    if (dst.size() != buffer[read_idx].size()) return false;
    std::copy(buffer[read_idx].begin(), buffer[read_idx].end(), dst.begin());
    return true;
}

bool io_buffer_cpu::restore(std::span<const float> src)
{
    // Restore the read page directly — the consumer reads it on the next step,
    // and the producer overwrites the write page anyway, so there is nothing to
    // swap (design §5.3).
    if (src.size() != buffer[read_idx].size()) return false;
    std::copy(src.begin(), src.end(), buffer[read_idx].begin());
    return true;
}

} // sim
