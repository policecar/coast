//
// Created by jk on the ngm3 GPU port (S3).
//

#ifdef COAST_WITH_GPU

#include "sim_core/io_buffer_gpu.h"

#include <cstring>

namespace sim {

// ---------------------------------------------------------- cpu_to_gpu ------

io_buffer_cpu_to_gpu::io_buffer_cpu_to_gpu(std::size_t size) :
    cpu_{ std::vector<float>(size), std::vector<float>(size) },
    size_(size),
    write_idx_(0),
    read_idx_(1)
{}

std::span<float> io_buffer_cpu_to_gpu::cur_write_buffer() {
    return { cpu_[write_idx_].begin(), cpu_[write_idx_].end() };
}

std::span<const float> io_buffer_cpu_to_gpu::cur_read_buffer() {
    return { cpu_[read_idx_].begin(), cpu_[read_idx_].end() };
}

bool io_buffer_cpu_to_gpu::snapshot(std::span<float> dst) {
    // The CPU mirror holds the read-page content (it is the source env uploads
    // into the GPU read page), so a plain copy reproduces what the GPU consumer
    // reads.
    if (dst.size() != size_) return false;
    const std::size_t bytes = size_ * sizeof(float);
    if (bytes > 0) std::memcpy(dst.data(), cpu_[read_idx_].data(), bytes);
    return true;
}

bool io_buffer_cpu_to_gpu::restore(std::span<const float> src) {
    // Restore the read page directly (no swap): refresh the CPU mirror so
    // cur_read_buffer()/fingerprints match, and upload into the device READ side
    // so the downstream GPU consumer reads the restored data on the next step.
    if (src.size() != size_) return false;
    const std::size_t bytes = size_ * sizeof(float);
    if (bytes > 0) {
        std::memcpy(cpu_[read_idx_].data(), src.data(), bytes);
        if (gpu_backed()) {
            device_->upload(pp_buffer_, std::as_bytes(src), coast::gpu::access::read);
        }
    }
    return true;
}

// ---------------------------------------------------------- gpu_to_cpu ------

io_buffer_gpu_to_cpu::io_buffer_gpu_to_cpu(std::size_t size) :
    cpu_{ std::vector<float>(size), std::vector<float>(size) },
    size_(size),
    write_idx_(0),
    read_idx_(1)
{}

std::span<float> io_buffer_gpu_to_cpu::cur_write_buffer() {
    return {};   // the producer is a GPU kernel; no CPU write page
}

std::span<const float> io_buffer_gpu_to_cpu::cur_read_buffer() {
    return { cpu_[read_idx_].begin(), cpu_[read_idx_].end() };
}

void io_buffer_gpu_to_cpu::device_to_cpu_readback() {
    // Read the device READ side (env has advanced the generation, so it holds
    // this step's fresh GPU output) into cpu_[write_idx_] — the page swap_buffer()
    // then publishes as the read page. No-op until env has registered the page.
    if (!gpu_backed() || size_ == 0) return;
    const std::span<float> dst{cpu_[write_idx_].data(), size_};
    (void)device_->readback(pp_buffer_, std::as_writable_bytes(dst));
}

bool io_buffer_gpu_to_cpu::snapshot(std::span<float> dst) {
    // CPU read page holds the read-back GPU output the CPU consumer reads.
    if (dst.size() != size_) return false;
    const std::size_t bytes = size_ * sizeof(float);
    if (bytes > 0) std::memcpy(dst.data(), cpu_[read_idx_].data(), bytes);
    return true;
}

bool io_buffer_gpu_to_cpu::restore(std::span<const float> src) {
    // The consumer is CPU-side and reads the CPU read page; restore it directly
    // (no swap). The GPU page is the writer kernel's output, re-derived next step,
    // so it needs no upload.
    if (src.size() != size_) return false;
    const std::size_t bytes = size_ * sizeof(float);
    if (bytes > 0) std::memcpy(cpu_[read_idx_].data(), src.data(), bytes);
    return true;
}

// ---------------------------------------------------------------- gpu -------

io_buffer_gpu::io_buffer_gpu(std::size_t size) : size_(size) {}

bool io_buffer_gpu::snapshot(std::span<float> dst) {
    // Read page lives only on the GPU; reuse the device read-page readback.
    return read_back_sync(dst);
}

bool io_buffer_gpu::restore(std::span<const float> src) {
    // Upload a restored state page into the device READ side (state load),
    // symmetric to snapshot()/read_back_sync(). No swap: the consumer reads the
    // read side next step. Returns false on size mismatch / when not yet
    // device-backed; never throws.
    if (src.size() != size_ || !gpu_backed()) return false;
    device_->upload(pp_buffer_, std::as_bytes(src), coast::gpu::access::read);
    return true;
}

} // sim

#endif // COAST_WITH_GPU
