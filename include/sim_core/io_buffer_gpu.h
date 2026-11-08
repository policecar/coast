//
// Created by jk on the ngm3 GPU port (S3).
//

#ifndef COAST_SIM_CORE_IO_BUFFER_GPU_H
#define COAST_SIM_CORE_IO_BUFFER_GPU_H

#ifdef COAST_WITH_GPU

#include <array>
#include <vector>
#include <cstdint>
#include <span>

#include "sim_core/io_buffer.h"

namespace sim {

// The three GPU io_buffer variants. All share the io_buffer double-buffer
// contract: write_idx / read_idx flipped by swap_buffer(), with one step of
// latency. The GPU page itself is owned by the gpu_device (registered by env via
// gpu_register()); these variants are thin CPU-side handles that hold only the
// opaque seam state (in the base) plus, where a CPU side exists, the host mirror
// pages. All cross-device data movement is served by the device through the
// agnostic facade (upload / readback) — there are NO SDK types here.
//
// Index convention (identical to io_buffer_cpu): during a step the producer
// touches the write_idx page, the consumer touches the read_idx page; the swap
// then sets read_idx = old write_idx so the consumer reads the freshly produced
// page on the next step.
//
// GPU pages are zero-initialized by env at registration so step-0 consumers see
// zeros — matching io_buffer_cpu's one-step-lag exactly.

// CPU writer -> GPU reader. The CPU pair holds the producer's output (and keeps
// the buffer CPU-visible for fingerprinting/logging); env uploads the just-
// written CPU page into the device WRITE page each swap so the GPU consumer reads
// it next step.
class io_buffer_cpu_to_gpu final : public io_buffer {

    std::array<std::vector<float>,2> cpu_;
    std::size_t                      size_;
    uint8_t                          write_idx_;
    uint8_t                          read_idx_;

public:

    explicit io_buffer_cpu_to_gpu(std::size_t size);

    void swap_buffer() noexcept override
    {
        read_idx_  =  write_idx_;
        write_idx_ = (write_idx_ + 1) & 1;
    }

    [[nodiscard]] std::size_t size() const override { return size_; }

    [[nodiscard]] std::span<float>       cur_write_buffer() override;
    [[nodiscard]] std::span<const float> cur_read_buffer()  override;

    // Read page = the CPU mirror (snapshot) plus the device READ page (restore):
    // the mirror keeps cur_read_buffer()/fingerprints coherent, while the device
    // read page is what the downstream GPU consumer actually reads. No swap.
    [[nodiscard]] bool snapshot(std::span<float> dst)       override;
    [[nodiscard]] bool restore (std::span<const float> src) override;
};

// GPU writer -> CPU reader. The kernel writes the device page; device_to_cpu_-
// readback() pulls the device READ side into the CPU read mirror so the CPU
// consumer sees it next step.
class io_buffer_gpu_to_cpu final : public io_buffer {

    std::array<std::vector<float>,2> cpu_;
    std::size_t                      size_;
    uint8_t                          write_idx_;
    uint8_t                          read_idx_;

public:

    explicit io_buffer_gpu_to_cpu(std::size_t size);

    void swap_buffer() noexcept override
    {
        read_idx_  =  write_idx_;
        write_idx_ = (write_idx_ + 1) & 1;
    }

    [[nodiscard]] std::size_t size() const override { return size_; }

    [[nodiscard]] std::span<float>       cur_write_buffer() override;   // empty: GPU writes
    [[nodiscard]] std::span<const float> cur_read_buffer()  override;   // the read mirror

    // Pull the device READ-side page (this step's fresh GPU output, after env's
    // advance_generation()) into cpu_[write_idx_] — the page swap_buffer() then
    // makes the read page, so the CPU consumer reads it next step.
    void device_to_cpu_readback() override;

    // Read page = the CPU read page (the read-back GPU output the CPU consumer
    // reads). The GPU page is re-derived by the writer kernel each step, so
    // restore touches only the CPU side. No swap.
    [[nodiscard]] bool snapshot(std::span<float> dst)       override;
    [[nodiscard]] bool restore (std::span<const float> src) override;
};

// GPU writer -> GPU reader. No CPU side; data never leaves the GPU. The device
// owns both ping-pong pages and resolves the current page per generation, so the
// swap is an opaque generation tick (env::swap_io drives advance_generation) and
// the CPU spans are empty. On-demand CPU readback (logging) / state save+load go
// through the device via the base read_back_sync() + restore() below.
class io_buffer_gpu final : public io_buffer {

    std::size_t size_;

public:

    explicit io_buffer_gpu(std::size_t size);

    void swap_buffer() noexcept override {}   // page flip is the device's generation tick

    [[nodiscard]] std::size_t size() const override { return size_; }

    [[nodiscard]] std::span<float>       cur_write_buffer() override { return {}; }
    [[nodiscard]] std::span<const float> cur_read_buffer()  override { return {}; }

    // read_back_sync() is the base default (device readback of the READ side).

    // Read page lives only on the GPU: snapshot reads it back (device readback),
    // restore uploads straight into the device READ page. No swap.
    [[nodiscard]] bool snapshot(std::span<float> dst)       override;
    [[nodiscard]] bool restore (std::span<const float> src) override;
};

} // sim

#endif // COAST_WITH_GPU

#endif // COAST_SIM_CORE_IO_BUFFER_GPU_H
