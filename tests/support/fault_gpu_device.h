// tests/support/fault_gpu_device.h
//
// An in-memory gpu_device for fault injection (#234). Buffers are plain host byte
// arrays (one page each: the tests that use it never step, so the ping-pong
// generation is irrelevant), uploads and readbacks are memcpy, compute is a
// no-op. Needs no adapter, so a test built on it runs in the sandbox and in a
// GPU_BACKEND=null build alike.
//
// Fault injection: `fail_read` makes the N-th readback (1-based, counted in
// `reads`) report failure, as gpu_device::readback does on a failed transfer;
// `fail_if` fails every readback it returns true for. A failed readback leaves
// `dst` untouched, like the WebGPU backend's failed map. `uploads` counts the
// uploads, so a test can tell whether anything was pushed to the device.

#ifndef COAST_TESTS_SUPPORT_FAULT_GPU_DEVICE_H
#define COAST_TESTS_SUPPORT_FAULT_GPU_DEVICE_H

#include "core/gpu/gpu_device.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace coast::test {

struct fault_gpu_device final : coast::gpu::gpu_device {
    using buffer_id = coast::gpu::buffer_id;

    std::vector<std::vector<std::byte>> buffers;
    std::size_t reads     = 0;   // readbacks issued so far
    std::size_t uploads   = 0;   // uploads issued so far
    std::size_t fail_read = 0;   // 1-based readback to fail; 0 = none
    std::function<bool(buffer_id, std::size_t bytes)> fail_if;   // fail these too

    std::expected<void, core::step_error> check() override { return {}; }
    void wait_idle() override {}
    [[nodiscard]] std::string adapter_name() const override { return "fault injection"; }

    buffer_id register_buffer(coast::gpu::topology, std::size_t bytes, bool) override {
        buffers.emplace_back(bytes, std::byte{0});
        return static_cast<buffer_id>(buffers.size() - 1);
    }
    coast::gpu::kernel_id register_kernel(std::string_view, std::string_view) override {
        return coast::gpu::kernel_id{};
    }
    coast::gpu::plan_id register_plan() override { return coast::gpu::plan_id{}; }
    void build_plan(coast::gpu::plan_id, const coast::gpu::step_plan&) override {}
    void mark_dirty(coast::gpu::plan_id) override {}
    void begin_step() override {}
    void enqueue(coast::gpu::plan_id) override {}
    void submit_step() override {}
    void rendezvous_step() override {}

    void upload(buffer_id id, std::span<const std::byte> data, coast::gpu::access) override {
        auto& b = at(id);
        if (data.size() > b.size()) throw std::logic_error("fault_gpu_device: upload too large");
        std::copy(data.begin(), data.end(), b.begin());
        ++uploads;
    }

    [[nodiscard]] bool readback(buffer_id id, std::span<std::byte> dst) override {
        auto& b = at(id);
        if (dst.size() != b.size()) throw std::logic_error("fault_gpu_device: dst size != buffer size");
        ++reads;
        if (reads == fail_read) return false;
        if (fail_if && fail_if(id, dst.size())) return false;
        std::copy(b.begin(), b.end(), dst.begin());
        return true;
    }

    void enqueue_transfers() override {}
    void complete_transfers() override {}
    void advance_generation() override {}

    // Typed view of a buffer's contents.
    template <class T>
    [[nodiscard]] std::span<T> as(buffer_id id) {
        auto& b = at(id);
        return { reinterpret_cast<T*>(b.data()), b.size() / sizeof(T) };
    }

    std::vector<std::byte>& at(buffer_id id) { return buffers.at(static_cast<std::size_t>(id)); }
};

} // namespace coast::test

#endif // COAST_TESTS_SUPPORT_FAULT_GPU_DEVICE_H
