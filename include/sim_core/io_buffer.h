//
// Created by jk on 04.09.25.
//

#ifndef IO_BUFFER_H
#define IO_BUFFER_H

#include <array>
#include <vector>
#include <cstdint>
#include <span>

#include "core/gpu/gpu_device.h" // agnostic seam: opaque buffer_id / topology / gpu_device

namespace sim {

// Abstract double-buffered float channel with one step of latency between the
// write page and the read page. Writers touch cur_write_buffer(), readers touch
// cur_read_buffer(); swap_buffer() flips the two at the step boundary.
//
// S3 of the GPU port promotes this to a virtual base with four concrete
// variants selected by topology (writer device x cross-device readers):
//   - io_buffer_cpu          : CPU-only (this file)
//   - io_buffer_cpu_to_gpu   : CPU writer, GPU reader   (io_buffer_gpu.{h,cpp})
//   - io_buffer_gpu_to_cpu   : GPU writer, CPU reader   (io_buffer_gpu.{h,cpp})
//   - io_buffer_gpu          : GPU writer, GPU reader   (io_buffer_gpu.{h,cpp})
//
// The base is fully backend-agnostic: it crosses the GPU seam only through the
// opaque gpu_device facade (buffer_id / topology) below. Only the GPU *variant*
// translation units are gated behind COAST_WITH_GPU.
class io_buffer {

public:

    virtual ~io_buffer() = default;

    // Flip write/read pages at the step boundary.
    virtual void swap_buffer() = 0;

    [[nodiscard]] virtual std::size_t size() const = 0;

    // CPU-visible spans. GPU-only variants return empty spans.
    [[nodiscard]] virtual std::span<float>       cur_write_buffer() = 0;
    [[nodiscard]] virtual std::span<const float> cur_read_buffer()  = 0;

    // On-demand synchronous readback of the current read page into dst, for
    // CPU-side consumers (e.g. buffer logging) of a GPU-resident buffer that
    // keeps no standing CPU mirror. Default: not applicable — CPU and mirrored
    // variants already expose their read page through cur_read_buffer(), so this
    // returns false and the caller uses that span. Only io_buffer_gpu overrides
    // it. Must NOT throw: it runs inside a sim-thread hook, and a transient GPU
    // failure should degrade logging gracefully rather than abort the sim.
    // Expects dst.size() == size(); fills dst and returns true on success.
    //
    // New gpu_device path: once env has registered this buffer's GPU page
    // (gpu_register), the default delegates to device_->readback() of the READ
    // side — the published page a consumer sees at the boundary. Legacy variants
    // that still carry their own WGPU pages override this; their override prefers
    // the device path when gpu_backed() (the new pages are the live ones) and
    // falls back to the legacy page otherwise (direct, non-env construction in
    // tests). CPU-only / mirrored variants need no readback and return false.
    [[nodiscard]] virtual bool read_back_sync(std::span<float> dst) {
        if (!gpu_backed()) return false;
        return device_->readback(pp_buffer_, std::as_writable_bytes(dst));
    }

    // New gpu_device path swap-time transfer for a GPU-writer / CPU-reader buffer:
    // pull this step's device output into the CPU read mirror so the CPU consumer
    // sees it next step. env calls it AFTER advance_generation() (the device READ
    // side is then this step's fresh output) and BEFORE swap_buffer() (so the page
    // it fills becomes the read page after the flip). Default: no-op — only the
    // gpu_to_cpu variant needs it; cpu->gpu / gpu->gpu / CPU buffers do nothing.
    virtual void device_to_cpu_readback() {}

    // --- State snapshot / restore (full-state save / load) -------------------
    //
    // Symmetric pair for state persistence. Both target the *read page* — the
    // only buffer state that is coherent at the step boundary, since the write
    // page is overwritten by its producer on every step. save reads the read
    // page via snapshot(); load writes the read page directly via restore(),
    // with no swap to perform or forget (this eliminates the prior
    // write-page+swap restore bug, design §5.3). Each variant reads/writes
    // whatever physically backs its read page:
    //   - io_buffer_cpu / io_buffer_gpu_to_cpu : the CPU read page,
    //   - io_buffer_cpu_to_gpu                 : the CPU mirror *and* the GPU
    //                                            read page (the GPU consumer
    //                                            reads the latter),
    //   - io_buffer_gpu                        : the GPU read page (readback for
    //                                            snapshot, upload for restore).
    // They run on the sim thread (or while the sim is parked) so any wgpu-queue
    // op is correctly serialised. dst/src must have size() elements; both return
    // false on a size mismatch or a failed GPU transfer and must NOT throw — they
    // run in the state-IO path, where a transient GPU fault should degrade to a
    // warning rather than abort save/load.

    // Copy the current read page into dst (state save).
    [[nodiscard]] virtual bool snapshot(std::span<float> dst) = 0;

    // Write src into the current read page (state load). No swap.
    [[nodiscard]] virtual bool restore(std::span<const float> src) = 0;

    // --- Opaque GPU seam (gpu-abstraction-layer refactor) --------------------
    //
    // These carry the backend-agnostic identity of this buffer's GPU page(s) for
    // the gpu_device path. env registers each GPU io_buffer at build time
    // (gpu_register) and the advance_step / transfer path resolves the page via
    // the device using pp_buffer_. CPU-only buffers leave these at their invalid
    // defaults.
    //
    // pp_buffer_ is a ping-pong buffer_id: the device keeps one physical page per
    // generation and resolves the current page at enqueue()/transfer time, so the
    // io_buffer needs only this single opaque id (not a write/read pair).

    void gpu_register(coast::gpu::gpu_device& dev,
                      coast::gpu::buffer_id   page,
                      coast::gpu::topology    topo) noexcept
    {
        device_    = &dev;
        pp_buffer_ = page;
        topo_      = topo;
    }

    [[nodiscard]] coast::gpu::buffer_id gpu_page_id()  const noexcept { return pp_buffer_; }
    [[nodiscard]] coast::gpu::topology  gpu_topology() const noexcept { return topo_; }
    [[nodiscard]] bool                  gpu_backed()   const noexcept { return device_ != nullptr; }

protected:
    coast::gpu::gpu_device* device_    = nullptr;
    coast::gpu::buffer_id   pp_buffer_ = coast::gpu::invalid_buffer;
    coast::gpu::topology    topo_      = coast::gpu::topology::gpu_resident;
};

// CPU-only variant: behaviorally identical to the pre-S3 concrete class. Two
// std::vector pages flipped by an index swap, no derived stats (stats moved to
// hd_ngm2_tools::compute_input_stats in S0).
class io_buffer_cpu final : public io_buffer {

    std::array<std::vector<float>,2> buffer;
    uint8_t write_idx;
    uint8_t read_idx;

public:

    explicit io_buffer_cpu(std::size_t size);

    void swap_buffer() noexcept override
    {
        read_idx  =  write_idx;
        write_idx = (write_idx + 1) & 1;
    }

    [[nodiscard]] std::size_t size() const override { return buffer[0].size(); }

    [[nodiscard]] std::span<float>       cur_write_buffer() override;
    [[nodiscard]] std::span<const float> cur_read_buffer()  override;

    [[nodiscard]] bool snapshot(std::span<float> dst)       override;
    [[nodiscard]] bool restore (std::span<const float> src) override;

};

} // sim

#endif //IO_BUFFER_H
