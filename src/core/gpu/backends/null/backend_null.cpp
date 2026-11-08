//
// Created by jk on the fixing-gpu-abstraction cycle (null backend).
//
// A host-only, dependency-free gpu_device. It carries NO GPU SDK: buffers are
// plain host byte arrays, copies are memcpy, compute is a no-op, upload/readback
// are identity memcpy. Its sole purpose is to PROVE the agnostic seam is genuinely
// backend-neutral — it is the GPU-free build+run target (CI / portability asset).
// It performs no real computation: a config run on it executes the full plumbing
// (register → build_plan → per-step begin/enqueue/submit/rendezvous, swap-time
// uploads + generation flip) end-to-end without crashing, but its numerics are
// not meaningful (every compute pass is skipped, so GPU-produced buffers stay at
// their zero-init value).
//
// It still resolves ping-pong pages exactly as the WebGPU backend does (write =
// current page, read = opposite/older page) and executes copy ops via memcpy, so
// the data-flow shape across the seam is faithful even though compute is absent.
//

#ifdef COAST_WITH_NULL_GPU

#include "core/gpu/gpu_device.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace coast::gpu {

namespace {

// ---------------------------------------------------------------------------
// backend_null — the concrete gpu_device. Private to this TU; reached only
// through make_gpu_device().
// ---------------------------------------------------------------------------
class backend_null final : public gpu_device {
public:
    backend_null()           = default;
    ~backend_null() override = default;

    // ---- backend identity / fault / sync ----
    std::expected<void, core::step_error> check() override { return {}; }  // no async faults
    void wait_idle()      override {}                                   // nothing in flight
    [[nodiscard]] std::string adapter_name() const override { return "null"; }

    // ---- registration ----
    buffer_id register_buffer(topology /*topo*/, std::size_t bytes, bool ping_pong) override {
        buffer_rec rec{};
        rec.bytes     = bytes;
        rec.ping_pong = ping_pong;
        rec.pages[0].assign(bytes, std::byte{0});
        if (ping_pong) rec.pages[1].assign(bytes, std::byte{0});
        const auto id = static_cast<buffer_id>(buffers_.size());
        buffers_.push_back(std::move(rec));
        return id;
    }

    kernel_id register_kernel(std::string_view /*source*/, std::string_view /*entry*/) override {
        return static_cast<kernel_id>(kernels_++);                      // no compilation
    }

    plan_id register_plan() override {
        const auto id = static_cast<plan_id>(plans_.size());
        plans_.emplace_back();
        return id;
    }

    // Keep only the copy ops (replayed via memcpy at enqueue); compute is dropped.
    void build_plan(plan_id id, const step_plan& plan) override {
        plan_rec& rec = plan_at(id);
        rec.copies.clear();
        for (const plan_op& op : plan.ops)
            if (op.kind == op_kind::copy)
                rec.copies.push_back(op.copy);
    }

    void mark_dirty(plan_id /*id*/) override {}

    // ---- per-step batch ----
    void begin_step() override {}

    void enqueue(plan_id id) override {
        for (const plan_copy& c : plan_at(id).copies) {
            const std::vector<std::byte>& src = page(c.src, c.src_side);
            std::vector<std::byte>&       dst = page(c.dst, c.dst_side);
            if (c.bytes == 0) continue;
            if (c.src_offset + c.bytes > src.size() || c.dst_offset + c.bytes > dst.size())
                throw std::logic_error("backend_null::enqueue: copy out of bounds");
            std::memcpy(dst.data() + c.dst_offset, src.data() + c.src_offset,
                        static_cast<std::size_t>(c.bytes));
        }
    }

    void submit_step()     override {}
    void rendezvous_step() override {}

    // ---- generation tick ----
    void advance_generation() override { ++generation_; }

    // ---- data plane ----
    void upload(buffer_id id, std::span<const std::byte> data,
                access side = access::write) override {
        std::vector<std::byte>& pg = page(id, side);
        if (data.size() > pg.size())
            throw std::logic_error("backend_null::upload: data larger than buffer");
        std::memcpy(pg.data(), data.data(), data.size());
    }

    [[nodiscard]] bool readback(buffer_id id, std::span<std::byte> dst) override {
        const std::vector<std::byte>& pg = page(id, access::read);
        if (dst.size() != pg.size())
            throw std::logic_error("backend_null::readback: dst size != buffer size");
        if (!dst.empty()) std::memcpy(dst.data(), pg.data(), dst.size());
        return true;
    }

    void enqueue_transfers()  override {}
    void complete_transfers() override {}

private:
    struct buffer_rec {
        std::size_t                            bytes     = 0;
        bool                                   ping_pong = false;
        std::array<std::vector<std::byte>, 2>  pages{};   // [1] used only when ping_pong
    };
    struct plan_rec {
        std::vector<plan_copy> copies;
    };

    buffer_rec& buffer_at(buffer_id id) { return buffers_.at(static_cast<std::size_t>(id)); }
    plan_rec&   plan_at  (plan_id   id) { return plans_  .at(static_cast<std::size_t>(id)); }

    // Resolve a buffer's physical page for access `side` at the current generation,
    // mirroring the WebGPU backend: non-ping_pong always page 0; write side = the
    // current page (gen&1); read side = the opposite/older page.
    std::vector<std::byte>& page(buffer_id id, access side) {
        buffer_rec& b = buffer_at(id);
        if (!b.ping_pong) return b.pages[0];
        const std::uint64_t cur = generation_ & 1u;
        return b.pages[side == access::write ? cur : (cur ^ 1u)];
    }

    std::vector<buffer_rec> buffers_;        // indexed by buffer_id
    std::vector<plan_rec>   plans_;          // indexed by plan_id
    std::uint64_t           kernels_    = 0; // count only (no compilation)
    std::uint64_t           generation_ = 0;
};

}  // namespace

std::unique_ptr<gpu_device> make_gpu_device() {
    return std::make_unique<backend_null>();
}

}  // namespace coast::gpu

#endif  // COAST_WITH_NULL_GPU
