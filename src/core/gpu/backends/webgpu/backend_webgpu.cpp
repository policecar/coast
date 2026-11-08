//
// Created by jk on the gpu-abstraction-layer refactor (B1).
//
// The single translation unit that sees <webgpu/webgpu.h>. It implements the
// agnostic coast::gpu::gpu_device facade against wgpu-native, owning the
// gpu_context and translating opaque handles (buffer_id / kernel_id / plan_id)
// to wgpu resources.
//
// Landed so far: B1 skeleton (gpu_context ownership + buffer/kernel registries +
// generation counter); B2 build_plan (compile a step_plan into an ordered list
// of compiled ops — copies verbatim + compute passes with pre-built per-
// generation bind groups); B3 per-step batch (begin/enqueue/submit/rendezvous,
// command stream identical to today's record()). The data plane (B4), dirty/
// rebuild (B5) and swap transfers (F2) are stubbed to throw until their nodes
// land, so the surface compiles and links as a complete gpu_device.
//

#ifdef COAST_WITH_WEBGPU

#include "core/gpu/gpu_device.h"

#include "gpu_context.h"
#include "wgsl_module.h"
#include "wgpu_util.h"
#include "diag/diag.h"

#include <webgpu/webgpu.h>

#include <cstdio>
#include <cstring>
#include <expected>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace coast::gpu {

namespace {

// Map a buffer's role to wgpu usage flags. Storage buffers get CopyDst|CopySrc
// generously (the entities already allocate this way: gather targets need
// CopyDst, publish/readback sources need CopySrc); uniforms get Uniform|CopyDst.
// Refined per-topology only if a real need appears in B4/D2.
WGPUBufferUsage usage_for(topology t) {
    switch (t) {
        case topology::uniform:
            return static_cast<WGPUBufferUsage>(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        case topology::gpu_resident:
        case topology::cpu_to_gpu:
        case topology::gpu_to_cpu:
        case topology::gpu_to_gpu:
        default:
            return static_cast<WGPUBufferUsage>(
                WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
    }
}

WGPUBuffer make_buffer(WGPUDevice dev, std::uint64_t bytes, WGPUBufferUsage usage) {
    WGPUBufferDescriptor d = {};
    d.usage = usage;
    d.size  = bytes == 0 ? 1u : bytes;   // wgpu rejects zero-size; 1-byte placeholder
    WGPUBuffer buf = wgpuDeviceCreateBuffer(dev, &d);
    if (buf == nullptr)
        throw std::runtime_error("backend_webgpu: wgpuDeviceCreateBuffer returned null");
    return buf;
}

// Async-map completion latch for the synchronous readback path (mirrors the
// per-TU helper the entities use today; pumped via coast::gpu::pump_until).
struct map_result { bool done = false; bool ok = false; };
void on_map(WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
    auto* r = static_cast<map_result*>(ud1);
    r->ok   = (status == WGPUMapAsyncStatus_Success);
    r->done = true;
}

// ---------------------------------------------------------------------------
// backend_webgpu — the concrete gpu_device. Private to this TU; reached only
// through make_gpu_device(). Constructs and privately owns its own gpu_context
// (no SDK type crosses the agnostic seam).
// ---------------------------------------------------------------------------
class backend_webgpu final : public gpu_device {
public:
    backend_webgpu()
        : ctx_(std::make_unique<gpu_context>())
    {
        instance_ = ctx_->instance;
        device_   = ctx_->device;
        queue_    = ctx_->queue;
    }

    // ---- backend identity / fault / sync (folded off gpu_context) ----
    // The single firewall translating the context's throwing fault check into a
    // value on the agnostic channel. The driver detail (only available as the
    // exception's message) is logged here; the channel carries the category.
    std::expected<void, core::step_error> check() override {
        try {
            ctx_->check_or_throw();
            return {};
        } catch (const std::exception& e) {
            // This firewall converts a throwing fault into a value on the sim
            // thread. COAST_ERROR is non-throwing by construction (emit_guarded
            // swallows the std::format OOM that would otherwise escape and defeat
            // the firewall), so it is safe here unwrapped.
            COAST_ERROR("gpu", "device fault: {}", e.what());
            return std::unexpected(core::step_error{
                core::error_kind::device_fault, core::step_error::no_entity, COAST_SITE});
        }
    }
    void wait_idle()      override { ctx_->wait_idle(); }
    [[nodiscard]] std::string adapter_name() const override { return ctx_->adapter_name; }

    ~backend_webgpu() override {
        // Release wgpu resources BEFORE the device dies (ctx_ destroyed last,
        // below). Plan bind groups, pages, pipelines, then cached modules.
        for (auto& pl : plans_)
            release_plan(pl);
        for (auto& b : buffers_)
            for (WGPUBuffer page : b.pages)
                if (page) wgpuBufferRelease(page);
        for (auto& k : kernels_)
            if (k.pipeline) wgpuComputePipelineRelease(k.pipeline);
        for (auto& [src, mod] : modules_)
            if (mod) wgpuShaderModuleRelease(mod);
        // ctx_ (and the device) released here as the member unwinds.
    }

    // ---- registration (B1) ----

    buffer_id register_buffer(topology topo, std::size_t bytes, bool ping_pong) override {
        buffer_rec rec{};
        rec.topo      = topo;
        rec.bytes     = bytes;
        rec.ping_pong = ping_pong;
        const WGPUBufferUsage usage = usage_for(topo);
        rec.pages[0] = make_buffer(device_, bytes, usage);
        rec.pages[1] = ping_pong ? make_buffer(device_, bytes, usage) : nullptr;
        const auto id = static_cast<buffer_id>(buffers_.size());
        buffers_.push_back(rec);
        return id;
    }

    kernel_id register_kernel(std::string_view source, std::string_view entry) override {
        // Cache the compiled module by source so an entity's N entry points share
        // one compile (mirrors today's single-module / many-entry pattern).
        const std::string key(source);
        auto it = modules_.find(key);
        if (it == modules_.end()) {
            WGPUShaderModule mod = load_wgsl_module(instance_, device_, source, "gpu_device");
            it = modules_.emplace(key, mod).first;
        }
        WGPUComputePipelineDescriptor d = {};
        d.compute.module     = it->second;
        d.compute.entryPoint = sv_from(entry);
        WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(device_, &d);
        if (pipe == nullptr)
            throw std::runtime_error("backend_webgpu: pipeline '" + std::string(entry) + "' failed");
        const auto id = static_cast<kernel_id>(kernels_.size());
        kernels_.push_back(kernel_rec{it->second, pipe});
        return id;
    }

    // ---- plan compilation (B2) ----

    plan_id register_plan() override {
        const auto id = static_cast<plan_id>(plans_.size());
        plans_.emplace_back();
        return id;
    }

    // Compile a step_plan into an ordered list of compiled ops: copies stored
    // verbatim (ping_pong endpoints resolved per generation at enqueue time, B3);
    // compute passes compiled to a borrowed pipeline + PRE-BUILT per-generation
    // bind groups (one set per sub-dispatch). Pipelines are borrowed from kernels_
    // (owned there); the plan owns only its bind groups.
    //
    // RE-CALLABLE — this is the rebuild path the dirty/epoch mechanism (B5) drives:
    // on a genuine structural change (runtime plasticity) the entity bumps its
    // graph_epoch(); env (F3) detects it, calls mark_dirty(id), and re-emits a
    // FRESH step_plan here. A rebuild over a LIVE plan is a SYNCHRONIZING boundary
    // (design §5): in-flight GPU work may still reference the old bind groups, so
    // DRAIN before releasing them. The first build (no ops yet, device idle) does
    // not drain.
    void build_plan(plan_id id, const step_plan& plan) override {
        plan_rec& rec = plan_at(id);
        if (!rec.ops.empty())                      // rebuild over live resources: flush first
            ctx_->wait_idle();
        release_plan(rec);                         // release prior build's bind groups
        rec.ops.clear();
        rec.ops.reserve(plan.ops.size());

        for (const plan_op& op : plan.ops) {
            compiled_op cop{};
            cop.kind = op.kind;
            if (op.kind == op_kind::copy) {
                cop.copy = op.copy;
            } else {
                cop.pass.pipeline      = kernel_at(op.kernel).pipeline;
                cop.pass.uniform_gated = op.uniform_gated;
                cop.pass.dispatches.reserve(op.dispatches.size());
                for (const plan_dispatch& d : op.dispatches) {
                    compiled_dispatch cd{};
                    cd.dispatch   = d.dispatch;
                    cd.parity_dep = dispatch_is_parity_dependent(d);
                    cd.bg[0]      = make_bind_group(d.bindings, cop.pass.pipeline, /*gen=*/0);
                    cd.bg[1]      = cd.parity_dep
                                        ? make_bind_group(d.bindings, cop.pass.pipeline, /*gen=*/1)
                                        : nullptr;
                    cop.pass.dispatches.push_back(std::move(cd));
                }
            }
            rec.ops.push_back(std::move(cop));
        }
        rec.dirty = false;                         // freshly (re)built -> clean
    }

    // ---- per-step batch (B3) ----
    //
    // begin/enqueue/submit/rendezvous realize the env step contract. One encoder
    // spans the whole step: begin_step() opens it, each GPU entity's enqueue()
    // appends its plan's command stream, submit_step() finishes+submits WITHOUT
    // waiting (so the CPU entities run concurrently), rendezvous_step() is the
    // mandatory per-step coherence wait (BARRIER 1). The recorded stream is
    // identical to today's record(): copies and compute passes in plan order,
    // each compute pass setPipeline-once then setBindGroup+dispatch per sub-step.

    void begin_step() override {
        if (encoder_ != nullptr)
            throw std::logic_error("backend_webgpu::begin_step: a step is already open");
        encoder_ = wgpuDeviceCreateCommandEncoder(device_, nullptr);
        if (encoder_ == nullptr)
            throw std::runtime_error("backend_webgpu::begin_step: encoder creation failed");
    }

    void enqueue(plan_id id) override {
        if (encoder_ == nullptr)
            throw std::logic_error("backend_webgpu::enqueue: no open step (call begin_step)");
        const std::uint64_t gen = generation_;
        const plan_rec& rec = plan_at(id);
        for (const compiled_op& op : rec.ops) {
            if (op.kind == op_kind::copy) {
                const WGPUBuffer src = resolve_page(op.copy.src, gen, op.copy.src_side);
                const WGPUBuffer dst = resolve_page(op.copy.dst, gen, op.copy.dst_side);
                wgpuCommandEncoderCopyBufferToBuffer(
                    encoder_, src, op.copy.src_offset, dst, op.copy.dst_offset, op.copy.bytes);
            } else {
                WGPUComputePassEncoder pass =
                    wgpuCommandEncoderBeginComputePass(encoder_, nullptr);
                wgpuComputePassEncoderSetPipeline(pass, op.pass.pipeline);
                for (const compiled_dispatch& d : op.pass.dispatches) {
                    WGPUBindGroup bg = d.parity_dep ? d.bg[gen & 1u] : d.bg[0];
                    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                    wgpuComputePassEncoderDispatchWorkgroups(
                        pass, d.dispatch[0], d.dispatch[1], d.dispatch[2]);
                }
                wgpuComputePassEncoderEnd(pass);
                wgpuComputePassEncoderRelease(pass);
            }
        }
    }

    void submit_step() override {
        if (encoder_ == nullptr)
            throw std::logic_error("backend_webgpu::submit_step: no open step");
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder_, nullptr);
        wgpuQueueSubmit(queue_, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder_);
        encoder_ = nullptr;
    }

    // BARRIER 1: block until the submitted batch is coherent. This cycle's
    // backend realizes coherence as a full device wait (wgpuDevicePoll(wait)).
    void rendezvous_step() override { ctx_->wait_idle(); }

    // ---- generation tick (B1; trivial) ----

    void advance_generation() override { ++generation_; }

    // ---- not yet implemented (filled by later nodes) ----

    // Mark a plan dirty: the explicit "a rebuild boundary is coming" signal
    // (design §5 — rebuild must be a VISIBLE barrier, never silent). Set by env
    // (F3) when an entity's graph_epoch() changes; the paired build_plan()
    // re-emission then performs the drain + swap and clears the flag. A keep-warm
    // backend (M3) would read this to flush its double-buffered submit pipeline
    // before the rebuild. Dormant this cycle (no entity bumps its epoch); the path
    // is proven end-to-end by G7's forced-bump test.
    void mark_dirty(plan_id id) override { plan_at(id).dirty = true; }

    // ---- data plane (B4) ----

    // Stage `data` into the buffer the upcoming step will read. wgpuQueueWriteBuffer
    // is ordered ahead of the next queue submit, so a value written here is visible
    // to this step's enqueue()'d batch — byte-identical to today's per-step
    // QueueWriteBuffer of the entity uniforms. The uniform path: a uniform buffer is
    // non-ping_pong, so this targets its single page.
    void upload(buffer_id id, std::span<const std::byte> data,
                access side = access::write) override {
        buffer_rec& b = buffer_at(id);
        if (data.size() > b.bytes)
            throw std::logic_error("backend_webgpu::upload: data larger than buffer");
        WGPUBuffer page = resolve_page(id, generation_, side);
        wgpuQueueWriteBuffer(queue_, page, 0, data.data(), data.size());
    }

    // Synchronous GPU->CPU readback of `id`'s current page into `dst` (observers /
    // logging / state-save at a step boundary). Copies into a transient MapRead
    // staging buffer, maps, memcpy's out. Assumes the device is idle (true at a
    // boundary, after rendezvous_step()). Returns false on a map failure; the
    // staging buffer is always released. Mirrors the entities' readback_bytes().
    [[nodiscard]] bool readback(buffer_id id, std::span<std::byte> dst) override {
        buffer_rec& b = buffer_at(id);
        if (dst.size() != b.bytes)
            throw std::logic_error("backend_webgpu::readback: dst size != buffer size");
        const std::uint64_t bytes = static_cast<std::uint64_t>(dst.size());
        if (bytes == 0) return true;

        // READ side: readback serves boundary observers / logging / state-save,
        // which want the PUBLISHED page (what a consumer reads), not the write
        // page the next producer will overwrite. For a ping_pong buffer at
        // generation G that is the page produced at G-1 (resolve_page read side);
        // for non-ping_pong buffers it is the single page. Symmetric to upload(),
        // which targets the write side.
        WGPUBuffer src     = resolve_page(id, generation_, access::read);
        WGPUBuffer staging = make_buffer(device_, bytes,
            static_cast<WGPUBufferUsage>(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst));

        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, staging, 0, bytes);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(queue_, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);

        map_result mr;
        WGPUBufferMapCallbackInfo mc = {};
        mc.mode      = WGPUCallbackMode_AllowProcessEvents;
        mc.callback  = on_map;
        mc.userdata1 = &mr;
        wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, bytes, mc);
        pump_until(instance_, mr.done);
        if (!mr.ok) {
            wgpuBufferRelease(staging);
            return false;
        }
        const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, bytes);
        std::memcpy(dst.data(), mapped, static_cast<std::size_t>(bytes));
        wgpuBufferUnmap(staging);
        wgpuBufferRelease(staging);
        return true;
    }

    void enqueue_transfers()                    override { unimplemented("enqueue_transfers (F2)"); }
    void complete_transfers()                   override { unimplemented("complete_transfers (F2)"); }

private:
    [[noreturn]] static void unimplemented(const char* what) {
        throw std::logic_error(std::string("backend_webgpu::") + what + " not yet implemented");
    }

    struct buffer_rec {
        topology               topo      = topology::gpu_resident;
        std::size_t            bytes     = 0;
        bool                   ping_pong = false;
        std::array<WGPUBuffer,2> pages{nullptr, nullptr};   // [1] used only when ping_pong
    };
    struct kernel_rec {
        WGPUShaderModule    module   = nullptr;   // borrowed from modules_ (not owned here)
        WGPUComputePipeline pipeline = nullptr;   // owned
    };
    // A compiled sub-dispatch: pre-built bind groups (owned) + workgroup counts.
    // bg[g] is the bind group for generation g; bg[1] is null unless parity_dep.
    struct compiled_dispatch {
        std::array<std::uint32_t,3> dispatch{1,1,1};
        std::array<WGPUBindGroup,2> bg{nullptr, nullptr};
        bool                        parity_dep = false;
    };
    // A compiled compute pass: one borrowed pipeline + an ordered list of
    // sub-dispatches recorded within a SINGLE wgpu compute pass.
    struct compiled_pass {
        WGPUComputePipeline            pipeline      = nullptr;   // borrowed
        bool                           uniform_gated = false;
        std::vector<compiled_dispatch> dispatches;
    };
    // A compiled op, mirroring plan_op: a copy (verbatim, resolved per gen at
    // enqueue) or a compute pass.
    struct compiled_op {
        op_kind       kind = op_kind::copy;
        plan_copy     copy{};        // when kind == copy
        compiled_pass pass{};        // when kind == compute
    };
    struct plan_rec {
        std::vector<compiled_op> ops;
        bool                     dirty = false;   // B5: set by mark_dirty, cleared by build_plan
    };

    // Release every bind group a plan owns (idempotent; safe on an empty rec).
    static void release_plan(plan_rec& rec) {
        for (compiled_op& op : rec.ops)
            if (op.kind == op_kind::compute)
                for (compiled_dispatch& d : op.pass.dispatches)
                    for (WGPUBindGroup bg : d.bg)
                        if (bg) wgpuBindGroupRelease(bg);
    }

    // ---- registry accessors (bounds-checked: a bad id is a programmer error) ----
    buffer_rec& buffer_at(buffer_id id) { return buffers_.at(static_cast<std::size_t>(id)); }
    kernel_rec& kernel_at(kernel_id id) { return kernels_.at(static_cast<std::size_t>(id)); }
    plan_rec&   plan_at  (plan_id   id) { return plans_  .at(static_cast<std::size_t>(id)); }

    // Resolve a buffer's physical page for generation `gen` and access `side`
    // (non-ping_pong always page 0). The double-buffer convention: the WRITE side
    // is the current page (gen&1); the READ side is the OPPOSITE page ((gen&1)^1),
    // which holds what a producer wrote the previous generation -> one-step
    // latency across entities. A producer writing at gen G touches pages[G&1]; a
    // consumer reading the same buffer at gen G+1 touches pages[(G+1&1)^1] =
    // pages[G&1] = exactly that page.
    WGPUBuffer resolve_page(buffer_id id, std::uint64_t gen, access side) {
        buffer_rec& b = buffer_at(id);
        if (!b.ping_pong) return b.pages[0];
        const std::uint64_t cur = gen & 1u;
        return b.pages[side == access::write ? cur : (cur ^ 1u)];
    }

    // Boundary single-page resolve (upload/readback): targets the current/write
    // page, preserving the pre-side-tag behavior for the data plane.
    WGPUBuffer resolve_page(buffer_id id, std::uint64_t gen) {
        return resolve_page(id, gen, access::write);
    }

    bool dispatch_is_parity_dependent(const plan_dispatch& d) {
        for (const plan_binding& bd : d.bindings)
            if (buffer_at(bd.buffer).ping_pong) return true;
        return false;
    }

    // Build one dispatch's bind group against group 0 of the pipeline's
    // auto-layout, resolving each binding's buffer page for generation `gen`. The
    // whole buffer is bound (size = its byte size). plan_binding::mode selects the
    // ping_pong page side (write -> current page, read -> opposite/older page), the
    // same convention copies use; for non-ping_pong buffers mode is irrelevant.
    WGPUBindGroup make_bind_group(const std::vector<plan_binding>& bindings,
                                  WGPUComputePipeline pipe, std::uint64_t gen) {
        std::vector<WGPUBindGroupEntry> entries;
        entries.reserve(bindings.size());
        for (const plan_binding& bd : bindings) {
            WGPUBindGroupEntry e{};
            e.binding = bd.slot;
            e.buffer  = resolve_page(bd.buffer, gen, bd.mode);
            e.size    = buffer_at(bd.buffer).bytes;
            entries.push_back(e);
        }
        // wgpuComputePipelineGetBindGroupLayout returns an OWNED (ref-counted)
        // handle — release it once the bind group has captured it, or it leaks
        // one layout per bind group built (and build_plan is the re-callable
        // rebuild path, so it would leak continuously).
        WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipe, 0);
        WGPUBindGroupDescriptor d = {};
        d.layout     = layout;
        d.entryCount = entries.size();
        d.entries    = entries.data();
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device_, &d);
        wgpuBindGroupLayoutRelease(layout);
        if (bg == nullptr)
            throw std::runtime_error("backend_webgpu: bind group creation failed");
        return bg;
    }

    std::unique_ptr<gpu_context> ctx_;            // owns instance/device/queue
    WGPUInstance instance_ = nullptr;             // borrowed convenience handles
    WGPUDevice   device_   = nullptr;
    WGPUQueue    queue_    = nullptr;

    std::vector<buffer_rec> buffers_;             // indexed by buffer_id
    std::vector<kernel_rec> kernels_;             // indexed by kernel_id
    std::vector<plan_rec>   plans_;               // indexed by plan_id
    std::unordered_map<std::string, WGPUShaderModule> modules_;  // source -> module (owned)

    std::uint64_t generation_ = 0;                // ping-pong tick (modulus private)

    WGPUCommandEncoder encoder_ = nullptr;        // open step encoder (begin..submit)
};

}  // namespace

std::unique_ptr<gpu_device> make_gpu_device() {
    return std::make_unique<backend_webgpu>();
}

}  // namespace coast::gpu

#endif  // COAST_WITH_WEBGPU
