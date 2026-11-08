// Isolation probe for the env two-phase encoder dispatcher (S4 /
// develop.s4.trivial-gpu-entity-test).
//
// S1-S3 each isolated a new GPU mechanism in a standalone unsandboxed exe before
// integrating it through configs; this does the same for the dispatcher added to
// env::process() / env::swap_io() in S4. It bypasses JSON / sim_builder and
// hand-emplaces three trivial entities into a real sim::env, installs a
// gpu_context, and drives N steps:
//
//   gpu_inc  (gpu_io_entity, id G): each step the kernel writes write_side[i] =
//            read_side[i] + 1 into its own io_buffer_gpu_to_cpu output. The
//            device ping-pong makes this a clean global +1/step counter that the
//            device-driven swap reads back into the CPU read page. After N steps
//            G's CPU read page must equal N.
//   cpu_probe (io_entity,   id C): reads G and copies it into its own CPU output.
//            Proves a downstream CPU consumer sees GPU-produced data across the
//            device seam through the normal input wiring (one-step lag => N-1).
//   cpu_indep (io_entity,   id I): no inputs, writes a step counter. The
//            non-perturbation witness: its output sequence must be byte-identical
//            whether or not the GPU dispatch branches run.
//
// Diagnostic only; not in the production build path. Run UNSANDBOXED — the
// sandbox blocks /dev/dri so Vulkan enumerates zero adapters.

#include "core/gpu/gpu_device.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"
#include "sim_core/gpu_io_entity.h"
#include "sim_core/io_buffer.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kSize  = 64;    // elements per buffer
constexpr int         kSteps = 8;     // simulation steps to drive

constexpr sim::output_id_t kIdGpu  = 10;   // gpu_inc  output
constexpr sim::output_id_t kIdProbe = 20;  // cpu_probe output
constexpr sim::output_id_t kIdIndep = 30;  // cpu_indep output

// ---- trivial CPU entity: writes a step counter into its output --------------
// Independent of any input; the witness that the GPU dispatch leaves CPU work
// untouched.
struct cpu_indep_entity final : sim::io_entity {
    sim::io_buffer* out_ = nullptr;
    float           step_ = 0.0f;

    void set_output_buffer(sim::io_buffer& b) override { out_ = &b; }
    sim::step_result process() override {
        auto w = out_->cur_write_buffer();
        for (auto& v : w) v = step_;
        step_ += 1.0f;
        return {};
    }
    [[nodiscard]] sim::output_id_t get_outp_id()  const override { return kIdIndep; }
    [[nodiscard]] std::size_t      get_outp_size() const override { return kSize; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

// ---- trivial CPU entity: mirrors its single GPU input into its output --------
struct cpu_probe_entity final : sim::io_entity {
    sim::io_buffer* in_  = nullptr;   // the GPU-produced buffer (id kIdGpu)
    sim::io_buffer* out_ = nullptr;
    std::array<std::size_t,1> inp_ids_{ kIdGpu };

    void set_output_buffer(sim::io_buffer& b) override { out_ = &b; }
    void set_input_buffer(sim::output_id_t, sim::io_buffer& b) override { in_ = &b; }
    sim::step_result process() override {
        auto r = in_->cur_read_buffer();
        auto w = out_->cur_write_buffer();
        for (std::size_t i = 0; i < w.size(); ++i) w[i] = r[i];
        return {};
    }
    [[nodiscard]] sim::output_id_t get_outp_id()  const override { return kIdProbe; }
    [[nodiscard]] std::size_t      get_outp_size() const override { return kSize; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return inp_ids_; }
};

// ---- trivial GPU entity: increments its own output buffer each step ----------
// process() is host-only (no-op here). build_step_plan() emits one compute pass:
// write_side[i] = read_side[i] + 1 over its own ping-pong output (one buffer_id,
// the two sides distinguished by access). The device resolves the physical page
// per generation, so no per-step bind-group rebuild is needed.
struct gpu_inc_entity final : sim::gpu_io_entity {
    sim::io_buffer*     out_    = nullptr;

    // The kernel reads the io_buffer READ side (binding 0) and writes the WRITE
    // side (binding 1) of its own ping-pong output — both the same buffer_id,
    // distinguished by access side.
    coast::gpu::gpu_device* dev_     = nullptr;
    coast::gpu::kernel_id   k_inc_   = coast::gpu::invalid_kernel;
    coast::gpu::plan_id     plan_id_ = coast::gpu::invalid_plan;

    gpu_inc_entity() = default;

    gpu_inc_entity(gpu_inc_entity&& o) noexcept
        : sim::gpu_io_entity(std::move(o)),
          out_(o.out_), dev_(o.dev_), k_inc_(o.k_inc_), plan_id_(o.plan_id_)
    {}
    gpu_inc_entity& operator=(gpu_inc_entity&&) = delete;

    ~gpu_inc_entity() override = default;   // device owns the GPU resources

    void set_output_buffer(sim::io_buffer& b) override { out_ = &b; }
    sim::step_result process() override { /* host bookkeeping only; the kernel is the producer */ return {}; }

    void build_step_plan(coast::gpu::gpu_device& dev) override {
        using namespace coast::gpu;
        dev_ = &dev;
        static const char* kWgsl = R"(
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&dst)) { dst[i] = src[i] + 1.0; }
}
)";
        k_inc_ = dev.register_kernel(kWgsl, "main");
        const buffer_id out_page = out_->gpu_page_id();
        step_plan sp;
        plan_op op{};
        op.kind   = op_kind::compute;
        op.kernel = k_inc_;
        plan_dispatch d{};
        d.bindings = {{out_page, access::read, 0}, {out_page, access::write, 1}};
        d.dispatch = {(kSize + 63) / 64, 1, 1};
        op.dispatches.push_back(std::move(d));
        sp.ops.push_back(std::move(op));
        plan_id_ = dev.register_plan();
        dev.build_plan(plan_id_, sp);
    }
    [[nodiscard]] coast::gpu::plan_id plan() const override { return plan_id_; }

    [[nodiscard]] sim::output_id_t get_outp_id()  const override { return kIdGpu; }
    [[nodiscard]] std::size_t      get_outp_size() const override { return kSize; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

// Pull a buffer's CPU read page into a vector for assertions.
std::vector<float> read_page(sim::env& e, sim::output_id_t id) {
    auto opt = e.get_io_buffer(id);
    auto r = (*opt)->cur_read_buffer();
    return { r.begin(), r.end() };
}

bool all_equal(const std::vector<float>& v, float expected) {
    if (v.size() != kSize) { std::printf("    size %zu != %zu\n", v.size(), kSize); return false; }
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i] != expected) {
            std::printf("    [%zu] = %f, expected %f\n", i, v[i], expected);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    try {
        // ---- GPU-enabled env: gpu_inc + cpu_probe + cpu_indep -----------------
        sim::env env;
        env.emplace_back<gpu_inc_entity>();
        env.emplace_back<cpu_probe_entity>();
        env.emplace_back<cpu_indep_entity>();
        env.enable_gpu();                      // enable BEFORE init_io_buffers
        env.init_io_buffers();                 // builds a real io_buffer_gpu_to_cpu for G
        std::printf("adapter: %s\n\n", env.gpu_adapter_name().c_str());

        for (int s = 0; s < kSteps; ++s) {
            sim::throw_on_step_fault(env.process(), "env::process");
            sim::throw_on_step_fault(env.swap_io(), "env::swap_io");
        }

        bool ok = true;

        // 1. GPU kernel ran once per step and round-tripped to the CPU read page.
        auto g_read = read_page(env, kIdGpu);
        if (all_equal(g_read, static_cast<float>(kSteps))) {
            std::printf("  [gpu_inc]   PASS: G read page == %d (kernel + readback per step)\n", kSteps);
        } else {
            std::printf("  [gpu_inc]   FAIL: G read page != %d\n", kSteps); ok = false;
        }

        // 2. Downstream CPU consumer saw GPU data across the seam (one-step lag).
        auto c_read = read_page(env, kIdProbe);
        if (all_equal(c_read, static_cast<float>(kSteps - 1))) {
            std::printf("  [cpu_probe] PASS: C read page == %d (G one step behind)\n", kSteps - 1);
        } else {
            std::printf("  [cpu_probe] FAIL: C read page != %d\n", kSteps - 1); ok = false;
        }

        // ---- CPU-only reference: cpu_indep alone, no GPU dispatch -------------
        sim::env ref;
        ref.emplace_back<cpu_indep_entity>();
        ref.init_io_buffers();
        for (int s = 0; s < kSteps; ++s) {
            sim::throw_on_step_fault(ref.process(), "ref::process");
            sim::throw_on_step_fault(ref.swap_io(), "ref::swap_io");
        }

        // 3. Non-perturbation: cpu_indep output identical with/without GPU dispatch.
        auto i_gpu = read_page(env, kIdIndep);
        auto i_ref = read_page(ref, kIdIndep);
        if (i_gpu == i_ref && all_equal(i_gpu, static_cast<float>(kSteps - 1))) {
            std::printf("  [cpu_indep] PASS: byte-identical with/without GPU dispatch (== %d)\n", kSteps - 1);
        } else {
            std::printf("  [cpu_indep] FAIL: GPU dispatch perturbed CPU output\n"); ok = false;
        }

        std::printf("\n[coast-dispatcher-probe] %s\n", ok ? "ALL PASS" : "FAILURES");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[coast-dispatcher-probe] FAILED: %s\n", e.what());
        return 1;
    }
}
