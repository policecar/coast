// tests/integration_ngm3/gpu_forced_rebuild_test.cpp
//
// G7 — the dirty/rebuild path (plan node B5) proven end-to-end.
//
// In the current model the rebuild path is DORMANT: no production entity ever
// bumps its graph_epoch(), and env::process() does not yet react to an epoch
// change. The path still has to be *correct* the day a runtime-plasticity entity
// needs it, so this test exercises it directly through a dedicated, minimal GPU
// entity rather than reaching into the real ngm3 entity (which would have to be
// modified purely to serve a test).
//
// `rebuild_demo_entity` below doubles as the reference implementation for a
// future rebuild-capable gpu_io_entity. It shows the exact shape the feature
// needs:
//   * build_step_plan()  — register buffers/kernels ONCE, emit the plan, build it
//   * emit_plan()        — a PURE function of the entity's current structure; the
//                          single source of truth re-used by build and rebuild
//   * graph_epoch()      — bumped only on a genuine structural change
//   * rebuild_plan()     — re-emit the (possibly changed) plan and build_plan()
//                          it onto the SAME plan_id, leaving the entity's live
//                          device buffers untouched
//
// The harness plays the role env will eventually play: after each step's host
// bookkeeping it compares graph_epoch() to the last seen value and, on a change,
// calls mark_dirty() + rebuild_plan() before enqueuing the step. That env-side
// reaction is NOT wired into env::process() this cycle (by design — the path is
// dormant); this test is its executable specification.
//
// The probe is a ping-pong accumulator: acc_out[i] = acc_in[i] + add, with `add`
// driven by a per-step uniform and the two pages flipped each step via
// advance_generation(). It is deliberately STATEFUL and PARITY-DEPENDENT:
//   * stateful  — the result after N steps is the running sum, so a rebuild that
//                 silently dropped the live buffer contents would change the
//                 output (caught by the control-vs-rebuild comparison) and a
//                 degenerate no-op kernel would miss the closed-form sum.
//   * parity-dependent — binding the ping-pong buffer makes the dispatch build
//                 BOTH even/odd bind groups, so rebuild_plan() actually releases
//                 and recreates the per-generation resources B5 is responsible
//                 for. A rebuild that mishandled parity would diverge.
//
// GPU-labeled: needs a real device, excluded by `ctest -LE gpu`.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "core/gpu/gpu_device.h"
#include "sim_core/gpu_io_entity.h"

namespace {

using namespace coast::gpu;

constexpr std::uint32_t k_n     = 256;   // accumulator elements
constexpr std::uint32_t k_steps = 64;    // accumulation steps
constexpr std::uint32_t k_wgsz  = 64;    // WGSL workgroup_size

// Minimal WGSL: out[i] = in[i] + u.add. Params padded to 16 bytes (uniform
// alignment). in/out are the two pages of one ping-pong storage buffer.
constexpr std::string_view k_wgsl = R"WGSL(
struct Params { add : f32, n : u32, _pad0 : u32, _pad1 : u32 };
@group(0) @binding(0) var<storage, read>       acc_in  : array<f32>;
@group(0) @binding(1) var<storage, read_write> acc_out : array<f32>;
@group(0) @binding(2) var<uniform>             u       : Params;

@compute @workgroup_size(64)
fn acc_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= u.n) { return; }
    acc_out[i] = acc_in[i] + u.add;
}
)WGSL";

struct params_data_t {
    float         add   = 0.0f;
    std::uint32_t n     = 0;
    std::uint32_t pad0  = 0;
    std::uint32_t pad1  = 0;
};

// A rebuild-capable GPU entity (see file header). Derives from gpu_io_entity so
// it is a faithful example of the contract a real entity would implement; the
// io_entity pure-virtuals are stubbed because this test drives the device
// directly rather than through env's dispatcher.
class rebuild_demo_entity : public sim::gpu_io_entity {
public:
    // `rebuild_at_step` == npos means "never bump epoch" (the control entity).
    static constexpr std::uint32_t never = ~0u;

    explicit rebuild_demo_entity(std::uint32_t rebuild_at_step) noexcept
        : rebuild_at_(rebuild_at_step) {}

    // ---- register ONCE, then build the plan (mirrors neuron_group_gpu_t) -----
    void build_step_plan(gpu_device& dev) override {
        dev_ = &dev;

        // One ping-pong storage buffer (two pages); both pages start at zero.
        acc_ = dev.register_buffer(topology::gpu_resident, k_n * sizeof(float), /*ping_pong=*/true);
        const std::vector<float> zeros(k_n, 0.0f);
        dev.upload(acc_, std::as_bytes(std::span<const float>(zeros)), access::write);
        dev.upload(acc_, std::as_bytes(std::span<const float>(zeros)), access::read);

        uni_    = dev.register_buffer(topology::uniform, sizeof(params_data_t), /*ping_pong=*/false);
        kernel_ = dev.register_kernel(k_wgsl, "acc_main");

        plan_id_ = dev.register_plan();
        dev.build_plan(plan_id_, emit_plan());
    }

    // ---- per-step host bookkeeping: refresh the uniform (mirrors process()) --
    // The harness sets the step first via set_step(); process() pushes the
    // resulting uniform through the data plane, exactly as a real entity would.
    sim::step_result process() override {
        params_data_t pd{};
        pd.add = static_cast<float>(step_ + 1);   // add 1,2,3,... -> triangular sum
        pd.n   = k_n;
        dev_->upload(uni_, std::as_bytes(std::span<const params_data_t>(&pd, 1)));
        return {};
    }

    // Advance the host step counter and bump the structural epoch when the
    // configured rebuild step is reached. A real entity would bump here only on
    // a genuine graph change; we bump on a schedule to force the rebuild.
    void set_step(std::uint32_t step) noexcept {
        step_ = step;
        if (step == rebuild_at_) ++epoch_;
    }

    // ---- the forward-looking rebuild hook ------------------------------------
    // Re-emit the plan and rebuild it onto the SAME plan_id. The entity's device
    // buffers (acc_ pages, uniform) are NOT re-registered, so the live
    // accumulated state survives; only the plan's cached pipelines + per-
    // generation bind groups are released and recreated. This is the call env
    // will make after detecting an epoch change.
    void rebuild_plan(gpu_device& dev) {
        dev.build_plan(plan_id_, emit_plan());
    }

    [[nodiscard]] plan_id       plan()        const override { return plan_id_; }
    [[nodiscard]] std::uint64_t graph_epoch() const override { return epoch_; }
    [[nodiscard]] buffer_id     acc_buffer()  const noexcept { return acc_; }

    // ---- io_entity pure-virtual stubs (unused: driven at the device level) ---
    [[nodiscard]] sim::output_id_t              get_outp_id()   const override { return 0; }
    [[nodiscard]] std::size_t                   get_outp_size() const override { return k_n; }
    [[nodiscard]] std::span<const std::size_t>  get_inp_ids()   const override { return {}; }

private:
    // The single source of truth for this entity's per-step GPU work. PURE: a
    // function of the entity's current structure only. build and rebuild both
    // call it, so they can never drift.
    [[nodiscard]] step_plan emit_plan() const {
        step_plan sp{};
        plan_op op{};
        op.kind   = op_kind::compute;
        op.kernel = kernel_;
        plan_dispatch d{};
        d.bindings = {
            {acc_, access::read,  0},   // acc_in  : older page (cross-step state)
            {acc_, access::write, 1},   // acc_out : current page
            {uni_, access::read,  2},   // per-step add
        };
        d.dispatch = {(k_n + k_wgsz - 1) / k_wgsz, 1, 1};
        op.dispatches.push_back(std::move(d));
        sp.ops.push_back(std::move(op));
        return sp;
    }

    gpu_device*   dev_      = nullptr;
    buffer_id     acc_      = invalid_buffer;
    buffer_id     uni_      = invalid_buffer;
    kernel_id     kernel_   = invalid_kernel;
    plan_id       plan_id_  = invalid_plan;
    std::uint32_t step_     = 0;
    std::uint32_t rebuild_at_ = never;
    std::uint64_t epoch_    = 0;
};

// Drive the accumulator for k_steps. The harness mirrors env's step loop AND the
// epoch->mark_dirty->rebuild reaction env will eventually own. Returns the final
// accumulated buffer.
std::vector<float> run_accumulator(std::uint32_t rebuild_at_step) {
    auto dev = make_gpu_device();

    rebuild_demo_entity ent(rebuild_at_step);
    ent.build_step_plan(*dev);

    std::uint64_t last_epoch = ent.graph_epoch();
    for (std::uint32_t s = 0; s < k_steps; ++s) {
        ent.set_step(s);
        sim::throw_on_step_fault(ent.process(), "ent::process");  // refresh uniforms (env::process host part)

        // env's role: a structural change triggers a VISIBLE rebuild barrier.
        if (ent.graph_epoch() != last_epoch) {
            dev->mark_dirty(ent.plan());
            ent.rebuild_plan(*dev);
            last_epoch = ent.graph_epoch();
        }

        dev->begin_step();
        dev->enqueue(ent.plan());
        dev->submit_step();
        dev->rendezvous_step();              // BARRIER 1
        dev->advance_generation();           // flip ping-pong pages (the per-step swap)
    }

    std::vector<float> out(k_n);
    REQUIRE(dev->readback(ent.acc_buffer(),
                          std::as_writable_bytes(std::span<float>(out))));
    return out;
}

} // namespace

TEST_CASE("gpu forced-epoch rebuild is output-transparent (G7)") {
    // Control: epoch never changes -> no rebuild, plain N-step accumulation.
    const std::vector<float> control = run_accumulator(rebuild_demo_entity::never);

    // Rebuild: force a structural-epoch bump (and thus a live-plan rebuild) at
    // the midpoint. The drain + bind-group release/recreate must leave the
    // accumulated device state and the per-generation parity intact.
    const std::vector<float> rebuilt = run_accumulator(k_steps / 2);

    // (1) The rebuild changed nothing observable: byte-identical to no-rebuild.
    CHECK(rebuilt == control);

    // (2) The run actually accumulated state across all steps (so (1) is not the
    //     trivial "both zero" pass). add = 1,2,...,k_steps -> triangular sum,
    //     an exact integer in f32 (<= 2^24), so compare exactly.
    const float expected =
        static_cast<float>(k_steps) * static_cast<float>(k_steps + 1) / 2.0f;
    for (std::size_t i = 0; i < control.size(); ++i) {
        REQUIRE(control[i] == expected);
    }
}
