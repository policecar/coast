//
// Created by jk on the ngm3 GPU port (S5).
//

#ifdef COAST_WITH_GPU

#include "io/mnist_io_gpu.h"

#include "core/gpu/shared_prng.h"
#include "core/require.h"
#include "logging/state_io.h"

#include <format>
#include <span>
#include <stdexcept>
#include <string>

namespace mnist {

namespace {

constexpr std::size_t kImageSize = 784;   // 28*28, asserted against the dataset
                                          // (backend-agnostic — used by the ctor).

// ===========================================================================
// DEVICE-SPECIFIC kernel source + uniform layout. Selected by the GPU backend
// chosen at compile time (COAST_GPU_BACKEND). Today this is WebGPU/WGSL; the
// host-array `null` backend reuses the same std140-shaped struct (layout-agnostic,
// no-ops compute), so both share this arm.
//
// A future NATIVE backend (Metal / Vulkan / CUDA) MUST add its own arm here with
// the corresponding kernel source AND a matching `params_data_t`. It must ALSO
// audit the rest of this TU: process() / build_step_plan reference params_data_t
// by name and assume its layout, so differing device properties prompt a refactor,
// not just a new #ifdef arm.
// ===========================================================================
#if defined(COAST_WITH_WEBGPU) || defined(COAST_WITH_NULL_GPU)

// The compute kernel (img_main), embedded at build time —
// same #embed mechanism as core/gpu/shared_prng.cpp, path relative to this .cpp.
constexpr unsigned char kMnistGpuWgsl[] = {
#embed "mnist_io_gpu.wgsl"
};

// 16-byte uniform refreshed each step via the gpu_device data plane. Layout
// matches the WGSL `struct Params { idx:u32, key:u32, noise_lo:f32, noise_hi:f32 }`.
// One per-step upload of this struct; the constant noise bounds ride along.
struct params_data_t {
    std::uint32_t idx;
    std::uint32_t key;
    float         noise_lo;
    float         noise_hi;
};

#else
#  error "mnist GPU entity: the selected COAST_GPU_BACKEND has no device-specific \
kernel source / uniform layout. Add an arm above (see the DEVICE-SPECIFIC note)."
#endif // device-specific kernel source + uniform layout

} // namespace

mnist_io_gpu::mnist_io_gpu(const mnist_param_t& params, sim::output_id_t output_id) :
    mnist_db(params.image_file, params.label_file),
    params(params),
    id(output_id),
    seq_(dataset::build_sequencer(params, get_image_cnt()))
{
    if (params.noise_range.first > params.noise_range.second) {
        throw std::runtime_error("noise_range[0] must be <= noise_range[1]");
    }
    if (get_image_size() != kImageSize) {
        throw std::runtime_error("mnist_io_gpu: expected image size 784, got " +
                                 std::to_string(get_image_size()));
    }
    // S5 limitation: the kernel always reads the indexed image. The CPU
    // mnist_io's change_interval<=0 "static noise frame" mode is not ported;
    // configs that need it must use the CPU mnist_io. Documented; revisit if a
    // config requires it.
    if (seq_.change_interval() <= 0) {
        throw std::runtime_error("mnist_io_gpu requires change_interval > 0");
    }
}

mnist_io_gpu::mnist_io_gpu(mnist_io_gpu&& o) noexcept :
    sim::gpu_io_entity(std::move(o)),
    mnist_db(std::move(o)),
    params(o.params),
    output_buf(o.output_buf), id(o.id),
    seq_(std::move(o.seq_)), step_(o.step_),
    dev_(o.dev_),
    dataset_gpu_id_(o.dataset_gpu_id_),
    params_uniform_id_(o.params_uniform_id_),
    k_img_(o.k_img_), plan_id_(o.plan_id_)
{}

void mnist_io_gpu::set_output_buffer(sim::io_buffer& buf) {
    output_buf = &buf;
}

sim::step_result mnist_io_gpu::process() {
    COAST_REQUIRE(output_buf != nullptr, core::error_kind::contract_violation);
    // Refresh this step's uniform (current index + PRNG key) via the device data
    // plane, ordered ahead of this step's enqueue so the kernel sees it.
    params_data_t pd {};
    pd.idx      = static_cast<std::uint32_t>(seq_.current_index());
    pd.key      = step_ * 747796405u + std::to_underlying(params.random_seed);
    pd.noise_lo = static_cast<float>(params.noise_range.first);
    pd.noise_hi = static_cast<float>(params.noise_range.second);
    dev_->upload(params_uniform_id_, std::as_bytes(std::span<const params_data_t>(&pd, 1)));

    ++step_;

    // Advance the host cursor — byte-for-byte the mnist_io schedule.
    seq_.advance();

    return {};
}

// ---------------------------------------------------------------------------
// build_step_plan — emit the static per-step work as a backend-agnostic
// step_plan (E2). Registers + initializes the dataset / uniform device buffers,
// compiles the kernel, reads the output io_buffer page id, and builds the plan
// (one compute pass: image+noise).
// ---------------------------------------------------------------------------

void mnist_io_gpu::build_step_plan(coast::gpu::gpu_device& dev) {
    using namespace coast::gpu;
    dev_ = &dev;

    // ---- register + init the device buffers --------------------------------
    // Dataset: the full normalized set, uploaded once (gpu_resident).
    const auto all = get_all_norm_images();
    dataset_gpu_id_ = dev.register_buffer(
        topology::gpu_resident,
        static_cast<std::size_t>(all.size()) * sizeof(float), /*ping_pong=*/false);
    dev.upload(dataset_gpu_id_, std::as_bytes(std::span(all)));

    // params uniform: registered only — process() writes it each step before the
    // enqueue.
    params_uniform_id_ = dev.register_buffer(topology::uniform, sizeof(params_data_t), false);

    // ---- compile the kernel -------------------------------------------------
    // DEVICE-SPECIFIC kernel generation — paired with the kernel-source/uniform
    // arm at the top of this TU.
#if defined(COAST_WITH_WEBGPU) || defined(COAST_WITH_NULL_GPU)
    const std::string src = std::string(coast::gpu::shared_prng_wgsl()) +
        std::string(reinterpret_cast<const char*>(kMnistGpuWgsl), sizeof(kMnistGpuWgsl));
    k_img_ = dev.register_kernel(src, "img_main");
#else
#  error "mnist GPU entity: no kernel generation for the selected COAST_GPU_BACKEND."
#endif

    // ---- output io_buffer page id (ping-pong; registered by env at F) -------
    const buffer_id out_page = output_buf->gpu_page_id();

    // ---- emit the op list ---------------------------------------------------
    step_plan sp;

    // img_main — dataset read + noise + clamp -> output WRITE page (this
    // step's current page). dispatch ((784+63)/64, 1, 1).
    {
        plan_op op{};
        op.kind   = op_kind::compute;
        op.kernel = k_img_;
        plan_dispatch d{};
        d.bindings = {
            {dataset_gpu_id_,    access::read,  0},
            {out_page,           access::write, 1},
            {params_uniform_id_, access::read,  2},
        };
        d.dispatch = {static_cast<std::uint32_t>((kImageSize + 63) / 64), 1, 1};
        op.dispatches.push_back(std::move(d));
        sp.ops.push_back(std::move(op));
    }

    plan_id_ = dev.register_plan();
    dev.build_plan(plan_id_, sp);
}

sim::output_id_t mnist_io_gpu::get_outp_id() const {
    return id;
}

std::size_t mnist_io_gpu::get_outp_size() const {
    return get_image_size();
}

std::string mnist_io_gpu::status_str() const {
    std::string status { "MNIST IO GPU" };
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(seq_.current_epoch());
    status += " | idx: " + std::to_string(seq_.current_index());
    return status;
}

void mnist_io_gpu::save_state(logging::state_writer& writer) const {
    seq_.save_state(writer);
    writer.write("step", static_cast<std::uint64_t>(step_));
    dataset::save_run_binding(writer, params.run);
}

void mnist_io_gpu::load_state(logging::state_reader& reader) {
    if (dataset::restores_sequence(reader, params.run, std::format("mnist_gpu {}", get_outp_id()))) {
        // The sequencer owns the cursor and wraps a saved position into the
        // loaded dataset; on the GPU path this also prevents an out-of-range
        // index uniform (OOB read of dataset_gpu) when loading onto a smaller
        // eval set.
        seq_.load_state(reader);
        step_ = static_cast<std::uint32_t>(reader.read_uint64("step", step_));
    }
}

} // mnist

#endif // COAST_WITH_GPU
