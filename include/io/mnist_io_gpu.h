//
// Created by jk on the ngm3 GPU port (S5).
//

#ifndef MNIST_IO_GPU_H
#define MNIST_IO_GPU_H

#ifdef COAST_WITH_GPU

#include <cstdint>
#include <span>
#include <string>

#include "sim_core/gpu_io_entity.h"
#include "io/mnist_db.h"
#include "io/mnist_cfg.h"
#include "io/sample_sequencer.h"
#include "core/entity_element_macros.h"
#include "core/gpu/gpu_device.h"   // agnostic seam: opaque buffer/kernel/plan ids

namespace mnist {

// GPU sibling of mnist_io (flat, no shared base — the two diverge entirely in how
// they produce their output). Derives gpu_io_entity (the step_plan hook) + mnist_db
// (the dataset). build_step_plan() registers the dataset (uploaded once) and
// uniform device buffers, compiles the kernel, and emits one compute pass
// (image+noise); per step process() advances the host cursor and refreshes the
// 16-byte uniform via the device data plane.
//
// Host bookkeeping (cur_idx / cur_epoch / ci_cnt advance, change_interval) is
// byte-for-byte the same logic as mnist_io so a config can swap CPU<->GPU dataset
// without the index schedule changing. The PIXEL VALUES differ: GPU noise is the
// stateless PCG hash, not the CPU mt19937 stream. S5 bar is sane/in-range.
//
// Compiles only under COAST_WITH_GPU.
class mnist_io_gpu : public sim::gpu_io_entity, public mnist_db {

    const mnist_param_t params;

public:
    ELEMENT_TYPE(mnist_io_gpu, "mnist_input_gpu")

    sim::io_buffer*  output_buf = nullptr;   // variant chosen by env topology pre-pass
    sim::output_id_t id;

    // host cursor — identical schedule to mnist_io (composed sequencer)
    dataset::sample_sequencer seq_;
    std::uint32_t step_ = 0;                 // PRNG key time-component

    mnist_io_gpu(const mnist_param_t& params, sim::output_id_t output_id);

    // Movable (env stores entities in a vector). No move-assign needed.
    mnist_io_gpu(mnist_io_gpu&& o) noexcept;
    mnist_io_gpu& operator=(mnist_io_gpu&&) = delete;
    mnist_io_gpu(const mnist_io_gpu&)            = delete;
    mnist_io_gpu& operator=(const mnist_io_gpu&) = delete;

    ~mnist_io_gpu() override = default;   // device owns the GPU resources

    void set_output_buffer(sim::io_buffer&) override;

    sim::step_result process() override;                    // host bookkeeping + uniform refresh

    // E2 (gpu-abstraction-layer): emit this entity's static per-step work as a
    // backend-agnostic step_plan through the gpu_device facade. Called once at build
    // time by env. Registers the dataset / uniform device buffers (+ initializes
    // them), compiles the kernel, reads the output io_buffer page id, emits the
    // compute pass, and builds the plan.
    void build_step_plan(coast::gpu::gpu_device& dev) override;

    // The plan id registered by build_step_plan (invalid until then). env (F3)
    // enqueues this per step.
    [[nodiscard]] coast::gpu::plan_id plan() const override { return plan_id_; }

    // Structural-change epoch for the dirty/rebuild path (design §8). The dataset
    // entity never changes structure at runtime -> constant 0 (dormant).
    [[nodiscard]] std::uint64_t graph_epoch() const override { return 0; }

    [[nodiscard]] sim::output_id_t get_outp_id() const override;
    [[nodiscard]] std::size_t      get_outp_size() const override;
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }

    [[nodiscard]] std::string status_str() const override;

    // Host-cursor accessors (used by the gpu isolation probe).
    [[nodiscard]] std::size_t current_index() const { return seq_.current_index(); }
    [[nodiscard]] std::size_t current_epoch() const { return seq_.current_epoch(); }

    void save_state(logging::state_writer& writer) const override;
    void load_state(logging::state_reader& reader) override;

    [[nodiscard]] std::optional<seed_t> effective_seed() const override {
        return params.random_seed;
    }

private:
    // ---- agnostic gpu_device handles (E2; populated by build_step_plan) -------
    // The device buffers/kernels/plan backing this entity's per-step work. All
    // invalid/null until build_step_plan() runs at build time.
    coast::gpu::gpu_device* dev_ = nullptr;
    coast::gpu::buffer_id dataset_gpu_id_   = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id params_uniform_id_ = coast::gpu::invalid_buffer;
    coast::gpu::kernel_id k_img_ = coast::gpu::invalid_kernel;
    coast::gpu::plan_id   plan_id_ = coast::gpu::invalid_plan;
};

} // mnist

#endif // COAST_WITH_GPU

#endif //MNIST_IO_GPU_H
