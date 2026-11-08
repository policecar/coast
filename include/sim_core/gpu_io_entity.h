//
// Created by jk on the ngm3 GPU port (S3).
//

#ifndef COAST_SIM_CORE_GPU_IO_ENTITY_H
#define COAST_SIM_CORE_GPU_IO_ENTITY_H

#include <cstdint>

#include "sim_core/io_entity.h"
#include "core/gpu/gpu_device.h"   // agnostic seam: gpu_device / plan_id

namespace sim {

// Intermediate base for entities whose work runs on the GPU. Adds the per-step
// step_plan hook on top of the existing io_entity contract.
//
// The dispatcher (env::process()) identifies GPU entities structurally via
// dynamic_cast<gpu_io_entity*> rather than a device-kind tag: the type
// hierarchy already carries device-awareness, so a parallel enum would only
// risk drift. io_entity::process() is still called every step on GPU entities
// for host-side bookkeeping (advance counters, refresh small uniforms via the
// gpu_device data plane); the per-step GPU work itself is the static step_plan
// built once by build_step_plan() and replayed by the device each step.
//
// This header (and every *_gpu entity that derives from it) is agnostic of any
// GPU SDK — it crosses the seam only through the opaque gpu_device facade.
class gpu_io_entity : public io_entity {

public:

    // ---- gpu_device path (gpu-abstraction-layer; env drives this at F) -------
    // Build this entity's static per-step step_plan on `dev` (register buffers +
    // kernels, emit the op list, register_plan + build_plan). Called once at build
    // time by env, after the io_buffers are wired + device-registered. Default
    // no-op so non-migrated gpu_io_entity subclasses (e.g. the stage-probe test
    // entities) still compile until they are ported.
    virtual void build_step_plan(coast::gpu::gpu_device&) {}

    // The plan id registered by build_step_plan (env enqueues it per step).
    // invalid_plan until build_step_plan runs / for un-migrated entities.
    [[nodiscard]] virtual coast::gpu::plan_id plan() const { return coast::gpu::invalid_plan; }

    // Structural-change epoch for the dirty/rebuild path (design §8). Constant for
    // entities with a static graph; bumped only on a genuine runtime structure
    // change. Dormant this cycle.
    [[nodiscard]] virtual std::uint64_t graph_epoch() const { return 0; }

    // Called by the dispatcher every step AFTER the phase-1 submit + wait_idle,
    // when the device is idle and this step's kernel outputs are resident. The
    // hook to read back GPU-computed probe values and fan them into the probe
    // system. Default no-op (a GPU entity that exposes no probes need not
    // override). IMPLEMENTERS: gate ONLY the readback on any_probe_active();
    // the PROBE() statements themselves MUST stay on the unconditional path so
    // their registration statics fire (otherwise the probes are never
    // discoverable and can never be activated). See
    // ngm3::neuron_group_gpu_t::read_probes for the canonical shape.
    virtual void read_probes() {}
};

} // sim

#endif // COAST_SIM_CORE_GPU_IO_ENTITY_H
