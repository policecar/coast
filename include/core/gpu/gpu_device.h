//
// Created by jk on the gpu-abstraction-layer refactor (A1/A2/A3).
//
// The one agnostic seam between sim_core / GPU entities and the GPU backend.
// `sim_core` and the GPU entities describe their per-step work as a
// backend-agnostic `step_plan` (POD, no SDK types) and drive it through the
// virtual `gpu_device` facade, which exposes only opaque handles
// (buffer_id / kernel_id / plan_id). The single backend translation unit
// (backend_webgpu.cpp) is the only code that includes <webgpu/webgpu.h>.
//
// This header contains NO SDK types and is safe to include anywhere in the
// agnostic core. It is the canary surface for the "zero <webgpu> in the
// agnostic core" success criterion.
//
// The step contract this facade serves is documented on sim::env
// (include/sim_core/sim_env.h): two global barriers per step. BARRIER 1 (every
// entity finished, before swap) is the GPU rendezvous joined with the CPU
// dispatch; BARRIER 2 (every buffer swapped, before next step) is enforced by
// the serial sim loop plus the device's blocking transfer completion. Both are
// mandatory; this facade does not weaken them.
//

#ifndef COAST_CORE_GPU_GPU_DEVICE_H
#define COAST_CORE_GPU_GPU_DEVICE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/step_error.h"  // device faults surface on the value-based channel

namespace coast::gpu {

// ===========================================================================
// A1 — Opaque handles
// ---------------------------------------------------------------------------
// Strong-typedef POD ids (enum class : u32). Type-safe: a buffer_id cannot be
// passed where a kernel_id is wanted; a plain int cannot be silently assigned
// in. Agnostic — no SDK types. The backend maps each id to its internal
// resource registry.
// ===========================================================================

enum class buffer_id : std::uint32_t {};
enum class kernel_id : std::uint32_t {};
enum class plan_id   : std::uint32_t {};

inline constexpr buffer_id invalid_buffer = static_cast<buffer_id>(~0u);
inline constexpr kernel_id invalid_kernel = static_cast<kernel_id>(~0u);
inline constexpr plan_id   invalid_plan   = static_cast<plan_id>(~0u);

// ===========================================================================
// A2 — step_plan POD
// ---------------------------------------------------------------------------
// A flat, backend-agnostic description of one entity's per-step GPU work. It is
// STATIC across steps (§13.4 audit, benchmarks/m3_gpu_timing/AUDIT_record_state.md:
// per-step CPU->GPU traffic is uniforms-only, dataset GPU-resident). Per-step
// variation enters through two channels only, neither of which mutates the plan:
//   1. per-step uniforms — refreshed via gpu_device::upload() (backend-buffered);
//   2. the global generation (ping-pong) — advanced once per step at the swap by
//      advance_generation(); the backend resolves a ping_pong-registered
//      buffer_id to its current physical page. The plan stays generation-agnostic.
//
// Audit refinement: generation affects COPY-OP endpoints (e.g. ngm3's input
// gather source, the group_outp publish source), not only compute bind groups.
// Hence copies are first-class plan elements with ordinary buffer_ids; any of
// them may reference a ping_pong buffer and the backend handles even/odd.
// ===========================================================================

// How a registered buffer moves data relative to the GPU. Set once at
// register_buffer(); drives the swap-time cross-device transfers. (This mirrors
// the io_buffer topology — the transfer is fundamentally an io_buffer property
// the device merely SERVES; a future variant could move data over a network.)
enum class topology : std::uint8_t {
    gpu_resident,   // lives on the GPU, no per-step CPU traffic (dataset, banks, scratch)
    cpu_to_gpu,     // CPU writer, GPU reader   (io_buffer page)
    gpu_to_cpu,     // GPU writer, CPU reader   (io_buffer page)
    gpu_to_gpu,     // GPU writer, GPU reader   (io_buffer page)
    uniform,        // small per-step uniform, refreshed via upload()
};

enum class access : std::uint8_t { read, write };

// One bound buffer in a compute dispatch -> a WGSL @binding slot in group 0.
struct plan_binding {
    buffer_id     buffer;
    access        mode;
    std::uint32_t slot;        // @binding(slot)
};

// One CopyBufferToBuffer. src and/or dst may be ping_pong buffers; the backend
// resolves the physical page per generation at enqueue() time.
//
// src_side / dst_side select WHICH page of a ping_pong endpoint to touch, exactly
// as plan_binding::mode does for compute bindings (non-ping_pong endpoints ignore
// it). This is IRREDUCIBLE for copies: src vs dst does not determine the side. A
// gather copy reads an UPSTREAM io_buffer produced last step -> read side (the
// opposite/older page, giving one-step latency); a publish copy reads THIS
// entity's own output written this step -> write side (the current page). Both
// are `src`, so the side must be carried explicitly. Default write = current page
// (= the pre-side-tag behavior); the entity sets src_side=read for cross-step
// reads (ngm3's input gather).
struct plan_copy {
    buffer_id     src;
    buffer_id     dst;
    std::uint64_t src_offset;
    std::uint64_t dst_offset;
    std::uint64_t bytes;
    access        src_side = access::write;   // which page of a ping_pong src
    access        dst_side = access::write;   // which page of a ping_pong dst
};

// One dispatch within a compute pass: a group-0 bind group + workgroup counts.
// A single compute pass may issue SEVERAL of these under ONE pipeline, switching
// only the bind group between them (e.g. ngm3 stats / adapt / branch dispatch
// the same kernel once per bank, proximal then apical). Modeling this faithfully
// keeps the recorded command stream identical to today's record() — one compute
// pass with N dispatches — rather than splitting it into N single-dispatch passes.
struct plan_dispatch {
    std::vector<plan_binding>   bindings;     // group-0 bindings for this dispatch
    std::array<std::uint32_t,3> dispatch;     // workgroup counts (x,y,z)
};

// What a plan_op carries.
enum class op_kind : std::uint8_t { copy, compute };

// One step operation, executed in PLAN ORDER. Either a buffer copy or a single
// compute pass holding >=1 dispatch. A strictly ordered list (rather than
// pre/passes/post buckets) is REQUIRED, not stylistic: ngm3 publishes its
// post-inhibition output (out_gpu -> group_outp_) into a stable buffer BETWEEN
// compute passes, and the winner + adapt passes that follow READ that buffer.
// A bucketed model has nowhere to place a copy mid-stream; deferring it past the
// passes would feed those readers stale data.
struct plan_op {
    op_kind kind;

    // ---- kind == copy ----
    plan_copy copy{};

    // ---- kind == compute ----
    kernel_id                  kernel{};      // one pipeline for the whole pass
    std::vector<plan_dispatch> dispatches;    // >=1, recorded in order in ONE pass
    // When true, this pass is ALWAYS recorded and the kernel early-outs on a
    // per-step gate uniform (e.g. ngm3 branch: step % branch_interval != 0 ->
    // return before any buffer write). Replaces today's host-side conditional
    // append, so the command stream is structurally static (plan nodes C1/C2).
    bool                       uniform_gated = false;
};

// The whole per-step plan for one entity: a strictly ordered list of copies and
// compute passes. Flat POD; no SDK types. STATIC across steps — per-step
// variation enters ONLY via uniforms (upload()) and the ping-pong generation
// (advance_generation()); neither mutates the plan (see the section banner).
struct step_plan {
    std::vector<plan_op> ops;
};

// ===========================================================================
// A3 — gpu_device virtual facade
// ---------------------------------------------------------------------------
// The single virtual seam (the one approved OOP deviation, design §12). The
// agnostic side holds a gpu_device* and never sees an SDK type.
//
// Per-step usage, realizing the env step contract WITHOUT weakening either
// barrier (see sim_env.h):
//
//   // env::process() — entities process in parallel; GPU batch in flight
//   dev.begin_step();
//   for (e : gpu_entities) { e.refresh_uniforms(); dev.enqueue(e.plan()); }
//   dev.submit_step();              // GPU runs, no wait...
//   dispatch_cpu_entities();        // ...concurrently with the CPU entities
//   dev.rendezvous_step();          // BARRIER 1 (GPU side): mandatory, every step
//   for (e : gpu_entities) e.read_probes();
//
//   // env::swap_io() — flip pages; move cross-boundary data during the flip
//   dev.enqueue_transfers();
//   dev.complete_transfers();       // blocking; with the serial loop => BARRIER 2
//   dev.advance_generation();       // page flip (opaque generation tick)
//
// The split begin/enqueue/submit/rendezvous exists to overlap the GPU batch
// with the CPU entities WITHIN a step (entity independence) — not to pipeline
// across steps. rendezvous_step() is a full, mandatory coherence wait every
// step; it makes NO data-flow assumptions (a GPU output may feed a CPU entity
// or a cross-device buffer, and we cannot know, so we always synchronize).
// ===========================================================================

struct gpu_device {
    virtual ~gpu_device() = default;

    // ---- backend identity / fault / sync (folded off gpu_context) ----

    // Surface a recorded async backend fault on the calling (sim) thread at a
    // phase boundary as a value (device_fault), not by throwing. Returns success
    // if the backend records no fault. Called by env at the start of
    // process()/swap_io() so a device-loss / validation error halts the run on
    // the same value-based channel as CPU entity faults rather than unwinding out
    // of the sim thread. The backend logs the driver detail; the channel carries
    // the category (entity_id == no_entity — a device fault maps to no entity).
    virtual std::expected<void, core::step_error> check() = 0;

    // Block until all submitted GPU work has completed. Used outside the per-step
    // rendezvous (e.g. teardown, one-shot boundary readbacks).
    virtual void wait_idle() = 0;

    // Human-readable backend / adapter identity for the run manifest
    // (e.g. "Radeon 8060S [Vulkan]" or "null"). Const: pure query.
    [[nodiscard]] virtual std::string adapter_name() const = 0;

    // ---- registration (build time) ----

    // Register a buffer of `bytes`. `ping_pong` => the backend keeps one physical
    // page per generation and resolves the current page at enqueue()/transfer
    // time. Returns an opaque id.
    virtual buffer_id register_buffer(topology topo, std::size_t bytes, bool ping_pong) = 0;

    // Compile (or look up) a compute kernel from WGSL `source` + `entry` point.
    virtual kernel_id register_kernel(std::string_view source, std::string_view entry) = 0;

    // Mint an empty plan slot. The caller (a GPU entity, via env at build time)
    // then populates/refreshes it with build_plan(). Symmetric with
    // register_buffer/register_kernel — an opaque id minted by the device, not
    // chosen by the caller. (Interface addition discovered at B2: build_plan
    // takes a plan_id but nothing else hands one out.)
    virtual plan_id register_plan() = 0;

    // (Re)build the backend resources for `id` from `plan`: cached pipelines and
    // PRE-BUILT per-generation bind groups (removes today's per-step bind-group
    // rebuild). RE-CALLABLE / idempotent: a rebuild releases and recreates this
    // plan's owned resources (plan node B5; dormant in the current model — no
    // entity bumps its epoch — but proven end-to-end by the forced-bump test G7).
    virtual void build_plan(plan_id id, const step_plan& plan) = 0;

    // Mark a plan dirty so the next begin_step() rebuilds it before enqueue.
    // Dormant this cycle.
    virtual void mark_dirty(plan_id id) = 0;

    // ---- per-step batch (env::process) ----

    virtual void begin_step()        = 0;  // open the step's command recording
    virtual void enqueue(plan_id id) = 0;  // record this plan's copies + passes (no submit)
    virtual void submit_step()       = 0;  // finish + submit the batch; DO NOT wait

    // BARRIER 1 (GPU side): block until this step's submitted batch is coherent.
    // MANDATORY and unconditional, every step. This cycle's backend implements
    // it as a device wait_idle. (Not specified as a drain only so a future
    // backend could realize coherence via an event instead — but it remains a
    // full per-step synchronization; nothing here is deferred across steps.)
    virtual void rendezvous_step()   = 0;

    // ---- data plane (plan node B4) ----

    // Stage `data` into the backend's buffering for `id`, ordered before the
    // consuming step. The per-step uniform / cpu_to_gpu path targets the WRITE side
    // (default). `side` selects the page for a ping_pong buffer (non-ping_pong
    // ignores it): state-restore writes the READ side so the published page a
    // consumer reads next matches, symmetric with readback() reading the read side.
    virtual void upload(buffer_id id, std::span<const std::byte> data,
                        access side = access::write) = 0;

    // Read `id`'s current contents into `dst` at a step boundary (observers /
    // logging / state-save). Returns false on a transfer failure. Expects
    // dst.size() == the buffer's byte size.
    [[nodiscard]] virtual bool readback(buffer_id id, std::span<std::byte> dst) = 0;

    // ---- swap phase (env::swap_io; plan node F2) ----
    // Serves the io_buffer "the swap may move data" contract for GPU-backed
    // buffers (cpu_to_gpu uploads, gpu_to_cpu copy-to-staging+map).

    // Record + submit every registered buffer's cross-device swap movement for
    // this step.
    virtual void enqueue_transfers()  = 0;

    // Block until the transfers have landed on the read-side. Together with the
    // serial sim loop (swap_io returns before the next process) this realizes
    // BARRIER 2: the next step cannot start until every buffer has swapped.
    virtual void complete_transfers() = 0;

    // Advance the global generation by one OPAQUE tick: flips which physical page
    // a ping_pong buffer resolves to. Called once per step at the swap. The
    // modulus is backend-private (2 today).
    virtual void advance_generation() = 0;

    gpu_device(const gpu_device&)            = delete;
    gpu_device& operator=(const gpu_device&) = delete;
    gpu_device(gpu_device&&)                 = delete;
    gpu_device& operator=(gpu_device&&)      = delete;

protected:
    gpu_device() = default;
};

// Factory: construct the GPU device for whichever backend CMake compiled in
// (selected by COAST_GPU_BACKEND). Backend-neutral and parameterless — the
// backend creates and privately owns its own context internally; no SDK type
// crosses this seam. The only seam-construction entry point the agnostic side
// (env, F3) uses. Exactly one backend translation unit defines this symbol
// (src/core/gpu/backends/<backend>/...); compiling two would be a loud
// duplicate-symbol link error, not a silent one.
std::unique_ptr<gpu_device> make_gpu_device();

}  // namespace coast::gpu

#endif  // COAST_CORE_GPU_GPU_DEVICE_H
