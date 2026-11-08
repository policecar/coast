//
// Created by jk on 04.09.25.
//

#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>

#include "sim_core/sim_env.h"
#include "diag/diag.h"

#ifdef COAST_WITH_GPU
#include "sim_core/gpu_io_entity.h"          // dynamic_cast device classification + plan()/build_step_plan()
#include "sim_core/io_buffer_gpu.h"          // the three thin GPU io_buffer handle variants
#include "core/gpu/gpu_device.h"    // make_gpu_device + the agnostic device facade
// No webgpu header at all: sim_env.cpp drives the GPU only through the agnostic
// gpu_device seam, and the backend now owns its own context privately (no
// gpu_context type crosses this TU). The G5/SC1 canary asserts zero <webgpu>
// (and zero wgpu/webgpu) matches here.
#endif

namespace sim {

namespace {

// An entity runs on the GPU iff it derives from gpu_io_entity. The device of a
// buffer is read structurally from its writer/readers rather than from a tag, so
// the type hierarchy stays the single source of truth. In CPU-only builds (or
// configs with no GPU entities) this is always false and every buffer is CPU.
bool entity_is_gpu([[maybe_unused]] io_entity& e)
{
#ifdef COAST_WITH_GPU
    return dynamic_cast<gpu_io_entity*>(&e) != nullptr;
#else
    return false;
#endif
}

} // namespace

env::env()  = default;
env::~env() = default;

#ifdef COAST_WITH_GPU
void env::enable_gpu()
{
    // Mark that this run uses the GPU. The actual device (and the backend's
    // private context) is created lazily in init_io_buffers() via make_gpu_device(),
    // once the buffer topology is known. Idempotent-checked: a double enable is a
    // build-wiring error.
    if (gpu_enabled_) {
        throw std::runtime_error("env::enable_gpu: GPU already enabled");
    }
    gpu_enabled_ = true;
}

std::string env::gpu_adapter_name() const
{
    // Backend/adapter identity for the run manifest; empty before the device is
    // built or in CPU-only runs.
    return gpu_dev_ ? gpu_dev_->adapter_name() : std::string{};
}
#endif

void env::init_io_buffers()
{
    // Topology pre-pass (DI5 / IDD §5.1): classify each buffer's variant as a
    // pure function of (writer device, any GPU reader, any CPU reader), then
    // construct the matching concrete io_buffer. Wiring happens only after all
    // buffers exist, so set_input_buffer never sees a half-built map.

    struct topo_t {
        bool        has_writer     = false;
        bool        writer_gpu     = false;
        bool        any_gpu_reader = false;
        bool        any_cpu_reader = false;
        std::size_t size           = 0;
    };
    std::unordered_map<std::size_t, topo_t> topo;

    // Pass 1: writers (one per buffer id — duplicates are a config error).
    for (auto &io_ent : iterate_entities()) {
        const std::size_t id = io_ent.get_outp_id();
        topo_t &t = topo[id];
        if (t.has_writer) {
            throw std::runtime_error("Duplicate io_entity ID: " + std::to_string(id));
        }
        t.has_writer = true;
        t.writer_gpu = entity_is_gpu(io_ent);
        t.size       = io_ent.get_outp_size();
    }

    // Pass 2: readers — record whether each buffer feeds a GPU and/or CPU reader.
    for (auto &io_ent : iterate_entities()) {
        const bool reader_gpu = entity_is_gpu(io_ent);
        for (auto inp_id : io_ent.get_inp_ids()) {
            auto it = topo.find(inp_id);
            if (it == topo.end()) {
                throw std::runtime_error("Missing io_entity ID: " + std::to_string(inp_id));
            }
            if (reader_gpu) it->second.any_gpu_reader = true;
            else            it->second.any_cpu_reader = true;
        }
    }

    // Pass 3: construct the variant selected by the §5.1 table. The variants are
    // thin CPU-side handles (the device owns the GPU pages, registered below); the
    // three cross-/all-GPU variants still require a gpu_context to be installed
    // (the device is built from it), so a GPU buffer with none is a hard
    // configuration error rather than a silent mis-selection.
    for (auto &[id, t] : topo) {
        std::unique_ptr<io_buffer> buf;
        if (!t.writer_gpu && !t.any_gpu_reader) {
            buf = std::make_unique<io_buffer_cpu>(t.size);           // CPU -> CPU
        } else {
#ifdef COAST_WITH_GPU
            if (!gpu_enabled_) {
                throw std::runtime_error("io_buffer id " + std::to_string(id) +
                    " needs a GPU variant but the GPU is not enabled "
                    "(call env::enable_gpu before init_io_buffers)");
            }
            if (!t.writer_gpu) {                                     // CPU -> GPU
                buf = std::make_unique<io_buffer_cpu_to_gpu>(t.size);
            } else if (t.any_cpu_reader) {                           // GPU -> CPU
                buf = std::make_unique<io_buffer_gpu_to_cpu>(t.size);
            } else {                                                 // GPU -> GPU only
                buf = std::make_unique<io_buffer_gpu>(t.size);
            }
#else
            throw std::runtime_error("io_buffer id " + std::to_string(id) +
                " requires a GPU variant, but this build has COAST_WITH_GPU=OFF");
#endif
        }
        io_buffers.emplace(id, std::move(buf));
    }

    // Wire writers, then readers, against the now-complete buffer map.
    for (auto &io_ent : iterate_entities()) {
        io_ent.set_output_buffer(*io_buffers.at(io_ent.get_outp_id()));
    }
    for (auto &io_ent : iterate_entities()) {
        for (auto inp_id : io_ent.get_inp_ids()) {
            io_ent.set_input_buffer(inp_id, *io_buffers.at(inp_id));
        }
    }

#ifdef COAST_WITH_GPU
    // gpu-abstraction-layer: build the agnostic device path. The backend creates
    // and privately owns its own context inside make_gpu_device() — nothing of the
    // SDK is named here. Register a ping-pong device page per GPU io_buffer,
    // zero-init both pages, hand each buffer its opaque page id, then build every
    // GPU entity's step_plan (their io_buffer page ids now resolve).
    if (gpu_enabled_) {
        gpu_dev_ = coast::gpu::make_gpu_device();

        std::vector<coast::gpu::buffer_id> gpu_pages;
        std::vector<std::size_t>           gpu_elems;
        for (auto &[id, t] : topo) {
            if (!t.writer_gpu && !t.any_gpu_reader) continue;   // CPU -> CPU: no device page
            coast::gpu::topology dtopo;
            if (!t.writer_gpu)         dtopo = coast::gpu::topology::cpu_to_gpu;
            else if (t.any_cpu_reader) dtopo = coast::gpu::topology::gpu_to_cpu;
            else                       dtopo = coast::gpu::topology::gpu_to_gpu;
            const coast::gpu::buffer_id page =
                gpu_dev_->register_buffer(dtopo, t.size * sizeof(float), /*ping_pong=*/true);
            io_buffers.at(id)->gpu_register(*gpu_dev_, page, dtopo);
            gpu_pages.push_back(page);
            gpu_elems.push_back(t.size);
        }

        // Zero-init BOTH ping-pong pages (step-0 read-side consumers see zeros, as
        // the legacy variants did). upload() targets the current (write) page; one
        // advance_generation flips which physical page is current, so two passes
        // around it cover both pages. The two advances leave the generation parity
        // at 0 — the state the first step expects.
        const auto zero_pages = [&] {
            for (std::size_t i = 0; i < gpu_pages.size(); ++i) {
                const std::vector<float> zeros(gpu_elems[i], 0.0f);
                gpu_dev_->upload(gpu_pages[i], std::as_bytes(std::span<const float>(zeros)));
            }
        };
        zero_pages();
        gpu_dev_->advance_generation();
        zero_pages();
        gpu_dev_->advance_generation();

        for (auto &io_ent : iterate_entities()) {
            if (auto* g = dynamic_cast<gpu_io_entity*>(&io_ent)) {
                g->build_step_plan(*gpu_dev_);
            }
        }
    }
#endif
}

entity_range env::iterate_entities()
{
    std::vector<io_entity*> ptrs;
    for (auto &ev : entities | std::views::values ) {
        std::size_t ev_size = ev->get_size();
        for (std::size_t i = 0; i < ev_size; ++i)
            ptrs.push_back(&ev->as_base(i));
    }
    return entity_range(std::move(ptrs));
}

io_entity* env::get_entity(output_id_t output_id)
{
    for (auto &ev : entities | std::views::values ) {
        std::size_t ev_size = ev->get_size();
        for (std::size_t i = 0; i < ev_size; ++i) {
            io_entity &cur_entity = ev->as_base(i);
            if (cur_entity.get_outp_id() == output_id)
                return &cur_entity;
        }
    }
    return nullptr;
}

void env::process_pending_hooks()
{
    // Fast path: no pending work
    if (!has_pending_hooks_.load(std::memory_order_acquire)) {
        return;
    }

    // Lock order: active, then pending (see sim_env.h).
    std::lock_guard active_lock(active_hooks_mutex_);
    std::lock_guard lock(pending_hooks_mutex_);

    // Process additions
    for (auto& [id, func] : pending_pre_process_hooks_) {
        pre_process_hooks_.emplace(id, std::move(func));
    }
    pending_pre_process_hooks_.clear();

    for (auto& [id, func] : pending_post_process_hooks_) {
        post_process_hooks_.emplace(id, std::move(func));
    }
    pending_post_process_hooks_.clear();

    for (auto& [id, func] : pending_pre_swap_hooks_) {
        pre_swap_hooks_.emplace(id, std::move(func));
    }
    pending_pre_swap_hooks_.clear();

    for (auto& [id, func] : pending_post_swap_hooks_) {
        post_swap_hooks_.emplace(id, std::move(func));
    }
    pending_post_swap_hooks_.clear();

    for (auto& [id, func] : pending_post_swap_barrier_hooks_) {
        post_swap_barrier_hooks_.emplace(id, std::move(func));
    }
    pending_post_swap_barrier_hooks_.clear();

    // Process removals
    for (auto id : pending_remove_pre_process_) {
        pre_process_hooks_.erase(id);
    }
    pending_remove_pre_process_.clear();

    for (auto id : pending_remove_post_process_) {
        post_process_hooks_.erase(id);
    }
    pending_remove_post_process_.clear();

    for (auto id : pending_remove_pre_swap_) {
        pre_swap_hooks_.erase(id);
    }
    pending_remove_pre_swap_.clear();

    for (auto id : pending_remove_post_swap_) {
        post_swap_hooks_.erase(id);
    }
    pending_remove_post_swap_.clear();

    for (auto id : pending_remove_post_swap_barrier_) {
        post_swap_barrier_hooks_.erase(id);
    }
    pending_remove_post_swap_barrier_.clear();

    has_pending_hooks_.store(false, std::memory_order_release);
}

// Dispatch the collected CPU entities in parallel. Uses parallel_for + a
// persistent partitioner (cpu_partitioner_) rather than parallel_for_each so the
// outer dispatch carries a stable thread->subrange mapping across steps — the
// same cross-call affinity the ngm2/ngm3 inner static_partitioner loops depend
// on. Each entity writes only its own output buffer, so any chunking/ordering is
// bit-identical.
sim::step_result env::dispatch_cpu_entities()
{
    // Allocation-free fault sink. The happy path does exactly one relaxed atomic
    // store (reset) + one load per step and touches NOTHING else shared — no
    // per-step container construction, no per-thread slots. The detail is written
    // under cpu_fault_mutex_ only on the cold !res branch (faults are rare), where
    // the smallest entity_id wins for a scheduling-independent (deterministic)
    // choice. See design/error_handling_plan.md.
    cpu_fault_.store(false, std::memory_order_relaxed);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, cpu_dispatch_.size()),
        [this](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                // A throw from process() joins the same reduction as a returned
                // fault, so it is stamped with the entity's id and does not
                // unwind out of the parallel_for (which would cancel the other
                // entities and, with a GPU batch in flight, skip the rendezvous).
                step_result res;
                try {
                    res = cpu_dispatch_[i]->process();
                } catch (...) {
                    COAST_ERROR("sim", "entity {} threw in process(): {}",
                                cpu_dispatch_[i]->get_outp_id(),
                                core::current_exception_what());
                    res = std::unexpected(core::step_error{
                        core::error_kind::exception, core::step_error::no_entity, COAST_SITE});
                }
                if (!res) [[unlikely]] {
                    core::step_error e = res.error();
                    e.entity_id = cpu_dispatch_[i]->get_outp_id();  // stamp identity
                    std::lock_guard lock(cpu_fault_mutex_);
                    if (!cpu_fault_err_ || e.entity_id < cpu_fault_err_->entity_id)
                        cpu_fault_err_ = e;
                    cpu_fault_.store(true, std::memory_order_relaxed);
                }
            }
        },
        cpu_partitioner_);

    if (cpu_fault_.load(std::memory_order_relaxed)) {
        std::lock_guard lock(cpu_fault_mutex_);
        const auto e = cpu_fault_err_;
        cpu_fault_err_.reset();   // consume, so the next step starts clean
        if (e)
            return std::unexpected(*e);
    }
    return {};
}

sim::step_result env::process()
{
    // Process any pending hook changes before iterating
    process_pending_hooks();

    run_hook_phase({&pre_process_hooks_});

    // First hot-path fault seen this step (success until one is recorded). The
    // post-process hooks still run on a faulted step so the last state is visible
    // to the GUI; the controller halts on the value returned at the end.
    sim::step_result result;

    // Collect this step's entities into the reusable dispatch scratch. Entities
    // are mutually independent within a process() pass (the io_buffers carry one
    // step of latency: every entity reads the read-side pages and writes the
    // write-side pages; the swap happens only in swap_io() after process()
    // returns). So the serial loop enforced no required ordering — CPU entities
    // can run concurrently, and the GPU batch can be in flight at the same time.
    cpu_dispatch_.clear();
#ifdef COAST_WITH_GPU
    gpu_dispatch_.clear();
    for (auto &io_ent : iterate_entities()) {
        if (auto* g = dynamic_cast<gpu_io_entity*>(&io_ent)) {
            gpu_dispatch_.push_back(g);
        } else {
            cpu_dispatch_.push_back(&io_ent);
        }
    }

    if (gpu_dev_) {
        // GPU host bookkeeping stays SERIAL (each g->process() writes the per-step
        // uniforms its plan reads). The plans are enqueued onto one batch and
        // submitted WITHOUT waiting, so the GPU drains concurrently with the CPU
        // entities; rendezvous_step() is the mandatory per-step coherence wait.
        if (auto r = gpu_dev_->check(); !r && result) result = r;
        gpu_dev_->begin_step();
        for (auto* g : gpu_dispatch_) {
            if (auto r = g->process(); !r && result) result = r;
            gpu_dev_->enqueue(g->plan());
        }
        gpu_dev_->submit_step();

        // CPU entities run in parallel while the GPU batch is in flight. A faulted
        // entity reports through dispatch_cpu_entities()'s reduced step_result; the
        // batch still drains and rendezvous below, then the fault halts the run.
        if (auto r = dispatch_cpu_entities(); !r && result) result = r;

        gpu_dev_->rendezvous_step();   // BARRIER 1 (GPU side): mandatory every step

        // Device coherent and this step's kernel outputs resident: let each GPU
        // entity read back probe values and fan them into the probe system.
        // read_probes() gates its own readback internally so the registration
        // statics still fire every step.
        for (auto* g : gpu_dispatch_) {
            g->read_probes();
        }
    } else {
        // CPU-only config inside a WGPU build (no GPU device): every entity is in
        // cpu_dispatch_, gpu_dispatch_ is empty.
        result = dispatch_cpu_entities();
    }
#else
    for (auto &io_ent : iterate_entities()) {
        cpu_dispatch_.push_back(&io_ent);
    }
    result = dispatch_cpu_entities();
#endif

    run_hook_phase({&post_process_hooks_});

    return result;
}

sim::step_result env::swap_io()
{
    // Process any pending hook changes before iterating
    process_pending_hooks();

    run_hook_phase({&pre_swap_hooks_});

#ifdef COAST_WITH_GPU
    // Phase 2: cross-device movement, then the page flip. cpu_to_gpu uploads the
    // just-written CPU mirror to the device WRITE page BEFORE the generation flip
    // (so the GPU reader reads it on the next step). gpu_to_gpu needs no transfer —
    // the generation flip alone carries the one-step lag. advance_generation() is
    // the device's page flip. CPU-only buffers skip this entirely.
    if (gpu_dev_) {
        // A device fault here aborts the swap (mirrors the prior check_or_throw):
        // skip the uploads/flip/hooks and halt on the returned value.
        if (auto r = gpu_dev_->check(); !r) return r;
        // cpu->gpu: upload the just-written CPU mirror to the device WRITE page
        // BEFORE the generation flip (so the GPU reader reads it next step).
        for (auto &buf : io_buffers | std::views::values) {
            if (buf->gpu_backed() &&
                buf->gpu_topology() == coast::gpu::topology::cpu_to_gpu) {
                gpu_dev_->upload(buf->gpu_page_id(),
                                 std::as_bytes(buf->cur_write_buffer()));
            }
        }
        gpu_dev_->advance_generation();   // device page flip (one-step lag)
        // gpu->cpu: AFTER the flip the device READ side holds this step's fresh
        // output; pull it into the CPU read mirror (the buffer fills the page that
        // the swap_buffer() loop below then publishes as the read page).
        for (auto &buf : io_buffers | std::views::values) {
            if (buf->gpu_backed() &&
                buf->gpu_topology() == coast::gpu::topology::gpu_to_cpu) {
                buf->device_to_cpu_readback();
            }
        }
    }
#endif

    for (auto &buf : io_buffers | std::views::values ) {
        buf->swap_buffer();
    }

    // One phase: a removal must not fall between a two-phase visualization's
    // signal (post_swap) and its wait (barrier), or the signal would be left
    // unanswered.
    run_hook_phase({&post_swap_hooks_, &post_swap_barrier_hooks_});

    return {};
}

namespace {
// The env whose hook phase this thread is currently running (null outside one).
// Lets remove_hook() recognise a removal from inside a hook, which must not
// wait for the phase it is part of.
thread_local const env* tl_hook_phase_env = nullptr;
} // namespace

void env::run_hook_phase(std::initializer_list<hook_map_t*> maps)
{
    // Held for the whole phase, so a concurrent remove_hook() waits until no hook
    // of this phase is running. Both guards unwind if a hook throws.
    std::lock_guard lock(active_hooks_mutex_);
    struct phase_scope {
        const env* prev;
        explicit phase_scope(const env* e) : prev(tl_hook_phase_env) { tl_hook_phase_env = e; }
        ~phase_scope() { tl_hook_phase_env = prev; }
        phase_scope(const phase_scope&) = delete;
        phase_scope& operator=(const phase_scope&) = delete;
    } scope(this);

    for (auto* m : maps) {
        for (auto& [id, func] : *m) {
            func();
        }
    }
}

void env::remove_hook(hook_map_t& active, std::vector<pending_hook_t>& pending_add,
                      std::vector<std::size_t>& pending_remove, const std::size_t id)
{
    if (tl_hook_phase_env == this) {
        // Inside one of our own hooks: this thread holds active_hooks_mutex_ and
        // is iterating the maps, so neither wait nor erase. Defer to the next
        // process()/swap_io() boundary.
        std::lock_guard lock(pending_hooks_mutex_);
        pending_remove.push_back(id);
        has_pending_hooks_.store(true, std::memory_order_release);
        return;
    }

    // Destroy the removed callables only after both locks are released: a
    // closure's destructor may run arbitrary code (a captured shared_ptr's last
    // release), which must not run under the hook locks.
    hook_map_t::node_type removed_active;
    hook_fn_t             removed_pending;
    {
        // Announce the wait, so hooks that wait for another thread (possibly
        // this one) give up instead of running out their timeouts.
        struct waiting_scope {
            std::atomic<int>& n;
            explicit waiting_scope(std::atomic<int>& c) : n(c) { n.fetch_add(1, std::memory_order_acq_rel); }
            ~waiting_scope() { n.fetch_sub(1, std::memory_order_acq_rel); }
            waiting_scope(const waiting_scope&) = delete;
            waiting_scope& operator=(const waiting_scope&) = delete;
        };
        std::unique_lock active_lock(active_hooks_mutex_, std::defer_lock);
        {
            waiting_scope waiting(removals_waiting_);
            active_lock.lock();                              // waits out a running phase
        }
        std::lock_guard lock(pending_hooks_mutex_);
        removed_active = active.extract(id);
        // Not applied yet: drop it from the pending additions so it never starts.
        for (auto it = pending_add.begin(); it != pending_add.end(); ++it) {
            if (it->id == id) {
                removed_pending = std::move(it->func);
                pending_add.erase(it);
                break;
            }
        }
    }
}

std::size_t env::set_pre_process_hook(std::function<void()> func)
{
    std::lock_guard lock(pending_hooks_mutex_);
    std::size_t id = next_hook_id_++;
    pending_pre_process_hooks_.push_back({id, std::move(func)});
    has_pending_hooks_.store(true, std::memory_order_release);
    return id;
}

std::size_t env::set_post_process_hook(std::function<void()> func)
{
    std::lock_guard lock(pending_hooks_mutex_);
    std::size_t id = next_hook_id_++;
    pending_post_process_hooks_.push_back({id, std::move(func)});
    has_pending_hooks_.store(true, std::memory_order_release);
    return id;
}

std::size_t env::set_pre_swap_hook(std::function<void()> func)
{
    std::lock_guard lock(pending_hooks_mutex_);
    std::size_t id = next_hook_id_++;
    pending_pre_swap_hooks_.push_back({id, std::move(func)});
    has_pending_hooks_.store(true, std::memory_order_release);
    return id;
}

std::size_t env::set_post_swap_hook(std::function<void()> func)
{
    std::lock_guard lock(pending_hooks_mutex_);
    std::size_t id = next_hook_id_++;
    pending_post_swap_hooks_.push_back({id, std::move(func)});
    has_pending_hooks_.store(true, std::memory_order_release);
    return id;
}

void env::remove_pre_process_hook(const std::size_t id)
{
    remove_hook(pre_process_hooks_, pending_pre_process_hooks_, pending_remove_pre_process_, id);
}

void env::remove_post_process_hook(const std::size_t id)
{
    remove_hook(post_process_hooks_, pending_post_process_hooks_, pending_remove_post_process_, id);
}

void env::remove_pre_swap_hook(const std::size_t id)
{
    remove_hook(pre_swap_hooks_, pending_pre_swap_hooks_, pending_remove_pre_swap_, id);
}

void env::remove_post_swap_hook(const std::size_t id)
{
    remove_hook(post_swap_hooks_, pending_post_swap_hooks_, pending_remove_post_swap_, id);
}

std::size_t env::set_post_swap_barrier_hook(std::function<void()> func)
{
    std::lock_guard lock(pending_hooks_mutex_);
    std::size_t id = next_hook_id_++;
    pending_post_swap_barrier_hooks_.push_back({id, std::move(func)});
    has_pending_hooks_.store(true, std::memory_order_release);
    return id;
}

void env::remove_post_swap_barrier_hook(const std::size_t id)
{
    remove_hook(post_swap_barrier_hooks_, pending_post_swap_barrier_hooks_, pending_remove_post_swap_barrier_, id);
}

} // sim
