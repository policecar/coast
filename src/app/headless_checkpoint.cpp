#include "app/headless_checkpoint.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "core/local_time.h"
#include "diag/diag.h"
#include "diag/progress.h"
#include "logging/logging_manager.h"
#include "logging/state_logger.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_thread_controller.h"

namespace coast::app {

headless_checkpointer::headless_checkpointer(std::uint64_t               interval,
                                             std::uint64_t               start_step,
                                             std::uint64_t               target_step,
                                             std::filesystem::path       snapshot_dir,
                                             std::string                 config_name,
                                             sim::env&                   env,
                                             logging::logging_manager&   lm,
                                             sim::sim_thread_controller& ctrl)
  : interval_(interval), start_step_(start_step), target_step_(target_step),
    snapshot_dir_(std::move(snapshot_dir)), config_name_(std::move(config_name)),
    env_(env), lm_(lm), ctrl_(ctrl)
{
    if (interval_ == 0)
        throw std::invalid_argument("headless_checkpointer: interval must be >= 1");
}

void headless_checkpointer::write(std::uint64_t completed)
{
    logging::mem_group tree = logging::state_logger::extract_state(env_, completed);

    const std::filesystem::path full_path = core::snapshot_path(snapshot_dir_, config_name_, completed);

    logging::state_save_config scfg;
    scfg.output_path           = full_path;

    auto fut = lm_.serialize_state(std::move(tree), scfg);
    {
        std::lock_guard lock(mtx_);
        pending_.push_back({ std::move(fut), full_path, completed });
    }
    // Stamped HERE, after the write is genuinely submitted (see last_written_).
    last_written_.store(completed, std::memory_order_relaxed);

    // One composed write: the poll loop prints progress lines from the caller
    // thread concurrently, and a multi-token << sequence could tear.
    std::cout << ("  Checkpoint @ step " + std::to_string(completed)
                  + ": " + full_path.filename().string() + "\n");
    diag::emit_progress({ diag::run_phase::checkpoint, completed, start_step_, target_step_,
                          ctrl_.get_actual_sps(), lm_.combined_stats().frames_written });
}

void headless_checkpointer::on_post_swap() noexcept
{
    // do_step() increments step_count AFTER swap_io(), so at the post-swap
    // boundary the number of COMPLETED steps is count + 1.
    const std::uint64_t completed = ctrl_.get_step_count() + 1;
    if (completed <= start_step_) return;
    if ((completed - start_step_) % interval_ != 0) return;

    // COAST_WARN is non-throwing by construction (emit_guarded swallows the
    // std::format OOM that would otherwise escape -> std::terminate), so it is
    // safe here, bare on the sim jthread.
    try {
        write(completed);
    } catch (const std::exception& e) {
        COAST_WARN("checkpoint", "checkpoint @ step {} failed: {} (skipping, run continues)",
                   completed, e.what());
    } catch (...) {
        COAST_WARN("checkpoint", "checkpoint @ step {} failed (unknown exception; skipping, "
                                 "run continues)", completed);
    }
}

void headless_checkpointer::write_final() noexcept
{
    std::uint64_t completed = 0;
    try {
        ctrl_.run_quiesced([&] {
            // Sampled INSIDE the quiesce, so the checkpoint is labelled with the
            // step its extracted state is actually at: the sim thread may have
            // advanced a few steps between the pause request and quiescence.
            completed = ctrl_.get_step_count();
            if (completed > start_step_ &&
                completed != last_written_.load(std::memory_order_relaxed))
                write(completed);
        });
    } catch (const std::exception& e) {
        COAST_WARN("checkpoint", "save-on-term checkpoint @ step {} failed: {}",
                   completed, e.what());
    } catch (...) {
        COAST_WARN("checkpoint", "save-on-term checkpoint failed (unknown exception)");
    }
}

void headless_checkpointer::collect(bool wait,
                                    const std::function<void(const written_t&)>& on_written)
{
    // Move the finished entries out under the lock, then get() them outside it,
    // so the sim thread's next push never waits on a slow callback.
    std::vector<pending_t> done;
    {
        std::lock_guard lock(mtx_);
        auto keep = pending_.begin();
        for (auto it = pending_.begin(); it != pending_.end(); ++it) {
            const bool ready = wait ||
                it->fut.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
            if (ready) {
                done.push_back(std::move(*it));
            } else {
                if (keep != it) *keep = std::move(*it);
                ++keep;
            }
        }
        pending_.erase(keep, pending_.end());
    }
    for (auto& p : done) {
        const logging::log_result result = p.fut.get();
        if (result) {
            on_written({ p.path, p.step });
        } else {
            COAST_WARN("checkpoint", "checkpoint write failed ({}): {}",
                       p.path.filename().string(), result.error_message);
        }
    }
}

} // namespace coast::app
