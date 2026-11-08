//
// COAST run manifest writer.
// Emits coast_runtime.json (schema_version "1") on headless run completion.
// Records build provenance (git SHA + dirty + build type captured at CMake
// configure time), seed/step counts, exit reason, wall-clock + UTC timestamps,
// hostname, the resolved observe block, and observe_outputs[] +
// savestate_outputs[]. This file is the stable interface a downstream analysis
// harness reads to bind a run's outputs to the exact binary that produced them.
//

#ifndef COAST_SIM_CORE_RUN_MANIFEST_H
#define COAST_SIM_CORE_RUN_MANIFEST_H

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "core/seed.h"

namespace loader { struct global_cfg_t; }

namespace sim {

/// Reasons recorded in `coast_runtime.json::exit_reason`. A scoped enum so the
/// orchestrator can distinguish a controlled fatal from a segfault/OOM-kill
/// (which leaves no manifest at all).
enum class exit_reason_t : std::uint8_t {
    completed_n_steps,    ///< n_steps_requested reached cleanly.
    fatal_error,          ///< Caught std::exception / setup failure in run_headless().
    user_signal,          ///< SIGTERM caught under --save-on-term interrupted the run (SIGINT is not caught).
    step_budget_exceeded  ///< Reserved for future wall-clock / step budget enforcement.
};

[[nodiscard]] std::string to_string(exit_reason_t r);

/// One entry in `savestate_outputs[]` (always array semantics so future
/// --checkpoint <N> intermediate snapshots can append without schema change).
///
/// PATH FRAME (shared by every path field in the manifest — here and
/// `observe.output_dir`): a path at or below the manifest's own directory is
/// recorded RELATIVE with a leading `"./"` (resolve it against the manifest
/// location — the manifest + its outputs are a relocatable bundle); a path
/// outside that directory is recorded ABSOLUTE (never `"../.."`). A consumer
/// therefore resolves every path field the same way: leading `"./"` → relative
/// to the manifest, otherwise absolute. `path` here is populated raw by
/// run_headless; write_run_manifest applies the frame.
struct savestate_output_t {
    std::string   path;
    std::uint64_t step{0};
    std::uint64_t bytes{0};
};

/// Effective RNG seed of one active stochastic node (config seed + global +
/// per-node offsets folded in), sourced from the built entity so the manifest
/// needn't reconstruct coast's seed-composition rule.
struct node_seed_t {
    std::uint64_t id{0};              ///< entity output id (== config node id)
    std::string   label;              ///< config node label (may be empty)
    seed_t        effective_seed{};
    std::optional<seed_t> inhibition_seed; ///< io_entity::inhibition_seed(); written only when set
};

/// Populated incrementally by run_headless() during the run, written to file at
/// a single emit point. Populating incrementally (rather than lazily at emit
/// time) keeps the writer from reading possibly-error-state subsystems.
struct run_manifest_inputs_t {
    std::filesystem::path             output_path;          ///< from --emit-run-manifest (empty = disabled)
    seed_offset_t                     seed_offset{};        ///< from --seed-offset
    /// From --steps. Meaningful only when `unlimited` is false — read the two
    /// together, never `n_steps_requested` alone. (Schema-additive: `unlimited`
    /// was introduced when `--steps 0` stopped meaning "run forever"; a consumer
    /// that ignores it sees the same numbers as before for every finite run.)
    std::uint64_t                     n_steps_requested{0};

    /// From --unlimited.
    ///
    /// `true` occurs only together with `--save-on-term`: cli::parse rejects
    /// `--unlimited --emit-run-manifest` without it, because an unlimited run ends
    /// only by being killed and the manifest is written on the way out. With
    /// `--save-on-term`, a SIGTERM takes the normal exit path and the manifest is
    /// emitted with `unlimited=true` (set in run_headless) and, if the signal
    /// interrupted the run, `exit_reason=user_signal`. This is coastd's
    /// configuration for unbounded jobs.
    bool                              unlimited{false};
    std::uint64_t                     n_steps_completed{0}; ///< controller->get_step_count() at exit
    exit_reason_t                     exit_reason{exit_reason_t::completed_n_steps};
    double                            wall_clock_seconds{0.0};
    std::string                       started_at_utc;       ///< ISO-8601 Z, e.g. "2026-05-18T15:30:00Z"
    std::string                       completed_at_utc;
    std::string                       gpu_adapter;          ///< gpu_context::adapter_name ("<device> [Vulkan|Metal]") if a GPU was used; empty = CPU-only run
    std::vector<savestate_output_t>   savestate_outputs;
    /// The run's id as it ran (sim::env::run_id: "auto" resolved, or adopted from
    /// the checkpoint by --resume); its checkpoints carry it (SPEC.md S.5.14).
    std::string                       run_id;
    /// Resolved OBSERVE output directory — where the .observe.h5 files actually
    /// land (the config's observe leaf placed under config_base, or an absolute
    /// override). Recorded instead of the raw config leaf so the manifest points a
    /// downstream harness at the real directory even when the config omits the leaf
    /// (empty leaf -> app default under config_base). Populated by run_headless;
    /// recorded under the shared PATH FRAME rule (see savestate_output_t): "./…"
    /// when it lands under the manifest dir, else absolute.
    std::filesystem::path             observe_output_dir;
    /// Effective RNG seed per active stochastic node, populated by run_headless
    /// from the built entities so a consumer needn't reconstruct coast's
    /// seed-composition rule. Each entry matches one entity whose effective_seed()
    /// returns a value (i.e. non-nullopt).
    std::vector<node_seed_t>          node_seeds;
};

// NOTE: the manifest deliberately does NOT enumerate observe HDF5 outputs.
// The observe writer is asynchronous and libhdf5 buffers each group's file
// until the writer is closed by sim_instance teardown (detach()), which runs
// *after* run_headless() returns — so at the manifest's emit point neither the
// on-disk byte size nor the flushed row count is authoritative. The manifest
// records the resolved observe *config* (output_dir + groups) instead; a
// downstream harness enumerates the actual .observe.h5 files after the process
// exits (fully flushed) to record authoritative sizes, row counts, and SHA256.
// savestate_outputs[] is safe to record here because state saves are written
// synchronously before the call returns.

/// Build an ISO-8601 UTC timestamp ("YYYY-MM-DDTHH:MM:SSZ") from `tp`.
[[nodiscard]] std::string iso8601_utc(std::chrono::system_clock::time_point tp);

/// Atomic, exception-safe write of `inputs` (+ cfg.observe section) to
/// inputs.output_path. Catches its own throws (does NOT propagate), writes to a
/// tmpfile then std::filesystem::rename. Returns true on success, false on
/// failure (with a stderr diagnostic).
bool write_run_manifest(const run_manifest_inputs_t& inputs,
                        const loader::global_cfg_t&  cfg) noexcept;

} // namespace sim

#endif // COAST_SIM_CORE_RUN_MANIFEST_H
