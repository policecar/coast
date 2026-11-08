#include "app/run_provenance.h"

#include <system_error>

#include "cli/options.h"
#include "observe/observe_registry.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_thread_controller.h"

namespace coast::app {

namespace {

// One entry per entity with an effective seed (sim_core/run_manifest.h
// node_seed_t), labelled with its built node's label — so a node created from a
// node_set carries its derived "<base_label>.<template label>".
std::vector<sim::node_seed_t> collect_node_seeds(sim_instance& sim)
{
    // Read from the built entity, so the manifest needn't reconstruct coast's
    // seed-composition rule (config seed + --seed-offset + per-node offset).
    std::vector<sim::node_seed_t> out;
    const auto& builder = sim.get_builder();
    for (auto& e : sim.get_sim_env().iterate_entities()) {
        const auto seed = e.effective_seed();
        if (!seed) continue;
        const auto  oid  = e.get_outp_id();
        const auto* node = builder.get_node(oid);   // output id == node id
        out.push_back({ static_cast<std::uint64_t>(oid), node ? node->label : std::string{},
                        *seed, e.inhibition_seed() });
    }
    return out;
}

} // namespace

run_provenance::run_provenance(sim_instance& sim, const cli::options& opts)
  : sim_(sim), t0_(std::chrono::steady_clock::now())
{
    m_.output_path = opts.emit_run_manifest;   // empty = disabled
    if (m_.output_path.empty()) return;

    // The RESOLVED observe output directory, so provenance points at the real
    // .observe.h5 location. The observe registry, attached by sim_instance during
    // build_env, already holds it.
    m_.observe_output_dir = observe::observe_registry::instance().output_dir();
    m_.seed_offset        = opts.seed_offset;
    m_.unlimited          = (opts.steps_mode == cli::options::run_length::unlimited);
    m_.n_steps_requested  = m_.unlimited ? 0 : opts.max_steps;
    m_.exit_reason        = sim::exit_reason_t::fatal_error;   // the run sets the real one
    m_.started_at_utc     = sim::iso8601_utc(std::chrono::system_clock::now());
#ifdef COAST_WITH_GPU
    if (sim.get_sim_env().has_gpu())
        m_.gpu_adapter = sim.get_sim_env().gpu_adapter_name();
#endif
    m_.node_seeds = collect_node_seeds(sim);
}

run_provenance::~run_provenance()
{
    if (m_.output_path.empty()) return;
    try {
        if (const auto* c = sim_.get_thread_controller())
            m_.n_steps_completed = c->get_step_count();
        m_.completed_at_utc   = sim::iso8601_utc(std::chrono::system_clock::now());
        m_.wall_clock_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_).count();
        m_.run_id = sim_.get_sim_env().run_id();
        (void)sim::write_run_manifest(m_, sim_.get_config());   // reports its own failure
    } catch (...) {
        // A manifest failure must never break run teardown.
    }
}

void run_provenance::record_savestate(const std::filesystem::path& p, std::uint64_t step)
{
    if (m_.output_path.empty()) return;
    std::error_code sz_ec;
    const auto bytes = std::filesystem::file_size(p, sz_ec);
    std::error_code abs_ec;
    const auto abs = std::filesystem::absolute(p, abs_ec).lexically_normal();
    m_.savestate_outputs.push_back(sim::savestate_output_t{
        (abs_ec ? p : abs).generic_string(),
        step,
        sz_ec ? 0u : static_cast<std::uint64_t>(bytes)
    });
}

} // namespace coast::app
