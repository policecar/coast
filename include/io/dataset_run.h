//
// What a dataset loader knows about the run it belongs to, and the one rule that
// decides whether a checkpoint load restores its sequence (SPEC.md S.5.14).
//
// Every checkpoint names the run that wrote it (metadata/run_id, core/run_id.h). A
// dataset restores its cursor and RNGs only when the loading run has the same
// run_id (state_reader::continues_run): a coastd resume reads the id coastd wrote
// into the job's config, so it continues; a new run ("auto", or another id) starts
// fresh — a frozen trial from a trained state, training on another subset. To
// continue a checkpoint's sequence in a new run, name its run_id in the config.
//
// When it restores, it checks the dataset it was saved with (its identity record)
// against its own and warns on a difference: continuing a sequence on other data is
// almost certainly a mistake, but not one the load can refuse on its own.
//

#ifndef COAST_DATASET_RUN_H
#define COAST_DATASET_RUN_H

#include <string>
#include <string_view>

namespace logging {
class state_writer;
class state_reader;
}

namespace dataset {

// Set by sim::register_entity_handlers at construction; never read from a config
// (no JSON key). Default-constructed in tests and tools that build a loader by hand:
// no identity record.
struct run_binding_t {
    std::string identity;   // the loader's resolved params (no seed) as JSON; "" = none
};

// Writes the identity record into the current (the loader's) group; nothing when
// `run.identity` is empty.
void save_run_binding(logging::state_writer& w, const run_binding_t& run);

// Whether this load restores the sequence: it continues the run that wrote the
// checkpoint (state_reader::continues_run). When it does, checks the identity record
// (a warning naming `who` on a difference; a checkpoint without one is noted on the
// reader).
[[nodiscard]] bool restores_sequence(logging::state_reader& r, const run_binding_t& run,
                                     std::string_view who);

} // namespace dataset

#endif // COAST_DATASET_RUN_H
