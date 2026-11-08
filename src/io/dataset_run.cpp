#include "io/dataset_run.h"

#include "logging/config_record.h"
#include "logging/state_io.h"

namespace dataset {

namespace {

logging::config_record identity_record(const run_binding_t& run)
{
    return { { .name = "dataset", .value = run.identity } };
}

} // namespace

void save_run_binding(logging::state_writer& w, const run_binding_t& run)
{
    if (!run.identity.empty())
        logging::write_config_record(w, identity_record(run));
}

bool restores_sequence(logging::state_reader& r, const run_binding_t& run, std::string_view who)
{
    if (!r.continues_run())
        return false;
    if (!run.identity.empty())
        logging::check_config_record(r, identity_record(run), who);
    return true;
}

} // namespace dataset
