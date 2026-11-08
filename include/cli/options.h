#ifndef COAST_CLI_OPTIONS_H
#define COAST_CLI_OPTIONS_H

#include <cstdint>
#include <string>

#include "core/seed.h"
#include "diag/level.h"

namespace coast::cli {

struct options {
    std::string config_path;

    // Application-level config file (--app-config). Empty = discover via
    // $COAST_APP_CONFIG > ./coast.json > built-in defaults. Governs WHERE outputs
    // go (config_base) and app assets; distinct from the per-experiment config.
    std::string app_config_path;

    bool        auto_start = false;
    bool        headless   = false;   // effective mode (also implied by -n/--unlimited)
    bool        show_help  = false;

    // True only when -H/--headless appeared on the command line. Distinct from
    // `headless`, which -n/--unlimited also set: a headless BUILD uses this to
    // decide whether to announce that it is headless-only, and the user silences
    // that notice by saying -H explicitly. Inferring it from `headless` would
    // make the notice unreachable, since every accepted headless run sets that.
    bool        headless_flag_given = false;

    // How long a headless run lasts. Deliberately NOT a sentinel: a bare count
    // cannot express "no limit", because the value that would have to mean it
    // (0) is also a perfectly typo-able argument, and a run that silently
    // becomes unbounded is expensive to notice. The mode is therefore explicit
    // and the parser rejects every ambiguous form:
    //   -n/--steps <count>  -> finite    (count == 0 is a parse error)
    //   --unlimited         -> unlimited (run until interrupted)
    //   headless with neither -> a parse error, not a default
    // `max_steps` is meaningful only when steps_mode == finite; read it through
    // the mode, never on its own.
    enum class run_length : std::uint8_t { unset, finite, unlimited };
    run_length    steps_mode = run_length::unset;
    std::uint64_t max_steps  = 0;

    // --init: write a default ./coast.json (app config template) and exit. A
    // pure CLI action handled in main() before anything is built; never
    // overwrites an existing coast.json.
    bool        init       = false;

    // Diagnostic verbosity. Resolved by cli::parse from --log-level / -v / -q;
    // applied by main() via diag::set_level. Default: info and above.
    diag::level log_level = diag::level::info;

    // Diagnostic sink target for --log-file: empty or "stderr" = default stderr
    // sink, "stdout" = stdout sink, anything else = append to that file. Applied
    // by main() via diag::set_log_file / set_stream_sink.
    std::string log_file;

    // Logging / state
    std::string buffer_log_path;
    std::string state_save_path;
    std::string state_load_path;
    bool          resume_latest        = false;
    std::uint64_t checkpoint_interval  = 0;

    // --save-on-term: catch SIGTERM, stop the run and take the normal exit path
    // (exit_reason=user_signal): --log-buffers is closed, --save-state and the
    // manifest are written, and, with --checkpoint, one final checkpoint at the
    // current step goes into the snapshot dir. Lets a supervisor stop a run
    // gracefully and resume it from near where it stopped instead of the
    // previous checkpoint. Without --checkpoint only that final checkpoint is
    // skipped (with a warning). SIGINT is not caught. Headless runs only; a GUI
    // run never arms the handler. Off by default (hard-kill + resume is safe).
    bool          save_on_term         = false;

    // RNG
    seed_offset_t seed_offset = {};

    // Provenance
    std::string emit_run_manifest;        // path for coast_runtime.json (headless only; empty = disabled)
};

} // namespace coast::cli

#endif // COAST_CLI_OPTIONS_H
