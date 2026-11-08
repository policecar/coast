#ifndef COAST_APP_BUILD_ENV_H
#define COAST_APP_BUILD_ENV_H

#include <memory>
#include <stdexcept>

namespace coast { class sim_instance; }
namespace hdf5  { class executor; }

namespace coast::cli { struct options; }

namespace coast::paths { struct app_config; }

namespace coast::app {

// The state build_env was asked to restore (--load-state, or the snapshot
// --resume found) could not be loaded — as opposed to a configuration failure.
// The message starts "Failed to load state from <path>: "; for a snapshot found
// by --resume it ends with the remedy (drop --resume, or move the directory).
struct state_load_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Build a fully configured sim_instance from CLI options.
//
//   - Loads JSON config (opts.config_path)
//   - Constructs sim_instance with the given hdf5 executor and seed_offset
//   - Resolves --resume / --load-state: if either is requested and a snapshot
//     is found, restores state and step counter.
//
// Returns nullptr iff opts.config_path is empty (GUI mode legitimately starts
// without a loaded simulation). In headless mode the caller must treat
// nullptr as an error.
//
// The app_config governs WHERE per-config outputs (snapshots) are resolved:
// the resume-snapshot lookup routes the config's snapshot_dir leaf through
// app_cfg.paths.config_base (see coast::paths::resolve_config_output).
//
// Throws state_load_error when the state cannot be loaded (unreadable file, a
// checkpoint that does not fit, a corrupt one), std::runtime_error on any other
// well-defined setup failure (missing config file, malformed JSON, etc.).
std::unique_ptr<sim_instance> build_env(const cli::options& opts,
                                        const paths::app_config&   app_cfg,
                                        hdf5::executor&     h5_exec);

// Convenience overload: build with the built-in default app_config. Used by
// in-process tests that don't exercise config_base output resolution (they
// fingerprint state or drive --load-state, never the default snapshot layout).
std::unique_ptr<sim_instance> build_env(const cli::options& opts,
                                        hdf5::executor&     h5_exec);

} // namespace coast::app

#endif // COAST_APP_BUILD_ENV_H
