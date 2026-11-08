#include "cli/parse.h"

#include <charconv>
#include <cstdint>
#include <ostream>
#include <string>
#include <system_error>

#include "diag/level.h"

namespace coast::cli {

void print_usage(std::ostream& out, const char* program_name) {
    out << "COAST - Neural Simulation\n\n";
#ifndef COAST_WITH_GUI
    // Headless build (-DCOAST_WITH_GUI=OFF): say so up front, because several
    // flags below and every "start GUI" example read differently without it.
    out << "This binary was built WITHOUT GUI support: it always runs headless,\n";
    out << "whether or not --headless is given. Every run therefore needs an\n";
    out << "explicit length — -n/--steps <count> or --unlimited.\n\n";
#endif
    out << "Usage: " << program_name << " [OPTIONS] [CONFIG_FILE]\n\n";
    out << "Options:\n";
    out << "  -c, --config <path>   Path to configuration file\n";
#ifdef COAST_WITH_GUI
    out << "  -r, --run             Start simulation immediately\n";
    out << "  -H, --headless        Run without GUI (implies --run)\n";
#else
    // -r is omitted, not forgotten: it is still accepted, but every run this
    // binary accepts is headless and headless runs auto-start, so listing it
    // would advertise a no-op.
    out << "  -H, --headless        Accepted and implied by this build; passing it\n";
    out << "                        only suppresses the headless-build notice\n";
#endif
    out << "  -n, --steps <count>   Run for <count> steps then exit (implies --headless).\n";
    out << "                        <count> must be >= 1\n";
    out << "  --unlimited           Run until interrupted, with no step limit\n";
    out << "                        (implies --headless). Mutually exclusive with -n.\n";
    out << "                        Rejects --save-state/--emit-run-manifest/--log-buffers\n";
    out << "                        unless --save-on-term is given: nothing else catches\n";
    out << "                        the signal that ends the run, so those outputs would\n";
    out << "                        never be written. Use --checkpoint instead\n";
    out << "  -h, --help            Show this help message\n";
    out << "  --app-config <path>   Application config (where outputs go, assets);\n";
    out << "                        else $COAST_APP_CONFIG > ./coast.json > built-in defaults\n";
    out << "  --init                Write a default ./coast.json (app config template)\n";
    out << "                        and exit; never overwrites an existing one\n";
    out << "\nDiagnostics options:\n";
    out << "  -v, --verbose         Lower the log threshold one level (repeatable)\n";
    out << "  -q, --quiet           Raise the log threshold one level (repeatable)\n";
    out << "  --log-level <level>   trace|debug|info|warn|error|off (default: info)\n";
    out << "  --log-file <target>   Diagnostics sink: stderr (default) | stdout | <path>\n";
    out << "                        (append; keywords case-insensitive). Run narration and\n";
    out << "                        @coast/progress lines stay on stdout regardless.\n";
    out << "\nState options:\n";
    out << "  --resume              Resume from latest snapshot in snapshot directory\n";
    out << "  --load-state <path>   Load initial state from specific HDF5 file\n";
    out << "  --save-state <path>   Save final state to HDF5 file (headless only)\n";
    out << "  --checkpoint <steps>  Auto-save checkpoint every N steps (headless only)\n";
    out << "  --save-on-term        Catch SIGTERM (headless only): stop the run, write a\n";
    out << "                        final checkpoint, and exit through the normal path\n";
    out << "                        (needs --checkpoint for the final write). SIGINT is\n";
    out << "                        NOT caught\n";
    out << "\nRNG options:\n";
    out << "  --seed-offset <int>   Offset added to all random seeds (modulo 2^32)\n";
    out << "\nLogging options:\n";
    out << "  --log-buffers <path>  Log IO buffers to HDF5 file\n";
    out << "\nProvenance options:\n";
    out << "  --emit-run-manifest <path>\n";
    out << "                        Write a coast_runtime.json run manifest on exit\n";
    out << "                        (headless only): git SHA, seed, steps, timing,\n";
    out << "                        observe config + outputs\n";
    out << "\nExamples:\n";
#ifdef COAST_WITH_GUI
    out << "  " << program_name << " config.json                    # Load config, start GUI\n";
    out << "  " << program_name << " -r config.json                 # Load config, auto-start\n";
#endif
    out << "  " << program_name << " --headless -n 1000 config.json # Run 1000 steps, no GUI\n";
    out << "  " << program_name << " --unlimited config.json        # Run until interrupted\n";
    out << "  " << program_name << " -n 50000 --log-buffers run.h5 config.json\n";
    out << "                                           # Log buffer data during run\n";
    out << "  " << program_name << " -n 10000 --save-state checkpoint.h5 config.json\n";
    out << "                                           # Save state after 10k steps\n";
    out << "  " << program_name << " --load-state checkpoint.h5 -n 10000 config.json\n";
    out << "                                           # Resume from specific checkpoint\n";
    out << "  " << program_name << " --resume -n 10000 config.json\n";
    out << "                                           # Resume from latest snapshot\n";
    out << "  " << program_name << " -n 100000 --checkpoint 10000 config.json\n";
    out << "                                           # Save checkpoint every 10k steps\n";
}

namespace {

void needs_value(const std::string& arg, int i, int argc) {
    if (i + 1 >= argc) {
        throw parse_error("Option '" + arg + "' requires a value.");
    }
}

// Strict integer parsing for option values.
//
// std::stoull / std::stoi were the previous idiom here and are wrong on three
// counts, each of which was reachable from the command line:
//   * they throw std::invalid_argument / std::out_of_range, which are NOT
//     parse_error — main()'s handler misses them, so `-n abc` aborted the
//     process (SIGABRT, exit 134) instead of printing a usage error;
//   * they stop at the first character that does not fit, silently ignoring the
//     rest, so `-n 1e6` ran ONE step;
//   * for an unsigned target they accept a minus sign and WRAP, so `-n -5`
//     became 18446744073709551611 steps — precisely the unbounded run that
//     rejecting `-n 0` exists to prevent.
// std::from_chars fixes all three: it refuses a sign for unsigned types, reports
// overflow as result_out_of_range rather than wrapping, and tells us where it
// stopped so we can insist the WHOLE token was consumed.
template <typename T>
T parse_int_value(const std::string& arg, const char* text)
{
    const std::string token{text};
    T                 value{};
    const char* const first = token.c_str();
    const char* const last  = first + token.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);

    // `ptr != last` is the trailing-garbage check: from_chars stops at the first
    // character it cannot use and reports success for the prefix, so without this
    // "1e6" would parse as 1.
    if (ec != std::errc{} || ptr != last) {
        std::string msg = "Value for '";
        msg += arg;
        msg += (ec == std::errc::result_out_of_range) ? "' is out of range: '"
                                                      : "' must be a whole number, got '";
        msg += token;
        msg += "'.";
        throw parse_error(msg);
    }
    return value;
}

// Shift a log level by `delta` steps. Negative = more verbose (-v), positive =
// quieter (-q). Repeatable flags compose.
//
// The clamp floor for -q is `error`, NOT `off`: no amount of -q may silence a
// fatal error message (which is the one line that explains a non-zero exit).
// Total silence is reserved for an explicit `--log-level off` — a deliberate
// "I mean it" choice, never an accident of stacking -q. -v clamps at trace.
diag::level step_level(diag::level l, int delta) {
    constexpr int lo = static_cast<int>(diag::level::trace);
    constexpr int hi = static_cast<int>(diag::level::error);
    // An explicit `off` is sticky: a later -q can't make it quieter (nothing is
    // quieter than off), and it must not silently un-silence. Only -v (louder)
    // steps it back into range — to `error`. This protects the "off = I-mean-it
    // silence" invariant from BOTH sides (the -q clamp at `error` protects it
    // from the other direction, so -q alone can never reach off by accident).
    if (l == diag::level::off)
        return delta < 0 ? diag::level::error : diag::level::off;
    int v = static_cast<int>(l) + delta;
    v = v < lo ? lo : (v > hi ? hi : v);
    return static_cast<diag::level>(v);
}

} // namespace

options parse(int argc, char** argv) {
    options opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            opts.show_help = true;
        } else if (arg == "--init") {
            opts.init = true;
        } else if (arg == "-r" || arg == "--run") {
            opts.auto_start = true;
        } else if (arg == "-H" || arg == "--headless") {
            opts.headless            = true;
            opts.headless_flag_given = true;
            opts.auto_start          = true;
        } else if (arg == "-c" || arg == "--config") {
            needs_value(arg, i, argc);
            opts.config_path = argv[++i];
        } else if (arg == "-n" || arg == "--steps") {
            needs_value(arg, i, argc);
            const std::string count = argv[++i];
            // Conflict FIRST, value second. The other order reports the value
            // problem for `--unlimited -n 0` and helpfully advises passing
            // --unlimited — the flag already on the command line.
            if (opts.steps_mode == options::run_length::unlimited) {
                throw parse_error("Options '--unlimited' and '" + arg +
                                  "' are mutually exclusive.");
            }
            // parse_int_value, not stoull: a negative count would otherwise wrap
            // to a near-infinite one, which is exactly the outcome the zero check
            // below exists to prevent.
            opts.max_steps = parse_int_value<std::uint64_t>(arg, count.c_str());
            if (opts.max_steps == 0) {
                // 0 used to mean "run forever". Rejecting it is the point of the
                // explicit mode: a mistyped or computed-to-zero count must not
                // start an unbounded job.
                std::string msg = "Option '";
                msg += arg;
                msg += "' requires a count of at least 1 (got '";
                msg += count;
                msg += "'). To run without a step limit, pass --unlimited.";
                throw parse_error(msg);
            }
            opts.steps_mode = options::run_length::finite;
            opts.headless   = true;
            opts.auto_start = true;
        } else if (arg == "--unlimited") {
            // NOTE on "run until interrupted": only SIGTERM can end such a run
            // gracefully, and only with --save-on-term. Then the run takes the
            // normal exit path (manifest, --save-state, closed --log-buffers).
            // SIGINT is never caught, and SIGTERM without --save-on-term hits the
            // default disposition: either kills the run outright, with no run
            // manifest, no --save-state, and an unclosed --log-buffers file. The
            // output-flag rejection below enforces this.
            if (opts.steps_mode == options::run_length::finite) {
                throw parse_error("Options '-n/--steps' and '--unlimited' are "
                                  "mutually exclusive.");
            }
            opts.steps_mode = options::run_length::unlimited;
            opts.headless   = true;
            opts.auto_start = true;
        } else if (arg == "--log-buffers") {
            needs_value(arg, i, argc);
            opts.buffer_log_path = argv[++i];
        } else if (arg == "--save-state") {
            needs_value(arg, i, argc);
            opts.state_save_path = argv[++i];
        } else if (arg == "--load-state") {
            needs_value(arg, i, argc);
            opts.state_load_path = argv[++i];
        } else if (arg == "--resume") {
            opts.resume_latest = true;
        } else if (arg == "--checkpoint") {
            needs_value(arg, i, argc);
            opts.checkpoint_interval = parse_int_value<std::uint64_t>(arg, argv[++i]);
        } else if (arg == "--save-on-term") {
            opts.save_on_term = true;
        } else if (arg == "--seed-offset") {
            // Parsed signed so a negative value reaches the range check and is
            // refused there, by the rule a config file's seeds follow (core/seed.h).
            needs_value(arg, i, argc);
            const char* const text = argv[++i];
            const auto bits = core::seed_bits(parse_int_value<std::int64_t>(arg, text));
            if (!bits)
                throw parse_error("Value for '" + arg + "' must be in [0, 4294967295], got '"
                                  + std::string(text) + "'.");
            opts.seed_offset = seed_offset_t{ *bits };
        } else if (arg == "--emit-run-manifest") {
            needs_value(arg, i, argc);
            opts.emit_run_manifest = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            opts.log_level = step_level(opts.log_level, -1);
        } else if (arg == "-q" || arg == "--quiet") {
            opts.log_level = step_level(opts.log_level, +1);
        } else if (arg == "--log-level") {
            needs_value(arg, i, argc);
            const std::string val = argv[++i];
            const auto lvl = diag::level_from(val);
            if (!lvl) {
                throw parse_error("Invalid --log-level '" + val +
                                  "' (expected trace|debug|info|warn|error|off).");
            }
            opts.log_level = *lvl;
        } else if (arg == "--log-file") {
            needs_value(arg, i, argc);
            opts.log_file = argv[++i];
        } else if (arg == "--app-config") {
            needs_value(arg, i, argc);
            opts.app_config_path = argv[++i];
        } else if (!arg.empty() && arg[0] != '-' && opts.config_path.empty()) {
            // Positional: config path
            opts.config_path = arg;
        } else {
            throw parse_error("Unknown option: " + arg);
        }
    }

    // A headless run must say how long it runs. There is no default: the old
    // default was "forever", which is the one outcome a user never wants by
    // accident. Since -n/--unlimited both set `headless`, this fires exactly when
    // -H/--headless was given with no length. --help and --init exit before
    // anything runs, so they are exempt.
    //
    // Deliberately build-INDEPENDENT. A headless build promotes every run to
    // headless, so it needs the same guarantee for a command line with no -H at
    // all — but that promotion happens in main(), and enforcing it here would
    // make cli::parse behave differently per build configuration, which would in
    // turn make its unit tests untestable in one of them. main() applies the same
    // rule with the same message at the point where it does the promoting.
    if (opts.headless && !opts.show_help && !opts.init &&
        opts.steps_mode == options::run_length::unset) {
        throw parse_error(headless_needs_length_msg);
    }

    // An --unlimited run only ends when it is killed, and without --save-on-term
    // nothing catches the signal (SIGINT is never caught), so end-of-run outputs
    // are not merely missing — they are actively destroyed. Measured on a 6-second run: --save-state and
    // --emit-run-manifest produce no file at all (both are written on the way out,
    // which never happens), and --log-buffers leaves a 48 MB HDF5 file that h5ls
    // reports as **NOT FOUND** because it was never closed.
    //
    // Accepting these combinations silently would repeat, in a new place, exactly
    // the mistake that removing the `-n 0` sentinel was meant to fix: a long run
    // that quietly produces nothing usable is expensive to discover. So they are a
    // parse error, not a warning.
    //
    // --checkpoint is deliberately NOT in this list: its snapshots are written and
    // closed DURING the run, so an --unlimited run checkpointing every N steps
    // leaves a usable trail of completed files. It is the supported way to get
    // output out of an unbounded run today.
    //
    // --save-on-term is the exemption, and it is the case this block anticipated:
    // with it, SIGTERM stops the loop and the run takes the NORMAL exit path —
    // buffer logging is closed, --save-state is written, the manifest is emitted
    // (with exit_reason=user_signal). The outputs are honest again, so refusing
    // them would block the one configuration that delivers them. This is what
    // coastd's unbounded jobs use: --unlimited --save-on-term --save-state
    // --emit-run-manifest, terminated by the daemon's SIGTERM.
    //
    // The exemption is deliberately narrow: SIGINT (Ctrl-C) is still unhandled, so
    // an interactively-killed run loses these outputs even with --save-on-term.
    // The flag is a contract with a supervisor that sends SIGTERM, not a general
    // promise, and it says so in --help.
    //
    // --help and --init are exempt for the same reason as the run-length check
    // above: they exit before anything runs, so no output can be lost. Without the
    // exemption, appending --help to a rejected command line — the obvious way to
    // go looking for the right flag — reprints the rejection, whose own last line
    // advises using --help.
    if (opts.steps_mode == options::run_length::unlimited &&
        !opts.save_on_term && !opts.show_help && !opts.init) {
        std::string offenders;
        const auto note = [&offenders](const char* flag) {
            if (!offenders.empty()) offenders += ", ";
            offenders += flag;
        };
        if (!opts.state_save_path.empty())   note("--save-state");
        if (!opts.emit_run_manifest.empty()) note("--emit-run-manifest");
        if (!opts.buffer_log_path.empty())   note("--log-buffers");

        if (!offenders.empty()) {
            std::string msg = "Option '--unlimited' cannot be combined with ";
            msg += offenders;
            msg += ": an unlimited run ends only by being killed, and without "
                   "--save-on-term nothing catches that signal, so those outputs "
                   "would be lost or left corrupt. Add --save-on-term (a SIGTERM "
                   "then ends the run through the normal exit path, writing them), "
                   "use -n/--steps <count> for a run whose outputs you "
                   "need, or --checkpoint <steps>, whose snapshots are completed "
                   "during the run.";
            throw parse_error(msg);
        }
    }

    return opts;
}

} // namespace coast::cli
