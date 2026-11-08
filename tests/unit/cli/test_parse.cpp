#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <array>
#include <string>
#include <vector>

#include "cli/parse.h"
#include "diag/diag.h"

namespace {

// Build an argv from C++ strings. Lives in test scope to keep things readable.
struct argv_holder {
    std::vector<std::string> store;
    std::vector<char*>       argv;

    explicit argv_holder(std::initializer_list<std::string> items) {
        store.assign(items.begin(), items.end());
        argv.reserve(store.size());
        for (auto& s : store) argv.push_back(s.data());
    }
    int    argc() { return static_cast<int>(argv.size()); }
    char** data() { return argv.data(); }
};

coast::cli::options run(std::initializer_list<std::string> items) {
    argv_holder h{items};
    return coast::cli::parse(h.argc(), h.data());
}

} // namespace

TEST_CASE("cli::parse: empty argv yields default options") {
    auto o = run({"coast"});
    CHECK(o.config_path.empty());
    CHECK_FALSE(o.auto_start);
    CHECK_FALSE(o.headless);
    CHECK_FALSE(o.show_help);
    CHECK(o.max_steps           == 0);
    CHECK(o.steps_mode          == coast::cli::options::run_length::unset);
    CHECK_FALSE(o.headless_flag_given);
    CHECK(o.seed_offset         == seed_offset_t{ 0 });
    CHECK(o.checkpoint_interval == 0);
    CHECK(o.resume_latest       == false);
}

TEST_CASE("cli::parse: positional argument → config_path") {
    auto o = run({"coast", "configs/mnist.json"});
    CHECK(o.config_path == "configs/mnist.json");
    CHECK_FALSE(o.headless);
}

TEST_CASE("cli::parse: -c / --config") {
    CHECK(run({"coast", "-c",       "x.json"}).config_path == "x.json");
    CHECK(run({"coast", "--config", "y.json"}).config_path == "y.json");
}

TEST_CASE("cli::parse: -r / --run sets auto_start only") {
    auto o = run({"coast", "-r", "x.json"});
    CHECK(o.auto_start);
    CHECK_FALSE(o.headless);
}

TEST_CASE("cli::parse: -H / --headless implies auto_start") {
    // -H needs a run length like any other headless run, hence the -n here.
    auto o = run({"coast", "-H", "-n", "1", "x.json"});
    CHECK(o.headless);
    CHECK(o.headless_flag_given);
    CHECK(o.auto_start);
}

TEST_CASE("cli::parse: -n / --steps implies headless and auto_start") {
    auto o = run({"coast", "-n", "12345", "x.json"});
    CHECK(o.max_steps  == 12345);
    CHECK(o.steps_mode == coast::cli::options::run_length::finite);
    CHECK(o.headless);
    CHECK(o.auto_start);
    // -n does not count as saying "-H": a headless build still announces itself.
    CHECK_FALSE(o.headless_flag_given);
}

// --- run length: explicit, never a sentinel ---------------------------------
//
// `-n 0` used to mean "run forever". These pin the replacement contract so the
// sentinel cannot come back by accident.

TEST_CASE("cli::parse: --unlimited implies headless and auto_start") {
    auto o = run({"coast", "--unlimited", "x.json"});
    CHECK(o.steps_mode == coast::cli::options::run_length::unlimited);
    CHECK(o.headless);
    CHECK(o.auto_start);
}

TEST_CASE("cli::parse: -n 0 is rejected, not treated as unlimited") {
    CHECK_THROWS_AS(run({"coast", "-n", "0", "x.json"}),      coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--steps", "0", "x.json"}), coast::cli::parse_error);
}

TEST_CASE("cli::parse: a malformed -n is a parse_error, never a wrap or an abort") {
    // Each of these used to escape std::stoull's semantics in its own way:
    //   "abc"  -> std::invalid_argument, uncaught -> SIGABRT
    //   huge   -> std::out_of_range,     uncaught -> SIGABRT
    //   "-5"   -> silently wrapped to 18446744073709551611 (an unbounded run!)
    //   "1e6"  -> silently truncated to 1
    //   "5x"   -> silently truncated to 5
    // All must now surface as parse_error, which main() turns into a usage
    // message and exit 1.
    CHECK_THROWS_AS(run({"coast", "-n", "abc", "x.json"}), coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n", "",    "x.json"}), coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n", "-5",  "x.json"}), coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n", "1e6", "x.json"}), coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n", "5x",  "x.json"}), coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n", "+5",  "x.json"}), coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n", " 5",  "x.json"}), coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n", "99999999999999999999999", "x.json"}),
                    coast::cli::parse_error);
    // The valid form still works, and the boundary value is accepted.
    CHECK(run({"coast", "-n", "1", "x.json"}).max_steps == 1);
}

TEST_CASE("cli::parse: --checkpoint and --seed-offset validate the same way") {
    CHECK_THROWS_AS(run({"coast", "-n", "1", "--checkpoint", "abc", "x.json"}),
                    coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n", "1", "--checkpoint", "-5", "x.json"}),
                    coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n", "1", "--seed-offset", "abc", "x.json"}),
                    coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n", "1", "--seed-offset", "7q", "x.json"}),
                    coast::cli::parse_error);
    CHECK(run({"coast", "-n", "1", "--checkpoint", "50", "x.json"}).checkpoint_interval == 50);
    // A negative --seed-offset is refused: one seed rule, [0, 2^32 - 1] (core/seed.h).
    CHECK_THROWS_AS(run({"coast", "-n", "1", "--seed-offset", "-3", "x.json"}),
                    coast::cli::parse_error);
    CHECK(run({"coast", "-n", "1", "--seed-offset", "3",  "x.json"}).seed_offset == seed_offset_t{ 3 });
}

TEST_CASE("cli::parse: a headless run without a length is rejected") {
    CHECK_THROWS_AS(run({"coast", "-H", "x.json"}),         coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--headless", "x.json"}), coast::cli::parse_error);
}

TEST_CASE("cli::parse: -n and --unlimited are mutually exclusive, either order") {
    CHECK_THROWS_AS(run({"coast", "-n", "10", "--unlimited", "x.json"}),
                    coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--unlimited", "-n", "10", "x.json"}),
                    coast::cli::parse_error);
}

TEST_CASE("cli::parse: the conflict is reported before the value is judged") {
    // `--unlimited -n 0` must not answer "pass --unlimited" — the flag is already
    // there. The conflict check runs first so the message names the real problem.
    try {
        run({"coast", "--unlimited", "-n", "0", "x.json"});
        FAIL("expected a parse_error");
    } catch (const coast::cli::parse_error& e) {
        const std::string what = e.what();
        CHECK(what.find("mutually exclusive") != std::string::npos);
    }
}

TEST_CASE("cli::parse: --unlimited refuses outputs it would destroy") {
    // No signal handler exists, so an unlimited run never writes these: they are
    // rejected rather than silently producing nothing (or a corrupt file).
    CHECK_THROWS_AS(run({"coast", "--unlimited", "--save-state", "s.h5", "x.json"}),
                    coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--unlimited", "--emit-run-manifest", "m.json", "x.json"}),
                    coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--unlimited", "--log-buffers", "b.h5", "x.json"}),
                    coast::cli::parse_error);
    // Order must not matter — the check runs after the whole argv is consumed.
    CHECK_THROWS_AS(run({"coast", "--save-state", "s.h5", "--unlimited", "x.json"}),
                    coast::cli::parse_error);
    // …but a finite run may use all of them, and --checkpoint is allowed even
    // unlimited: its snapshots are completed during the run, not at the end.
    CHECK_NOTHROW(run({"coast", "-n", "10", "--save-state", "s.h5",
                       "--emit-run-manifest", "m.json", "--log-buffers", "b.h5", "x.json"}));
    CHECK_NOTHROW(run({"coast", "--unlimited", "--checkpoint", "50", "x.json"}));
}

TEST_CASE("cli::parse: --save-on-term makes --unlimited outputs honest again") {
    // The rejection above exists because nothing catches the signal that ends an
    // unlimited run. --save-on-term does catch it (SIGTERM), and the run then
    // leaves through the NORMAL exit path — buffer log closed, state saved,
    // manifest written. So the exact combination coastd launches its unbounded
    // jobs with must parse.
    CHECK_NOTHROW(run({"coast", "--unlimited", "--save-on-term", "--save-state", "s.h5",
                       "--emit-run-manifest", "m.json", "--log-buffers", "b.h5",
                       "--checkpoint", "5000", "x.json"}));
    // Order must not matter here either — save_on_term is read after the whole
    // argv is consumed, so it lifts the rejection regardless of where it sits.
    CHECK_NOTHROW(run({"coast", "--save-state", "s.h5", "--save-on-term",
                       "--unlimited", "x.json"}));
    // Removing --save-on-term restores the rejection: the exemption is the flag,
    // not the flag's neighbours.
    CHECK_THROWS_AS(run({"coast", "--unlimited", "--save-state", "s.h5", "x.json"}),
                    coast::cli::parse_error);
    // It does not disturb the run-length rules it sits next to.
    CHECK_THROWS_AS(run({"coast", "--save-on-term", "-n", "0", "x.json"}),
                    coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--save-on-term", "--headless", "x.json"}),
                    coast::cli::parse_error);
}

TEST_CASE("cli::parse: --help and --init escape the --unlimited output rejection") {
    // Appending --help to a rejected command line is the obvious way to go looking
    // for the right flag; it must print usage, not reprint the rejection whose own
    // last line says "use --help". Both exit before anything runs, so no output
    // can be lost and the rejection has nothing to protect.
    CHECK(run({"coast", "--unlimited", "--save-state", "s.h5", "--help"}).show_help);
    CHECK(run({"coast", "--unlimited", "--log-buffers", "b.h5", "--help"}).show_help);
    CHECK(run({"coast", "--unlimited", "--emit-run-manifest", "m.json", "-h"}).show_help);
    CHECK(run({"coast", "--unlimited", "--save-state", "s.h5", "--init"}).init);
    // The same command line without --help/--init is still rejected.
    CHECK_THROWS_AS(run({"coast", "--unlimited", "--save-state", "s.h5"}),
                    coast::cli::parse_error);
}

TEST_CASE("cli::parse: --help is exempt from the run-length requirement") {
    // --help must print usage even when the command line says nothing else.
    CHECK(run({"coast", "--help"}).show_help);
    CHECK(run({"coast", "-H", "--help"}).show_help);
}

TEST_CASE("cli::parse: the run-length rule is build-independent") {
    // A command line with no -H parses fine in EVERY build configuration, even
    // though a COAST_WITH_GUI=OFF binary will go on to reject it in main(). The
    // rule that varies by build lives there, not here — otherwise these very
    // tests would mean different things in the two configurations.
    CHECK_NOTHROW(run({"coast"}));
    CHECK_NOTHROW(run({"coast", "x.json"}));
    CHECK_NOTHROW(run({"coast", "-r", "x.json"}));
}

TEST_CASE("cli::parse: -h / --help") {
    CHECK(run({"coast", "-h"}).show_help);
    CHECK(run({"coast", "--help"}).show_help);
}

TEST_CASE("cli::parse: state-related flags") {
    auto o = run({
        "coast",
        "--load-state", "snap.h5",
        "--save-state", "out.h5",
        "--resume",
        "--checkpoint", "1000",
        "x.json"
    });
    CHECK(o.state_load_path     == "snap.h5");
    CHECK(o.state_save_path     == "out.h5");
    CHECK(o.resume_latest);
    CHECK(o.checkpoint_interval == 1000);
}

TEST_CASE("cli::parse: --log-buffers") {
    auto o = run({"coast", "--log-buffers", "run.h5", "x.json"});
    CHECK(o.buffer_log_path == "run.h5");
}

TEST_CASE("cli::parse: --seed-offset takes [0, 2^32 - 1], the rule of core/seed.h") {
    CHECK(run({"coast", "--seed-offset",           "0", "x.json"}).seed_offset == seed_offset_t{ 0 });
    CHECK(run({"coast", "--seed-offset",           "7", "x.json"}).seed_offset == seed_offset_t{ 7 });
    CHECK(run({"coast", "--seed-offset",  "4294967295", "x.json"}).seed_offset == seed_offset_t{ 4294967295u });
    CHECK_THROWS_AS(run({"coast", "--seed-offset", "-1", "x.json"}),          coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--seed-offset",  "4294967296", "x.json"}), coast::cli::parse_error);
}

TEST_CASE("cli::parse: missing value throws parse_error") {
    CHECK_THROWS_AS(run({"coast", "-c"}),             coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--config"}),       coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-n"}),             coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--steps"}),        coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--load-state"}),   coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--save-state"}),   coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--checkpoint"}),   coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--seed-offset"}),  coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--log-buffers"}),  coast::cli::parse_error);
}

TEST_CASE("cli::parse: unknown option throws parse_error") {
    CHECK_THROWS_AS(run({"coast", "--bogus"}), coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "-z"}),      coast::cli::parse_error);
}

TEST_CASE("cli::parse: second positional after config_path set is not silently consumed") {
    // First positional wins; second one is unexpected. The current parser
    // accepts it as garbage in the "else if (arg[0] != '-' && config_path.empty())"
    // path → falls through to throw. This pins the behavior.
    CHECK_THROWS_AS(run({"coast", "a.json", "b.json"}), coast::cli::parse_error);
}

TEST_CASE("cli::parse: flag order does not matter") {
    auto a = run({"coast", "x.json", "-n", "100"});
    auto b = run({"coast", "-n", "100", "x.json"});
    CHECK(a.config_path == b.config_path);
    CHECK(a.max_steps   == b.max_steps);
    CHECK(a.headless    == b.headless);
}

// --- diagnostics flags ------------------------------------------------------

TEST_CASE("cli::parse: default diagnostics = info level, no log file") {
    auto o = run({"coast"});
    CHECK(o.log_level == coast::diag::level::info);
    CHECK(o.log_file.empty());
}

TEST_CASE("cli::parse: --log-level sets each level explicitly") {
    CHECK(run({"coast", "--log-level", "trace"}).log_level == coast::diag::level::trace);
    CHECK(run({"coast", "--log-level", "debug"}).log_level == coast::diag::level::debug);
    CHECK(run({"coast", "--log-level", "info"}).log_level  == coast::diag::level::info);
    CHECK(run({"coast", "--log-level", "warn"}).log_level  == coast::diag::level::warn);
    CHECK(run({"coast", "--log-level", "error"}).log_level == coast::diag::level::error);
    CHECK(run({"coast", "--log-level", "off"}).log_level   == coast::diag::level::off);
}

TEST_CASE("cli::parse: --log-level is case-insensitive") {
    CHECK(run({"coast", "--log-level", "INFO"}).log_level  == coast::diag::level::info);
    CHECK(run({"coast", "--log-level", "Debug"}).log_level == coast::diag::level::debug);
}

TEST_CASE("cli::parse: invalid --log-level throws parse_error") {
    CHECK_THROWS_AS(run({"coast", "--log-level", "bogus"}), coast::cli::parse_error);
}

TEST_CASE("cli::parse: -v/--verbose lowers threshold, clamps at trace") {
    CHECK(run({"coast", "-v"}).log_level            == coast::diag::level::debug); // info→debug
    CHECK(run({"coast", "--verbose"}).log_level     == coast::diag::level::debug);
    CHECK(run({"coast", "-v", "-v"}).log_level      == coast::diag::level::trace); // →trace
    CHECK(run({"coast", "-v","-v","-v","-v"}).log_level == coast::diag::level::trace); // clamp
}

TEST_CASE("cli::parse: -q/--quiet raises threshold, clamps at error (never off)") {
    CHECK(run({"coast", "-q"}).log_level            == coast::diag::level::warn);  // info→warn
    CHECK(run({"coast", "--quiet"}).log_level       == coast::diag::level::warn);
    CHECK(run({"coast", "-q", "-q"}).log_level      == coast::diag::level::error); // →error
    // A fatal error must survive any amount of -q — the clamp is `error`, not `off`.
    CHECK(run({"coast", "-q","-q","-q","-q"}).log_level == coast::diag::level::error);
}

TEST_CASE("cli::parse: -v and -q compose in order") {
    // info --(-q -q)--> error --(-v)--> warn
    CHECK(run({"coast", "-q", "-q", "-v"}).log_level == coast::diag::level::warn);
}

TEST_CASE("cli::parse: an explicit --log-level off is sticky against later -q") {
    // -q must not un-silence an explicit off (order-independent).
    CHECK(run({"coast", "--log-level", "off", "-q"}).log_level      == coast::diag::level::off);
    CHECK(run({"coast", "--log-level", "off", "-q", "-q"}).log_level == coast::diag::level::off);
    CHECK(run({"coast", "-q", "--log-level", "off"}).log_level       == coast::diag::level::off);
    // -v (louder) is the only way to step off back into range → error.
    CHECK(run({"coast", "--log-level", "off", "-v"}).log_level      == coast::diag::level::error);
}

TEST_CASE("cli::parse: --log-file targets") {
    CHECK(run({"coast", "--log-file", "stdout"}).log_file     == "stdout");
    CHECK(run({"coast", "--log-file", "/tmp/x.log"}).log_file == "/tmp/x.log");
}

TEST_CASE("cli::parse: diagnostics flags missing value throws") {
    CHECK_THROWS_AS(run({"coast", "--log-level"}), coast::cli::parse_error);
    CHECK_THROWS_AS(run({"coast", "--log-file"}),  coast::cli::parse_error);
}
