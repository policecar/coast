// B-tier subprocess tests against the built `coast` binary.
//
// These tests spawn the actual user-facing executable, exercising the real
// CLI surface (argv parsing, exit codes, stdout/stderr) and file-system side
// effects of options like --save-state, --load-state, --checkpoint, and
// --log-buffers. They do NOT re-validate simulation state — that's A-tier's
// job. Coverage focus here is the harness around the simulation.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "await_progress.h"
#include "subprocess.h"
#include "hdf5/executor.h"
#include <hdf5.h>
#include "logging/state_logger.h"

// The coast::simc façade lives in coast_gui (its run dashboard and log viewers
// are ImGui windows), so the simc cases at the end of this file exist only in a
// GUI build. Everything above them drives the `coast` binary through its CLI and
// runs in every configuration.
#ifdef COAST_WITH_GUI
#include "sim_console/sim_console_api.h"
#endif

namespace fs = std::filesystem;
using coast::test::run;
using coast::test::subprocess_result;

namespace {

// Per-test scratch directory under $TMPDIR. Wiped on destruction so a failed
// test doesn't leak files into the developer's machine.
struct scratch_dir {
    fs::path path;
    explicit scratch_dir(std::string_view label) {
        path = fs::temp_directory_path() /
               ("coast_btier_" + std::string(label) + "_" +
                std::to_string(::getpid()) + "_" +
                std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~scratch_dir() { std::error_code ec; fs::remove_all(path, ec); }
};

// The last "@coast/progress" line of a run's stdout, or "" if there is none.
std::string last_progress_line(const std::string& out) {
    std::istringstream in(out);
    std::string line, last;
    while (std::getline(in, line))
        if (line.starts_with("@coast/progress ")) last = line;
    return last;
}

// True iff `needle` appears anywhere in `haystack`. Tiny convenience.
bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// Read a small text file (used for inspecting generated configs / logs).
std::string read_file(const fs::path& p) {
    std::ifstream in(p);
    std::ostringstream s; s << in.rdbuf();
    return s.str();
}

// Write a big-endian uint32 — the on-disk encoding of IDX header fields.
void write_be32(std::ostream& os, std::uint32_t v) {
    const unsigned char b[4] = {
        static_cast<unsigned char>((v >> 24) & 0xFFu),
        static_cast<unsigned char>((v >> 16) & 0xFFu),
        static_cast<unsigned char>((v >> 8)  & 0xFFu),
        static_cast<unsigned char>( v        & 0xFFu),
    };
    os.write(reinterpret_cast<const char*>(b), 4);
}

// Emit a synthetic MNIST-format dataset (IDX image + label files) of `count`
// 28x28 images. Pixel/label values follow a deterministic, non-degenerate
// pattern so entropy and learning have signal; the regression only cares about
// the dataset *size*, not its content.
void write_synthetic_mnist(const fs::path& img_path, const fs::path& lbl_path,
                           std::uint32_t count) {
    constexpr std::uint32_t rows = 28, cols = 28;
    const std::size_t pixels = static_cast<std::size_t>(rows) * cols;
    {
        std::ofstream img(img_path, std::ios::binary);
        write_be32(img, 0x00000803u);   // magic: unsigned byte, 3 dimensions
        write_be32(img, count);
        write_be32(img, rows);
        write_be32(img, cols);
        std::vector<char> buf(static_cast<std::size_t>(count) * pixels);
        for (std::uint32_t n = 0; n < count; ++n) {
            for (std::size_t p = 0; p < pixels; ++p) {
                buf[n * pixels + p] = static_cast<char>((n * 31u + p * 7u) & 0xFFu);
            }
        }
        img.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
    {
        std::ofstream lbl(lbl_path, std::ios::binary);
        write_be32(lbl, 0x00000801u);   // magic: unsigned byte, 1 dimension
        write_be32(lbl, count);
        std::vector<char> buf(count);
        for (std::uint32_t n = 0; n < count; ++n) {
            buf[n] = static_cast<char>(n % 10u);
        }
        lbl.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
}

// Derive a config from configs/mnist.json that points the mnist dataset at the
// given IDX files, written into `dir`. Only the dataset paths differ from the
// repo config, so the network structure (and thus saved state layout) matches.
fs::path write_mnist_config_for(const fs::path& dir, std::string_view tag,
                                const fs::path& img, const fs::path& lbl) {
    auto j = nlohmann::json::parse(read_file(fs::path{COAST_REPO_ROOT} / "configs" / "mnist.json"));
    for (const char* sub : {"max_interval", "default"}) {
        j["datasets"]["mnist"][sub]["image_file"] = img.string();
        j["datasets"]["mnist"][sub]["label_file"] = lbl.string();
    }
    const auto cfg = dir / ("mnist_" + std::string(tag) + ".json");
    std::ofstream out(cfg);
    out << j.dump(2);
    return cfg;
}

// As write_mnist_config_for, but also sets `snapshot_dir` to the given value
// (relative or absolute). Used by the path-resolution tests below, which need a
// config that is runnable from any working directory (absolute dataset paths)
// while its snapshot_dir stays a controlled relative/absolute string.
fs::path write_mnist_config_with_snapshot_dir(const fs::path& dir,
                                              std::string_view tag,
                                              const fs::path& img,
                                              const fs::path& lbl,
                                              const std::string& snapshot_dir) {
    const auto cfg = write_mnist_config_for(dir, tag, img, lbl);
    auto j = nlohmann::json::parse(read_file(cfg));
    j["snapshot_dir"] = snapshot_dir;
    { std::ofstream out(cfg); out << j.dump(2); }
    return cfg;
}

// Rewrites the neuron count of a config written by the helpers above (the one
// group of configs/mnist.json), in place.
void set_neuron_count(const fs::path& cfg, int count) {
    auto j = nlohmann::json::parse(read_file(cfg));
    j["ngm2"]["neuron_groups"]["L4-mnist"]["neurons"][0]["neuron_count"] = count;
    { std::ofstream out(cfg); out << j.dump(2); }
}

} // namespace

// ---------------------------------------------------------------------------
// 1. --help is a clean stdout banner with exit 0.

TEST_CASE("b_tier.cli_help: --help prints usage and exits 0") {
    auto r = run(COAST_BIN, {"--help"});
    CHECK(r.exit_code == 0);
    CHECK(contains(r.stdout_text, "Usage:"));
    CHECK(contains(r.stdout_text, "--headless"));
    CHECK(r.stderr_text.empty());
}

// ---------------------------------------------------------------------------
// 2. Headless mode without a config errors out clearly.

TEST_CASE("b_tier.cli_missing_config: --headless without config fails with helpful stderr") {
    auto r = run(COAST_BIN, {"--headless", "-n", "100"});
    CHECK(r.exit_code != 0);
    CHECK(contains(r.stderr_text, "configuration file required"));
}

// ---------------------------------------------------------------------------
// 3. Unknown CLI flags trip the parser.

TEST_CASE("b_tier.cli_unknown_flag: --bogus exits non-zero with diagnostic") {
    auto r = run(COAST_BIN, {"--bogus"});
    CHECK(r.exit_code != 0);
    CHECK(contains(r.stderr_text, "Unknown option"));
}

// ---------------------------------------------------------------------------
// 3. A node listing the same input channel name twice is refused at build.
//
// sim_builder keys a node's inputs by channel name, so a second link under the
// same name used to overwrite the first, dropping it without a word.

TEST_CASE("b_tier.config_duplicate_input_channel: a repeated channel name fails the build") {
    scratch_dir sd{"dup_channel"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 16);
    const auto cfg = write_mnist_config_for(sd.path, "dupchan", img, lbl);
    auto j = nlohmann::json::parse(read_file(cfg));
    // two different sources under one name: a dedup of identical links would
    // not catch this, only a check on the name does
    auto& inputs = j["sim"]["nodes"][1]["inputs"];
    REQUIRE(inputs == nlohmann::json::parse(R"([[0, "ga"]])"));
    inputs.push_back(nlohmann::json::array({1, "ga"}));
    { std::ofstream out(cfg); out << j.dump(2); }

    auto r = run(COAST_BIN, {"--headless", "-n", "1", cfg.string()});
    CHECK(r.exit_code != 0);
    CHECK(contains(r.stderr_text, "lists input channel \"ga\" more than once"));
}

// ---------------------------------------------------------------------------
// 3a. A malformed numeric option must FAIL CLEANLY, not abort the process.
//
// The parser used std::stoull/std::stoi, whose exceptions are not parse_error,
// so main()'s handler missed them and `-n abc` died on an uncaught exception:
// SIGABRT, exit 134, core dumped. Worse, `-n -5` did not throw at all — it
// wrapped to 18446744073709551611 steps, an effectively unbounded run. Both are
// checked here against the REAL binary, because "exit 1 with a message" versus
// "killed by a signal" is a distinction only a subprocess test can see.

TEST_CASE("b_tier.cli_bad_number: a malformed -n exits 1 with a message, never a signal") {
    for (const char* bad : {"abc", "-5", "1e6", "99999999999999999999999"}) {
        CAPTURE(bad);
        auto r = run(COAST_BIN, {"-n", bad, "configs/mnist.json"});
        // Exactly 1: a shell reports a signal death as 128+signo (134 for
        // SIGABRT), so this pins "clean rejection", not merely "non-zero".
        CHECK(r.exit_code == 1);
        CHECK(contains(r.stderr_text, "Value for '-n'"));
        // And it must not have started running anything.
        CHECK_FALSE(contains(r.stdout_text, "Starting simulation"));
    }
}

// ---------------------------------------------------------------------------
// 3b. Diagnostics flags (diag facility): --log-file, --log-level, -q, --help.

TEST_CASE("b_tier.log_file: --log-file routes diagnostics to a file, not stderr") {
    scratch_dir sd{"logfile"};
    const auto logp = sd.path / "diag.log";
    // Bad config → a fatal COAST_ERROR after the sink is configured.
    auto r = run(COAST_BIN, {"--headless", "-n", "1",
                             "--log-file", logp.string(), "does_not_exist.json"});
    CHECK(r.exit_code != 0);
    REQUIRE(fs::exists(logp));
    CHECK(contains(read_file(logp), "[ERROR]"));          // diagnostic landed in the file
    CHECK_FALSE(contains(r.stderr_text, "[ERROR]"));      // and not on stderr
}

TEST_CASE("b_tier.log_level_off: --log-level off silences the fatal diagnostic") {
    auto r = run(COAST_BIN, {"--headless", "-n", "1",
                             "--log-level", "off", "does_not_exist.json"});
    CHECK(r.exit_code != 0);
    CHECK(r.stderr_text.empty());                         // explicit off = full silence
}

TEST_CASE("b_tier.quiet_keeps_fatal: stacked -q cannot silence a fatal error") {
    auto r = run(COAST_BIN, {"--headless", "-n", "1",
                             "-q", "-q", "-q", "does_not_exist.json"});
    CHECK(r.exit_code != 0);
    CHECK(contains(r.stderr_text, "failed to load configuration"));  // floored at error
}

TEST_CASE("b_tier.help_before_logfile: --help ignores a bad --log-file") {
    auto r = run(COAST_BIN, {"--help", "--log-file", "/nonexistent_dir/x.log"});
    CHECK(r.exit_code == 0);
    CHECK(contains(r.stdout_text, "Usage:"));
    CHECK(r.stderr_text.empty());                         // no spurious sink-open warning
}

TEST_CASE("b_tier.log_file_keyword_case: --log-file STDOUT is the stream, not a file") {
    scratch_dir sd{"logfile_case"};
    // Run in the scratch cwd so a wrong (case-sensitive) impl would create its
    // stray "STDOUT" file there, not in the repo. Bad config → a fatal error.
    auto r = run(COAST_BIN,
                 {"--headless", "-n", "1", "--log-file", "STDOUT", "does_not_exist.json"},
                 {}, sd.path.string());
    CHECK(r.exit_code != 0);
    CHECK(contains(r.stdout_text, "[ERROR]"));            // routed to the stdout STREAM
    CHECK_FALSE(fs::exists(sd.path / "STDOUT"));          // not a file literally named STDOUT
}

// ---------------------------------------------------------------------------
// 4. The shortest real run completes successfully.

TEST_CASE("b_tier.headless_short_run: -n 500 mnist runs and exits cleanly") {
    auto r = run(COAST_BIN, {"--headless", "-n", "500", "configs/mnist.json"});
    CHECK(r.exit_code == 0);
    CHECK(contains(r.stdout_text, "Simulation completed"));
    CHECK(contains(r.stdout_text, "500"));   // total steps reported
}

// ---------------------------------------------------------------------------
// 5. save-state followed by load-state advances the restored step counter.

TEST_CASE("b_tier.save_load_roundtrip: --save-state then --load-state preserves step count") {
    scratch_dir sd{"save_load"};
    const auto snap = sd.path / "snap.h5";

    auto save = run(COAST_BIN, {
        "--headless", "-n", "200",
        "--save-state", snap.string(),
        "configs/mnist.json"
    });
    REQUIRE(save.exit_code == 0);
    REQUIRE(fs::exists(snap));

    auto load = run(COAST_BIN, {
        "--headless",
        "--load-state", snap.string(),
        "-n", "100",
        "configs/mnist.json"
    });
    CHECK(load.exit_code == 0);
    // build_env's state-load notices are diagnostics → diag default sink (stderr),
    // lower-cased. The run narration ("Total steps: …") stays on stdout.
    CHECK(contains(load.stderr_text, "loading state from"));
    CHECK(contains(load.stderr_text, "restored step: 200"));
    // After 100 more steps from step 200, we expect step 300 at completion.
    CHECK(contains(load.stdout_text, "Total steps: 300"));
}

// ---------------------------------------------------------------------------
// 5b. ngm2.5 save-state / load-state round-trip through the real binary's HDF5
//     path. The ngm2.5 neuron_group has its own checkpoint schema (flat arena
//     SoA + directory + hash-key state), distinct from ngm2's; this is the only
//     B-tier coverage that drives it end-to-end through save/load.

TEST_CASE("b_tier.ngm25_save_load_roundtrip: ngm2.5 --save-state then --load-state preserves step count") {
    scratch_dir sd{"ngm25_save_load"};
    const auto snap = sd.path / "snap.h5";

    auto save = run(COAST_BIN, {
        "--headless", "-n", "200",
        "--save-state", snap.string(),
        "configs/mnist_ngm25.json"
    });
    REQUIRE(save.exit_code == 0);
    REQUIRE(fs::exists(snap));

    auto load = run(COAST_BIN, {
        "--headless",
        "--load-state", snap.string(),
        "-n", "100",
        "configs/mnist_ngm25.json"
    });
    CHECK(load.exit_code == 0);
    // build_env's state-load notices are diagnostics → diag default sink (stderr),
    // lower-cased (matching the ngm2 save_load_roundtrip case above). The run
    // narration ("Total steps: …") stays on stdout.
    CHECK(contains(load.stderr_text, "loading state from"));
    CHECK(contains(load.stderr_text, "restored step: 200"));
    CHECK(contains(load.stdout_text, "Total steps: 300"));
}

// ---------------------------------------------------------------------------
// 6. --checkpoint creates snapshot files at the requested interval.
//
// Uses a dynamically-generated config that redirects `snapshot_dir` into a
// per-test scratch dir, so the test does not pollute the repo.

TEST_CASE("b_tier.checkpoint_creates_files: --checkpoint writes snapshots into snapshot_dir") {
    scratch_dir sd{"ckpt"};
    const auto snap_dir = sd.path / "snapshots";
    fs::create_directories(snap_dir);

    // configs/mnist.json with snapshot_dir pointed at the scratch dir.
    auto j = nlohmann::json::parse(read_file(fs::path{COAST_REPO_ROOT} / "configs" / "mnist.json"));
    j["snapshot_dir"] = snap_dir.string();
    const auto cfg = sd.path / "mnist_ckpt.json";
    {
        std::ofstream out(cfg);
        out << j.dump(2);
    }

    auto r = run(COAST_BIN, {
        "--headless", "-n", "300",
        "--checkpoint", "100",
        cfg.string()
    });
    REQUIRE(r.exit_code == 0);
    CHECK(contains(r.stdout_text, "Checkpoint @ step"));

    // The checkpoint cadence is enforced by a per-step sim-thread hook, so it is
    // deterministic: -n 300 --checkpoint 100 writes EXACTLY three snapshots, at
    // steps 100, 200 and 300 — independent of machine speed. (The previous
    // poll-loop trigger could be outrun on a fast machine at low -n, yielding
    // fewer files; this is the regression test for that.)
    std::set<std::uint64_t> steps;
    for (const auto& e : fs::directory_iterator(snap_dir)) {
        if (e.is_regular_file() && e.path().extension() == ".h5") {
            // filename: <stem>_<YYYYMMDD>_<HHMMSS>_<step>.h5 — step is the last
            // underscore-delimited token of the stem.
            const std::string stem = e.path().stem().string();
            const auto pos = stem.rfind('_');
            REQUIRE(pos != std::string::npos);
            steps.insert(std::stoull(stem.substr(pos + 1)));
        }
    }
    CHECK(steps == std::set<std::uint64_t>{100, 200, 300});
}

// ---------------------------------------------------------------------------
// 6a. config_base contract: a per-config snapshot_dir leaf is placed under the
// app config's `config_base` (default `<config_dir>/<config>/`), i.e. next to
// the config file, isolated per config — NOT under the process CWD.
//
// This is the Phase-2 location-policy layer on top of the Phase-1 semantics fix.
// Phase 1 made snapshot_dir CWD-verbatim (no site assumes); Phase 2 restores
// config-relative placement, but now by explicit, visible, overridable policy.
// The config lives in dir A with `"snapshot_dir": "snaps"` (a relative leaf); the
// child runs from an unrelated dir B. The default config_base anchors to the
// config's own directory, so "snaps" resolves under `A/<stem>/snaps`, regardless
// of the CWD — the property that makes write and resume agree from anywhere.
//
// Uses a synthetic dataset with absolute paths so the config is runnable from any
// CWD (real configs/*.json use dataset paths relative to the repo root).

static bool dir_has_h5(const fs::path& d) {
    if (!fs::exists(d)) return false;
    for (const auto& e : fs::directory_iterator(d))
        if (e.is_regular_file() && e.path().extension() == ".h5") return true;
    return false;
}

// Write a minimal app config (coast.json content) that overrides only
// paths.config_base. _WITH_DEFAULT means every other field keeps its built-in
// default, so a partial file is a valid app config.
static fs::path write_app_config(const fs::path& dir, std::string_view name,
                                 const std::string& config_base) {
    nlohmann::json j;
    j["paths"]["config_base"] = config_base;
    const auto p = dir / std::string(name);
    std::ofstream out(p);
    out << j.dump(2);
    return p;
}

TEST_CASE("b_tier.snapshot_dir_under_config_base: a relative snapshot_dir lands under <config_dir>/<config>/, not the CWD") {
    scratch_dir sd{"snap_base"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);

    const auto dir_a = sd.path / "A";   // config lives here
    const auto dir_b = sd.path / "B";   // child runs from here (unrelated CWD)
    fs::create_directories(dir_a);
    fs::create_directories(dir_b);

    // Config file: A/mnist_basetest.json → stem "mnist_basetest".
    const auto cfg =
        write_mnist_config_with_snapshot_dir(dir_a, "basetest", img, lbl, "snaps");

    auto r = run(COAST_BIN,
                 {"--headless", "-n", "100", "--checkpoint", "50", cfg.string()},
                 {}, dir_b.string());
    REQUIRE(r.exit_code == 0);

    // Default config_base "<config_dir>/<config>/" → A/mnist_basetest/snaps.
    CHECK(dir_has_h5(dir_a / "mnist_basetest" / "snaps"));
    // NOT under the CWD, and NOT the bare A/snaps (the <config>/ isolation level).
    CHECK(!fs::exists(dir_b / "snaps"));
    CHECK(!fs::exists(dir_a / "snaps"));
}

// ---------------------------------------------------------------------------
// 6b. config_base default layout: a config that OMITS snapshot_dir gets the
// app default leaf ("snapshots") under `<config_dir>/<config>/`.

TEST_CASE("b_tier.snapshot_default_layout: an omitted snapshot_dir defaults to <config_dir>/<config>/snapshots") {
    scratch_dir sd{"snap_default"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);

    const auto dir_a = sd.path / "A";
    const auto dir_b = sd.path / "B";
    fs::create_directories(dir_a);
    fs::create_directories(dir_b);

    // write_mnist_config_for derives from configs/mnist.json, which has NO
    // snapshot_dir key → the config omits it → the default leaf applies.
    const auto cfg = write_mnist_config_for(dir_a, "deftest", img, lbl);

    auto r = run(COAST_BIN,
                 {"--headless", "-n", "100", "--checkpoint", "50", cfg.string()},
                 {}, dir_b.string());
    REQUIRE(r.exit_code == 0);

    CHECK(dir_has_h5(dir_a / "mnist_deftest" / "snapshots"));
    CHECK(!fs::exists(dir_b / "snapshots"));
}

// ---------------------------------------------------------------------------
// 6c. config_base contract: an ABSOLUTE snapshot_dir leaf still wins verbatim,
// bypassing config_base entirely (an explicit absolute output location overrides
// the base — the property that keeps simc's absolute run dirs working).

TEST_CASE("b_tier.snapshot_dir_absolute_verbatim: an absolute snapshot_dir bypasses config_base") {
    scratch_dir sd{"snap_abs"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);

    const auto dir_a    = sd.path / "A";
    const auto dir_b    = sd.path / "B";
    const auto abs_snap = sd.path / "abs_snaps";
    fs::create_directories(dir_a);
    fs::create_directories(dir_b);

    const auto cfg = write_mnist_config_with_snapshot_dir(
        dir_a, "abstest", img, lbl, abs_snap.string());

    auto r = run(COAST_BIN,
                 {"--headless", "-n", "100", "--checkpoint", "50", cfg.string()},
                 {}, dir_b.string());
    REQUIRE(r.exit_code == 0);

    CHECK(dir_has_h5(abs_snap));                       // used as-is (leaf wins)
    CHECK(!fs::exists(dir_b / "abs_snaps"));           // not re-rooted under the CWD
    CHECK(!fs::exists(dir_a / "mnist_abstest"));       // not placed under config_base
}

// ---------------------------------------------------------------------------
// 6d. config_base contract: write and resume agree, and the agreement is
// CWD-INDEPENDENT. A checkpoint written from one CWD must be found by a later
// --resume run from a DIFFERENT CWD, because config_base anchors the snapshot
// location to the config's own directory, not the process working directory.

TEST_CASE("b_tier.resume_config_anchored: --resume finds a checkpoint written from a different CWD") {
    scratch_dir sd{"resume_base"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);

    const auto dir_a = sd.path / "A";   // config lives here
    const auto dir_b = sd.path / "B";   // write runs from here
    const auto dir_c = sd.path / "C";   // resume runs from here (different CWD)
    fs::create_directories(dir_a);
    fs::create_directories(dir_b);
    fs::create_directories(dir_c);

    const auto cfg = write_mnist_config_with_snapshot_dir(
        dir_a, "resumetest", img, lbl, "snaps");

    // Write a checkpoint from CWD B → lands under A/mnist_resumetest/snaps.
    auto w = run(COAST_BIN,
                 {"--headless", "-n", "100", "--checkpoint", "50", cfg.string()},
                 {}, dir_b.string());
    REQUIRE(w.exit_code == 0);
    REQUIRE(dir_has_h5(dir_a / "mnist_resumetest" / "snaps"));

    // Resume from a DIFFERENT CWD (C): config_base anchors to the config dir, so
    // write and resume resolve to the same place regardless of CWD.
    auto rr = run(COAST_BIN,
                  {"--headless", "-n", "1", "--resume", cfg.string()},
                  {}, dir_c.string());
    REQUIRE(rr.exit_code == 0);
    // The resume notice is a diag message → stderr, lowercase (build_env.cpp).
    CHECK(contains(rr.stderr_text, "found snapshot to resume"));
}

// ---------------------------------------------------------------------------
// 6e. The §3.0 invariant (design): config_base NEVER rebases user-typed CLI
// paths. A relative --save-state / --log-buffers resolves against the process
// CWD verbatim, NOT under <config_dir>/<config>/. This guards against re-opening,
// for a new site, the "a relative value silently means relative-to-config-dir"
// bug class the Phase-1 semantics fix eliminated.

TEST_CASE("b_tier.cli_paths_verbatim_not_rebased: relative --save-state / --log-buffers stay under the CWD") {
    scratch_dir sd{"cli_verbatim"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);

    const auto dir_a = sd.path / "A";   // config lives here
    const auto dir_b = sd.path / "B";   // child runs from here
    fs::create_directories(dir_a);
    fs::create_directories(dir_b);

    const auto cfg = write_mnist_config_for(dir_a, "cliverb", img, lbl);

    // Relative CLI paths — must resolve against CWD (B), never config_base.
    auto r = run(COAST_BIN,
                 {"--headless", "-n", "200",
                  "--save-state", "out.h5",
                  "--log-buffers", "buf.h5",
                  cfg.string()},
                 {}, dir_b.string());
    REQUIRE(r.exit_code == 0);

    CHECK(fs::exists(dir_b / "out.h5"));   // CLI path verbatim under CWD
    CHECK(fs::exists(dir_b / "buf.h5"));
    // NOT rebased under config_base.
    CHECK(!fs::exists(dir_a / "mnist_cliverb" / "out.h5"));
    CHECK(!fs::exists(dir_a / "mnist_cliverb" / "buf.h5"));
}

// ---------------------------------------------------------------------------
// 6f. App-config override: --app-config with a custom `config_base` relocates
// where a config's outputs land. The base can point anywhere (here an absolute
// dir), and the per-config `<config>/` isolation is preserved.

TEST_CASE("b_tier.app_config_flag_relocates_base: --app-config config_base relocates snapshots") {
    scratch_dir sd{"app_relocate"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);

    const auto dir_a = sd.path / "A";
    const auto dir_b = sd.path / "B";
    const auto runs  = sd.path / "runs";   // relocation target (absolute base)
    fs::create_directories(dir_a);
    fs::create_directories(dir_b);

    const auto cfg = write_mnist_config_for(dir_a, "apptest", img, lbl);
    // config_base points at an absolute dir with the <config> token → per-config
    // isolation under runs/.
    const auto app = write_app_config(sd.path, "app.json",
                                      (runs / "<config>").string() + "/");

    auto r = run(COAST_BIN,
                 {"--headless", "-n", "100", "--checkpoint", "50",
                  "--app-config", app.string(), cfg.string()},
                 {}, dir_b.string());
    REQUIRE(r.exit_code == 0);

    // Default leaf "snapshots" under the overridden base runs/mnist_apptest/.
    CHECK(dir_has_h5(runs / "mnist_apptest" / "snapshots"));
    // NOT at the built-in default location next to the config.
    CHECK(!fs::exists(dir_a / "mnist_apptest"));
}

// ---------------------------------------------------------------------------
// 6g. App-config discovery precedence: --app-config > $COAST_APP_CONFIG > ./coast.json.
// Each source sets a DISTINCT config_base so the snapshot location reveals which
// one the binary honored.

TEST_CASE("b_tier.app_config_discovery_precedence: flag beats env beats ./coast.json") {
    scratch_dir sd{"app_precedence"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);

    const auto dir_a = sd.path / "A";   // config lives here
    const auto cwd   = sd.path / "cwd"; // child runs here; holds ./coast.json
    const auto base_flag = sd.path / "from_flag";
    const auto base_env  = sd.path / "from_env";
    const auto base_file = sd.path / "from_file";
    fs::create_directories(dir_a);
    fs::create_directories(cwd);

    const auto cfg = write_mnist_config_for(dir_a, "prectest", img, lbl);

    const auto app_flag = write_app_config(sd.path, "flag.json",
                                           (base_flag / "<config>").string() + "/");
    const auto app_env  = write_app_config(sd.path, "env.json",
                                           (base_env / "<config>").string() + "/");
    // ./coast.json lives in the child's CWD (the default discovery path).
    write_app_config(cwd, "coast.json", (base_file / "<config>").string() + "/");

    // (i) env set, no flag, ./coast.json present → env wins over the file.
    {
        auto r = run(COAST_BIN,
                     {"--headless", "-n", "50", "--checkpoint", "50", cfg.string()},
                     {{"COAST_APP_CONFIG", app_env.string()}}, cwd.string());
        REQUIRE(r.exit_code == 0);
        CHECK(dir_has_h5(base_env / "mnist_prectest" / "snapshots"));
        CHECK(!fs::exists(base_file / "mnist_prectest"));
    }

    // (ii) flag set AND env set AND ./coast.json present → flag wins over both.
    {
        auto r = run(COAST_BIN,
                     {"--headless", "-n", "50", "--checkpoint", "50",
                      "--app-config", app_flag.string(), cfg.string()},
                     {{"COAST_APP_CONFIG", app_env.string()}}, cwd.string());
        REQUIRE(r.exit_code == 0);
        CHECK(dir_has_h5(base_flag / "mnist_prectest" / "snapshots"));
    }

    // (iii) no flag, no env → the ./coast.json in the CWD is discovered and used.
    // ($COAST_APP_CONFIG is cleared for the whole suite in main(), so this is
    // hermetic against a developer's exported value.)
    {
        auto r = run(COAST_BIN,
                     {"--headless", "-n", "50", "--checkpoint", "50", cfg.string()},
                     {}, cwd.string());
        REQUIRE(r.exit_code == 0);
        CHECK(dir_has_h5(base_file / "mnist_prectest" / "snapshots"));
    }
}

// ---------------------------------------------------------------------------
// 6h. A missing app config named EXPLICITLY (--app-config) is a hard error;
// an absent auto-discovered ./coast.json is not (the file is optional).

TEST_CASE("b_tier.app_config_missing_explicit_fails: --app-config to a missing file exits non-zero") {
    auto r = run(COAST_BIN, {"--headless", "-n", "1",
                             "--app-config", "does_not_exist_app.json",
                             "configs/mnist.json"});
    CHECK(r.exit_code != 0);
    CHECK(contains(r.stderr_text, "app config"));
}

// 6i. A stale $COAST_APP_CONFIG (env var pointing at a missing file — e.g. a file
// exported once and later moved) is a SOFT failure: the binary warns and falls
// through to ./coast.json / defaults rather than refusing to start. Without the
// soft-env handling this run exits non-zero (env used to be strict), so this is a
// real regression guard. It also proves fall-through: the discovered ./coast.json
// still wins, i.e. the broken env var doesn't shadow the next source.

TEST_CASE("b_tier.app_config_stale_env_falls_through: missing $COAST_APP_CONFIG warns, uses ./coast.json") {
    scratch_dir sd{"app_stale_env"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);

    const auto dir_a    = sd.path / "A";     // config lives here
    const auto cwd      = sd.path / "cwd";   // child runs here; holds ./coast.json
    const auto base_file = sd.path / "from_file";
    fs::create_directories(dir_a);
    fs::create_directories(cwd);

    const auto cfg = write_mnist_config_for(dir_a, "staletest", img, lbl);
    write_app_config(cwd, "coast.json", (base_file / "<config>").string() + "/");

    const auto missing = (sd.path / "stale_moved_away.json").string();
    REQUIRE(!fs::exists(missing));

    auto r = run(COAST_BIN,
                 {"--headless", "-n", "50", "--checkpoint", "50", cfg.string()},
                 {{"COAST_APP_CONFIG", missing}}, cwd.string());

    REQUIRE(r.exit_code == 0);                                   // teeth: was non-zero
    CHECK(contains(r.stderr_text, "COAST_APP_CONFIG"));              // warned about the stale var
    CHECK(dir_has_h5(base_file / "mnist_staletest" / "snapshots")); // fell through to ./coast.json
}

// 6j. `coast --init` writes a default ./coast.json template and exits, and never
// clobbers an existing one. The written file must be a valid app config that the
// binary itself can load back (round-trip through discovery).

TEST_CASE("b_tier.init_writes_default_coast_json: --init bootstraps a valid template, no clobber") {
    scratch_dir sd{"app_init"};
    fs::create_directories(sd.path);
    const auto coast_json = sd.path / "coast.json";
    REQUIRE(!fs::exists(coast_json));

    // First --init: writes the file and exits 0.
    auto r1 = run(COAST_BIN, {"--init"}, {}, sd.path.string());
    REQUIRE(r1.exit_code == 0);
    REQUIRE(fs::exists(coast_json));
    {
        const auto j = nlohmann::json::parse(read_file(coast_json));
        REQUIRE(j.contains("paths"));
        CHECK(j["paths"].contains("config_base"));               // full template, not sparse
    }

    // The written file is a usable app config that discovery honors — prove it by
    // editing config_base to a DISTINCT location (the default would be
    // indistinguishable from no file) and checking the output lands there.
    {
        auto j = nlohmann::json::parse(read_file(coast_json));
        // A distinct base so this run's output is unmistakably driven by the
        // discovered ./coast.json (the default base would be indistinguishable).
        j["paths"]["config_base"] = "from_init/<config>/";
        std::ofstream(coast_json) << j.dump(2);
    }
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);
    const auto cfg = write_mnist_config_for(sd.path, "inittest", img, lbl);
    auto r2 = run(COAST_BIN, {"--headless", "-n", "50", "--checkpoint", "50", cfg.string()},
                  {}, sd.path.string());  // $COAST_APP_CONFIG cleared suite-wide in main()
    REQUIRE(r2.exit_code == 0);
    CHECK(dir_has_h5(sd.path / "from_init" / "mnist_inittest" / "snapshots")); // edited coast.json honored

    // Second --init must NOT clobber: mutate the file, re-run, confirm untouched.
    { std::ofstream(coast_json, std::ios::app) << "\n// sentinel\n"; }
    const auto before = fs::last_write_time(coast_json);
    auto r3 = run(COAST_BIN, {"--init"}, {}, sd.path.string());
    CHECK(r3.exit_code == 0);
    CHECK(fs::last_write_time(coast_json) == before);            // left untouched
}

// ---------------------------------------------------------------------------
// 7. --log-buffers produces a non-empty HDF5 file.

TEST_CASE("b_tier.log_buffers_creates_file: --log-buffers writes a populated HDF5 file") {
    scratch_dir sd{"logbuf"};
    const auto log_path = sd.path / "buffers.h5";

    auto r = run(COAST_BIN, {
        "--headless", "-n", "500",
        "--log-buffers", log_path.string(),
        "configs/mnist.json"
    });
    REQUIRE(r.exit_code == 0);
    CHECK(contains(r.stdout_text, "Starting buffer logging"));
    CHECK(contains(r.stdout_text, "Buffer logging stopped"));
    // The terminal event comes after the stop and counts every frame, including
    // the tail the stop flushed (PR #228).
    const auto last = last_progress_line(r.stdout_text);
    CHECK(contains(last, "phase=done "));
    CHECK(contains(last, " logged=500"));

    REQUIRE(fs::exists(log_path));
    // Empty HDF5 file ≠ no logging — file should have some real content.
    // 1 KiB is a safe lower bound for "the writer actually did something".
    CHECK(fs::file_size(log_path) > 1024u);
}

// 7b. A run that exits 1 must not end its progress feed with phase=done.
//
// PR #228 review: phase=done was emitted before buffer logging was stopped and
// --save-state written, so a failure in either exited 1 after a "done" line.
// The buffer-log failure is injected with a file-size limit (SIGXFSZ ignored, so
// the write fails with EFBIG instead of killing the process), set in a /bin/sh
// wrapper so only the child is limited. ~9.5 MB of log against a limit of
// 512 blocks (256 or 512 KiB, depending on the shell's block unit).

TEST_CASE("b_tier.failed_run_terminal_progress: failed log / save-state end on phase=failed") {
    scratch_dir sd{"term_progress"};

    SUBCASE("buffer log write fails") {
        const auto log_path = sd.path / "buffers.h5";
        auto r = run("/bin/sh", {
            "-c", "ulimit -f 512 && trap '' XFSZ && exec \"$0\" \"$@\"",
            COAST_BIN, "--headless", "-n", "3000",
            "--log-buffers", log_path.string(), "configs/mnist.json"
        });
        CHECK(r.exit_code == 1);
        CHECK(contains(r.stderr_text, "buffer logging failed"));
        const auto last = last_progress_line(r.stdout_text);
        CAPTURE(last);
        CHECK(contains(last, "phase=failed "));
        CHECK_FALSE(contains(r.stdout_text, "phase=done "));
    }
    SUBCASE("--save-state write fails") {
        // The state writer creates missing parent directories, so make the
        // parent a regular file: that cannot be created over.
        const auto not_a_dir = sd.path / "not_a_dir";
        { std::ofstream(not_a_dir) << "x"; }
        auto r = run(COAST_BIN, {
            "--headless", "-n", "50",
            "--save-state", (not_a_dir / "s.h5").string(),
            "configs/mnist.json"
        });
        CHECK(r.exit_code == 1);
        CHECK(contains(r.stderr_text, "failed to save state"));
        const auto last = last_progress_line(r.stdout_text);
        CAPTURE(last);
        CHECK(contains(last, "phase=failed "));
        CHECK_FALSE(contains(r.stdout_text, "phase=done "));
    }
}

// ---------------------------------------------------------------------------
// 8. Loading a trained model onto a SMALLER dataset must not read past the
//    image buffer.
//
// Regression for the loader out-of-range fix: the saved cur_idx reflects the
// (larger) training dataset's length. Restored onto a smaller eval dataset it
// was left out of range, and the next get_norm_image() read past
// norm_img_data — under Release the only guard was an assert (compiled out),
// so this was a segfault / silent out-of-bounds read. load_state now wraps
// cur_idx into the loaded dataset.
//
// Mechanism this test relies on: with the wrap in place the run completes
// cleanly (exit 0). If the wrap regresses, get_norm_image() now throws
// std::out_of_range, which run_headless turns into a non-zero exit — so this
// CHECK fails deterministically rather than depending on UB happening to
// crash. MNIST is the representative loader (emnist/kmnist/color_mnist share
// mnist_db::get_norm_image).
//
// The cursor is restored only by a load that continues the run that saved it
// (SPEC.md S.5.14), so the eval config names the training run's run_id. Shuffle is
// off: with it, an unwrapped position indexes the 16-entry permutation out of range,
// which is UB that may happen to yield a valid index and pass.

namespace {   // defined with section 8b
logging::mem_group state_tree(const fs::path& h5);
std::string run_id_of(const logging::mem_group& tree);
void patch(const fs::path& cfg, const std::function<void(nlohmann::json&)>& f);
}

TEST_CASE("b_tier.load_smaller_dataset: --load-state onto a smaller dataset exits cleanly") {
    scratch_dir sd{"load_smaller"};

    // A 'train' dataset large enough that 200 steps never wrap (cur_idx == 200),
    // and an 'eval' dataset far smaller than that index.
    const auto train_img = sd.path / "train-images";
    const auto train_lbl = sd.path / "train-labels";
    const auto eval_img  = sd.path / "eval-images";
    const auto eval_lbl  = sd.path / "eval-labels";
    write_synthetic_mnist(train_img, train_lbl, 300);
    write_synthetic_mnist(eval_img,  eval_lbl,  16);

    const auto train_cfg = write_mnist_config_for(sd.path, "train", train_img, train_lbl);
    const auto eval_cfg  = write_mnist_config_for(sd.path, "eval",  eval_img,  eval_lbl);
    for (const auto& cfg : { train_cfg, eval_cfg })
        patch(cfg, [](nlohmann::json& j) {
            for (auto& [name, d] : j["datasets"]["mnist"].items()) d["shuffle"] = false;
        });
    const auto snap = sd.path / "snap.h5";

    // Train 200 steps on the 300-image dataset and snapshot. cur_idx == 200.
    auto save = run(COAST_BIN, {
        "--headless", "-n", "200",
        "--save-state", snap.string(),
        train_cfg.string()
    });
    REQUIRE(save.exit_code == 0);
    REQUIRE(fs::exists(snap));
    patch(eval_cfg, [&](nlohmann::json& j) { j["run_id"] = run_id_of(state_tree(snap)); });

    // Restore onto the 16-image dataset and keep running. cur_idx 200 is out of
    // range for 16 images and must be wrapped; pre-fix this read ~0.6 MB past a
    // ~50 KB buffer (segfault / UB).
    auto load = run(COAST_BIN, {
        "--headless",
        "--load-state", snap.string(),
        "-n", "100",
        eval_cfg.string()
    });
    CHECK(load.exit_code == 0);
    CHECK(contains(load.stderr_text, "continues: datasets continue their sequences"));
    CHECK(contains(load.stderr_text, "restored step: 200"));  // diag notice → stderr
    CHECK(contains(load.stdout_text, "Total steps: 300"));
    CHECK(contains(load.stdout_text, "Simulation completed"));
}

// ---------------------------------------------------------------------------
// 8b. Continuation is the run's (SPEC.md S.5.14): a load continues the datasets'
//     sequences iff the checkpoint was written by the same run_id. A frozen trial
//     named by its own id, loaded from a trained state ("auto"), starts fresh; its
//     own checkpoint continues it — so a trial interrupted and resumed ends where one
//     run straight through does. Before, a frozen trial's resume restarted its
//     sequence (restore_sequence_from_checkpoint=false). Training on another subset
//     from a trained checkpoint starts fresh too, and --resume of an "auto" config
//     takes the checkpoint's id.

namespace {

logging::mem_group state_tree(const fs::path& h5)
{
    hdf5::executor ex;
    auto pr = logging::state_logger::parse_state(h5, ex);
    REQUIRE_MESSAGE(pr.result.success, pr.result.error_message);
    return pr.tree;
}

// The dataset entities of a checkpoint, by output id: those with a sequencer cursor.
std::map<std::string, logging::mem_group> dataset_states(const logging::mem_group& tree)
{
    std::map<std::string, logging::mem_group> r;
    for (const auto& [id, g] : tree.groups.at("entities").groups)
        if (g.fields.contains("seq_pos"))
            r.emplace(id, g);
    return r;
}

std::string run_id_of(const logging::mem_group& tree)
{
    return tree.groups.at("metadata").attrs.at("run_id");
}

void patch(const fs::path& cfg, const std::function<void(nlohmann::json&)>& f)
{
    auto j = nlohmann::json::parse(read_file(cfg));
    f(j);
    std::ofstream out(cfg);
    out << j.dump(2);
}

void ok(const subprocess_result& r)
{
    INFO(r.stdout_text << r.stderr_text);
    REQUIRE(r.exit_code == 0);
}

} // namespace

TEST_CASE("b_tier.frozen_resume_continues: a frozen trial resumed from its own checkpoint "
          "equals one run straight through") {
    scratch_dir sd{"frozen_resume"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 64);
    const auto train_cfg  = write_mnist_config_for(sd.path, "train", img, lbl);
    const auto frozen_cfg = write_mnist_config_for(sd.path, "frozen", img, lbl);
    patch(frozen_cfg, [](nlohmann::json& j) { j["frozen"] = true; j["run_id"] = "trial-1"; });
    const auto trained = sd.path / "trained.h5", full = sd.path / "full.h5",
               half = sd.path / "half.h5", resumed = sd.path / "resumed.h5",
               fresh = sd.path / "fresh.h5";
    ok(run(COAST_BIN, {"-n", "50", "--save-state", trained.string(), train_cfg.string()}));
    ok(run(COAST_BIN, {"--load-state", trained.string(), "-n", "60",
                       "--save-state", full.string(), frozen_cfg.string()}));
    ok(run(COAST_BIN, {"--load-state", trained.string(), "-n", "30",
                       "--save-state", half.string(), frozen_cfg.string()}));
    const auto resume = run(COAST_BIN, {"--load-state", half.string(), "-n", "30",
                                        "--save-state", resumed.string(), frozen_cfg.string()});
    ok(resume);
    CHECK(contains(resume.stderr_text, "run trial-1 continues"));
    ok(run(COAST_BIN, {"-n", "60", "--save-state", fresh.string(), frozen_cfg.string()}));

    const auto full_tree = state_tree(full);
    CHECK(run_id_of(full_tree) == "trial-1");
    const auto trained_id = run_id_of(state_tree(trained));
    CHECK(trained_id.size() == 32);   // "auto": a fresh id
    CHECK(trained_id != "trial-1");

    // Interrupted and resumed == straight through: every entity, every buffer.
    const auto resumed_tree = state_tree(resumed);
    CHECK(resumed_tree.groups.at("entities") == full_tree.groups.at("entities"));
    CHECK(resumed_tree.groups.at("io_buffers") == full_tree.groups.at("io_buffers"));

    // From the trained state (another run), the datasets started fresh: as from no state.
    const auto ds = dataset_states(full_tree);
    REQUIRE_FALSE(ds.empty());
    CHECK(ds == dataset_states(state_tree(fresh)));
}

// The observe files of a frozen trial that a crash interrupted after a checkpoint
// and that was resumed from it hold exactly what one run straight through writes:
// the rows before the checkpoint kept, the steps after it dropped and run again.
TEST_CASE("b_tier.frozen_resume_continues_observe: a resumed run's observe files equal "
          "those of one run straight through") {
    scratch_dir sd{"frozen_resume_observe"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 64);
    const auto train_cfg = write_mnist_config_for(sd.path, "train", img, lbl);
    const auto observed = [&](std::string_view tag) {
        const auto cfg = write_mnist_config_for(sd.path, tag, img, lbl);
        patch(cfg, [&](nlohmann::json& j) {
            j["frozen"] = true;
            j["run_id"] = "trial-obs";
            j["snapshot_dir"] = (sd.path / tag / "snapshots").string();
            j["observe"] = { { "output_dir", (sd.path / tag / "observe").string() },
                             { "groups", { { "input", { { "ranges", { { 0, 1000000 } } },
                                                       { "combine", "or" } } } } } };
        });
        return cfg;
    };
    const auto crashed_cfg = observed("crashed"), straight_cfg = observed("straight");
    const auto trained = sd.path / "trained.h5";
    ok(run(COAST_BIN, {"-n", "50", "--save-state", trained.string(), train_cfg.string()}));
    // Straight through: steps 50..79.
    ok(run(COAST_BIN, {"--load-state", trained.string(), "-n", "30", straight_cfg.string()}));
    // Interrupted: a checkpoint at step 70, the run "dies" at 80 (its rows 70..79
    // were written but are past the checkpoint); resumed from step 70 to 80.
    ok(run(COAST_BIN, {"--load-state", trained.string(), "-n", "30", "--checkpoint", "20",
                       crashed_cfg.string()}));
    fs::path ckpt;
    for (const auto& e : fs::directory_iterator(sd.path / "crashed" / "snapshots"))
        if (e.path().filename().string().ends_with("_70.h5")) ckpt = e.path();
    REQUIRE_MESSAGE(!ckpt.empty(), "no checkpoint at step 70");
    const auto resume = run(COAST_BIN, {"--load-state", ckpt.string(), "-n", "10",
                                        crashed_cfg.string()});
    ok(resume);
    CHECK(contains(resume.stderr_text, "run trial-obs continues from step 70"));

    const auto datasets = [](const fs::path& h5) {
        hdf5::executor ex;
        return ex.submit_sync([&] {
            std::map<std::string, std::vector<std::uint64_t>> r;
            const hid_t f = H5Fopen(h5.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            REQUIRE(f >= 0);
            for (const char* name : { "/mnist_input[0]__index", "/mnist_input[0]__index__step_index" }) {
                const hid_t d = H5Dopen2(f, name, H5P_DEFAULT);
                REQUIRE(d >= 0);
                const hid_t sp = H5Dget_space(d);
                hsize_t n = 0;
                H5Sget_simple_extent_dims(sp, &n, nullptr);
                std::vector<std::uint64_t> v(static_cast<std::size_t>(n));
                if (n > 0) REQUIRE(H5Dread(d, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()) >= 0);
                r.emplace(name, std::move(v));
                H5Sclose(sp);
                H5Dclose(d);
            }
            H5Fclose(f);
            return r;
        });
    };
    const auto got  = datasets(sd.path / "crashed" / "observe" / "input.observe.h5");
    const auto want = datasets(sd.path / "straight" / "observe" / "input.observe.h5");
    REQUIRE(want.at("/mnist_input[0]__index__step_index").size() == 30);
    CHECK(want.at("/mnist_input[0]__index__step_index").front() == 50);
    CHECK(got == want);
}

TEST_CASE("b_tier.new_subset_starts_fresh: training on another index_manifest from a "
          "trained checkpoint starts its sequence fresh, unless it names the run_id") {
    scratch_dir sd{"new_subset"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 64);
    const auto subset = sd.path / "subset.txt";
    { std::ofstream(subset) << "3 9 12 40 41 7\n"; }
    const auto train_cfg  = write_mnist_config_for(sd.path, "train", img, lbl);
    const auto subset_cfg = write_mnist_config_for(sd.path, "subset", img, lbl);
    patch(subset_cfg, [&](nlohmann::json& j) {
        for (auto& [name, d] : j["datasets"]["mnist"].items()) d["index_manifest"] = subset.string();
    });
    const auto trained = sd.path / "trained.h5", tuned = sd.path / "tuned.h5";
    ok(run(COAST_BIN, {"-n", "50", "--save-state", trained.string(), train_cfg.string()}));
    // "auto": another run, so the subset's sequence starts at its beginning instead of
    // refusing the training cursor, which has no position on the subset.
    const auto r = run(COAST_BIN, {"--load-state", trained.string(), "-n", "20",
                                   "--save-state", tuned.string(), subset_cfg.string()});
    ok(r);
    CHECK(contains(r.stderr_text, "datasets keep their own sequences"));
    // Naming the trained run's id asks to continue: the cursor has no position on the
    // subset, so the load refuses, naming why.
    const auto id = run_id_of(state_tree(trained));
    patch(subset_cfg, [&](nlohmann::json& j) { j["run_id"] = id; });
    const auto cont = run(COAST_BIN, {"--load-state", trained.string(), "-n", "20", subset_cfg.string()});
    CHECK(cont.exit_code != 0);
    CHECK(contains(cont.stdout_text + cont.stderr_text, "continues its sequence only on the manifest"));
}

TEST_CASE("b_tier.resume_adopts_run_id: --resume of an \"auto\" config continues its "
          "checkpoint's run") {
    scratch_dir sd{"resume_adopt"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 64);
    const auto cfg = write_mnist_config_with_snapshot_dir(sd.path, "resume", img, lbl,
                                                          (sd.path / "snaps").string());
    const auto straight = sd.path / "straight.h5", resumed = sd.path / "resumed.h5";
    ok(run(COAST_BIN, {"-n", "60", "--save-state", straight.string(), cfg.string()}));
    ok(run(COAST_BIN, {"-n", "30", "--checkpoint", "30", cfg.string()}));
    const auto r = run(COAST_BIN, {"--resume", "-n", "30", "--save-state", resumed.string(),
                                   cfg.string()});
    ok(r);
    CHECK(contains(r.stderr_text, "continues: datasets continue their sequences"));
    const auto rt = state_tree(resumed);
    CHECK(dataset_states(rt) == dataset_states(state_tree(straight)));
    CHECK(rt.groups.at("entities") == state_tree(straight).groups.at("entities"));
}

// ---------------------------------------------------------------------------
// 9. --emit-run-manifest writes a schema-"1" coast_runtime.json with the
//    expected build/run provenance, and is opt-in (no flag ⇒ no file).

TEST_CASE("b_tier.run_manifest_emits: --emit-run-manifest writes a valid coast_runtime.json") {
    scratch_dir sd{"manifest"};
    const auto manifest = sd.path / "coast_runtime.json";

    auto r = run(COAST_BIN, {
        "--headless", "-n", "200",
        "--seed-offset", "3",
        "--emit-run-manifest", manifest.string(),
        "configs/mnist.json"
    });
    REQUIRE(r.exit_code == 0);
    REQUIRE(fs::exists(manifest));

    auto j = nlohmann::json::parse(read_file(manifest));

    // Build provenance.
    CHECK(j.at("schema_version").get<std::string>() == "1");
    const auto sha = j.at("coast_git_sha").get<std::string>();
    CHECK((sha.size() == 40 || sha == "unknown"));  // full hex, or non-git build
    CHECK(j.contains("coast_git_dirty"));
    CHECK(j.contains("coast_build_type"));
    CHECK_FALSE(j.at("coast_compiler").get<std::string>().empty());
    CHECK_FALSE(j.at("coast_target_cpu").get<std::string>().empty());   // "unknown" at worst
#if defined(__GNUC__)   // GCC and (Apple)Clang: cmake/coast_build_info.cmake asks both
    CHECK(j.at("coast_target_cpu").get<std::string>() != "unknown");
#endif

    // Run facts.
    CHECK(j.at("seed_offset").get<int>() == 3);
    CHECK(j.at("n_steps_requested").get<std::uint64_t>() == 200u);
    CHECK(j.at("n_steps_completed").get<std::uint64_t>() >= 200u);  // may overshoot a few
    CHECK(j.at("exit_reason").get<std::string>() == "completed_n_steps");
    CHECK(j.at("wall_clock_seconds").get<double>() >= 0.0);
    CHECK(j.contains("started_at_utc"));
    CHECK(j.contains("completed_at_utc"));
    CHECK(j.contains("hostname"));

    // Resolved observe config block is mirrored; savestate_outputs is always an
    // array. observe_outputs is intentionally absent — the async observe writer
    // is unflushed at emit time, so a downstream harness enumerates those files
    // after the process exits.
    CHECK(j.contains("observe"));
    // output_dir is the RESOLVED observe directory, not the raw config leaf.
    // configs/mnist.json omits observe.output_dir, so the app default leaf
    // ("observe") lands under the default config_base "<config_dir>/<config>/",
    // i.e. <repo>/configs/mnist/observe (run from the repo root). The manifest is
    // written into a scratch dir OUTSIDE that tree, so under the manifest path-frame
    // rule the observe dir is recorded ABSOLUTE (not climbed to with "../..").
    {
        const auto obs_dir = j.at("observe").at("output_dir").get<std::string>();
        const auto expect  = (fs::path{COAST_REPO_ROOT} / "configs" / "mnist" / "observe")
                                 .lexically_normal().generic_string();
        CHECK(fs::path{obs_dir}.is_absolute());
        CHECK(obs_dir == expect);
    }
    CHECK(j.contains("datasets"));
    CHECK(j.at("datasets").contains("mnist"));
    REQUIRE(j.contains("savestate_outputs"));
    CHECK(j.at("savestate_outputs").is_array());
    CHECK(j.at("savestate_outputs").empty());       // no --save-state in this run
    CHECK_FALSE(j.contains("observe_outputs"));

    // Effective RNG seeds — at least one stochastic entity (the mnist dataset).
    CHECK(j.contains("node_seeds"));
    REQUIRE(j.at("node_seeds").is_array());
    CHECK_FALSE(j.at("node_seeds").empty());
    CHECK(j.at("node_seeds")[0].contains("effective_seed"));
}

TEST_CASE("b_tier.run_manifest_savestate: --save-state records a savestate_outputs entry") {
    scratch_dir sd{"manifest_ss"};
    const auto manifest = sd.path / "coast_runtime.json";
    const auto state    = sd.path / "final.h5";

    auto r = run(COAST_BIN, {
        "--headless", "-n", "150",
        "--save-state", state.string(),
        "--emit-run-manifest", manifest.string(),
        "configs/mnist.json"
    });
    REQUIRE(r.exit_code == 0);
    REQUIRE(fs::exists(manifest));
    REQUIRE(fs::exists(state));

    auto j = nlohmann::json::parse(read_file(manifest));
    REQUIRE(j.at("savestate_outputs").is_array());
    REQUIRE(j.at("savestate_outputs").size() == 1);
    const auto& entry = j.at("savestate_outputs").at(0);
    CHECK(entry.at("step").get<std::uint64_t>() >= 150u);
    // State saves are synchronous, so the recorded byte size is authoritative.
    CHECK(entry.at("bytes").get<std::uint64_t>() == fs::file_size(state));
}

// A resume restores each group's random state from the checkpoint (SPEC.md
// M.7.8), so the manifest of the resumed run must name the seed that state came
// from, not the one --seed-offset would compose (#239, M.2.4). Checked for the
// ngm2 group (node 1 of configs/mnist.json).
TEST_CASE("b_tier.run_manifest_resumed_seed: a resume under another --seed-offset reports the checkpoint's group seed") {
    scratch_dir sd{"manifest_resume_seed"};
    const auto state = sd.path / "snap.h5";
    const auto m0    = sd.path / "save.json";
    const auto m1    = sd.path / "resume.json";

    auto read_json = [](const fs::path& p) {
        nlohmann::json j;
        std::ifstream in(p);
        REQUIRE(in.good());
        in >> j;
        return j;
    };
    auto group_seed = [](const nlohmann::json& j) -> std::uint64_t {
        for (const auto& e : j.at("node_seeds"))
            if (e.at("id").get<std::uint64_t>() == 1) return e.at("effective_seed").get<std::uint64_t>();
        FAIL("no node_seeds entry for node 1");
        return 0;
    };

    auto save = run(COAST_BIN, {
        "--headless", "-n", "100",
        "--save-state", state.string(),
        "--emit-run-manifest", m0.string(),
        "configs/mnist.json"
    });
    REQUIRE(save.exit_code == 0);
    const auto saved_seed = group_seed(read_json(m0));

    auto resume = run(COAST_BIN, {
        "--headless", "-n", "50",
        "--seed-offset", "5",
        "--load-state", state.string(),
        "--emit-run-manifest", m1.string(),
        "configs/mnist.json"
    });
    REQUIRE(resume.exit_code == 0);
    const auto j1 = read_json(m1);
    CHECK(j1.at("seed_offset").get<int>() == 5);
    CHECK(group_seed(j1) == saved_seed);
}

TEST_CASE("b_tier.run_manifest_optin: absent --emit-run-manifest writes no manifest") {
    scratch_dir sd{"manifest_optin"};
    const auto manifest = sd.path / "coast_runtime.json";

    auto r = run(COAST_BIN, {"--headless", "-n", "100", "configs/mnist.json"});
    REQUIRE(r.exit_code == 0);
    CHECK_FALSE(fs::exists(manifest));
}

// A failed periodic checkpoint write is reported while the run is going, not
// only at exit (an --unlimited run may last days). snapshot_dir is read-only, so
// every write fails; the test waits for the warning, then stops the run with
// SIGTERM (--save-on-term).
// The highest step=N an "@coast/progress" line in `out` reports (0 if none).
static std::uint64_t last_reported_step(const std::string& out) {
    std::uint64_t best = 0;
    for (std::size_t p = out.find(" step="); p != std::string::npos; p = out.find(" step=", p + 1))
        best = std::max<std::uint64_t>(best, std::strtoull(out.c_str() + p + 6, nullptr, 10));
    return best;
}

TEST_CASE("b_tier.checkpoint_write_failure_reported_during_run") {
    scratch_dir sd{"ckpt_fail_live"};
    const auto ro = sd.path / "ro";
    fs::create_directories(ro);
    fs::permissions(ro, fs::perms::owner_read | fs::perms::owner_exec, fs::perm_options::replace);
    struct restore_perms {   // so scratch_dir can remove it
        fs::path p;
        ~restore_perms() { std::error_code ec; fs::permissions(p, fs::perms::owner_all, fs::perm_options::replace, ec); }
    } restore{ ro };
    if (::access(ro.c_str(), W_OK) == 0) {   // e.g. running as root
        MESSAGE("skipped: cannot make a directory unwritable here");
        return;
    }
    const auto cfg = sd.path / "mnist_ro.json";
    {
        auto j = nlohmann::json::parse(read_file(fs::path{COAST_REPO_ROOT} / "configs" / "mnist.json"));
        j["snapshot_dir"] = ro.string();
        std::ofstream out(cfg);
        out << j.dump(2);
    }

    int pfd[2];
    REQUIRE(::pipe(pfd) == 0);
    const pid_t pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        ::dup2(pfd[1], STDOUT_FILENO);
        ::dup2(pfd[1], STDERR_FILENO);
        ::close(pfd[0]);
        ::close(pfd[1]);
        const std::string bin = COAST_BIN, c = cfg.string();
        ::execl(bin.c_str(), bin.c_str(), "--headless", "--unlimited", "--save-on-term",
                "--checkpoint", "50", c.c_str(), static_cast<char*>(nullptr));
        ::_exit(127);
    }
    ::close(pfd[1]);

    // Read until the warning shows up while the child is alive. The child's own
    // output is the heartbeat: the wait gives up only if it writes nothing for
    // the stall limit (frozen), never because the machine is slow.
    std::string out;
    bool seen_while_running = false;
    bool child_gone         = false;
    (void)coast_test::await_progress(
        [&] {
            pollfd pf{ pfd[0], POLLIN, 0 };
            if (::poll(&pf, 1, 200) > 0) {
                char buf[4096];
                const ssize_t n = ::read(pfd[0], buf, sizeof buf);
                if (n <= 0) { child_gone = true; return true; }   // it must not exit before SIGTERM
                out.append(buf, static_cast<std::size_t>(n));
            }
            seen_while_running = out.find("checkpoint write failed") != std::string::npos
                              && ::waitpid(pid, nullptr, WNOHANG) == 0;
            // An unlimited run keeps writing progress whether or not the warning
            // comes, so its output alone would never end the wait: past 20
            // checkpoint intervals (--checkpoint 50) without it, it never will.
            return seen_while_running || last_reported_step(out) > 20 * 50;
        },
        [&] { return out.size(); });

    ::kill(pid, SIGTERM);
    for (;;) {   // drain so the child never blocks on a full pipe
        char buf[4096];
        const ssize_t n = ::read(pfd[0], buf, sizeof buf);
        if (n <= 0) break;
        out.append(buf, static_cast<std::size_t>(n));
    }
    ::close(pfd[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);

    INFO("output: ", out);
    CHECK_FALSE(child_gone);   // it exited before the SIGTERM
    CHECK(seen_while_running);
    CHECK(WIFEXITED(status));   // a graceful --save-on-term stop, not a kill
}

TEST_CASE("b_tier.run_manifest_node_set_labels: node_seeds label node_set copies by their "
          "derived label") {
    // configs/mnist.json plus a node_set copying both of its nodes (0 "mnist",
    // 1 "group 1") at base_id 10: the copies are 10 "copy.mnist" and 11
    // "copy.group 1", wired 10 -> 11. node_seeds used to look labels up in the
    // config's direct nodes only, so the copies were recorded with empty labels.
    scratch_dir sd{"manifest_node_set"};
    const auto manifest = sd.path / "coast_runtime.json";
    const auto cfg      = sd.path / "mnist_node_set.json";
    {
        auto j = nlohmann::json::parse(read_file(fs::path{COAST_REPO_ROOT} / "configs" / "mnist.json"));
        j["sim"]["node_sets"] = nlohmann::json::array({ {
            { "base_id", 10 }, { "base_label", "copy" }, { "nodes", { 0, 1 } },
            { "node_deltas", nlohmann::json::array() } } });
        std::ofstream out(cfg);
        out << j.dump(2);
    }

    auto r = run(COAST_BIN, {"--headless", "-n", "20",
                             "--emit-run-manifest", manifest.string(), cfg.string()});
    INFO("stderr: ", r.stderr_text);
    REQUIRE(r.exit_code == 0);
    auto j = nlohmann::json::parse(read_file(manifest));
    std::map<std::uint64_t, std::string> labels;
    for (const auto& e : j.at("node_seeds"))
        labels[e.at("id").get<std::uint64_t>()] = e.at("label").get<std::string>();
    INFO("node_seeds: ", j.at("node_seeds").dump());
    REQUIRE(labels.size() == 4);
    CHECK(labels.at(0)  == "mnist");
    CHECK(labels.at(1)  == "group 1");
    CHECK(labels.at(10) == "copy.mnist");
    CHECK(labels.at(11) == "copy.group 1");
}

TEST_CASE("b_tier.observe_dtype_lock: a conflicting site re-resolve aborts") {
    // The observe dtype/shape lock (resolve_site) is a hard invariant enforced
    // with std::abort(). An abort cannot be caught in-process, so a helper
    // binary trips it and we assert abnormal termination here. The stderr check
    // confirms the abort came from *our* guard, not some unrelated crash.
    auto r = run(OBSERVE_DTYPE_MISMATCH_BIN, {});
    CHECK(r.exit_code == -1);  // abort() -> process did not terminate normally
    CHECK(contains(r.stderr_text, "dtype/shape lock violated"));
}

TEST_CASE("b_tier.observe_emit_dtype_lock: a conflicting emit() cache-hit aborts") {
    // Behavior-change coverage (this PR): observe_registry::emit() now enforces
    // the dtype/shape lock on its CACHE-HIT path, not just via resolve_site() on a
    // cache miss. This closes a buffer-over-read into H5Dwrite when a site is
    // re-emitted with a wider value than it was registered with, and it applies to
    // the already-shipped ngm2/ngm3 emit paths as well as ngm2.5.
    //
    // The guard sits behind emit()'s active-group gate, so the helper builds a real
    // (synthetic-dataset) observing sim, steps it to make the group active, then
    // re-emits a float scalar site as a double -> cache hit -> abort. As above the
    // stderr check confirms the abort came from our guard, not an unrelated crash.
    auto r = run(OBSERVE_EMIT_DTYPE_MISMATCH_BIN, {});
    CHECK(r.exit_code == -1);  // abort() -> process did not terminate normally
    CHECK(contains(r.stderr_text, "dtype/shape lock violated"));
}

TEST_CASE("b_tier.load_corrupt_state: a non-HDF5 state file fails cleanly, no crash") {
    scratch_dir sd{"corrupt_state"};
    const auto bad = sd.path / "garbage.h5";
    {
        std::ofstream os(bad, std::ios::binary);
        // 4 KiB of clearly non-HDF5 bytes — no HDF5 superblock signature.
        for (int i = 0; i < 4096; ++i) os.put(static_cast<char>(0xA5));
    }

    auto r = run(COAST_BIN,
                 {"--headless", "-n", "10", "--load-state", bad.string(), "configs/mnist.json"});

    // Contract: a corrupt state file is a controlled failure — a clean non-zero
    // exit (1), NOT a signal/abort/segv (which subprocess_result reports as -1).
    CHECK(r.exit_code == 1);
    CHECK(contains(r.stdout_text + r.stderr_text, "Failed to load state"));
}

// A checkpoint from a group of another shape is refused at startup, naming the
// field, and reported as a state failure — the configuration itself loaded.
// It used to be skipped silently while the step counter was restored, so the
// run claimed to resume step 100 on fresh weights.
TEST_CASE("b_tier.load_state_misfit: --load-state of another shape exits 1 naming the field") {
    scratch_dir sd{"load_misfit"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);
    const auto train_cfg = write_mnist_config_for(sd.path, "train", img, lbl);
    const auto other_cfg = write_mnist_config_for(sd.path, "other", img, lbl);
    set_neuron_count(other_cfg, 48);
    const auto snap = sd.path / "snap.h5";

    REQUIRE(run(COAST_BIN, {"--headless", "-n", "100", "--save-state", snap.string(),
                            train_cfg.string()}).exit_code == 0);

    auto r = run(COAST_BIN, {"--headless", "-n", "10", "--load-state", snap.string(),
                             other_cfg.string()});
    const auto out = r.stdout_text + r.stderr_text;
    INFO(out);
    CHECK(r.exit_code == 1);
    CHECK(contains(out, "Failed to load state from " + snap.string()));
    CHECK(contains(out, "checkpoint does not fit this group — "
                        "neuron_count 50 (checkpoint) vs 48 (configured)"));
    CHECK(!contains(out, "failed to load configuration"));
    CHECK(!contains(out, "drop --resume"));   // the user named this file
}

// The snapshot --resume picks up was not named by the user: after a change of
// shape the error must say where it came from and how to get past it.
TEST_CASE("b_tier.resume_misfit_names_remedy: --resume after a shape change names the remedy") {
    scratch_dir sd{"resume_misfit"};
    const auto img = sd.path / "images";
    const auto lbl = sd.path / "labels";
    write_synthetic_mnist(img, lbl, 128);
    const auto snaps = sd.path / "snaps";
    const auto cfg = write_mnist_config_with_snapshot_dir(sd.path, "resumefit", img, lbl,
                                                          snaps.string());

    REQUIRE(run(COAST_BIN, {"--headless", "-n", "100", "--checkpoint", "50",
                            cfg.string()}).exit_code == 0);
    REQUIRE(dir_has_h5(snaps));

    set_neuron_count(cfg, 48);
    auto r = run(COAST_BIN, {"--headless", "-n", "1", "--resume", cfg.string()});
    const auto out = r.stdout_text + r.stderr_text;
    INFO(out);
    CHECK(r.exit_code == 1);
    CHECK(contains(out, "checkpoint does not fit this group"));
    CHECK(contains(out, "found by --resume in " + snaps.string()));
    CHECK(contains(out, "drop --resume to start fresh, or move that directory aside"));
    CHECK(!contains(out, "failed to load configuration"));
}

// ---------------------------------------------------------------------------
#ifdef COAST_WITH_GUI
// 9. The coast::simc sim-control façade: edit a config, run() it into a run
//    directory, and step() it in chunks via a child `coast` process. This is
//    the console-driven parameter-search path (design/sim_console_api.md).
//
//    COAST_SIMC_BINARY points the façade at the built `coast` (in production it
//    would spawn itself, but this test host is not coast). Uses a synthetic
//    dataset so it needs no downloaded data.

TEST_CASE("b_tier.simc_facade: config edit + run + chunked step chains state") {
    scratch_dir sd{"simc"};
    const auto img = sd.path / "img-idx3";
    const auto lbl = sd.path / "lbl-idx1";
    write_synthetic_mnist(img, lbl, 128);
    const auto base_cfg = write_mnist_config_for(sd.path, "simc", img, lbl);

    // Point the façade at the built binary (it would otherwise spawn this test).
    ::setenv("COAST_SIMC_BINARY", COAST_BIN, /*overwrite=*/1);

    namespace simc = coast::simc;

    // Load + edit: a hyperparameter by JSON pointer, plus run_dir sugar.
    const auto run_dir = sd.path / "run";
    auto c = simc::config(base_cfg.string());
    c.set("/ngm2/dendrites/default/default_primary_learning_rate", 0.02)
     .set("/note", "string-literal-value")  // guards the const char*/bool trap
     .run_dir(run_dir.string())
     .seed(0);
    // Integer overload trap: a bare int AND unsigned/size_t must all compile
    // (they were ambiguous across double/int64_t/bool before the integral
    // template) and store as integers.
    const std::size_t sz = 7;
    c.set("/probe_int", 42).set("/probe_sz", sz);
    CHECK(c.get_number("/probe_int") == 42.0);
    CHECK(c.get_number("/probe_sz") == 7.0);
    CHECK(contains(c.dump(), "0.02"));                       // edit round-trips
    CHECK(contains(c.dump(), "\"output_dir\""));             // run_dir wired observe
    // A string literal must be stored as a STRING, not silently coerced to bool
    // (const char* -> bool would win overload resolution without the guard).
    CHECK(c.get_string("/note") == "string-literal-value");

    // Materialize: writes config.json into the run dir, nothing stepped yet.
    auto h = simc::run(c);
    CHECK(fs::exists(run_dir / "config.json"));
    CHECK(h->config_path() == (run_dir / "config.json").string());
    CHECK(h->total_steps() == 0);
    CHECK(h->state_path().empty());

    // First chunk: runs a real child, saves state, writes a timestamped log.
    CHECK(h->step(10) == 10);
    REQUIRE(fs::exists(h->state_path()));
    REQUIRE(fs::exists(h->last_log()));
    CHECK(contains(read_file(h->last_log()), "Simulation completed"));
    const auto log1   = h->last_log();
    const auto state1 = h->state_path();
    CHECK(contains(state1, "state_001.h5"));                 // per-chunk naming

    // Second chunk: chains --load-state, cumulative count continues, and a
    // DISTINCT log file is written (timestamp + chunk index differ).
    CHECK(h->step(10) == 20);
    CHECK(h->total_steps() == 20);
    REQUIRE(fs::exists(h->last_log()));
    CHECK(h->last_log() != log1);
    // Both logs live under <run_dir>/logs/.
    CHECK(contains(h->last_log(), (run_dir / "logs").string()));
    // Rolling history: the second chunk saved a fresh state file and the first
    // one is retained immutably (not overwritten) — the introspection payoff.
    const auto state2 = h->state_path();
    CHECK(state2 != state1);
    CHECK(contains(state2, "state_002.h5"));
    CHECK(fs::exists(state1));                                // earlier state kept
    CHECK(fs::exists(state2));

    // step(0) is a guarded no-op (raw -n 0 would mean UNLIMITED).
    CHECK(h->step(0) == 20);

    // save_state copies the latest checkpoint out.
    const auto out = sd.path / "final.h5";
    h->save_state(out);
    CHECK(fs::exists(out));

    // A nonzero child exit surfaces as an exception. Point the dataset at a
    // missing file (stored as a real string thanks to the const char* overload)
    // so the child fails to load and exits nonzero.
    auto bad = simc::config(base_cfg.string());
    bad.set("/datasets/mnist/default/image_file", "/nonexistent/path")
       .set("/datasets/mnist/max_interval/image_file", "/nonexistent/path")
       .run_dir((sd.path / "bad").string());
    CHECK(bad.get_string("/datasets/mnist/default/image_file") == "/nonexistent/path");
    auto hb = simc::run(bad);
    CHECK_THROWS_AS(hb->step(5), std::runtime_error);
}

// 9b. Two runs resolving to the SAME run dir must not collide: the second gets
//     a running-number suffix so their config.json / state files / logs stay
//     separate. Materialization only (no child), so this stays fast.
TEST_CASE("b_tier.simc_run_dir_collision: same target dir gets a _NNN suffix") {
    scratch_dir sd{"simc_collide"};
    const auto  img = sd.path / "img-idx3";
    const auto  lbl = sd.path / "lbl-idx1";
    write_synthetic_mnist(img, lbl, 128);
    const auto base_cfg = write_mnist_config_for(sd.path, "simc", img, lbl);
    ::setenv("COAST_SIMC_BINARY", COAST_BIN, /*overwrite=*/1);

    namespace simc = coast::simc;
    const auto target = (sd.path / "run").string();

    auto c1 = simc::config(base_cfg.string());
    c1.run_dir(target);
    auto h1 = simc::run(c1);

    auto c2 = simc::config(base_cfg.string());
    c2.run_dir(target);
    auto h2 = simc::run(c2);

    // First claim keeps the exact path; the collision is redirected.
    CHECK(h1->run_dir() == target);
    CHECK(h2->run_dir() != target);
    CHECK(h2->run_dir() == target + "_001");
    // Both are real, distinct dirs each with their own config.json.
    CHECK(fs::exists(fs::path(h1->run_dir()) / "config.json"));
    CHECK(fs::exists(fs::path(h2->run_dir()) / "config.json"));
    CHECK(h1->config_path() != h2->config_path());
}

// 9c. run() must write ABSOLUTE observe/snapshot paths into config.json. coast
//     resolves a relative snapshot_dir against the config's parent but uses
//     observe/output_dir as-is (CWD-relative), so a relative "<run_dir>/…" would
//     double-nest snapshots to <run_dir>/<run_dir>/snapshots under --checkpoint.
//     Reproduce the trigger with a RELATIVE run dir (chdir into scratch so it
//     lands there, not in the repo). Materialization only.
TEST_CASE("b_tier.simc_abs_output_paths: run() writes absolute observe/snapshot dirs") {
    scratch_dir sd{"simc_abs"};
    const auto  img = sd.path / "img-idx3";
    const auto  lbl = sd.path / "lbl-idx1";
    write_synthetic_mnist(img, lbl, 128);
    const auto base_cfg = write_mnist_config_for(sd.path, "simc", img, lbl);

    namespace simc = coast::simc;

    // Restore CWD on every exit path (incl. a REQUIRE throw).
    struct cwd_guard {
        fs::path old;
        cwd_guard() : old(fs::current_path()) {}
        ~cwd_guard() { std::error_code ec; fs::current_path(old, ec); }
    } guard;
    fs::current_path(sd.path);

    auto c = simc::config(base_cfg.string());   // base_cfg is absolute — CWD-safe
    c.run_dir("rel_run");                        // RELATIVE — the bug's trigger
    auto h = simc::run(c);

    auto j = nlohmann::json::parse(read_file(h->config_path()));
    const std::string snap = j.at("snapshot_dir").get<std::string>();
    const std::string obs  = j.at("observe").at("output_dir").get<std::string>();

    // Absolute → coast's is_relative() branch is skipped, so no double-nest.
    CHECK(fs::path(snap).is_absolute());
    CHECK(fs::path(obs).is_absolute());
    CHECK(fs::path(snap) == fs::absolute(fs::path("rel_run") / "snapshots"));
    CHECK(fs::path(obs) == fs::absolute(fs::path("rel_run") / "observe"));
}

// ---------------------------------------------------------------------------
// 10. step_async: two concurrent runs, non-blocking, tracked in the registry;
//     wait_all_runs() synchronizes; state chains back into each handle.

TEST_CASE("b_tier.simc_async: concurrent step_async runs are tracked and chained") {
    scratch_dir sd{"simc_async"};
    const auto img = sd.path / "img-idx3";
    const auto lbl = sd.path / "lbl-idx1";
    write_synthetic_mnist(img, lbl, 128);
    const auto base_cfg = write_mnist_config_for(sd.path, "async", img, lbl);
    ::setenv("COAST_SIMC_BINARY", COAST_BIN, /*overwrite=*/1);

    namespace simc = coast::simc;
    simc::clear_finished_runs(); // isolate from any prior test's rows

    auto a = simc::config(base_cfg.string());
    auto b = simc::config(base_cfg.string());
    auto ha = simc::run(a, (sd.path / "run_a").string());
    auto hb = simc::run(b, (sd.path / "run_b").string());

    // Launch both without blocking (show_status=false: no GUI in this test).
    ha->step_async(10, /*show_status=*/false);
    hb->step_async(10, /*show_status=*/false);
    CHECK(ha->running());
    CHECK(hb->running());
    CHECK(simc::active_run_count() == 2);

    // A second step on an in-flight handle is rejected.
    CHECK_THROWS_AS(ha->step_async(10, false), std::runtime_error);
    CHECK_THROWS_AS(ha->step(10), std::runtime_error);

    simc::wait_all_runs(); // blocks until both children exit
    CHECK(simc::active_run_count() == 0);
    CHECK_FALSE(ha->running());

    // State chained back into each handle (as a sync step would).
    CHECK(ha->total_steps() == 10);
    CHECK(hb->total_steps() == 10);
    REQUIRE(fs::exists(ha->state_path()));
    REQUIRE(fs::exists(hb->state_path()));
    REQUIRE(fs::exists(ha->last_log()));

    // A later sync step continues from the async result (10 -> 20).
    CHECK(ha->step(10) == 20);

    simc::clear_finished_runs();
    CHECK(simc::active_run_count() == 0);
}

// 10b. A step_async whose child fails (nonzero exit) must clear step_in_flight
//      via on_done(false) WITHOUT chaining state or advancing the count — the
//      async mirror of step()'s throw path. Also covers save_state's "no state
//      yet" throw branch (the failed handle never produced a state).
TEST_CASE("b_tier.simc_async_failure: a failed async run clears the guard, chains nothing") {
    scratch_dir sd{"simc_async_fail"};
    const auto  img = sd.path / "img-idx3";
    const auto  lbl = sd.path / "lbl-idx1";
    write_synthetic_mnist(img, lbl, 128);
    const auto base_cfg = write_mnist_config_for(sd.path, "asyncfail", img, lbl);
    ::setenv("COAST_SIMC_BINARY", COAST_BIN, /*overwrite=*/1);

    namespace simc = coast::simc;
    simc::clear_finished_runs();

    // Point the dataset at a missing file so the child exits nonzero at load.
    auto c = simc::config(base_cfg.string());
    c.set("/datasets/mnist/default/image_file", "/nonexistent/path")
     .set("/datasets/mnist/max_interval/image_file", "/nonexistent/path");
    auto h = simc::run(c, (sd.path / "run_fail").string());

    h->step_async(10, /*show_status=*/false);
    CHECK(h->running());
    simc::wait_all_runs(); // blocks until the child exits (with failure)

    // The guard is cleared — a fresh step is accepted, not rejected as in-flight.
    CHECK_FALSE(h->running());
    CHECK(simc::active_run_count() == 0);
    // Nothing chained: no step count, no state.
    CHECK(h->total_steps() == 0);
    CHECK(h->state_path().empty());
    // save_state before any successful step throws rather than copying nothing.
    CHECK_THROWS_AS(h->save_state(sd.path / "out.h5"), std::runtime_error);
    // A log was still written (the child's failure diagnostics land there).
    REQUIRE(fs::exists(h->last_log()));

    simc::clear_finished_runs();
}

// 11. A relative run dir whose name starts with '-' must still step: coast's CLI
//     parser reads a positional beginning with '-' as an unknown option, so the
//     façade prefixes "./". chdir into scratch so the relative dir lands there,
//     not in the repo.
TEST_CASE("b_tier.simc_leading_dash: a relative run dir starting with '-' still steps") {
    scratch_dir sd{"simc_dash"};
    const auto  img = sd.path / "img-idx3";
    const auto  lbl = sd.path / "lbl-idx1";
    write_synthetic_mnist(img, lbl, 128);
    const auto base_cfg = write_mnist_config_for(sd.path, "dash", img, lbl);
    ::setenv("COAST_SIMC_BINARY", COAST_BIN, /*overwrite=*/1);

    struct cwd_guard {
        fs::path old;
        cwd_guard() : old(fs::current_path()) {}
        ~cwd_guard() { std::error_code ec; fs::current_path(old, ec); }
    } guard;
    fs::current_path(sd.path);

    namespace simc = coast::simc;
    auto c = simc::config(base_cfg.string()); // base_cfg is absolute — CWD-safe
    auto h = simc::run(c, "-baseline");        // RELATIVE dir, leading dash
    // Without the "./" guard the child fails with "Unknown option:
    // -baseline/config.json" and step() throws; with it, the run completes.
    CHECK(h->step(1) == 1);
}

#ifdef __linux__
// 12. closefrom: the spawned child must inherit only fds 0/1/2, not the console's
//     open HDF5 handles etc. Stand in a helper "coast" that dumps /proc/self/fd
//     (spawn_child redirects its stdout to the step log); open an extra fd duped
//     high in this process and assert the child never saw it. Linux-only — the
//     closefrom file action and /proc/self/fd are Linux/glibc.
TEST_CASE("b_tier.simc_closefrom: spawned child does not inherit parent fds") {
    scratch_dir sd{"simc_closefrom"};
    const auto  img = sd.path / "img-idx3";
    const auto  lbl = sd.path / "lbl-idx1";
    write_synthetic_mnist(img, lbl, 128);
    const auto base_cfg = write_mnist_config_for(sd.path, "cf", img, lbl);

    // Stand-in binary: ignore all args, dump this process's open fds, exit 0.
    const auto helper = sd.path / "fd_dump.sh";
    { std::ofstream o(helper); o << "#!/bin/sh\nls /proc/self/fd\n"; }
    fs::permissions(helper, fs::perms::owner_all | fs::perms::group_read |
                                fs::perms::others_read);
    ::setenv("COAST_SIMC_BINARY", helper.string().c_str(), /*overwrite=*/1);

    // An extra fd in THIS process, duped to a distinctive high number so it can't
    // collide with the child's own low fds. F_DUPFD clears close-on-exec, so
    // WITHOUT closefrom the child would inherit it.
    const int raw = ::open(base_cfg.string().c_str(), O_RDONLY);
    REQUIRE(raw >= 0);
    const int hi = ::fcntl(raw, F_DUPFD, 50);
    REQUIRE(hi >= 50);

    namespace simc = coast::simc;
    auto c = simc::config(base_cfg.string());
    auto h = simc::run(c, (sd.path / "run_cf").string());
    CHECK(h->step(1) == 1); // helper exits 0 -> step succeeds

    ::close(raw);
    ::close(hi);
    ::setenv("COAST_SIMC_BINARY", COAST_BIN, /*overwrite=*/1); // restore for others

    const std::string log = read_file(h->last_log());
    CHECK(contains(log, "0"));                              // child listed its fds
    CHECK_FALSE(contains(log, std::to_string(hi)));         // but NOT the inherited one
}
#endif

// 12b. Async-child fd hygiene, from the caller's side (the hang path simc_async
// can't reach, exercised end-to-end).
//
// The console wraps every REPL cell in an fd_capture (session.cpp): it dup2's a
// pipe's write end onto stdout/stderr and drains the read end on a reader thread,
// then stop() closes the write end and joins the reader on EOF. A step_async
// child that INHERITED that write end keeps the pipe open past the cell, so the
// reader never sees EOF and stop()'s join blocks until the (possibly long) child
// exits — silently turning the async run synchronous. Two layers prevent that:
// session.cpp CLOEXEC's its raw pipe ends, and spawn_child closes every inherited
// fd >= 3 in the child (Linux close_range; on macOS the equivalent spawnattr).
//
// b_tier.simc_async runs step_async with NO capture active, so it can't catch a
// re-leak. This test puts a capture-shaped pipe in place around a step_async and
// asserts the read end reaches EOF while the child is still alive. Crucially it
// leaves the pipe ends WITHOUT close-on-exec: that makes spawn_child's own fd
// closing the ONLY thing that can stop the child inheriting the write end, so the
// test genuinely exercises that closing (verified by mutation: with it removed
// AND the pipe non-CLOEXEC, the read end never EOFs and this test fails). Plain
// POSIX, so it runs on macOS too — where close_range is absent and the macOS
// spawnattr is what must satisfy it.
// For the cases whose stand-in child (`exec sleep 3600`) lives until it is
// cancelled: on EVERY exit — a REQUIRE failing before the explicit cancel
// included — cancel and reap all runs, restore the concurrency cap, and point
// the façade back at the real coast. Otherwise the sleeper keeps running, the
// next case's (global) wait_all_runs() waits out its hour, and later cases
// would spawn the stand-in. Declare it first, so it is destroyed last.
struct simc_sleeper_guard {
    int saved_cap = coast::simc::max_concurrent_runs();
    simc_sleeper_guard() = default;
    simc_sleeper_guard(const simc_sleeper_guard&)            = delete;
    simc_sleeper_guard& operator=(const simc_sleeper_guard&) = delete;
    ~simc_sleeper_guard() {
        coast::simc::cancel_all_runs();
        coast::simc::wait_all_runs();
        coast::simc::set_max_concurrent_runs(saved_cap);
        ::setenv("COAST_SIMC_BINARY", COAST_BIN, /*overwrite=*/1);
    }
};

TEST_CASE("b_tier.simc_async_capture_eof: a step_async child does not pin the caller's stdout pipe") {
    scratch_dir sd{"simc_async_cap"};
    const auto  img = sd.path / "img-idx3";
    const auto  lbl = sd.path / "lbl-idx1";
    write_synthetic_mnist(img, lbl, 128);
    const auto base_cfg = write_mnist_config_for(sd.path, "cap", img, lbl);

    // Stand-in child that lives until it is cancelled below: its exit would close
    // any inherited write end too and hide the leak (a false pass), so it must not
    // exit on its own — not after a real coast run's length, nor after a fixed
    // sleep a slow machine can outlast. `exec` makes `sleep` the child itself, so
    // the cancel's SIGTERM ends it. The child ignores its args.
    const auto helper = sd.path / "sleeper.sh";
    { std::ofstream o(helper); o << "#!/bin/sh\nexec sleep 3600\n"; }
    fs::permissions(helper, fs::perms::owner_all | fs::perms::group_read |
                                fs::perms::others_read);
    ::setenv("COAST_SIMC_BINARY", helper.string().c_str(), /*overwrite=*/1);
    const simc_sleeper_guard sleeper_guard;

    namespace simc = coast::simc;
    auto c = simc::config(base_cfg.string());
    auto h = simc::run(c, (sd.path / "run_cap").string());

    // A capture-shaped pipe: dup2 the write end onto stdout+stderr, as fd_capture
    // does. Deliberately NO close-on-exec on the raw ends here (fd_capture does set
    // it) — see the header comment: this is what forces spawn_child to be the sole
    // fd-closer, so the test actually exercises spawn_child rather than its own
    // CLOEXEC. The dup2'd 1/2 are what the child would redirect to its own log.
    int pfd[2];
    REQUIRE(::pipe(pfd) == 0);
    const int read_fd  = pfd[0];
    const int write_fd = pfd[1];
    const int saved_out = ::dup(STDOUT_FILENO);
    const int saved_err = ::dup(STDERR_FILENO);
    REQUIRE(saved_out >= 0);
    REQUIRE(saved_err >= 0);

    // Restore stdout/stderr on every exit path (incl. a step_async throw) so a
    // failure here can't corrupt the rest of the suite's console output.
    struct fd_restore {
        int  out, err;
        bool done = false;
        void run() {
            if (done) return;
            ::dup2(out, STDOUT_FILENO);
            ::dup2(err, STDERR_FILENO);
            ::close(out);
            ::close(err);
            done = true;
        }
        ~fd_restore() { run(); }
    } restorer{saved_out, saved_err};

    ::dup2(write_fd, STDOUT_FILENO);
    ::dup2(write_fd, STDERR_FILENO);

    // Launch the async child WHILE the capture is active — the exact moment a
    // regression would leak the write end into the child. spawn_child runs
    // synchronously inside step_async, so the child's fd table is fixed once this
    // returns; there is no spawn-on-a-background-thread race to lose.
    h->step_async(1, /*show_status=*/false);

    // Tear down the capture as fd_capture::stop() does: restore 1/2, then close
    // our write end. After this THIS process holds no write end; if the child
    // holds one (regression), the pipe stays open and read_fd never EOFs.
    restorer.run();
    ::close(write_fd);

    // No write end exists any more unless the still-running child inherited
    // one, and EOF on a pipe is pending the moment its last write end closes —
    // so a ZERO-timeout poll decides it, with no window: readable + read()==0 is
    // EOF -> the child did NOT inherit the write end (PASS); not readable means a
    // write end is open in the child (FAIL).
    struct pollfd pf {};
    pf.fd     = read_fd;
    pf.events = POLLIN;
    const int pr = ::poll(&pf, 1, /*timeout_ms=*/0);
    ::setenv("COAST_SIMC_BINARY", COAST_BIN, /*overwrite=*/1); // restore for others

    REQUIRE(pr >= 0);        // poll itself didn't error
    CHECK(pr == 1);          // read end is readable (EOF pending) ...
    char          byte = 0;
    const ssize_t n    = (pr == 1) ? ::read(read_fd, &byte, 1) : -1;
    CHECK(n == 0);           // ... and it was EOF, not stray child output
    ::close(read_fd);

    h->cancel();           // the sleeper does not end on its own
    simc::wait_all_runs(); // reap it so the suite leaves no child behind
}

// 13. Process pool: step_async never fails or blocks when the concurrency cap is
//     reached — excess runs QUEUE and drain automatically as running children
//     exit. With the cap pinned to 1, three launched runs must be exactly one
//     running + two queued (deterministic: reaping/draining happens only inside
//     poll_runs(), which nothing calls between the launches and the snapshot), and
//     wait_all_runs() must drain the queue so all three complete and chain state.
TEST_CASE("b_tier.simc_pool: step_async queues past the cap and drains in order") {
    scratch_dir sd{"simc_pool"};
    const auto  img = sd.path / "img-idx3";
    const auto  lbl = sd.path / "lbl-idx1";
    write_synthetic_mnist(img, lbl, 128);
    const auto base_cfg = write_mnist_config_for(sd.path, "pool", img, lbl);
    ::setenv("COAST_SIMC_BINARY", COAST_BIN, /*overwrite=*/1);

    namespace simc = coast::simc;
    simc::clear_finished_runs();
    const int saved_cap = simc::max_concurrent_runs();
    simc::set_max_concurrent_runs(0);            // clamps to 1
    CHECK(simc::max_concurrent_runs() == 1);

    // Three independent handles, each stepping a small run.
    std::vector<std::shared_ptr<simc::sim_run>> hs;
    for (int i = 0; i < 3; ++i) {
        auto c = simc::config(base_cfg.string());
        hs.push_back(simc::run(c, (sd.path / ("run_" + std::to_string(i))).string()));
    }
    for (auto& h : hs) h->step_async(5, /*show_status=*/false);

    // Snapshot before any poll: 1 running, 2 queued, 3 active. Every handle counts
    // as running() — a queued run still holds its handle's in-flight guard.
    CHECK(simc::active_run_count() == 3);
    CHECK(simc::queued_run_count() == 2);
    CHECK(simc::active_run_count() - simc::queued_run_count() == 1); // exactly one child
    for (auto& h : hs) CHECK(h->running());

    simc::wait_all_runs(); // drains the queue one slot at a time, reaps all three

    CHECK(simc::active_run_count() == 0);
    CHECK(simc::queued_run_count() == 0);
    for (auto& h : hs) {
        CHECK_FALSE(h->running());
        CHECK(h->total_steps() == 5);      // each ran its 5 steps
        REQUIRE(fs::exists(h->state_path())); // and chained its state
    }

    simc::set_max_concurrent_runs(saved_cap); // restore for later tests
    simc::clear_finished_runs();
}

// 14. Cancellation: a queued run is dropped before it spawns; a running run is
//     signalled and reaped as cancelled; cancel_all() clears the board. In every
//     case the handle's in-flight guard is released and nothing chains. A sleeper
//     stand-in that lives until it is cancelled keeps a "running" child alive by
//     construction, not for a fixed time a slow machine could outlast.
TEST_CASE("b_tier.simc_cancel: cancel drops a queued run and stops a running one") {
    scratch_dir sd{"simc_cancel"};
    const auto  img = sd.path / "img-idx3";
    const auto  lbl = sd.path / "lbl-idx1";
    write_synthetic_mnist(img, lbl, 128);
    const auto base_cfg = write_mnist_config_for(sd.path, "cancel", img, lbl);

    const auto helper = sd.path / "sleeper.sh";
    // `exec`: the SIGTERM of a cancel reaches `sleep` itself; every run below is
    // cancelled, so none is left to wait out the hour.
    { std::ofstream o(helper); o << "#!/bin/sh\nexec sleep 3600\n"; }
    fs::permissions(helper, fs::perms::owner_all | fs::perms::group_read |
                                fs::perms::others_read);
    ::setenv("COAST_SIMC_BINARY", helper.string().c_str(), /*overwrite=*/1);
    const simc_sleeper_guard sleeper_guard;

    namespace simc = coast::simc;
    simc::clear_finished_runs();
    const int saved_cap = simc::max_concurrent_runs();
    simc::set_max_concurrent_runs(1); // force the second run to queue

    auto ca = simc::config(base_cfg.string());
    auto cb = simc::config(base_cfg.string());
    auto ha = simc::run(ca, (sd.path / "run_a").string());
    auto hb = simc::run(cb, (sd.path / "run_b").string());
    ha->step_async(1, /*show_status=*/false); // spawns the sleeper (running)
    hb->step_async(1, /*show_status=*/false); // no slot -> queued
    REQUIRE(simc::active_run_count() == 2);
    REQUIRE(simc::queued_run_count() == 1);

    // Cancel the QUEUED run: dropped immediately, guard cleared, never spawns.
    hb->cancel();
    CHECK_FALSE(hb->running());
    CHECK(simc::queued_run_count() == 0);
    CHECK(simc::active_run_count() == 1); // only ha's running child remains

    // Cancel the RUNNING run: SIGTERM now, finalized on the next poll.
    CHECK(ha->running());
    ha->cancel();
    simc::wait_all_runs();               // reaps the signalled child
    CHECK_FALSE(ha->running());
    CHECK(simc::active_run_count() == 0);
    CHECK(ha->total_steps() == 0);       // cancelled chains nothing
    CHECK(hb->total_steps() == 0);

    // cancel_all(): one running + two queued, all cleared in one call.
    simc::clear_finished_runs();
    std::vector<std::shared_ptr<simc::sim_run>> hs;
    for (int i = 0; i < 3; ++i) {
        auto c = simc::config(base_cfg.string());
        hs.push_back(simc::run(c, (sd.path / ("c_" + std::to_string(i))).string()));
        hs.back()->step_async(1, /*show_status=*/false);
    }
    REQUIRE(simc::active_run_count() == 3);
    simc::cancel_all_runs();
    simc::wait_all_runs();
    CHECK(simc::active_run_count() == 0);
    for (auto& h : hs) CHECK_FALSE(h->running());

    simc::set_max_concurrent_runs(saved_cap);
    simc::clear_finished_runs();
    ::setenv("COAST_SIMC_BINARY", COAST_BIN, /*overwrite=*/1);
}

#endif // COAST_WITH_GUI — end of the coast::simc façade cases

// ---------------------------------------------------------------------------
// 12. Run-length enforcement that only exists in a headless BUILD.
//
// cli::parse is deliberately build-independent: it rejects an explicit -H with no
// length, but a command line with no -H at all parses fine everywhere (there is a
// unit test pinning exactly that). The build-dependent half lives in main(), which
// promotes every run to headless when COAST_WITH_GUI is off — and must therefore
// apply the same run-length rule.
//
// That rule is the entire reason the `-n 0`-means-unlimited sentinel was removed:
// without it, `coast config.json` — the command that opens a window in a GUI
// build — silently starts an UNBOUNDED run here. Only a subprocess test can see
// it, because it is a property of main() in one build configuration.

#ifndef COAST_WITH_GUI
TEST_CASE("b_tier.headless_build_requires_length: a bare config is rejected, not run forever") {
    auto r = run(COAST_BIN, {"configs/mnist.json"});
    CHECK(r.exit_code == 1);
    CHECK(contains(r.stderr_text, "needs an explicit length"));
    // The decisive assertion: it must not have started stepping.
    CHECK_FALSE(contains(r.stdout_text, "Starting simulation"));
}

TEST_CASE("b_tier.headless_build_notice: absent only when -H was given") {
    // Without -H the build announces itself; with -H the user has said they know.
    auto quiet = run(COAST_BIN, {"-H", "-n", "1", "configs/mnist.json"});
    CHECK(quiet.exit_code == 0);
    CHECK_FALSE(contains(quiet.stderr_text, "built without GUI support"));

    auto noisy = run(COAST_BIN, {"-n", "1", "configs/mnist.json"});
    CHECK(noisy.exit_code == 0);
    CHECK(contains(noisy.stderr_text, "built without GUI support"));
}
#endif // !COAST_WITH_GUI

// ---------------------------------------------------------------------------
// 13. --unlimited must refuse the outputs it would destroy.
//
// An unlimited run ends only by being killed, and there is no signal handler, so
// --save-state / --emit-run-manifest never run at all and --log-buffers leaves an
// unclosed (unreadable) HDF5 file. Accepting these silently would recreate the
// very footgun that removing the -n 0 sentinel closed. --checkpoint is the
// exception and must keep working: its snapshots complete during the run.

TEST_CASE("b_tier.unlimited_rejects_doomed_outputs: end-of-run outputs are refused") {
    for (const char* flag : {"--save-state", "--emit-run-manifest", "--log-buffers"}) {
        CAPTURE(flag);
        auto r = run(COAST_BIN, {"--unlimited", flag, "out.h5", "configs/mnist.json"});
        CHECK(r.exit_code == 1);
        CHECK(contains(r.stderr_text, "cannot be combined with"));
        CHECK_FALSE(contains(r.stdout_text, "Starting simulation"));
    }
}

TEST_CASE("b_tier.unlimited_allows_checkpoint: --checkpoint survives an unlimited run") {
    // Parse-level only: --checkpoint must NOT be rejected. The run itself is
    // unbounded, so this asserts the rejection does not fire, via the one signal
    // available without starting it — a different failure than the one above.
    auto r = run(COAST_BIN, {"--unlimited", "--checkpoint", "50", "/nonexistent.json"});
    CHECK(r.exit_code == 1);
    CHECK_FALSE(contains(r.stderr_text, "cannot be combined with"));
    CHECK(contains(r.stderr_text, "Cannot open config file"));
}

int main(int argc, char** argv)
{
    // Hermetic subprocess environment. This feature actively encourages exporting
    // $COAST_APP_CONFIG, so the developer running these tests is the one most
    // likely to have it set — and every coast child inherits the parent env
    // (build_envp copies environ). A leaked value would beat ./coast.json and the
    // built-in default config_base, breaking discovery/default-layout cases in a
    // way that looks like a real regression. Clear it once for the whole suite
    // (covers ctest AND a direct ./test_b_tier run). Cases that need a value set it
    // explicitly, which overrides this.
    ::unsetenv("COAST_APP_CONFIG");
    return doctest::Context(argc, argv).run();
}
