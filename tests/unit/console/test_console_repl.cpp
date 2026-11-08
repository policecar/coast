// console::repl — the embedded clang-repl interpreter driving the console_api
// facade end to end: interpreted macro code parses console_api.h, calls the
// compiled facade, iterates real HDF5 data, and writes results back to the
// host. Also verifies the exception guard (an uncaught throw in a macro is
// reported, not fatal) and the .x macro-file path.
//
// Requires the host exe to be linked with -rdynamic (set in CMake) so the JIT
// resolves the facade symbols.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "console/repl.h"
#include "campaign/campaign_api.h"
#include "coastd_console/coastd_console_api.h"
#include "h5browse/console_api.h"
#include "sim_console/sim_console_api.h"
#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include <hdf5.h>

#include "../coastd/test_daemon.h"   // OS-assigned daemon ports + reaping guard

#include <arpa/inet.h>
#include <csignal>
#include <netinet/in.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

extern char** environ;
#ifndef COASTD_BIN
#define COASTD_BIN "coastd"
#endif

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t n_act = 2000; // act[i] = i % 100 -> sum = 20*4950 = 99000

fs::path make_test_file(hdf5::executor& exec)
{
    // pid-suffixed, like every scratch file here: two concurrent runs of this
    // binary otherwise truncate (or fail to lock) each other's file.
    const fs::path  p = fs::temp_directory_path() /
                       ("h5b_console_repl_test_" + std::to_string(::getpid()) + ".h5");
    std::error_code ec;
    fs::remove(p, ec);
    exec.submit_sync([&] {
        const hid_t fid = H5Fcreate(p.string().c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(fid >= 0);
        {
            const hid_t md = H5Gcreate2(fid, "metadata", H5P_DEFAULT, H5P_DEFAULT,
                                        H5P_DEFAULT);
            REQUIRE(hdf5::write_attr(md, "type", std::string_view{"observe_log"}) >= 0);
            REQUIRE(hdf5::write_attr(md, "schema_version", std::string_view{"1.0"}) >= 0);
            H5Gclose(md);
        }
        const hsize_t dims[1]  = {n_act};
        const hsize_t chunk[1] = {256};
        const hid_t   sp   = H5Screate_simple(1, dims, nullptr);
        const hid_t   dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        std::vector<float> vals(n_act);
        for (std::uint64_t i = 0; i < n_act; ++i) vals[i] = static_cast<float>(i % 100);
        const hid_t d = H5Dcreate2(fid, "act", H5T_NATIVE_FLOAT, sp, H5P_DEFAULT,
                                   dcpl, H5P_DEFAULT);
        H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals.data());
        H5Dclose(d);
        H5Pclose(dcpl);
        H5Sclose(sp);
        H5Fclose(fid);
    });
    return p;
}

// Bind a host pointer as a named double* global in the interpreter.
console::exec_result bind_double_ptr(console::repl& r, const char* name, double* p)
{
    char buf[160];
    std::snprintf(buf, sizeof buf,
                  "double* %s = reinterpret_cast<double*>(std::uintptr_t(%lluULL));",
                  name, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(p)));
    return r.exec(buf);
}

} // namespace

TEST_CASE("console::repl: interpreter drives the facade over real data")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);
    h5b::console::set_executor(exec); // wire open()'s executor

    console::repl r;
    REQUIRE_MESSAGE(r.ready(), r.init_error());

    // Open the file the way a real cell does — via open(), into a shared handle.
    REQUIRE(r.exec_guarded("auto f = open(\"" + path.string() + "\");").ok);

    // A macro that iterates the dataset through the facade and writes the sum
    // back to a host double.
    double out = -1.0;
    REQUIRE(bind_double_ptr(r, "__out", &out).ok);
    const auto run = r.exec_guarded(
        "{ auto d = f->dataset(\"act\");"
        "  double s = 0; for (auto row : d.scan()) s += row.at(0);"
        "  *__out = s; }");
    INFO(run.message);
    CHECK(run.ok);
    CHECK(out == doctest::Approx(99000.0));

    // Discovery through the interpreter too.
    double cnt = -1.0;
    REQUIRE(bind_double_ptr(r, "__cnt", &cnt).ok);
    CHECK(r.exec_guarded("*__cnt = (double)f->dataset(\"act\").rows();").ok);
    CHECK(cnt == doctest::Approx(static_cast<double>(n_act)));

    // The sim-control façade is reachable through the `simc` alias the preamble
    // declares (regression: without `namespace simc = coast::simc;` a cell must
    // spell out coast::simc::, so a copy-pasted `simc::…` example failed to
    // compile). Referencing a coast::simc symbol from C++ here pulls
    // sim_console_api.o into this (non-whole-archive) test exe so the JIT can
    // resolve it — the real `coast` exe gets it via --whole-archive. This also
    // proves the alias resolves and the compiled symbol links via -rdynamic.
    CHECK(coast::simc::active_run_count() == 0); // C++ ref: pull the object in
    double cwd_len = -1.0;
    REQUIRE(bind_double_ptr(r, "__cwd", &cwd_len).ok);
    const auto simc_call = r.exec_guarded("*__cwd = (double)simc::cwd().size();");
    INFO(simc_call.message);
    CHECK(simc_call.ok);
    CHECK(cwd_len > 0.0);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::repl: an uncaught macro exception is reported, not fatal")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);
    h5b::console::set_executor(exec);

    console::repl r;
    REQUIRE(r.ready());
    REQUIRE(r.exec_guarded("auto f = open(\"" + path.string() + "\");").ok);

    // Unknown dataset -> facade throws std::runtime_error. exec_guarded must
    // catch it across the JIT boundary and return an error (process survives).
    const auto bad = r.exec_guarded("(void)f->dataset(\"no_such_dataset\").rows();");
    CHECK_FALSE(bad.ok);
    CHECK(bad.message.find("no_such_dataset") != std::string::npos);

    // The interpreter is still usable afterwards.
    double out = -1.0;
    REQUIRE(bind_double_ptr(r, "__out", &out).ok);
    CHECK(r.exec_guarded("*__out = (double)f->dataset(\"act\").width();").ok);
    CHECK(out == doctest::Approx(1.0));

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::repl: a parse error preserves earlier declarations (no Undo)")
{
    console::repl r;
    REQUIRE_MESSAGE(r.ready(), r.init_error());

    // Declare a variable, then submit garbage that fails to PARSE.
    REQUIRE(r.exec_guarded("int keep_me = 7;").ok);
    CHECK_FALSE(r.exec_guarded("@@@ this does not parse;").ok);

    // A storm of parse errors must not corrupt the interpreter.
    for (int i = 0; i < 5; ++i) CHECK_FALSE(r.exec_guarded("@@@ bad;").ok);

    // keep_me must still be visible, and the session still usable.
    double out = -1.0;
    char   buf[128];
    std::snprintf(buf, sizeof buf,
                  "double* __k = reinterpret_cast<double*>(std::uintptr_t(%lluULL));",
                  static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(&out)));
    REQUIRE(r.exec(buf).ok);
    CHECK(r.exec_guarded("*__k = (double)keep_me;").ok);
    CHECK(out == doctest::Approx(7.0));
}

// Regression: a submitted block that pulls in a FRESH header via #include and
// then fails to parse must not corrupt the interpreter. Before exec_guarded
// isolated each #include into its own Parse() unit, a fresh header file-entry
// sharing a failed Parse() unit corrupted the preprocessor's source-file state,
// so the *next* cell lost the entire preamble and reported "use of undeclared
// identifier 'std'" (see split_into_units in repl.cpp). <deque> is chosen
// because it is NOT pulled in by the preamble, so its #include really enters the
// file; a pure-syntax error keeps the block from emitting symbols (which would
// entangle this with the separate, inherent clang-repl JIT-poisoning limitation).
TEST_CASE("console::repl: a fresh #include in a failing block does not corrupt later cells")
{
    console::repl r;
    REQUIRE_MESSAGE(r.ready(), r.init_error());

    // Block 1: enters a fresh header, then a syntax error aborts the parse.
    CHECK_FALSE(r.exec_guarded("#include <deque>\n@@@ does not parse;\n").ok);

    // Repeat: a storm of include+error blocks must not accumulate corruption.
    for (int i = 0; i < 3; ++i)
        CHECK_FALSE(r.exec_guarded("#include <deque>\n@@@ bad;\n").ok);

    // The preamble must still be intact: std is declared and a fresh std::deque
    // (needing the header committed by the isolated include unit) compiles/runs.
    double out = -1.0;
    char   buf[128];
    std::snprintf(buf, sizeof buf,
                  "double* __d = reinterpret_cast<double*>(std::uintptr_t(%lluULL));",
                  static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(&out)));
    REQUIRE(r.exec(buf).ok);
    const auto good =
        r.exec_guarded("std::deque<int> dq; dq.push_back(5); *__d = (double)dq.front();");
    INFO(good.message);
    CHECK(good.ok);
    CHECK(out == doctest::Approx(5.0));
}

// Regression: .L/.x run a USER-supplied file, so load() must be as hardened as
// cell input. It used to call the raw exec() — a leftover from when
// exec_guarded() wrapped its input and so could not host declarations — which
// left macro files with no exception guard and no #include isolation.
//
// Measured before the fix: this aborted the process (SIGABRT), taking the GUI
// and any unsaved simulation state with it. A file-scope initializer is a normal
// ROOT-style pattern and h5.dataset() throws on an unknown name, so a typo was
// enough. Revert load() to exec() and this test dies rather than fails.
TEST_CASE("console::repl: .L a throwing file-scope init is reported, not fatal")
{
    console::repl r;
    REQUIRE_MESSAGE(r.ready(), r.init_error());

    const fs::path macro = fs::temp_directory_path() /
                           ("throwing_init_macro_" + std::to_string(::getpid()) + ".cpp");
    {
        std::ofstream f(macro);
        f << "#include <stdexcept>\n"
          << "int boom() { throw std::runtime_error(\"file-scope boom\"); }\n"
          << "int g = boom();\n"
          << "void throwing_init_macro() {}\n";
    }

    const auto res = r.load(macro);
    CHECK_FALSE(res.ok);
    CHECK(res.message.find("file-scope boom") != std::string::npos);

    // The process is alive and the interpreter still usable.
    CHECK(r.exec_guarded("int after_throw = 1;").ok);

    std::error_code ec;
    fs::remove(macro, ec);
}

// Regression: the same #include-plus-parse-error poisoning covered above for
// cells, but reaching the interpreter through load(). This is the likeliest real
// macro-file shape of all — headers at the top, a typo below — and it used to
// bypass split_into_units entirely, wiping the preamble.
TEST_CASE("console::repl: .L with a fresh #include + parse error does not poison")
{
    console::repl r;
    REQUIRE_MESSAGE(r.ready(), r.init_error());

    const fs::path macro = fs::temp_directory_path() /
                           ("poison_macro_" + std::to_string(::getpid()) + ".cpp");
    {
        std::ofstream f(macro); // <deque> is not in the preamble -> a real file entry
        f << "#include <deque>\n"
          << "@@@ this does not parse;\n";
    }

    CHECK_FALSE(r.load(macro).ok);

    // The preamble must have survived: `std` still resolves.
    const auto after = r.exec_guarded("std::string alive = \"yes\";");
    INFO(after.message);
    CHECK(after.ok);

    std::error_code ec;
    fs::remove(macro, ec);
}

// Regression: isolating #include lines must never cut a block comment in half.
// split_into_units is line-based, so before it tracked comment state these two
// shapes were split into an unterminated /* and failed to parse — code clang
// accepts, rejected by our own pre-pass. Only lines wholly outside a comment are
// isolated now.
TEST_CASE("console::repl: an #include inside a block comment is not split")
{
    SUBCASE("commented-out include")
    {
        console::repl r;
        REQUIRE_MESSAGE(r.ready(), r.init_error());
        const auto res = r.exec_guarded("/* example usage:\n"
                                        "   #include <deque>\n"
                                        "*/\n"
                                        "int bc_val = 5;\n");
        INFO(res.message);
        CHECK(res.ok);
    }

    SUBCASE("include with a block comment opened on the same line")
    {
        console::repl r;
        REQUIRE_MESSAGE(r.ready(), r.init_error());
        const auto res = r.exec_guarded("#include <deque>  /* we need deque\n"
                                        "                     for reasons */\n"
                                        "std::deque<int> bc_dq;\n");
        INFO(res.message);
        CHECK(res.ok);
    }

    SUBCASE("a trailing // /* does not open a phantom comment")
    {
        console::repl r;
        REQUIRE_MESSAGE(r.ready(), r.init_error());
        // The include must still be ISOLATED here; if the `// /*` were taken as
        // opening a comment we would silently stop isolating and regress to the
        // poisoning-prone whole-block behaviour.
        const auto res = r.exec_guarded("int lc_a = 1; // /* not a comment\n"
                                        "#include <deque>\n"
                                        "std::deque<int> lc_dq;\n");
        INFO(res.message);
        CHECK(res.ok);
    }
}

TEST_CASE("console::repl: .x macro file loads and runs")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);
    h5b::console::set_executor(exec);

    console::repl r;
    REQUIRE(r.ready());

    double got = -1.0;
    REQUIRE(bind_double_ptr(r, "__got", &got).ok);

    // The macro opens its own file via open() — the console needs no file bound.
    // run_macro() calls the function named after the file's stem, so the pid
    // suffix goes into both.
    const std::string macro_fn = "probe_macro_" + std::to_string(::getpid());
    const fs::path    macro    = fs::temp_directory_path() / (macro_fn + ".cpp");
    {
        std::ofstream f(macro);
        f << "void " << macro_fn << "() {\n"
          << "    auto file = open(\"" << path.string() << "\");\n"
          << "    auto d = file->dataset(\"act\");\n"
          << "    double s = 0; for (auto row : d.scan(0, 100)) s += row.at(0);\n"
          << "    *__got = s;\n"
          << "}\n";
    }
    const auto res = r.run_macro(macro);
    INFO(res.message);
    CHECK(res.ok);
    // First 100 rows: act[i] = i % 100 -> sum 0..99 = 4950.
    CHECK(got == doctest::Approx(4950.0));

    std::error_code ec;
    fs::remove(macro, ec);
    fs::remove(path, ec);
}

// The coastd daemon-control façade, driven INTERPRETED over the SAME clang-repl
// path — the real proof the console API is programmatically usable, not just
// printable. It launches a live coastd and, from interpreted code, connects,
// submits, ITERATES s.jobs() (the compiled job_list handle — the fix for the
// std::vector<custom>-across-the-JIT limitation), reads job fields, and reads
// s.info() fields. Any JIT-materialization failure would make an exec !ok.
namespace {
// Force coastd_console_api.o into THIS test binary. It links coast_lib normally
// (not whole-archive like the real coast exe), so without a compiled reference
// the façade object — and the -rdynamic exports the JIT needs — would be dropped.
void* keep_coastd_facade_linked() {
    return reinterpret_cast<void*>(&coast::coastd::print_discovered);
}
} // namespace

// PLATFORM GATE — same as the real-daemon cases in tests/unit/coastd/
// (test_console.cpp, test_recovery.cpp, test_fault_matrix.cpp). This case launches
// a REAL coastd, and the daemon refuses to start where start_token_supported() is
// false (coastd/src/main.cpp; ADR-16): off Linux it exits 1 before binding, so the
// ports file never appears and the case can only fail, meaning "wrong OS". Skip it
// there rather than hand a Mac dev running the documented ctest red. The other
// cases in this file do not need the daemon and run everywhere.
// Spelled __linux__ rather than coastd::start_token_supported() for the reason
// test_console.cpp gives: this TU exercises the console façade and must not see
// coastd/include. Keep this list in sync with start_token_supported().
#if defined(__linux__) || defined(__APPLE__)
constexpr bool skip_off_linux = false;
#else
constexpr bool skip_off_linux = true;
#endif

TEST_CASE("console::repl: coastd façade is programmatically usable from the interpreter"
          * doctest::skip(skip_off_linux))
{
    volatile auto keep = keep_coastd_facade_linked();
    (void)keep;

    // --- launch a real coastd on a throwaway repo (no real coast build) --------
    // Unique per run: a fixed name let two overlapping runs delete each other's.
    std::string root_s = (fs::temp_directory_path() / "coastd_repl_XXXXXX").string();
    REQUIRE(::mkdtemp(root_s.data()) != nullptr);
    const fs::path root = root_s;
    std::error_code ec;
    const fs::path data = root / "data", repo = root / "repo";
    fs::create_directories(data); fs::create_directories(repo);
    REQUIRE(std::system(("git -C '" + repo.string() + "' init -q").c_str()) == 0);
    REQUIRE(std::system(("git -C '" + repo.string() +
                         "' -c user.email=t@t -c user.name=t commit -q --allow-empty -m i").c_str()) == 0);
    { std::ofstream(data / "good.json") << "{\"seed\":1}\n"; }

    // OS-assigned ports (test_daemon.h). This test used the fixed 53940, inside the
    // kernel's ephemeral range, and failed in CI whenever an outbound connection had
    // just used 53940 as its source port.
    const char* argv[] = {COASTD_BIN, "--data-root", data.c_str(), "--repo", repo.c_str(),
                          "--tcp-port", "0", "--udp-port", "0", "-q", nullptr};
    pid_t pid = 0;
    REQUIRE(::posix_spawn(&pid, COASTD_BIN, nullptr, nullptr,
                          const_cast<char* const*>(argv), environ) == 0);
    coastd_test::daemon_guard daemon(pid);   // reaped on every exit, failed REQUIREs included

    // The ports file appears only once the daemon accepts, so the single
    // interpreted connect() below succeeds first try (no retry-in-JIT noise).
    const auto ports = coastd_test::wait_for_ports(data, pid);
    REQUIRE_MESSAGE(ports.has_value(), "daemon exited or never published its ports");
    const std::string tcp = std::to_string(ports->tcp);

    console::repl r;
    REQUIRE_MESSAGE(r.ready(), r.init_error());   // preamble incl. the coastd header parsed

    // Connect from INTERPRETED code — one clean attempt (port is already up).
    const auto c = r.exec_guarded(
        // 60 s for connect and for each reply: the stall limit for a daemon we
        // run ourselves, not a performance budget (CLAUDE.md "Testing").
        "auto s = coastd::session::connect(\"localhost\", " + tcp + ", 60000, 60000);");
    INFO(c.message);
    REQUIRE(c.ok);   // JIT held a coastd::session (custom move-only type) by value

    // submit() returns std::string — resolves fine; keep the id in the interpreter.
    const std::string submit =
        "auto id = s.submit(\"HEAD\", \"" + (data / "good.json").string() + "\", \"gate\", \"\", 10);";
    { const auto e = r.exec_guarded(submit); INFO(e.message); CHECK(e.ok); }

    // Iterate the compiled job_list handle and read fields — the whole point.
    double n = -1.0;
    REQUIRE(bind_double_ptr(r, "__n", &n).ok);
    { const auto e = r.exec_guarded("*__n = (double)s.jobs().size();"); INFO(e.message); CHECK(e.ok); }
    CHECK(n == doctest::Approx(1.0));                 // one job, via job_list::size()

    n = -1.0;
    // range-for over the handle + read a std::string field's length back.
    const auto loop = r.exec_guarded(
        "{ double k = 0; for (auto j : s.jobs()) if (j.ref == \"HEAD\") k += (double)j.id.size(); *__n = k; }");
    INFO(loop.message);
    CHECK(loop.ok);
    CHECK(n > 0.0);                                   // iterated, matched ref, read id.size()

    n = -1.0;
    { const auto e = r.exec_guarded("*__n = (double)s.info().slots;"); INFO(e.message); CHECK(e.ok); }
    CHECK(n >= 1.0);                                  // s.info() -> host_info by value, field read

    CHECK(r.exec_guarded("coastd::print_discovered(1);").ok);   // no-daemon void path too

    CHECK(daemon.stop());   // a daemon that needs SIGKILL failed its shutdown contract
    fs::remove_all(root, ec);
}

// The campaign meta-loader through the `camp` alias the preamble declares. Like the
// simc case above, referencing a coast::campaign symbol from C++ pulls
// campaign_api.o into this (non-whole-archive) test exe so the JIT can resolve it;
// the real `coast` exe gets it via --whole-archive.
//
// The point of driving it from the interpreter rather than only from C++: the API is
// held across cells as a shared_ptr and a by-value selection, which is exactly the
// shape that breaks when a type's vtable or inline members do not survive the JIT
// boundary.
TEST_CASE("console::repl: the campaign meta-loader is reachable as camp::")
{
    std::error_code ec;
    const fs::path  root =
        fs::temp_directory_path() / ("coast_camp_repl_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);

    // A 2x2 factorial: two swept ngm25 leaves, plus a constant leaf that must NOT
    // become an axis.
    int n = 0;
    for (const char* plr : {"0.01", "0.05"}) {
        for (const char* li : {"1.0", "5.0"}) {
            const fs::path d = root / ("run_" + std::to_string(n++));
            fs::create_directories(d);
            std::ofstream(d / "config.json")
                << R"({"ngm25":{"neuron_groups":{"L4-mnist":{"primary_learning_rate":)" << plr
                << R"(,"local_inhibition_strength":)" << li << R"(}}},"sim":{"seed":7}})";
        }
    }

    // C++ ref: pull the object in, and check the compiled path agrees with the
    // interpreted one below.
    REQUIRE(coast::campaign::open(root)->size() == 4);

    console::repl r;
    REQUIRE_MESSAGE(r.ready(), r.init_error());   // preamble incl. campaign_api.h parsed

    double v = -1.0;
    REQUIRE(bind_double_ptr(r, "__v", &v).ok);

    { const auto e = r.exec_guarded("auto c = camp::open(\"" + root.string() + "\");");
      INFO(e.message); REQUIRE(e.ok); }                    // shared_ptr held across cells

    { const auto e = r.exec_guarded("*__v = (double)c->size();");
      INFO(e.message); CHECK(e.ok); }
    CHECK(v == doctest::Approx(4.0));

    v = -1.0;
    { const auto e = r.exec_guarded("*__v = (double)c->axes().size();");
      INFO(e.message); CHECK(e.ok); }
    CHECK(v == doctest::Approx(2.0));                      // the constant /sim/seed is not one

    // The query the whole feature exists for, held as a by-value selection.
    v = -1.0;
    { const auto e = r.exec_guarded(
          "auto sel = c->where(\"primary_learning_rate\", 0.01)"
          "             .where(\"local_inhibition_strength\", 5.0);");
      INFO(e.message); REQUIRE(e.ok); }
    { const auto e = r.exec_guarded("*__v = (double)sel.size();");
      INFO(e.message); CHECK(e.ok); }
    CHECK(v == doctest::Approx(1.0));

    // Iteration + the x-vector extraction a plot is built from.
    v = -1.0;
    { const auto e = r.exec_guarded(
          "{ auto s2 = c->where(\"local_inhibition_strength\", 5.0)"
          "              .sort_by(\"primary_learning_rate\");"
          "  auto xs = s2.values(\"primary_learning_rate\");"
          "  double acc = 0; for (const auto& rr : s2) acc += rr.number(\"primary_learning_rate\");"
          "  *__v = acc + xs.size(); }");
      INFO(e.message); CHECK(e.ok); }
    CHECK(v == doctest::Approx(0.01 + 0.05 + 2.0));

    // A bad value throws across the JIT boundary and is reported, not fatal — and the
    // message names the values that do exist.
    const auto bad = r.exec_guarded("(void)c->where(\"primary_learning_rate\", 0.02).size();");
    CHECK_FALSE(bad.ok);
    CHECK(bad.message.find("0.05") != std::string::npos);

    fs::remove_all(root, ec);
}
