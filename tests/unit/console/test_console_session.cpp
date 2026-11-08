// console::session — the threaded interpreter wrapper: async command queue,
// stdout/stderr capture into output_line records, multi-file open() (no file is
// required to run), and the exception-guarded error surface, all off the calling
// thread.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "console/session.h"
#include "gui_plot/console_plot.h"
#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include <hdf5.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path make_test_file(hdf5::executor& exec, const char* fname, std::uint64_t n,
                        float mod)
{
    const fs::path  p = fs::temp_directory_path() / fname;
    std::error_code ec;
    fs::remove(p, ec);
    exec.submit_sync([&] {
        const hid_t fid = H5Fcreate(p.string().c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(fid >= 0);
        const hid_t md = H5Gcreate2(fid, "metadata", H5P_DEFAULT, H5P_DEFAULT,
                                    H5P_DEFAULT);
        REQUIRE(hdf5::write_attr(md, "type", std::string_view{"observe_log"}) >= 0);
        REQUIRE(hdf5::write_attr(md, "schema_version", std::string_view{"1.0"}) >= 0);
        H5Gclose(md);
        const hsize_t dims[1]  = {n};
        const hsize_t chunk[1] = {std::min<std::uint64_t>(256, n)}; // chunk <= extent
        const hid_t   sp   = H5Screate_simple(1, dims, nullptr);
        const hid_t   dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        std::vector<float> vals(n);
        for (std::uint64_t i = 0; i < n; ++i) {
            vals[i] = static_cast<float>(std::uint64_t(i) % std::uint64_t(mod));
        }
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

bool any_line_contains(const std::vector<console::output_line>& lines,
                       console::output_line::kind k, std::string_view needle)
{
    for (const auto& l : lines) {
        if (l.k == k && l.text.find(needle) != std::string_view::npos) return true;
    }
    return false;
}

} // namespace

TEST_CASE("console::session: open, run macro over data, capture stdout")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec, "h5b_sess_a.h5", 2000, 100.0f);

    console::session sess(exec);
    const std::string v = sess.open_file(path);
    sess.wait_idle();
    CHECK(sess.current_file() == path.string());

    // Macro prints a computed sum; the session captures it off the process fd.
    sess.submit(
        "{ auto d = " + v + "->dataset(\"act\"); double s = 0;"
        "  for (auto row : d.scan()) s += row.at(0);"
        "  printf(\"SUM=%.1f\\n\", s); }");
    sess.wait_idle();

    const auto out = sess.drain_output();
    CHECK(any_line_contains(out, console::output_line::kind::input, "SUM")); // echoed input
    CHECK(any_line_contains(out, console::output_line::kind::output, "SUM=99000.0"));

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::session: uncaught macro exception surfaces as an error line")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec, "h5b_sess_b.h5", 500, 50.0f);

    console::session sess(exec);
    const std::string v = sess.open_file(path);
    sess.submit("(void)" + v + "->dataset(\"nope\").rows();");
    sess.wait_idle();

    const auto out = sess.drain_output();
    CHECK(any_line_contains(out, console::output_line::kind::error, "nope"));

    // Still usable.
    sess.submit("printf(\"alive=%llu\\n\", (unsigned long long)" + v +
                "->dataset(\"act\").rows());");
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::output,
                            "alive=500"));

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::session: bind_dataset escapes the dataset name (no code injection)")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec, "h5b_sess_inject.h5", 500, 50.0f);

    console::session sess(exec);
    const std::string v = sess.open_file(path);
    sess.wait_idle();
    (void)sess.drain_output();

    // A crafted HDF5 dataset name: spliced raw into the submitted statement it
    // would close the dataset("...") literal and execute an injected printf.
    // "act" is a real dataset, so a pre-escaping (raw) bind would succeed on it
    // and then run the payload. bind_dataset must escape the name so the whole
    // string is one (unknown) dataset lookup instead.
    const std::string malicious = "act\"); printf(\"PWNED_%d\", 1337); (void)(\"";
    sess.bind_dataset("d_inject", v, malicious);
    sess.wait_idle();

    // The payload must NOT have run. Its output would be the resolved "PWNED_1337";
    // the escaped name only ever carries the literal "PWNED_%d", so this exact
    // string appears iff the printf executed.
    const auto out = sess.drain_output();
    CHECK_FALSE(any_line_contains(out, console::output_line::kind::output, "PWNED_1337"));

    // And the interpreter is unharmed — a normal, properly-escaped bind still works.
    sess.bind_dataset("d_ok", v, "act");
    sess.submit("printf(\"rows=%llu\\n\", (unsigned long long)d_ok.rows());");
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::output,
                            "rows=500"));

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::session: declarations persist across submits, std::cout captured")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec, "h5b_sess_persist.h5", 300, 100.0f);

    console::session sess(exec);
    sess.open_file(path);
    sess.submit("#include <iostream>");
    sess.submit("int carry = 41;");               // must persist to the next submit
    sess.wait_idle();
    (void)sess.drain_output();

    // Uses a variable defined in a PRIOR submit, and std::cout (not printf).
    sess.submit("std::cout << \"carry=\" << (carry + 1) << std::endl;");
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::output,
                            "carry=42"));

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::session: reset rebuilds the interpreter, re-opens the files")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec, "h5b_sess_reset.h5", 400, 100.0f);

    console::session sess(exec);
    const std::string v = sess.open_file(path);
    sess.submit("int carry = 41;"); // a user definition that reset must drop
    sess.wait_idle();
    (void)sess.drain_output();

    sess.reset();
    sess.wait_idle();
    {
        const auto out = sess.drain_output();
        CHECK(any_line_contains(out, console::output_line::kind::info, "console reset"));
    }

    // The prior definition is gone (fresh interpreter): referencing it now fails.
    // The compiler diagnostic (naming 'carry') is captured onto an output line;
    // the error line carries the generic "Parsing failed".
    sess.submit("printf(\"c=%d\\n\", carry);");
    sess.wait_idle();
    {
        const auto out = sess.drain_output();
        CHECK(any_line_contains(out, console::output_line::kind::output, "carry"));
        CHECK(any_line_contains(out, console::output_line::kind::error, "Parsing failed"));
    }

    // The managed file variable is re-opened by reset, so it is still usable.
    CHECK(sess.current_file() == path.string());
    sess.submit("printf(\"rows=%llu\\n\", (unsigned long long)" + v +
                "->dataset(\"act\").rows());");
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::output,
                            "rows=400"));

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::session: interpreter-created plots survive a reset (JIT arena)")
{
    // A figure built in a macro via std::make_shared<...> has a control block whose
    // _M_dispose/_M_destroy are VIRTUAL, and whose vtable was emitted into that
    // interpreter's JIT module. reset() destroys the interpreter. The retaining JIT
    // arena (see repl.cpp) keeps the emitted code mapped, so the figure stays fully
    // usable afterwards and releases cleanly -- no registry clearing needed, and
    // plots are no longer collateral damage of a reset.
    //
    // With the arena disabled (COAST_REPL_RETAIN_JIT=0) the free_resources() below
    // dispatches through the unmapped vtable and this test dies with SIGSEGV.
    hdf5::executor exec;
    const fs::path path = make_test_file(exec, "h5b_sess_plotreset.h5", 200, 100.0f);

    console::session sess(exec);
    sess.open_file(path);
    // Create + show a figure from interpreter-JIT'd code -> registers in plot_manager.
    sess.submit("#include \"gui_plot/console_plot.h\"");
    sess.submit("auto _h = std::make_shared<coast::plot::histogram>(\"act\"); _h->show();");
    sess.wait_idle();
    (void)sess.drain_output();
    CHECK(coast::plot::plot_count() == 1);

    // Reset destroys interpreter #1 and its JIT module's owner. The figure SURVIVES:
    // the arena still has its code mapped.
    sess.reset();
    sess.wait_idle();
    (void)sess.drain_output();
    CHECK(coast::plot::plot_count() == 1);

    // App quit: this releases the figure, dispatching into the retired module's
    // vtable. Without the arena this is the SIGSEGV.
    coast::plot::free_resources();
    CHECK(coast::plot::plot_count() == 0);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::session: interpreter-created plots survive session destruction")
{
    // Same hazard via the teardown path nothing guards: ~session joins the worker
    // thread, whose thread_main runs the final interp.reset(). The shipping GUI
    // happens to call free_resources() before joining that thread, but nothing
    // enforces that ordering -- and here nothing clears the registry at all.
    hdf5::executor exec;
    const fs::path path = make_test_file(exec, "h5b_sess_plotdtor.h5", 200, 100.0f);

    {
        console::session sess(exec);
        sess.open_file(path);
        sess.submit("#include \"gui_plot/console_plot.h\"");
        sess.submit("auto _h = std::make_shared<coast::plot::histogram>(\"act\"); _h->show();");
        sess.wait_idle();
        (void)sess.drain_output();
        CHECK(coast::plot::plot_count() == 1);
    } // ~session -> thread_main -> interp.reset(); no release of any kind

    // The figure outlived its interpreter and is still registered and intact.
    CHECK(coast::plot::plot_count() == 1);

    coast::plot::free_resources();
    CHECK(coast::plot::plot_count() == 0);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::session: runs with no file open, and reset still works")
{
    // The console is usable from the start: no file needs to be opened first.
    hdf5::executor  exec;
    console::session sess(exec);
    sess.submit("printf(\"answer=%d\\n\", 6 * 7);");
    sess.wait_idle();
    CHECK(sess.current_file().empty());
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::output,
                            "answer=42"));

    // Reset with no file rebuilds the interpreter and remains usable.
    sess.reset();
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::info,
                            "console reset"));
    sess.submit("printf(\"still=%d\\n\", 1);");
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::output,
                            "still=1"));
}

TEST_CASE("console::session: several files are open at once, each in its own variable")
{
    hdf5::executor exec;
    const fs::path a = make_test_file(exec, "h5b_sess_c.h5", 300, 100.0f);
    const fs::path b = make_test_file(exec, "h5b_sess_d.h5", 700, 100.0f);

    console::session sess(exec);

    const std::string va = sess.open_file(a);
    const std::string vb = sess.open_file(b);
    sess.wait_idle();

    // Distinct variables were assigned, and both files show up as open.
    CHECK(va != vb);
    CHECK(sess.current_file() == b.string()); // most-recently opened
    CHECK(sess.open_files().size() == 2);

    // BOTH handles are live simultaneously — the whole point of dropping the
    // single global `h5`: read from a and b in the same statement.
    sess.submit("printf(\"a=%llu b=%llu\\n\", (unsigned long long)" + va +
                "->dataset(\"act\").rows(), (unsigned long long)" + vb +
                "->dataset(\"act\").rows());");
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::output,
                            "a=300 b=700"));

    // Re-opening an already-open path returns the same variable and does not
    // add a second entry.
    const std::string va2 = sess.open_file(a);
    CHECK(va2 == va);
    CHECK(sess.open_files().size() == 2);

    std::error_code ec;
    fs::remove(a, ec);
    fs::remove(b, ec);
}

TEST_CASE("console::session: same-stem files in different dirs get distinct variables")
{
    // Two files whose stems collide ("run") must not share one variable — the
    // second is disambiguated with a numeric suffix, and both stay usable.
    hdf5::executor  exec;
    const fs::path  base = fs::temp_directory_path() / "h5b_sess_collision";
    std::error_code ec;
    fs::create_directories(base / "a", ec);
    fs::create_directories(base / "b", ec);
    const fs::path a = make_test_file(exec, "h5b_sess_collision/a/run.h5", 100, 100.0f);
    const fs::path b = make_test_file(exec, "h5b_sess_collision/b/run.h5", 200, 100.0f);

    console::session  sess(exec);
    const std::string va = sess.open_file(a);
    const std::string vb = sess.open_file(b);
    sess.wait_idle();

    CHECK(va == "f_run");   // stem-derived base name
    CHECK(vb == "f_run_2"); // collision disambiguated
    CHECK(sess.open_files().size() == 2);

    // Each variable resolves to its own file.
    sess.submit("printf(\"a=%llu b=%llu\\n\", (unsigned long long)" + va +
                "->dataset(\"act\").rows(), (unsigned long long)" + vb +
                "->dataset(\"act\").rows());");
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::output,
                            "a=100 b=200"));

    fs::remove_all(base, ec);
}

TEST_CASE("console::session: reserve_var shares one namespace with file handles")
{
    // A caller that builds its own console variable (e.g. the browser binding a
    // dataset) reserves through the SAME registry as open_file(), so it cannot
    // collide with a file handle or another reserved name. The old bug: a file
    // run.h5 (-> f_run) and a dataset "foo/run" (-> f_run) both landed on f_run
    // with no cross-check, and the injected `auto f_run = f_run->dataset(...)`
    // self-referenced and failed to compile.
    hdf5::executor   exec;
    console::session sess(exec);
    const fs::path   path = make_test_file(exec, "h5b_sess_reserve.h5", 100, 100.0f);

    const std::string fv = sess.open_file(path); // f_h5b_sess_reserve
    CHECK(fv.rfind("f_", 0) == 0);

    // A base that collides with the live file handle is disambiguated, not dup'd.
    CHECK(sess.reserve_var(fv) == fv + "_2");
    CHECK(sess.reserve_var(fv) == fv + "_3");
    // A non-colliding base is returned unchanged, and repeats disambiguate too.
    CHECK(sess.reserve_var("ng1_out") == "ng1_out");
    CHECK(sess.reserve_var("ng1_out") == "ng1_out_2");

    sess.wait_idle();
    (void)sess.drain_output();

    // Reset re-opens files but drops transient reservations: the file handle
    // stays claimed, the dataset-bind names free up.
    sess.reset();
    sess.wait_idle();
    (void)sess.drain_output();

    CHECK(sess.open_files().size() == 1);          // file re-opened
    CHECK(sess.reserve_var(fv) == fv + "_2");       // file handle still reserved
    CHECK(sess.reserve_var("ng1_out") == "ng1_out"); // transient name freed

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::session: block_var / release_var manage the name registry")
{
    hdf5::executor   exec;
    console::session sess(exec);

    // block_var claims an exact name; a second claim of the same name fails, and
    // the allocator then skips it.
    CHECK(sess.block_var("myvar"));                 // claimed
    CHECK_FALSE(sess.block_var("myvar"));           // already taken
    CHECK(sess.reserve_var("myvar") == "myvar_2");  // allocator avoids the blocked name

    // release_var frees a held name; releasing again reports "not held".
    CHECK(sess.release_var("myvar"));               // was held
    CHECK_FALSE(sess.release_var("myvar"));         // now gone
    CHECK(sess.reserve_var("myvar") == "myvar");    // available again

    // release_var also frees a reserve_var()-allocated transient name.
    const std::string a = sess.reserve_var("ds_act"); // "ds_act"
    CHECK(a == "ds_act");
    CHECK(sess.release_var(a));
    CHECK(sess.reserve_var("ds_act") == "ds_act");  // reusable, no _2 suffix

    // A live file handle cannot be released out from under its variable.
    const fs::path    path = make_test_file(exec, "h5b_sess_block.h5", 50, 50.0f);
    const std::string fv   = sess.open_file(path);
    CHECK_FALSE(sess.release_var(fv));              // refused: still an open file
    sess.wait_idle();
    (void)sess.drain_output();
    CHECK(sess.open_files().size() == 1);           // handle untouched by the refusal

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console::session: a failed open frees the reservation for retry")
{
    // open() on a missing file throws in compiled code; the session must catch
    // it, report it, AND release the name reservation — at BOTH levels: the
    // session's own bookkeeping AND the interpreter's declaration table. If the
    // half-initialized `auto <var> = open(...)` declaration lingers in the
    // interpreter, retrying the same name fails to compile ("redefinition"), so
    // a user who fixes a typo'd path and re-runs the cell is permanently stuck.
    hdf5::executor  exec;
    console::session sess(exec);
    const char*     fname = "h5b_sess_retry_good.h5";
    const fs::path  path  = fs::temp_directory_path() / fname;
    std::error_code ec;
    fs::remove(path, ec); // make sure it does not exist on the first attempt

    const std::string v1 = sess.open_file(path);
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::error,
                            "open failed"));
    // The reservation was rolled back: nothing is listed as open.
    CHECK(sess.open_files().empty());
    CHECK(sess.current_file().empty());

    // The file now exists (the user fixed the path / it became available). The
    // retry reuses the SAME name (no _2 suffix), proving the session freed it...
    make_test_file(exec, fname, 640, 100.0f);
    const std::string v2 = sess.open_file(path);
    CHECK(v2 == v1);
    sess.wait_idle();
    (void)sess.drain_output();
    CHECK(sess.open_files().size() == 1); // opened this time

    // ...and the handle behind that reused name is actually usable — this is
    // what a leftover ghost declaration would break (the redefinition would fail
    // to parse, so open_files() above would already be empty and this read would
    // never fire). Reading a real dataset proves the interpreter was cleaned up.
    sess.submit("printf(\"retry_rows=%llu\\n\", (unsigned long long)" + v2 +
                "->dataset(\"act\").rows());");
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(),
                            console::output_line::kind::output, "retry_rows=640"));

    fs::remove(path, ec);
}

TEST_CASE("console::session: .I adds a runtime include path (no file needed)")
{
    hdf5::executor exec;

    // A header in a directory that is NOT on the interpreter's default -I.
    const fs::path dir = fs::temp_directory_path() / "console_inc_test";
    std::error_code ec;
    fs::create_directories(dir, ec);
    {
        std::ofstream f(dir / "console_extra.h");
        f << "#pragma once\ninline int extra_answer() { return 99; }\n";
    }

    console::session sess(exec);
    sess.add_include_path(dir.string());
    sess.submit("#include <console_extra.h>");
    sess.submit("printf(\"extra=%d\\n\", extra_answer());");
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::output,
                            "extra=99"));

    fs::remove_all(dir, ec);
}

TEST_CASE("console::session: .x macro file")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec, "h5b_sess_e.h5", 100, 100.0f);

    const fs::path macro = fs::temp_directory_path() / "sess_macro.cpp";
    {
        std::ofstream f(macro);
        f << "void sess_macro() {\n"
          << "    auto file = open(\"" << path.string() << "\");\n"
          << "    printf(\"macro rows=%llu\\n\", "
             "(unsigned long long)file->dataset(\"act\").rows());\n"
          << "}\n";
    }

    console::session sess(exec);
    sess.run_macro(macro);
    sess.wait_idle();
    CHECK(any_line_contains(sess.drain_output(), console::output_line::kind::output,
                            "macro rows=100"));

    std::error_code ec;
    fs::remove(macro, ec);
    fs::remove(path, ec);
}
