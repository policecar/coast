#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstdlib>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include "campaign/campaign_api.h"
// For the one fixture that must come from the REAL producer rather than a string
// literal — see campaign.manifest: a manifest from coast's own writer.
#include "json_cfg/json_loader.h"
#include "sim_core/run_manifest.h"

namespace fs = std::filesystem;
using namespace coast::campaign;

namespace {

// A unique scratch directory that cleans itself up. pid + counter (no randomness
// needed) so parallel ctest runs of this binary cannot collide.
struct scratch {
    fs::path root;
    scratch()
    {
        static std::atomic<int> ctr{0};
        root = fs::temp_directory_path() /
               ("coast_campaign_" + std::to_string(::getpid()) + "_" +
                std::to_string(ctr.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch()
    {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

void write_file(const fs::path& p, const std::string& text)
{
    fs::create_directories(p.parent_path());
    std::ofstream out(p);
    out << text;
}

// A config shaped like a real coast config: the swept ngm25 leaves alongside the
// dead ngm2/ngm3 defaults that carry the SAME leaf names. `plr_text` and `li_text`
// go in verbatim so a test can plant an exact numeric literal (e.g. a float32
// round-trip) rather than whatever a double formatter would produce.
std::string config_text(const std::string& plr_text, const std::string& li_text,
                        const std::string& run_dir)
{
    return R"({
  "ngm2":  { "dendrites":     { "default":   { "default_primary_learning_rate": 0.01 } },
             "neuron_groups": { "default":   { "default_local_inhibition_strength": 5.0 } } },
  "ngm3":  { "neuron_groups": { "default":   { "primary_learning_rate": 0.009999999776482582,
                                               "local_inhibition_strength": 5.0 } } },
  "ngm25": { "neuron_groups": { "L4-mnist": { "primary_learning_rate": )" +
           plr_text + R"(,
                                               "local_inhibition_strength": )" +
           li_text + R"( } } },
  "sim": { "nodes": [ { "id": 0, "entity_type": "ngm25_neuron_group_gpu" } ] },
  "observe": { "output_dir": "./observe" },
  "snapshot_dir": ")" +
           run_dir + R"(/snapshots"
})";
}

std::string manifest_text(const std::string& exit_reason, std::uint64_t done)
{
    return R"({"schema_version":"1","coast_git_sha":"deadbeef","exit_reason":")" +
           exit_reason + R"(","hostname":"testbox","n_steps_requested":600000,)" +
           R"("n_steps_completed":)" + std::to_string(done) +
           R"(,"wall_clock_seconds":359.25,"started_at_utc":"2026-07-31T19:11:06Z"})";
}

// A 3x3 factorial over primary_learning_rate x local_inhibition_strength, laid out
// the way coastd writes one: <root>/<run id>/{config.json,coast_runtime.json}.
void build_factorial(const fs::path& root)
{
    const char* plr[] = {"0.01", "0.05", "0.1"};
    const char* li[]  = {"1.0", "5.0", "10.0"};
    int         n     = 0;
    for (const char* p : plr) {
        for (const char* l : li) {
            const std::string id = "run_" + std::to_string(n++);
            write_file(root / id / "config.json", config_text(p, l, (root / id).string()));
            write_file(root / id / "coast_runtime.json", manifest_text("completed_n_steps", 600000));
        }
    }
}

} // namespace

TEST_CASE("campaign.axes: only leaves that differ become axes")
{
    scratch s;
    build_factorial(s.root);

    const auto c = open(s.root);
    CHECK(c->size() == 9);
    CHECK(c->issues().empty());

    // Exactly the two swept ngm25 leaves. The ngm2/ngm3 copies carry the same leaf
    // names but are identical across runs, so they never enter the namespace — this
    // is what makes the short names usable.
    REQUIRE(c->axes().size() == 2);

    const axis* plr = c->find_axis("primary_learning_rate");
    REQUIRE(plr != nullptr);
    CHECK(plr->pointer == "/ngm25/neuron_groups/L4-mnist/primary_learning_rate");
    REQUIRE(plr->values.size() == 3);
    CHECK(plr->values[0].num == doctest::Approx(0.01));
    CHECK(plr->values[2].num == doctest::Approx(0.1));

    const axis* li = c->find_axis("local_inhibition_strength");
    REQUIRE(li != nullptr);
    CHECK(li->pointer == "/ngm25/neuron_groups/L4-mnist/local_inhibition_strength");
    CHECK(li->values.size() == 3);

    // Constant leaves are absent from the axis set but still reachable by pointer.
    CHECK(c->find_axis("entity_type") == nullptr);
    CHECK(c->runs()[0].config_string("/sim/nodes/0/entity_type") == "ngm25_neuron_group_gpu");
}

TEST_CASE("campaign.axes: output locations are denied, not discovered")
{
    scratch s;
    build_factorial(s.root);

    // /snapshot_dir differs in all 9 runs (coastd writes each run's own directory
    // into its config) but is an output location, not a hyperparameter.
    const auto c = open(s.root);
    CHECK(c->find_axis("/snapshot_dir") == nullptr);
    CHECK(c->find_axis("snapshot_dir") == nullptr);

    // ...and a caller can extend the deny-list.
    scan_options o;
    o.ignore_pointers.push_back("/ngm25/neuron_groups/L4-mnist/local_inhibition_strength");
    const auto c2 = open(s.root, o);
    CHECK(c2->axes().size() == 1);
    CHECK(c2->find_axis("local_inhibition_strength") == nullptr);
    CHECK(c2->find_axis("primary_learning_rate") != nullptr);
}

TEST_CASE("campaign.query: filtering by swept parameters")
{
    scratch s;
    build_factorial(s.root);
    const auto c = open(s.root);

    CHECK(c->all().size() == 9);
    CHECK(c->where("primary_learning_rate", 0.01).size() == 3);

    const auto sel = c->where("primary_learning_rate", 0.01).where("local_inhibition_strength", 5.0);
    REQUIRE(sel.size() == 1);
    CHECK(sel[0].number("primary_learning_rate") == doctest::Approx(0.01));
    CHECK(sel[0].number("local_inhibition_strength") == doctest::Approx(5.0));

    // The full JSON Pointer is always accepted in place of the short name.
    CHECK(c->where("/ngm25/neuron_groups/L4-mnist/primary_learning_rate", 0.05).size() == 3);

    // Escape hatch: anything the axis filters cannot express.
    CHECK(c->where([](const run& r) { return r.number("local_inhibition_strength") > 4.0; })
              .size() == 6);

    // Range-for yields run handles.
    int seen = 0;
    for (const run& r : c->where("local_inhibition_strength", 10.0)) {
        CHECK(r.number("local_inhibition_strength") == doctest::Approx(10.0));
        ++seen;
    }
    CHECK(seen == 3);
}

TEST_CASE("campaign.query: tolerance absorbs a float32 round-trip")
{
    scratch s;
    // The value coast stores when the config round-tripped through float32. An exact
    // `== 0.01` would silently match nothing.
    write_file(s.root / "a" / "config.json",
               config_text("0.009999999776482582", "5.0", (s.root / "a").string()));
    write_file(s.root / "b" / "config.json", config_text("0.05", "5.0", (s.root / "b").string()));

    const auto c = open(s.root);
    REQUIRE(c->axes().size() == 1);
    CHECK(c->where("primary_learning_rate", 0.01).size() == 1);
    // ...while values a decade apart stay firmly distinct.
    CHECK(c->where("primary_learning_rate", 0.05).size() == 1);
}

TEST_CASE("campaign.query: a value that matches nothing is a message, not an empty result")
{
    scratch s;
    build_factorial(s.root);
    const auto c = open(s.root);

    std::string msg;
    try {
        (void)c->where("primary_learning_rate", 0.02);
    } catch (const std::exception& e) {
        msg = e.what();
    }
    REQUIRE(!msg.empty());
    // The message must name the values that DO exist — that is the whole point of
    // snapping to the discovered set rather than filtering silently.
    CHECK(msg.find("0.01") != std::string::npos);
    CHECK(msg.find("0.05") != std::string::npos);
    CHECK(msg.find("0.1") != std::string::npos);

    // An unknown axis name likewise lists the axes that exist.
    msg.clear();
    try {
        (void)c->where("learning_rate", 0.01);
    } catch (const std::exception& e) {
        msg = e.what();
    }
    REQUIRE(!msg.empty());
    CHECK(msg.find("primary_learning_rate") != std::string::npos);
}

TEST_CASE("campaign.names: a genuine collision qualifies from the root section down")
{
    scratch s;
    // Two DIFFERENT sections both sweep `primary_learning_rate`, so the short name is
    // ambiguous and must be qualified.
    for (int i = 0; i < 2; ++i) {
        const std::string a = i == 0 ? "0.01" : "0.05";
        const std::string b = i == 0 ? "0.2" : "0.3";
        write_file(s.root / ("r" + std::to_string(i)) / "config.json",
                   R"({"ngm25":{"neuron_groups":{"L4-mnist":{"primary_learning_rate":)" + a +
                       R"(}}},"ngm3":{"neuron_groups":{"default":{"primary_learning_rate":)" + b +
                       R"(}}}})");
    }

    const auto c = open(s.root);
    REQUIRE(c->axes().size() == 2);
    CHECK(c->find_axis("primary_learning_rate") == nullptr); // ambiguous: not assigned
    CHECK(c->find_axis("ngm25/primary_learning_rate") != nullptr);
    CHECK(c->find_axis("ngm3/primary_learning_rate") != nullptr);
    // The pointer resolves either way.
    CHECK(c->find_axis("/ngm25/neuron_groups/L4-mnist/primary_learning_rate") != nullptr);
    CHECK(c->where("ngm25/primary_learning_rate", 0.01).size() == 1);
}

TEST_CASE("campaign.names: qualification works at every pointer depth")
{
    scratch s;
    // TWO-component pointers. The candidate loop used to skip the root-plus-tail
    // candidate entirely at this depth, so both axes fell back to raw pointers and
    // the header's own headline example never happened. The old collision test used
    // 4-component pointers, where the loop did produce a qualified name, and passed.
    for (int i = 0; i < 2; ++i) {
        write_file(s.root / ("r" + std::to_string(i)) / "config.json",
                   R"({"ngm2":{"lr":)" + std::string(i == 0 ? "0.01" : "0.05") +
                       R"(},"ngm25":{"lr":)" + std::string(i == 0 ? "0.2" : "0.3") + R"(}})");
    }
    const auto c = open(s.root);
    REQUIRE(c->axes().size() == 2);
    CHECK(c->find_axis("lr") == nullptr); // ambiguous, so not assigned bare
    CHECK(c->find_axis("ngm2/lr") != nullptr);
    CHECK(c->find_axis("ngm25/lr") != nullptr);
    CHECK(c->where("ngm25/lr", 0.2).size() == 1);
}

TEST_CASE("campaign.names: a name freed by a deeper match is re-offered")
{
    scratch s;
    // A shallow and a deep axis both want "lr". The deep one can qualify as
    // "ngm25/lr"; once it does, "lr" is unambiguous and must go to the shallow one.
    // A single forward pass never revisited the level it had already passed, so the
    // shallow axis fell back to its pointer and the two names ended up differing only
    // by a leading slash.
    for (int i = 0; i < 2; ++i) {
        const std::string a = i == 0 ? "0.01" : "0.05";
        const std::string b = i == 0 ? "0.2" : "0.3";
        write_file(s.root / ("r" + std::to_string(i)) / "config.json",
                   R"({"ngm25":{"lr":)" + a + R"(,"groups":{"L4":{"lr":)" + b + R"(}}}})");
    }
    const auto c = open(s.root);
    REQUIRE(c->axes().size() == 2);
    CHECK(c->find_axis("lr") != nullptr);
    CHECK(c->find_axis("lr")->pointer == "/ngm25/lr");
    CHECK(c->find_axis("ngm25/L4/lr") != nullptr);
    CHECK(c->find_axis("ngm25/L4/lr")->pointer == "/ngm25/groups/L4/lr");
}

TEST_CASE("campaign.robustness: incomplete and unreadable runs")
{
    scratch s;
    build_factorial(s.root);

    // A run still in flight: config but no manifest yet.
    write_file(s.root / "inflight" / "config.json",
               config_text("0.01", "5.0", (s.root / "inflight").string()));
    // A run that died.
    write_file(s.root / "failed" / "config.json",
               config_text("0.05", "5.0", (s.root / "failed").string()));
    write_file(s.root / "failed" / "coast_runtime.json", manifest_text("child_failed", 1234));
    // A corrupt config.
    write_file(s.root / "corrupt" / "config.json", "{ this is not json");
    // A non-run file next to the runs (campaign notes).
    write_file(s.root / "NOTES.md", "# sweep notes\n");

    const auto c = open(s.root);
    CHECK(c->size() == 11); // 9 + inflight + failed; corrupt excluded

    // The corrupt run is excluded, but never silently: the issue names the run, the
    // file and the parse failure.
    REQUIRE(c->issues().size() == 1);
    CHECK(c->issues()[0].dir.filename() == "corrupt");
    CHECK(c->issues()[0].file == "config.json");
    CHECK(c->issues()[0].run_dropped);
    CHECK(c->issues()[0].message.find("parse error") != std::string::npos);
    CHECK(c->summary().find("corrupt") != std::string::npos);

    const auto inflight = c->where([](const run& r) { return r.id() == "inflight"; });
    REQUIRE(inflight.size() == 1);
    CHECK_FALSE(inflight[0].info().present);
    CHECK_FALSE(inflight[0].complete());
    // A run with no manifest yet is NORMAL, not an error — it must be distinguishable
    // from one whose manifest is damaged (see campaign.manifest_errors below).
    CHECK_FALSE(inflight[0].info().malformed());

    const auto failed = c->where([](const run& r) { return r.id() == "failed"; });
    REQUIRE(failed.size() == 1);
    CHECK(failed[0].info().present);
    CHECK(failed[0].info().exit_reason == "child_failed");
    CHECK(failed[0].info().steps_completed == 1234);
    CHECK_FALSE(failed[0].complete());

    const auto done = c->where([](const run& r) { return r.complete(); });
    CHECK(done.size() == 9);
    CHECK(done[0].info().steps_requested == 600000);
    CHECK(done[0].info().wall_clock_seconds == doctest::Approx(359.25));
}

TEST_CASE("campaign.manifest_errors: a wrong-typed field is reported, never coerced")
{
    scratch s;
    const auto plant = [&](const char* id, const std::string& manifest_body) {
        write_file(s.root / id / "config.json", config_text("0.01", "5.0", (s.root / id).string()));
        write_file(s.root / id / "coast_runtime.json", manifest_body);
    };

    // A well-formed baseline, so the campaign has an axis and the others have company.
    plant("good", manifest_text("completed_n_steps", 600000));
    // The writer types the step counts as uint64_t, so a float here means the file is
    // not ours. It must NOT be quietly converted (that conversion was also UB).
    plant("floaty", R"({"n_steps_completed": -1.5, "exit_reason": "completed_n_steps"})");
    // Wrong type on a string field.
    plant("numeric_reason", R"({"exit_reason": 42, "n_steps_completed": 10})");
    // Not an object at all.
    plant("array", R"([1, 2, 3])");
    // Unparseable.
    plant("garbage", "{ not json");
    // A manifest that omits fields entirely is LEGITIMATE — the schema is additive.
    plant("sparse", R"({"exit_reason": "completed_n_steps"})");
    // wall_clock_seconds is the only field read as a plain number rather than an
    // unsigned, and nothing exercised its type check.
    plant("stringy_clock", R"({"wall_clock_seconds": "fast", "exit_reason": "x"})");

    const auto c = open(s.root);
    CHECK(c->size() == 7); // every run kept: a bad manifest costs provenance, not data

    const auto info_of = [&](const char* id) {
        const auto sel = c->where([&](const run& r) { return r.id() == id; });
        REQUIRE(sel.size() == 1);
        return sel[0].info();
    };

    // The float is reported, names the field and the offending value, and is NOT
    // silently turned into a step count.
    const manifest f = info_of("floaty");
    CHECK(f.malformed());
    CHECK(f.error.find("n_steps_completed") != std::string::npos);
    CHECK(f.error.find("-1.5") != std::string::npos);
    CHECK(f.steps_completed == 0);

    const manifest nr = info_of("numeric_reason");
    CHECK(nr.malformed());
    CHECK(nr.error.find("exit_reason") != std::string::npos);
    CHECK(nr.exit_reason.empty()); // not coerced into a plausible-looking reason

    CHECK(info_of("array").malformed());
    CHECK(info_of("garbage").malformed());
    CHECK(info_of("garbage").error.find("parse error") != std::string::npos);

    // Absent fields are not an error.
    const manifest sp = info_of("sparse");
    CHECK_FALSE(sp.malformed());
    CHECK(sp.present);
    CHECK(sp.steps_completed == 0);

    const manifest sc = info_of("stringy_clock");
    CHECK(sc.malformed());
    CHECK(sc.error.find("wall_clock_seconds") != std::string::npos);
    CHECK(sc.wall_clock_seconds == 0.0);

    CHECK_FALSE(info_of("good").malformed());

    // All five bad manifests are reported, none of them drops its run, and the
    // summary prints them rather than counting them.
    const auto& is = c->issues();
    CHECK(is.size() == 5);
    for (const auto& i : is) {
        CHECK(i.file == "coast_runtime.json");
        CHECK_FALSE(i.run_dropped);
        CHECK_FALSE(i.message.empty());
    }
    const std::string sum = c->summary();
    CHECK(sum.find("floaty") != std::string::npos);
    CHECK(sum.find("n_steps_completed") != std::string::npos);
}

// THE FIXTURE THAT MATTERS: a manifest produced by coast's own writer, not by the
// string literal above it. The hand-built manifest_text() omits `gpu_adapter`, which is
// the one shape the strict reader accepted — so the whole suite stayed green while the
// reader rejected EVERY manifest a real CPU-only run writes, because write_run_manifest
// emits `gpu_adapter: null` there deliberately (run_manifest.cpp:178, pinned by
// test_run_manifest.cpp:94). Starting from the real producer is what catches that
// class; a DTO written by the test can only ever agree with the test.
TEST_CASE("campaign.manifest: a manifest from coast's own writer is well-formed")
{
    scratch s;
    build_factorial(s.root);

    const fs::path d = s.root / "run_0";
    sim::run_manifest_inputs_t in;
    in.output_path       = d / "coast_runtime.json";
    in.n_steps_requested = 600000;
    in.n_steps_completed = 600000;
    in.exit_reason       = sim::exit_reason_t::completed_n_steps;
    in.started_at_utc    = "2026-08-02T10:00:00Z";
    in.completed_at_utc  = "2026-08-02T10:06:00Z";
    in.gpu_adapter.clear(); // CPU-only -> the writer emits null
    loader::global_cfg_t cfg;
    REQUIRE(sim::write_run_manifest(in, cfg));

    // The premise, asserted rather than assumed: the file really does carry a null.
    {
        std::ifstream   is(d / "coast_runtime.json");
        nlohmann::json  j;
        is >> j;
        REQUIRE(j.at("gpu_adapter").is_null());
    }

    const auto c  = open(s.root);
    const auto r0 = c->where([](const run& r) { return r.id() == "run_0"; })[0];

    CHECK_FALSE(r0.info().malformed());
    CHECK(r0.info().error.empty());
    CHECK(r0.info().gpu_adapter.empty()); // null reads as "not applicable", not as text
    CHECK(r0.info().present);
    CHECK(r0.complete());
    CHECK(r0.info().steps_completed == 600000);
    // And nothing about this run is reported as a problem.
    for (const auto& i : c->issues()) CHECK(i.dir != d);
}

TEST_CASE("campaign.manifest: null is 'not applicable' on every field, not a wrong type")
{
    scratch s;
    build_factorial(s.root);
    write_file(s.root / "run_0" / "coast_runtime.json",
               R"({"exit_reason":"completed_n_steps","gpu_adapter":null,"hostname":null,)"
               R"("n_steps_completed":null,"wall_clock_seconds":null})");

    const auto c  = open(s.root);
    const auto r0 = c->where([](const run& r) { return r.id() == "run_0"; })[0];
    CHECK_FALSE(r0.info().malformed());
    CHECK(r0.info().steps_completed == 0);
    CHECK(r0.info().wall_clock_seconds == 0.0);
}

TEST_CASE("campaign.manifest: a malformed manifest is not evidence of completion")
{
    scratch s;
    build_factorial(s.root);
    // exit_reason parses fine; another field does not. The run must not count as
    // complete on the strength of a file the loader has already called damaged.
    write_file(s.root / "run_0" / "coast_runtime.json",
               R"({"exit_reason":"completed_n_steps","n_steps_completed":-1.5})");

    const auto c  = open(s.root);
    const auto r0 = c->where([](const run& r) { return r.id() == "run_0"; })[0];
    REQUIRE(r0.info().malformed());
    CHECK(r0.info().exit_reason == "completed_n_steps");
    CHECK_FALSE(r0.complete());
    CHECK(c->where([](const run& r) { return r.complete(); }).size() == 8);
}

// A wrong-typed field is rendered into the error message. Rendering it with a bare
// json::dump() made the reader inherit dump()'s recursion (parse and ~json in nlohmann
// 3.12 are iterative, dump is not) and its unbounded size. The manifest never passes
// through flatten(), so the config depth cap does not cover it: measured, a 180 kB
// manifest holding a 90 000-deep array segfaulted the process — an uncatchable death
// from one bad file, which is exactly what the cap exists to prevent.
TEST_CASE("campaign.manifest: a pathological wrong-typed value cannot kill or flood")
{
    scratch s;
    build_factorial(s.root);

    // Deep. 20 000 levels is well past what a recursive dump survives on an 8 MB stack
    // in a debug-instrumented build, and far below the file sizes involved.
    {
        std::string deep = "[";
        for (int i = 0; i < 20000; ++i) deep = "[" + deep;
        for (int i = 0; i < 20001; ++i) deep += "]";
        write_file(s.root / "run_0" / "coast_runtime.json",
                   R"({"hostname":)" + deep + R"(,"exit_reason":"completed_n_steps"})");
    }
    // Wide. A 2 MB value on a wrong-typed field would otherwise be copied into
    // manifest::error AND into the scan_issue, and then measured by ImGui's
    // CalcTextSize every frame.
    write_file(s.root / "run_1" / "coast_runtime.json",
               R"({"n_steps_completed":")" + std::string(2 * 1024 * 1024, 'x') +
                   R"(","exit_reason":"x"})");

    const auto c = open(s.root); // must return at all — the crash was here
    CHECK(c->size() == 9);

    const auto info_of = [&](const char* id) {
        return c->where([&](const run& r) { return r.id() == id; })[0].info();
    };
    const manifest deep = info_of("run_0");
    CHECK(deep.malformed());
    CHECK(deep.error.find("hostname") != std::string::npos);
    CHECK(deep.error.find("an array") != std::string::npos);
    CHECK(deep.error.size() < 200);
    // The size bound alone does NOT test the recursion guard — the length clip would
    // satisfy it while dump() had already walked (and overflowed) the whole tree. What
    // rules the recursion out is that a container is never rendered at all, so assert
    // exactly that: the type name is there and not one character of the contents.
    // (Mutation-checked: without this line, re-enabling the bare dump() went unnoticed.)
    CHECK(deep.error.find('[') == std::string::npos);

    const manifest wide = info_of("run_1");
    CHECK(wide.malformed());
    CHECK(wide.error.size() < 200);
    for (const auto& i : c->issues()) CHECK(i.message.size() < 200);
}

TEST_CASE("campaign.robustness: deeply nested JSON is refused, not fatal")
{
    scratch s;
    build_factorial(s.root);

    // 30 000 levels overflowed the stack and killed the process (measured); a SIGSEGV
    // is not catchable, so this defeated the whole "a bad file is reported" contract.
    // 5 000 is far below that but far above the cap, so it exercises the guard without
    // depending on the machine's stack size for the test to be meaningful.
    constexpr int deep = 5000;
    std::string   bomb(static_cast<std::size_t>(deep), '[');
    bomb += std::string(static_cast<std::size_t>(deep), ']');
    write_file(s.root / "bomb" / "config.json", "{\"x\": " + bomb + "}");

    const auto c = open(s.root); // must return rather than crash
    CHECK(c->size() == 9);       // the 9 good runs survive
    REQUIRE(c->issues().size() == 1);
    CHECK(c->issues()[0].dir.filename() == "bomb");
    CHECK(c->issues()[0].run_dropped);
    CHECK(c->issues()[0].message.find("nesting") != std::string::npos);
}

TEST_CASE("campaign.scan: a symlink cycle neither hangs nor duplicates runs")
{
    scratch s;
    write_file(s.root / "real" / "run1" / "config.json",
               config_text("0.01", "5.0", (s.root / "real" / "run1").string()));
    write_file(s.root / "real" / "run2" / "config.json",
               config_text("0.05", "5.0", (s.root / "real" / "run2").string()));

    // A link back to an ancestor. Without a visited-set the walk follows it (
    // directory_entry::is_directory resolves through status()) and rediscovers the
    // same run directories through every path that reaches them.
    std::error_code ec;
    fs::create_directory_symlink(s.root, s.root / "real" / "loop", ec);
    if (ec) return; // no symlink support here; nothing to assert

    const auto c = open(s.root);
    CHECK(c->size() == 2);
    // Ids are root-relative, so they are unique and stable.
    const auto ids = c->all().ids();
    CHECK(std::find(ids.begin(), ids.end(), "real/run1") != ids.end());
    CHECK(std::find(ids.begin(), ids.end(), "real/run2") != ids.end());
}

// The axes are discovered by DIFFING whatever the walk found, so a subtree the walk
// could not enter does not merely lose runs — it changes the answer. Measured before
// the fix: three runs a=1,2,3 with the third under an unreadable parent produced
// "1 run, no swept parameters (all configs identical)", a confident falsehood about a
// campaign with a 3-valued axis.
TEST_CASE("campaign.scan: an unenterable subtree is reported, never silently dropped")
{
    scratch s;
    for (int i = 0; i < 2; ++i) {
        const auto d = s.root / ("r" + std::to_string(i));
        write_file(d / "config.json", std::string(R"({"a":)") + std::to_string(i) + "}");
    }
    const auto hidden = s.root / "locked";
    write_file(hidden / "r2" / "config.json", R"({"a":9})");

    std::error_code ec;
    fs::permissions(hidden, fs::perms::none, ec);
    if (ec) return;
    // Running as root ignores the mode bits, and so would this test. Check, don't
    // assume: a silently-passing test here is worse than none.
    {
        std::error_code probe;
        fs::directory_iterator it(hidden, probe);
        if (!probe) { fs::permissions(hidden, fs::perms::owner_all, ec); return; }
    }

    const auto c = open(s.root);
    CHECK(c->size() == 2);
    REQUIRE(c->issues().size() == 1);
    const auto& i = c->issues()[0];
    CHECK(i.label == "locked");
    CHECK(i.file.empty());                                        // a walk issue, not a file
    CHECK_FALSE(i.run_dropped);                                   // how many is unknown
    CHECK(i.message.find("cannot be listed") != std::string::npos);
    CHECK(c->summary().find("locked") != std::string::npos);

    fs::permissions(hidden, fs::perms::owner_all, ec); // so scratch can clean up
}

TEST_CASE("campaign.scan: a symlink out of the root is refused and reported")
{
    scratch inside;
    scratch outside;
    write_file(inside.root / "r0" / "config.json", R"({"a":1})");
    write_file(inside.root / "r1" / "config.json", R"({"a":2})");
    // A foreign run, carrying a leaf the campaign has never heard of.
    write_file(outside.root / "foreign" / "config.json", R"({"a":3,"secret":"hunter2"})");

    std::error_code ec;
    fs::create_directory_symlink(outside.root, inside.root / "link_out", ec);
    if (ec) return;

    // The canonical/visited pair stops CYCLES, not ESCAPES: before the frame check this
    // pulled the foreign run in, merged `secret` into the axis namespace, and gave the
    // run an id of "../../<tmp>/foreign".
    const auto c = open(inside.root);
    CHECK(c->size() == 2);
    CHECK(c->find_axis("secret") == nullptr);
    for (const auto& id : c->all().ids()) CHECK(id.find("..") == std::string::npos);

    REQUIRE(c->issues().size() == 1);
    CHECK(c->issues()[0].message.find("outside the campaign root") != std::string::npos);
}

TEST_CASE("campaign.scan: the depth limit is reported, not applied in silence")
{
    scratch s;
    write_file(s.root / "shallow" / "config.json", R"({"a":1})");
    write_file(s.root / "one" / "two" / "three" / "config.json", R"({"a":2})");

    scan_options o;
    o.max_depth = 2; // reaches <root>/one/two, not the run below it

    const auto c = open(s.root, o);
    CHECK(c->size() == 1);
    REQUIRE(c->issues().size() == 1);
    CHECK(c->issues()[0].label == "one/two");
    CHECK(c->issues()[0].message.find("scan limit") != std::string::npos);
}

TEST_CASE("campaign.select: ids round-trip, collide-free across a nested layout")
{
    scratch s;
    // The same directory NAME under two parents — the layout the loader advertises
    // and the one a bare filename id cannot represent.
    write_file(s.root / "a" / "run1" / "config.json",
               config_text("0.01", "5.0", (s.root / "a" / "run1").string()));
    write_file(s.root / "b" / "run1" / "config.json",
               config_text("0.05", "5.0", (s.root / "b" / "run1").string()));

    const auto c = open(s.root);
    REQUIRE(c->size() == 2);

    const auto ids = c->all().ids();
    REQUIRE(ids.size() == 2);
    CHECK(ids[0] == "a/run1");
    CHECK(ids[1] == "b/run1");

    // Round-trip: ids() -> select() yields the same runs in the same order. This is
    // the GUI's exact hand-off to the console.
    const auto back = c->select(ids);
    REQUIRE(back.size() == 2);
    CHECK(back.ids() == ids);
    CHECK(back[0].number("primary_learning_rate") == doctest::Approx(0.01));
    CHECK(back[1].number("primary_learning_rate") == doctest::Approx(0.05));

    // Selecting both distinctly is what a bare-filename id could not do: it matched
    // the first "run1" twice and never reached the second.
    CHECK(c->select({"a/run1"}).size() == 1);
    CHECK(c->select({"a/run1"})[0].number("primary_learning_rate") == doctest::Approx(0.01));
    CHECK(c->select({"b/run1"})[0].number("primary_learning_rate") == doctest::Approx(0.05));

    // A repeated id selects the run once — duplicating it would double that run's
    // weight in any aggregate computed over the selection.
    CHECK(c->select({"a/run1", "a/run1"}).size() == 1);
    // An unknown id is dropped, not thrown, so a stale list degrades gracefully.
    CHECK(c->select({"a/run1", "does_not_exist"}).size() == 1);
    CHECK(c->select({}).empty());
}

TEST_CASE("campaign.lifetime: a run outliving its campaign reports, never corrupts")
{
    scratch s;
    build_factorial(s.root);

    // The most natural way to iterate in the REPL, and the one that used to be a
    // silent use-after-free: runs() hands out handles that own nothing.
    std::vector<run> orphans;
    run              one;
    {
        const auto c = open(s.root);
        orphans = c->runs();
        one     = c->all()[0];
        CHECK(orphans.size() == 9);
        CHECK_FALSE(one.id().empty()); // fine while the campaign is alive
    }
    // The campaign is gone. Every accessor must say so rather than read freed memory.
    CHECK_THROWS_AS((void)orphans[0].id(), std::runtime_error);
    CHECK_THROWS_AS((void)orphans[0].dir(), std::runtime_error);
    CHECK_THROWS_AS((void)orphans[0].info(), std::runtime_error);
    CHECK_THROWS_AS((void)one.number("primary_learning_rate"), std::runtime_error);
    CHECK_THROWS_AS((void)one.observe_groups(), std::runtime_error);

    std::string msg;
    try {
        (void)one.id();
    } catch (const std::exception& e) {
        msg = e.what();
    }
    CHECK(msg.find("outlived its campaign") != std::string::npos);

    // The guarantee is a TYPE, and only a type check can hold it. Handing back a
    // reference is not detectable at runtime — the mutated program reads freed memory,
    // which may or may not throw — so a CHECK cannot distinguish it. These do, at
    // compile time, and a revert to `const run&` fails the build here.
    static_assert(std::is_same_v<decltype(std::declval<const selection&>()[0]), run>,
                  "selection::operator[] must return by value: a reference into the "
                  "campaign's storage escapes run's weak_ptr guard");
    static_assert(std::is_same_v<decltype(std::declval<const selection&>().at(0)), run>,
                  "selection::at must return by value");
    static_assert(std::is_same_v<decltype(std::declval<const campaign&>().runs()),
                                 std::vector<run>>,
                  "campaign::runs must return by value");
    static_assert(std::is_same_v<decltype(*std::declval<const selection&>().begin()), run>,
                  "selection::iterator must dereference by value");

    // REFERENCE spellings, one character from the safe ones above. While every
    // accessor returned `const run&` into the campaign's own vector these bypassed the
    // guard entirely and read freed memory in silence — a weak_ptr cannot travel
    // through a reference. Returning by value makes `const auto&` bind to a
    // lifetime-extended temporary that still carries its guard — and makes the non-const
    // `auto&` spelling a compile error rather than a subtly wrong program.
    {
        run  by_ref_elem;
        auto keep = open(s.root);
        {
            const auto& r = keep->all()[0];   // was: a reference INTO keep->p_->runs
            const auto rs = keep->runs();
            by_ref_elem   = rs[0];
            CHECK_FALSE(r.id().empty());
        }
        keep.reset();
        CHECK_THROWS_AS((void)by_ref_elem.id(), std::runtime_error);
    }
    {
        auto  keep = open(s.root);
        const auto& rs = keep->runs();        // binds to a temporary vector
        CHECK(rs.size() == 9);
        CHECK_FALSE(rs[0].id().empty());
        keep.reset();
        CHECK_THROWS_AS((void)rs[0].id(), std::runtime_error);
    }

    // A selection, by contrast, OWNS its campaign: it stays usable after the local
    // handle is dropped, which is what makes `for (const run& r : sel)` always safe.
    selection sel;
    {
        const auto c = open(s.root);
        sel = c->where("primary_learning_rate", 0.01);
    }
    REQUIRE(sel.size() == 3);
    CHECK(sel[0].number("primary_learning_rate") == doctest::Approx(0.01));
    CHECK(sel.owner().size() == 9);
    for (const run& r : sel) CHECK_FALSE(r.id().empty());
}

TEST_CASE("campaign.query: a non-finite query value is reported, not matched")
{
    scratch s;
    build_factorial(s.root);
    const auto c = open(s.root);

    // inf made `scale` infinite, so `|a-b| <= rel_tol*scale` became inf <= inf — true
    // for every finite value, silently selecting the WHOLE campaign.
    const double inf = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS((void)c->where("primary_learning_rate", inf), std::runtime_error);
    CHECK_THROWS_AS((void)c->where("primary_learning_rate", -inf), std::runtime_error);
    CHECK_THROWS_AS((void)c->where("primary_learning_rate", std::nan("")),
                    std::runtime_error);
}

TEST_CASE("campaign.scan: nested roots, and leaves absent from some runs")
{
    scratch s;
    // Two campaigns under one parent, at different depths.
    build_factorial(s.root / "sweep_a");
    write_file(s.root / "deeper" / "sweep_b" / "r0" / "config.json",
               R"({"ngm25":{"neuron_groups":{"L4-mnist":{"primary_learning_rate":0.01}}},
                   "extra_knob": 7})");
    write_file(s.root / "deeper" / "sweep_b" / "r1" / "config.json",
               R"({"ngm25":{"neuron_groups":{"L4-mnist":{"primary_learning_rate":0.05}}},
                   "extra_knob": 9})");

    const auto c = open(s.root);
    CHECK(c->size() == 11);

    // A leaf present in only some runs is an axis whose value set includes <absent>,
    // rather than a scan failure.
    const axis* extra = c->find_axis("extra_knob");
    REQUIRE(extra != nullptr);
    REQUIRE(extra->values.size() == 3);
    CHECK(extra->values[0].k == value::kind::absent);
    CHECK(c->where("extra_knob", 7).size() == 1);

    int with_knob = 0;
    for (const run& r : c->all()) {
        if (r.has("extra_knob")) ++with_knob;
    }
    CHECK(with_knob == 2);

    // max_depth bounds the walk. sweep_a's runs sit at depth 2 and sweep_b's at
    // depth 3, so a cap of 2 finds the former only. (A run is recognised before the
    // cap is applied, which is why a flat campaign — runs directly under the root —
    // needs only 1.)
    scan_options shallow;
    shallow.max_depth = 2;
    CHECK(open(s.root, shallow)->size() == 9);
}

TEST_CASE("campaign.extract: sort_by / group_by / values line up for plotting")
{
    scratch s;
    build_factorial(s.root);
    const auto c = open(s.root);

    const auto sel = c->where("local_inhibition_strength", 5.0).sort_by("primary_learning_rate");
    REQUIRE(sel.size() == 3);

    const std::vector<double> x = sel.values("primary_learning_rate");
    REQUIRE(x.size() == sel.size());
    CHECK(std::is_sorted(x.begin(), x.end()));
    CHECK(x[0] == doctest::Approx(0.01));
    CHECK(x[2] == doctest::Approx(0.1));

    // The x-vector is element-for-element aligned with iteration order, which is what
    // makes "y from a metric, x from an axis" correct without bookkeeping.
    std::size_t i = 0;
    for (const run& r : sel) {
        CHECK(r.number("primary_learning_rate") == doctest::Approx(x[i]));
        ++i;
    }

    CHECK(sel.labels("primary_learning_rate")[0] == "0.01");
    CHECK(sel.dirs().size() == 3);

    // One group per distinct axis value, in the axis's own order.
    const auto groups = c->all().group_by("local_inhibition_strength");
    REQUIRE(groups.size() == 3);
    for (const auto& g : groups) CHECK(g.size() == 3);
    CHECK(groups[0][0].number("local_inhibition_strength") == doctest::Approx(1.0));
    CHECK(groups[2][0].number("local_inhibition_strength") == doctest::Approx(10.0));

    // EMPTY GROUPS ARE DROPPED. Every fixture in this suite is a full factorial, so
    // nothing exercised that; a narrowed selection does. The consequence a caller has
    // to know is that the result is then NOT index-aligned with the axis's value list —
    // reading a group's value by position would mislabel every curve after the gap.
    const auto narrowed = c->where("local_inhibition_strength", 1.0)
                              .where("primary_learning_rate", 0.01);
    REQUIRE(narrowed.size() == 1);
    const auto sparse = narrowed.group_by("local_inhibition_strength");
    REQUIRE(sparse.size() == 1); // not 3, and not two empties either side
    CHECK(sparse[0].size() == 1);
    CHECK(sparse[0][0].number("local_inhibition_strength") == doctest::Approx(1.0));
    REQUIRE(c->find_axis("local_inhibition_strength")->values.size() == 3); // the gap is real
}

TEST_CASE("campaign.open: a root with no runs is an error, not an empty campaign")
{
    scratch s;
    write_file(s.root / "notes.md", "nothing here");
    CHECK_THROWS_AS((void)open(s.root), std::runtime_error);
    CHECK_THROWS_AS((void)open(s.root / "does_not_exist"), std::runtime_error);
}

TEST_CASE("campaign.data: file access is lazy and reports missing outputs")
{
    scratch s;
    build_factorial(s.root);
    // Plant the file layout a real run has, without real HDF5 content: these cases
    // must be answered from the DIRECTORY, never by opening a file.
    write_file(s.root / "run_0" / "observe" / "population.observe.h5", "not really hdf5");
    write_file(s.root / "run_0" / "observe" / "single.observe.h5", "not really hdf5");
    write_file(s.root / "run_0" / "snapshots" / "cfg_20260731_211603_500000.h5", "x");
    write_file(s.root / "run_0" / "snapshots" / "cfg_20260731_220000_600000.h5", "x");

    const auto c  = open(s.root);
    const auto r0 = c->where([](const run& r) { return r.id() == "run_0"; })[0];

    const auto groups = r0.observe_groups();
    REQUIRE(groups.size() == 2);
    CHECK(groups[0] == "population");
    CHECK(groups[1] == "single");

    CHECK(r0.snapshots().size() == 2);
    CHECK(r0.snapshots()[0].filename().string() < r0.snapshots()[1].filename().string());

    // Ambiguous group: names the choices instead of guessing.
    std::string msg;
    try {
        (void)r0.observe();
    } catch (const std::exception& e) {
        msg = e.what();
    }
    REQUIRE(!msg.empty());
    CHECK(msg.find("population") != std::string::npos);
    CHECK(msg.find("single") != std::string::npos);

    // A run with no outputs at all reports that, without touching HDF5.
    const auto r1 = c->where([](const run& r) { return r.id() == "run_1"; })[0];
    CHECK(r1.observe_groups().empty());
    CHECK(r1.snapshots().empty());
    CHECK_THROWS_AS((void)r1.observe(), std::runtime_error);
    CHECK_THROWS_AS((void)r1.final_state(), std::runtime_error);
}

TEST_CASE("campaign.data: an unreadable output directory reports, it does not read empty")
{
    scratch s;
    build_factorial(s.root);
    const auto d = s.root / "run_0";
    write_file(d / "observe" / "population.observe.h5", "x");
    write_file(d / "snapshots" / "step_100.h5", "x");
    write_file(d / "snapshots" / "notes.txt", "x"); // must be filtered out by extension

    const auto c  = open(s.root);
    const auto r0 = c->where([](const run& r) { return r.id() == "run_0"; })[0];
    REQUIRE(r0.observe_groups().size() == 1);
    REQUIRE(r0.snapshots().size() == 1);
    CHECK(r0.snapshots()[0].filename() == "step_100.h5");

    // An ABSENT directory is genuinely empty; a run that has not written snapshots yet
    // is normal and must not throw.
    const auto r1 = c->where([](const run& r) { return r.id() == "run_1"; })[0];
    CHECK(r1.observe_groups().empty());
    CHECK(r1.snapshots().empty());

    // An UNREADABLE one is not empty — it is unknown, and saying "no observe files"
    // there is a false statement about the run. Returning {} for both made a loop over
    // snapshots() silently analyse zero checkpoints on an EIO/ESTALE directory.
    std::error_code ec;
    fs::permissions(d / "observe", fs::perms::none, ec);
    fs::permissions(d / "snapshots", fs::perms::none, ec);
    if (ec) return;
    {
        std::error_code probe;
        fs::directory_iterator it(d / "observe", probe);
        if (!probe) { // root, or a filesystem that ignores the mode bits
            fs::permissions(d / "observe", fs::perms::owner_all, ec);
            fs::permissions(d / "snapshots", fs::perms::owner_all, ec);
            return;
        }
    }

    CHECK_THROWS_AS((void)r0.observe_groups(), std::runtime_error);
    CHECK_THROWS_AS((void)r0.snapshots(), std::runtime_error);
    std::string why;
    try {
        (void)r0.observe_path();
    } catch (const std::exception& e) {
        why = e.what();
    }
    CHECK(why.find("cannot be listed") != std::string::npos);
    CHECK(why.find("no observe files") == std::string::npos); // the old, false wording

    fs::permissions(d / "observe", fs::perms::owner_all, ec);
    fs::permissions(d / "snapshots", fs::perms::owner_all, ec);
}

TEST_CASE("campaign.axes: text and boolean axes, and the const char*/bool trap")
{
    scratch s;
    // Two non-numeric axes. Nothing in the suite involved one before, so every
    // non-number path — discovery, ordering, label(), the string and bool filters —
    // was dark.
    const char* variants[] = {"max_softmax", "threshold_linear", "max_softmax"};
    const char* gpu[]      = {"true", "false", "false"};
    // `scale` is here so the integer-literal overload has a numeric axis to resolve
    // against; 2 in two runs, 3 in one.
    const char* scale[]    = {"2", "3", "2"};
    for (int i = 0; i < 3; ++i) {
        write_file(s.root / ("r" + std::to_string(i)) / "config.json",
                   std::string(R"({"inh":{"variant":")") + variants[i] +
                       R"(","on_gpu":)" + gpu[i] + R"(,"scale":)" + scale[i] + R"(}})");
    }

    const auto c = open(s.root);
    REQUIRE(c->axes().size() == 3);

    const axis* v = c->find_axis("variant");
    REQUIRE(v != nullptr);
    REQUIRE(v->values.size() == 2);
    CHECK(v->values[0].k == value::kind::text);
    CHECK(v->values[0].str == "max_softmax"); // lexicographic
    CHECK(v->values[1].str == "threshold_linear");
    CHECK(v->values[0].label() == "max_softmax");

    const axis* g = c->find_axis("on_gpu");
    REQUIRE(g != nullptr);
    REQUIRE(g->values.size() == 2);
    CHECK(g->values[0].k == value::kind::boolean);
    CHECK(g->values[0].flag == false); // false sorts before true
    CHECK(g->values[0].label() == "false");
    CHECK(g->values[1].label() == "true");

    // THE TRAP: a string literal is const char*, and const char* -> bool is a standard
    // conversion that outranks -> string_view. Without the exact-match overload this
    // silently filters on a BOOLEAN and would throw "no boolean value true" here, or
    // worse, quietly match nothing. This project has shipped exactly that bug before.
    // BOTH overload sets need covering: campaign::where and selection::where each
    // declare their own const char* exact match, and a mutation to either is invisible
    // to a test that only exercises the other.
    CHECK(c->where("variant", "max_softmax").size() == 2);            // campaign::where
    CHECK(c->all().where("variant", "max_softmax").size() == 2);      // selection::where
    CHECK(c->where("variant", "threshold_linear").size() == 1);
    CHECK(c->all().where("variant", "threshold_linear").size() == 1);
    CHECK(c->where("variant", std::string_view("max_softmax")).size() == 2);

    // Boolean filtering, and its precheck.
    CHECK(c->where("on_gpu", true).size() == 1);
    CHECK(c->where("on_gpu", false).size() == 2);
    CHECK(c->runs()[0].text("variant") == "max_softmax");

    // Wrong-kind and unmatched queries report rather than returning nothing.
    CHECK_THROWS_AS((void)c->where("variant", "no_such_variant"), std::runtime_error);
    CHECK_THROWS_AS((void)c->where("variant", 1.0), std::runtime_error);
    CHECK_THROWS_AS((void)c->runs()[0].number("variant"), std::runtime_error);
    CHECK_THROWS_AS((void)c->all().values("variant"), std::runtime_error);
    // ...but labels() renders any kind, which is what the GUI columns use.
    CHECK(c->all().labels("on_gpu").size() == 3);

    // The bool precheck, which the implementation calls the easiest silent-empty to
    // hit: an axis that HAS boolean values but not the one asked for must report, not
    // return an empty selection. Its numeric and string siblings are covered above.
    CHECK_THROWS_AS((void)c->where("variant", true), std::runtime_error);
    CHECK_THROWS_AS((void)c->all().where("variant", true), std::runtime_error);

    // An integer literal must route to the NUMERIC filter, not be ambiguous between
    // double and bool. The GUI emitter depends on this: std::format("{}", 5.0) prints
    // "5", so it emits `.where("axis", 5)`. This has to be COMPILED against a numeric
    // axis to mean anything — the previous version of this case asserted a find_axis
    // lookup underneath the same comment and exercised no overload resolution at all.
    CHECK(c->where("scale", 2).size() == 2);
    CHECK(c->all().where("scale", 2).size() == 2);
}

TEST_CASE("campaign.axes: discovery order follows config key order")
{
    scratch s;
    // "zeta" is written FIRST and "alpha" second, so an ordering that follows the file
    // differs from any alphabetical or hash order.
    for (int i = 0; i < 2; ++i) {
        write_file(s.root / ("r" + std::to_string(i)) / "config.json",
                   std::string(R"({"zeta":)") + (i ? "2" : "1") + R"(,"alpha":)" +
                       (i ? "20" : "10") + R"(})");
    }
    const auto c = open(s.root);
    REQUIRE(c->axes().size() == 2);
    CHECK(c->axes()[0].name == "zeta");
    CHECK(c->axes()[1].name == "alpha");
}

TEST_CASE("campaign.value: equality is exact and kind-aware")
{
    const auto num = [](double d) { value v; v.k = value::kind::number; v.num = d; return v; };
    const auto txt = [](const char* t) { value v; v.k = value::kind::text; v.str = t; return v; };
    const auto flg = [](bool b) { value v; v.k = value::kind::boolean; v.flag = b; return v; };
    const value absent;

    CHECK(num(0.01) == num(0.01));
    CHECK(num(0.01) != num(0.0100000001));
    CHECK(txt("a") == txt("a"));
    CHECK(txt("a") != txt("b"));
    CHECK(flg(true) == flg(true));
    CHECK(flg(true) != flg(false));
    CHECK(absent == value{});

    // Cross-kind is never equal, even where the renderings would agree. The GUI's
    // filter predicate relies on this: a mutation to compare only the kind ships a
    // panel where checking one value of an axis shows all of them.
    CHECK(num(0.0) != flg(false));
    CHECK(num(1.0) != flg(true));
    CHECK(flg(true) != txt("true"));
    CHECK(absent != txt(""));
    CHECK(absent != num(0.0));
}

TEST_CASE("campaign.axes: a single-run campaign has no axes")
{
    scratch s;
    write_file(s.root / "only" / "config.json", config_text("0.01", "5.0", (s.root / "only").string()));

    const auto c = open(s.root);
    CHECK(c->size() == 1);
    CHECK(c->axes().empty());
    CHECK(c->all().size() == 1);
    CHECK(c->summary().find("no swept parameters") != std::string::npos);

    // With no axes at all, an axis query must still say something useful.
    std::string msg;
    try {
        (void)c->where("anything", 1.0);
    } catch (const std::exception& e) {
        msg = e.what();
    }
    CHECK(msg.find("all configs are identical") != std::string::npos);
}

TEST_CASE("campaign.data: observe_path validates the group name")
{
    scratch s;
    build_factorial(s.root);
    write_file(s.root / "run_0" / "observe" / "population.observe.h5", "x");

    const auto c  = open(s.root);
    const auto r0 = c->where([](const run& r) { return r.id() == "run_0"; })[0];

    CHECK(r0.observe_path().filename() == "population.observe.h5");
    CHECK(r0.observe_path("population").filename() == "population.observe.h5");

    // A group name is a path COMPONENT. Without validation these escape the run
    // directory through operator/ — an absolute path replaces everything to its left.
    CHECK_THROWS_AS((void)r0.observe_path("/tmp/elsewhere"), std::runtime_error);
    CHECK_THROWS_AS((void)r0.observe_path("../../etc/passwd"), std::runtime_error);
    CHECK_THROWS_AS((void)r0.observe_path(".."), std::runtime_error);
    // An unknown but well-formed group is a plain not-found.
    CHECK_THROWS_AS((void)r0.observe_path("nope"), std::runtime_error);

    // THE CASE THAT MAKES THE ABOVE MEAN ANYTHING. A mutation check found that with
    // the validation deleted, every escape above still threw — from the exists() test
    // at the bottom, because none of those paths happen to exist. So the assertions
    // could not tell "rejected as a non-plain name" from "not found", and passed
    // against a stub with no validation at all. Here the traversal target DOES exist:
    // only the name check can reject it, and the message says which check fired.
    write_file(s.root / "run_1" / "observe" / "population.observe.h5", "x");
    std::string why;
    try {
        (void)r0.observe_path("../run_1/observe/population");
    } catch (const std::exception& e) {
        why = e.what();
    }
    CHECK(why.find("not a plain name") != std::string::npos);
}

TEST_CASE("campaign.handles: a default-constructed selection reports, never crashes")
{
    selection s;
    CHECK(s.size() == 0);
    CHECK(s.empty());
    CHECK(s.ids().empty());
    // These would have dereferenced a null campaign. `camp::selection s;` is a natural
    // thing to type in the REPL, where a segfault has no diagnostic at all.
    CHECK_THROWS_AS((void)s.owner(), std::runtime_error);
    CHECK_THROWS_AS((void)s.where("x", 1.0), std::runtime_error);
    CHECK_THROWS_AS((void)s.sort_by("x"), std::runtime_error);
    CHECK_THROWS_AS((void)s.values("x"), std::runtime_error);
    CHECK_THROWS_AS((void)s.at(0), std::runtime_error);

    run r;
    CHECK_THROWS_AS((void)r.id(), std::runtime_error);
}

TEST_CASE("campaign.handles: at() is range-checked on a BOUND selection")
{
    scratch sc;
    build_factorial(sc.root);
    const auto c = open(sc.root);

    // The unbound case above reaches its throw from the null-campaign guard and never
    // gets as far as the range check, so it leaves that check untested — deleting the
    // range check kept the whole suite green. A bound selection is the only shape that
    // exercises it, and at() exists precisely so a REPL typo reports instead of
    // reading past the end of the index vector.
    CHECK(c->all().at(0).id() == c->all()[0].id());
    CHECK_THROWS_AS((void)c->all().at(9), std::runtime_error);
    CHECK_THROWS_AS((void)c->all().at(9999), std::runtime_error);
    CHECK_THROWS_AS((void)c->where("primary_learning_rate", 0.01).at(3), std::runtime_error);
}

// --------------------------------------------------------------------------------
// The real campaign. Skipped when the NAS is not mounted, so this stays green on any
// machine; when it IS mounted this is the case that proves the design against the
// data it was built for.
TEST_CASE("campaign.real: the LRI_sweep campaign on the NAS")
{
    const fs::path root = "/mnt/NAS/coast-experiments/basic/ngm25_gpu/LRI_sweep";
    // CI sets COAST_NO_NAS: a test on the runner must not depend on the NAS.
    if (std::getenv("COAST_NO_NAS") != nullptr) {
        MESSAGE("skipped: COAST_NO_NAS is set");
        return;
    }
    std::error_code ec;
    if (!fs::is_directory(root, ec) || ec) {
        MESSAGE("skipped: " << root.string() << " is not mounted");
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto c  = open(root);
    const auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    CHECK(c->size() == 81);
    CHECK(c->issues().empty());

    // The four swept knobs, and nothing else — /snapshot_dir is denied.
    REQUIRE(c->axes().size() == 4);
    for (const auto& a : c->axes()) {
        CHECK(a.values.size() == 3);
        CHECK(a.pointer.starts_with("/ngm25/neuron_groups/L4-mnist/"));
    }

    // A full factorial: each single constraint keeps a third, two keep a ninth.
    CHECK(c->where("primary_learning_rate", 0.01).size() == 27);
    CHECK(c->where("primary_learning_rate", 0.01).where("local_inhibition_strength", 5.0).size() == 9);

    CHECK(c->runs()[0].complete());

    // Reported, not checked: the scan time is the NAS's latency, not the code's
    // (8.7-9.9 s measured on a loaded NAS, under 5 s on an idle one). Strict
    // laziness — no observe or snapshot file is opened — is checked without the
    // NAS, by "campaign.data: file access is lazy and reports missing outputs".
    MESSAGE("scan of " << c->size() << " runs took " << dt << " s");
}
