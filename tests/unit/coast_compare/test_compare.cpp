#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "compare.h"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace coast::compare;

namespace {

struct scratch_dir {
    fs::path path;
    explicit scratch_dir(std::string_view label) {
        path = fs::temp_directory_path() /
               ("coast_compare_" + std::string(label) + "_" +
                std::to_string(::getpid()) + "_" +
                std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~scratch_dir() { std::error_code ec; fs::remove_all(path, ec); }
};

void write_json(const fs::path& p, const json& j) {
    if (auto parent = p.parent_path(); !parent.empty()) {
        fs::create_directories(parent);
    }
    std::ofstream out(p);
    out << j.dump(2);
}

template <typename T>
void write_bin(const fs::path& p, const std::vector<T>& v) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(v.data()),
              static_cast<std::streamsize>(v.size() * sizeof(T)));
}

} // namespace

// ----- tolerance_set --------------------------------------------------------

TEST_CASE("tolerance_set: defaults and overrides") {
    tolerance_set ts;
    auto def = ts.get("anything");
    CHECK(def.rel       == doctest::Approx(1e-6));
    CHECK(def.abs_floor == doctest::Approx(1e-9));

    ts.set_override("g/x", {1e-4, 1e-7});
    CHECK(ts.get("g/x").rel       == doctest::Approx(1e-4));
    CHECK(ts.get("g/x").abs_floor == doctest::Approx(1e-7));
    CHECK(ts.get("g/y").rel       == doctest::Approx(1e-6));
}

TEST_CASE("tolerance_set: array-index suffix strips before lookup") {
    // Walker strips [N] before calling tols.get(). Verify the lookup itself
    // is exact (no built-in stripping). This is a contract check on the
    // documented behaviour.
    tolerance_set ts;
    ts.set_override("g/arr", {1e-3, 1e-9});
    CHECK(ts.get("g/arr").rel == doctest::Approx(1e-3));
    CHECK(ts.get("g/arr[0]").rel == doctest::Approx(1e-6));  // exact: no match
}

// ----- compare_against_goldens, structural cases ---------------------------

TEST_CASE("compare: no goldens directory → structural error") {
    scratch_dir sd{"no_dir"};
    write_json(sd.path / "cand.json", json{{"x", 1.0}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "missing_goldens", {});
    CHECK(outcome.kind == compare_outcome::result::structural_error);
    CHECK(exit_code_of(outcome) == 2);
    CHECK(outcome.goldens_considered.empty());
}

TEST_CASE("compare: empty goldens directory → structural error") {
    scratch_dir sd{"empty_dir"};
    fs::create_directories(sd.path / "goldens");
    write_json(sd.path / "cand.json", json{{"x", 1.0}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::structural_error);
    CHECK(exit_code_of(outcome) == 2);
}

// ----- compare_against_goldens, scalar match -------------------------------

TEST_CASE("compare: exact scalar match → matched, exit 0") {
    scratch_dir sd{"scalar_match"};
    write_json(sd.path / "cand.json",       json{{"x", 1.0}, {"y", 2}});
    write_json(sd.path / "goldens/20260512_aaa.json",
               json{{"x", 1.0}, {"y", 2}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::matched);
    CHECK(exit_code_of(outcome) == 0);
    CHECK(outcome.matched_golden->all_pass());
}

TEST_CASE("compare: scalar within tolerance → matched") {
    scratch_dir sd{"scalar_within_tol"};
    write_json(sd.path / "cand.json", json{{"x", 1.0 + 1e-8}});
    write_json(sd.path / "goldens/20260512_aaa.json", json{{"x", 1.0}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::matched);
}

TEST_CASE("compare: scalar outside tolerance → no_match, exit 1") {
    scratch_dir sd{"scalar_outside_tol"};
    write_json(sd.path / "cand.json", json{{"x", 1.0}});
    write_json(sd.path / "goldens/20260512_aaa.json", json{{"x", 2.0}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::no_match);
    CHECK(exit_code_of(outcome) == 1);
    REQUIRE(outcome.closest_golden.has_value());
    CHECK(outcome.closest_golden->failing_count() == 1);
    CHECK(outcome.closest_golden->fields[0].status == field_status::fail_tol);
    CHECK(outcome.closest_golden->fields[0].rel_delta == doctest::Approx(0.5));
}

// ----- multi-version matching ----------------------------------------------

TEST_CASE("compare: matches second-of-three goldens chronologically") {
    scratch_dir sd{"multi"};
    write_json(sd.path / "cand.json", json{{"x", 5.0}});
    // β-naming = chronological string-sort
    write_json(sd.path / "goldens/20260510_aaa.json", json{{"x", 1.0}});
    write_json(sd.path / "goldens/20260511_bbb.json", json{{"x", 5.0}});  // match
    write_json(sd.path / "goldens/20260512_ccc.json", json{{"x", 9.0}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::matched);
    CHECK(outcome.matched_golden->golden_path.filename().string()
          == "20260511_bbb.json");
    CHECK(outcome.goldens_considered.size() == 3);
}

TEST_CASE("compare: no match → closest by failing_count then rel_delta") {
    scratch_dir sd{"closest"};
    write_json(sd.path / "cand.json", json{{"x", 1.10}, {"y", 1.05}});
    // golden A: both fields off (2 failures)
    write_json(sd.path / "goldens/20260510_aaa.json", json{{"x", 9.0},  {"y", 9.0}});
    // golden B: only x off, large delta (1 failure)
    write_json(sd.path / "goldens/20260511_bbb.json", json{{"x", 0.5},  {"y", 1.05}});
    // golden C: only x off, smaller delta (1 failure, smaller rel_delta)
    write_json(sd.path / "goldens/20260512_ccc.json", json{{"x", 1.05}, {"y", 1.05}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::no_match);
    REQUIRE(outcome.closest_golden.has_value());
    CHECK(outcome.closest_golden->golden_path.filename().string()
          == "20260512_ccc.json");
}

// ----- nested paths & inline arrays ----------------------------------------

TEST_CASE("compare: nested object matches") {
    scratch_dir sd{"nested"};
    write_json(sd.path / "cand.json",
               json{{"g0", {{"act", 0.5}, {"spikes", 17}}}});
    write_json(sd.path / "goldens/aaa.json",
               json{{"g0", {{"act", 0.5}, {"spikes", 17}}}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::matched);
}

TEST_CASE("compare: missing field → structural error") {
    scratch_dir sd{"missing"};
    write_json(sd.path / "cand.json",        json{{"x", 1.0}});
    write_json(sd.path / "goldens/aaa.json", json{{"x", 1.0}, {"y", 2.0}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::structural_error);
    REQUIRE(outcome.closest_golden.has_value());
    CHECK(outcome.closest_golden->any_structural());
}

TEST_CASE("compare: extra field in candidate → structural error") {
    scratch_dir sd{"extra"};
    write_json(sd.path / "cand.json",        json{{"x", 1.0}, {"z", 3.0}});
    write_json(sd.path / "goldens/aaa.json", json{{"x", 1.0}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::structural_error);
}

TEST_CASE("compare: array length mismatch → structural") {
    scratch_dir sd{"alen"};
    write_json(sd.path / "cand.json",        json{{"a", {1.0, 2.0, 3.0}}});
    write_json(sd.path / "goldens/aaa.json", json{{"a", {1.0, 2.0}}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::structural_error);
}

TEST_CASE("compare: array strict: any element fail → field fail") {
    scratch_dir sd{"strict"};
    write_json(sd.path / "cand.json",
               json{{"a", {1.0, 2.0, 99.0, 4.0}}});           // [2] differs
    write_json(sd.path / "goldens/aaa.json",
               json{{"a", {1.0, 2.0, 3.0, 4.0}}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::no_match);
    REQUIRE(outcome.closest_golden.has_value());
    // Exactly one element should be flagged as fail_tol.
    int tol_fails = 0;
    for (const auto& f : outcome.closest_golden->fields) {
        if (f.status == field_status::fail_tol) ++tol_fails;
    }
    CHECK(tol_fails == 1);
}

// ----- NaN handling --------------------------------------------------------

TEST_CASE("compare: NaN in candidate scalar → structural error") {
    scratch_dir sd{"nan_cand"};
    json cand;  cand["x"] = std::numeric_limits<double>::quiet_NaN();
    write_json(sd.path / "cand.json", cand);
    write_json(sd.path / "goldens/aaa.json", json{{"x", 1.0}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::structural_error);
    REQUIRE(outcome.closest_golden.has_value());
    CHECK(outcome.closest_golden->any_structural());
}

TEST_CASE("compare: NaN in golden scalar → structural error") {
    scratch_dir sd{"nan_golden"};
    json golden;  golden["x"] = std::numeric_limits<double>::quiet_NaN();
    write_json(sd.path / "cand.json",        json{{"x", 1.0}});
    write_json(sd.path / "goldens/aaa.json", golden);
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::structural_error);
}

// ----- binary sidecar ------------------------------------------------------

TEST_CASE("compare: binary sidecar f64 match") {
    scratch_dir sd{"bin_match"};
    fs::create_directories(sd.path / "goldens");
    std::vector<double> v{1.0, 2.0, 3.0};
    write_bin(sd.path / "cand_w.bin", v);
    write_bin(sd.path / "goldens" / "gold_w.bin", v);
    write_json(sd.path / "cand.json",
        json{{"w", {{"binary", "cand_w.bin"}, {"dtype", "f64"}, {"shape", {3}}}}});
    write_json(sd.path / "goldens/aaa.json",
        json{{"w", {{"binary", "gold_w.bin"}, {"dtype", "f64"}, {"shape", {3}}}}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::matched);
}

TEST_CASE("compare: binary sidecar f64 differ → no_match") {
    scratch_dir sd{"bin_diff"};
    fs::create_directories(sd.path / "goldens");
    write_bin<double>(sd.path / "cand_w.bin", {1.0, 2.0, 3.0});
    write_bin<double>(sd.path / "goldens" / "gold_w.bin", {1.0, 2.0, 9.0});
    write_json(sd.path / "cand.json",
        json{{"w", {{"binary", "cand_w.bin"}, {"dtype", "f64"}, {"shape", {3}}}}});
    write_json(sd.path / "goldens/aaa.json",
        json{{"w", {{"binary", "gold_w.bin"}, {"dtype", "f64"}, {"shape", {3}}}}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::no_match);
}

TEST_CASE("compare: binary sidecar dtype mismatch → structural") {
    scratch_dir sd{"bin_dtype"};
    fs::create_directories(sd.path / "goldens");
    write_bin<double>(sd.path / "cand_w.bin", {1.0});
    write_bin<std::int64_t>(sd.path / "goldens" / "gold_w.bin", {1});
    write_json(sd.path / "cand.json",
        json{{"w", {{"binary", "cand_w.bin"}, {"dtype", "f64"}, {"shape", {1}}}}});
    write_json(sd.path / "goldens/aaa.json",
        json{{"w", {{"binary", "gold_w.bin"}, {"dtype", "i64"}, {"shape", {1}}}}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::structural_error);
}

TEST_CASE("compare: binary sidecar missing file → structural") {
    scratch_dir sd{"bin_miss"};
    fs::create_directories(sd.path / "goldens");
    write_bin<double>(sd.path / "cand_w.bin", {1.0});
    // gold_w.bin intentionally NOT written
    write_json(sd.path / "cand.json",
        json{{"w", {{"binary", "cand_w.bin"}, {"dtype", "f64"}, {"shape", {1}}}}});
    write_json(sd.path / "goldens/aaa.json",
        json{{"w", {{"binary", "gold_w.bin"}, {"dtype", "f64"}, {"shape", {1}}}}});
    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {});
    CHECK(outcome.kind == compare_outcome::result::structural_error);
}

// ----- per-field tolerance overrides ---------------------------------------

TEST_CASE("compare: per-field override loosens tolerance") {
    scratch_dir sd{"override"};
    write_json(sd.path / "cand.json", json{{"x", 1.1}});
    write_json(sd.path / "goldens/aaa.json", json{{"x", 1.0}});

    tolerance_set tols;
    // Default rel=1e-6 would fail (Δ ≈ 10%); override permits 20%.
    tols.set_override("x", {0.2, 1e-9});

    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", tols);
    CHECK(outcome.kind == compare_outcome::result::matched);
}

TEST_CASE("compare: array element inherits parent override via index strip") {
    scratch_dir sd{"override_arr"};
    write_json(sd.path / "cand.json",
               json{{"arr", {1.0, 2.0, 2.5}}});
    write_json(sd.path / "goldens/aaa.json",
               json{{"arr", {1.0, 2.0, 3.0}}});      // arr[2] off by 0.5

    tolerance_set tols;
    tols.set_override("arr", {0.3, 1e-9});           // 30% rel — covers it

    auto outcome = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", tols);
    CHECK(outcome.kind == compare_outcome::result::matched);
}

// ----- exit code mapping ---------------------------------------------------

TEST_CASE("exit_code_of: kind → code mapping") {
    compare_outcome m; m.kind = compare_outcome::result::matched;
    compare_outcome n; n.kind = compare_outcome::result::no_match;
    compare_outcome s; s.kind = compare_outcome::result::structural_error;
    CHECK(exit_code_of(m) == 0);
    CHECK(exit_code_of(n) == 1);
    CHECK(exit_code_of(s) == 2);
}

// ----- aggregate mode ------------------------------------------------------

TEST_CASE("aggregate: permuted array matches (mean/max/std identical); strict fails") {
    scratch_dir sd{"agg_perm"};
    write_json(sd.path / "cand.json",        json{{"a", {1.0, 2.0, 3.0, 4.0}}});
    write_json(sd.path / "goldens/aaa.json", json{{"a", {4.0, 3.0, 2.0, 1.0}}});

    // strict: element-wise → mismatch
    auto strict = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {}, compare_mode::strict);
    CHECK(strict.kind == compare_outcome::result::no_match);

    // aggregate: a permutation has identical (mean, max, std) → match
    auto agg = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {}, compare_mode::aggregate);
    CHECK(agg.kind == compare_outcome::result::matched);
}

TEST_CASE("aggregate: distribution shift is detected → no_match") {
    scratch_dir sd{"agg_shift"};
    write_json(sd.path / "cand.json",        json{{"a", {10.0, 10.0, 10.0}}});
    write_json(sd.path / "goldens/aaa.json", json{{"a", {1.0, 2.0, 3.0}}});
    auto agg = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {}, compare_mode::aggregate);
    CHECK(agg.kind == compare_outcome::result::no_match);
}

TEST_CASE("aggregate: binary sidecar is reduced to (mean,max,std)") {
    scratch_dir sd{"agg_bin"};
    fs::create_directories(sd.path / "goldens");
    std::vector<double> cand_v{1.0, 2.0, 3.0, 4.0};
    std::vector<double> gold_v{4.0, 3.0, 2.0, 1.0};   // permutation → same stats
    write_bin(sd.path / "cand_w.bin", cand_v);
    write_bin(sd.path / "goldens" / "gold_w.bin", gold_v);
    write_json(sd.path / "cand.json",
        json{{"w", {{"binary", "cand_w.bin"}, {"dtype", "f64"}, {"shape", {4}}}}});
    write_json(sd.path / "goldens/aaa.json",
        json{{"w", {{"binary", "gold_w.bin"}, {"dtype", "f64"}, {"shape", {4}}}}});
    auto agg = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {}, compare_mode::aggregate);
    CHECK(agg.kind == compare_outcome::result::matched);
}

TEST_CASE("aggregate: array length mismatch stays structural") {
    scratch_dir sd{"agg_len"};
    write_json(sd.path / "cand.json",        json{{"a", {1.0, 2.0, 3.0}}});
    write_json(sd.path / "goldens/aaa.json", json{{"a", {1.0, 2.0}}});
    auto agg = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {}, compare_mode::aggregate);
    CHECK(agg.kind == compare_outcome::result::structural_error);
}

TEST_CASE("aggregate: per-statistic std override loosens only std") {
    scratch_dir sd{"agg_perstat"};
    // Same mean (2) and max (4); std differs (sqrt2 ≈ 1.414 vs sqrt2.5 ≈ 1.581).
    write_json(sd.path / "cand.json",        json{{"a", {1.0, 4.0, 0.0, 3.0}}});
    write_json(sd.path / "goldens/aaa.json", json{{"a", {0.0, 4.0, 2.0, 2.0}}});

    // Tight default (1e-6) → the ~11% std delta fails.
    auto tight = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", {}, compare_mode::aggregate);
    CHECK(tight.kind == compare_outcome::result::no_match);

    // Loosen only std on path "a"; mean/max already match exactly.
    tolerance_set tols;
    agg_tolerance e;
    e.stddev = tolerance{0.3, 1e-9};
    tols.set_aggregate_override("a", e);
    auto loose = compare_against_goldens(
        sd.path / "cand.json", sd.path / "goldens", tols, compare_mode::aggregate);
    CHECK(loose.kind == compare_outcome::result::matched);
}

TEST_CASE("aggregate: tolerances.json aggregate block parses with full precedence") {
    scratch_dir sd{"agg_load"};
    json j;
    j["default"]              = {{"rel", 1e-6}, {"abs_floor", 1e-9}};
    j["aggregate"]["default"] = {{"rel", 1e-4}, {"abs_floor", 1e-9},
                                 {"std", {{"rel", 1e-2}}}};
    j["aggregate"]["overrides"]["buf/values"] =
        {{"rel", 1e-3}, {"max", {{"rel", 5e-2}}}};
    write_json(sd.path / "tol.json", j);

    auto ts = tolerance_set::load(sd.path / "tol.json");
    CHECK(ts.has_aggregate());
    // default base
    CHECK(ts.get_aggregate("other", agg_stat::mean).rel    == doctest::Approx(1e-4));
    // default per-stat std
    CHECK(ts.get_aggregate("other", agg_stat::stddev).rel  == doctest::Approx(1e-2));
    // override base applies to mean
    CHECK(ts.get_aggregate("buf/values", agg_stat::mean).rel == doctest::Approx(1e-3));
    // override per-stat max
    CHECK(ts.get_aggregate("buf/values", agg_stat::max).rel  == doctest::Approx(5e-2));
    // override std unset → override base (1e-3), NOT the default's std (1e-2)
    CHECK(ts.get_aggregate("buf/values", agg_stat::stddev).rel == doctest::Approx(1e-3));
    // scalar-field aggregate base lookup
    CHECK(ts.get_aggregate_base("buf/values").rel == doctest::Approx(1e-3));
    CHECK(ts.get_aggregate_base("other").rel      == doctest::Approx(1e-4));
    // strict path is untouched
    CHECK(ts.get("anything").rel == doctest::Approx(1e-6));
}

TEST_CASE("aggregate: no aggregate block → base falls back to strict default") {
    scratch_dir sd{"agg_fallback"};
    json j;
    j["default"] = {{"rel", 2e-3}, {"abs_floor", 1e-9}};
    write_json(sd.path / "tol.json", j);
    auto ts = tolerance_set::load(sd.path / "tol.json");
    CHECK_FALSE(ts.has_aggregate());
    CHECK(ts.get_aggregate("x", agg_stat::mean).rel == doctest::Approx(2e-3));
    CHECK(ts.get_aggregate_base("x").rel            == doctest::Approx(2e-3));
}
