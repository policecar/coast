// h5browse row_filter — all ops, AND combination, row range, edge cases.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/row_filter.h"
#include "h5browse/stats_kernel.h"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

using h5b::column_condition;
using h5b::condition_matches;
using h5b::filter_op;
using h5b::row_filter;
using h5b::row_matches;

namespace {
constexpr double qnan = std::numeric_limits<double>::quiet_NaN();
constexpr double inf  = std::numeric_limits<double>::infinity();
} // namespace

TEST_CASE("condition_matches: comparison ops")
{
    CHECK(condition_matches({0, filter_op::lt, 1.0}, 0.5));
    CHECK_FALSE(condition_matches({0, filter_op::lt, 1.0}, 1.0));
    CHECK(condition_matches({0, filter_op::le, 1.0}, 1.0));
    CHECK(condition_matches({0, filter_op::gt, 1.0}, 1.5));
    CHECK_FALSE(condition_matches({0, filter_op::gt, 1.0}, 1.0));
    CHECK(condition_matches({0, filter_op::ge, 1.0}, 1.0));
    CHECK(condition_matches({0, filter_op::eq, 2.0}, 2.0));
    CHECK_FALSE(condition_matches({0, filter_op::eq, 2.0}, 2.1));
    CHECK(condition_matches({0, filter_op::ne, 2.0}, 2.1));
}

TEST_CASE("condition_matches: IEEE semantics with NaN, explicit class tests")
{
    // Ordered comparisons with NaN are false; NaN != v is true.
    CHECK_FALSE(condition_matches({0, filter_op::lt, 1.0}, qnan));
    CHECK_FALSE(condition_matches({0, filter_op::ge, 1.0}, qnan));
    CHECK_FALSE(condition_matches({0, filter_op::eq, 1.0}, qnan));
    CHECK(condition_matches({0, filter_op::ne, 1.0}, qnan));

    CHECK(condition_matches({0, filter_op::is_nan, 0.0}, qnan));
    CHECK_FALSE(condition_matches({0, filter_op::is_nan, 0.0}, 1.0));
    CHECK(condition_matches({0, filter_op::is_inf, 0.0}, inf));
    CHECK(condition_matches({0, filter_op::is_inf, 0.0}, -inf));
    CHECK_FALSE(condition_matches({0, filter_op::is_inf, 0.0}, qnan));
    CHECK(condition_matches({0, filter_op::is_finite, 0.0}, 0.0));
    CHECK_FALSE(condition_matches({0, filter_op::is_finite, 0.0}, inf));
    CHECK_FALSE(condition_matches({0, filter_op::is_finite, 0.0}, qnan));
}

TEST_CASE("row_matches: AND combination and column bounds")
{
    const std::vector<double> row{1.0, 5.0, qnan};
    const std::span<const double> rs(row);

    row_filter f;
    f.conditions = {{0, filter_op::ge, 1.0}, {1, filter_op::lt, 10.0}};
    CHECK(row_matches(f, rs));

    f.conditions.push_back({1, filter_op::gt, 5.0}); // now fails
    CHECK_FALSE(row_matches(f, rs));

    // Condition on a column beyond the row width never matches.
    row_filter g;
    g.conditions = {{7, filter_op::is_finite, 0.0}};
    CHECK_FALSE(row_matches(g, rs));

    // Empty filter matches everything.
    CHECK(row_matches(row_filter{}, rs));
}

TEST_CASE("row_filter: row range is half-open and composes with conditions")
{
    row_filter f;
    f.row_begin = 10;
    f.row_end   = 20;
    CHECK_FALSE(f.row_in_range(9));
    CHECK(f.row_in_range(10));
    CHECK(f.row_in_range(19));
    CHECK_FALSE(f.row_in_range(20));

    row_filter open_end;
    open_end.row_begin = 5;
    CHECK(open_end.row_in_range(1'000'000'000));
    CHECK_FALSE(open_end.row_in_range(4));
}

TEST_CASE("collect_matching_rows: filter + range over a native-dtype block")
{
    // 10 rows x 3 cols of u16; keep rows where col1 > 40, rows [2, 8).
    std::vector<std::uint16_t> data(30);
    for (std::size_t r = 0; r < 10; ++r) {
        data[r * 3 + 0] = static_cast<std::uint16_t>(r);
        data[r * 3 + 1] = static_cast<std::uint16_t>(r * 10);
        data[r * 3 + 2] = 7;
    }
    row_filter f;
    f.row_begin  = 2;
    f.row_end    = 8;
    f.conditions = {{1, filter_op::gt, 40.0}};

    std::vector<std::uint64_t> out;
    h5b::collect_matching_rows<std::uint16_t>(
        out, std::span<const std::uint16_t>(data), 3, f, 0);
    CHECK(out == std::vector<std::uint64_t>{5, 6, 7});

    // Same block presented as a later batch: absolute indices shift.
    out.clear();
    row_filter g;
    g.conditions = {{1, filter_op::ge, 80.0}};
    h5b::collect_matching_rows<std::uint16_t>(
        out, std::span<const std::uint16_t>(data), 3, g, 100);
    CHECK(out == std::vector<std::uint64_t>{108, 109});
}

TEST_CASE("row_filter: describe provenance strings")
{
    CHECK(row_filter{}.describe() == "unfiltered");

    row_filter f;
    f.row_begin  = 100;
    f.row_end    = 200;
    f.conditions = {{3, filter_op::gt, 0.5}, {7, filter_op::is_finite, 0.0}};
    const std::string d = f.describe();
    CHECK(d.find("rows [100,200)") != std::string::npos);
    CHECK(d.find("col 3 > 0.5") != std::string::npos);
    CHECK(d.find("col 7 is_finite") != std::string::npos);
    CHECK(d.find(" & ") != std::string::npos);
}
