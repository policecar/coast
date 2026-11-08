#ifndef COAST_ROW_FILTER_H
#define COAST_ROW_FILTER_H

// Structured AND-only row filter (IDD D7). Doubles as an IR — a future
// text parser could compile to it — and serializes into log lines for
// provenance (describe()).
//
// Comparison semantics: the element is widened to double and compared
// against `value` with IEEE rules (any ordered comparison with NaN is
// false; NaN != v is true). Explicit NaN/Inf tests use the is_* ops.
// Integral values beyond 2^53 lose precision in the widening — acceptable
// for a triage filter, documented here.

#include "h5browse/scalar_type.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace h5b {

enum class filter_op : std::uint8_t {
    lt, le, gt, ge, eq, ne, is_nan, is_inf, is_finite
};

constexpr const char* filter_op_name(filter_op op)
{
    switch (op) {
        case filter_op::lt:        return "<";
        case filter_op::le:        return "<=";
        case filter_op::gt:        return ">";
        case filter_op::ge:        return ">=";
        case filter_op::eq:        return "==";
        case filter_op::ne:        return "!=";
        case filter_op::is_nan:    return "is_nan";
        case filter_op::is_inf:    return "is_inf";
        case filter_op::is_finite: return "is_finite";
    }
    std::unreachable();
}

struct column_condition {
    std::uint32_t column = 0;
    filter_op     op     = filter_op::is_finite;
    double        value  = 0.0; // ignored by the is_* ops
};

struct row_filter {
    // Half-open absolute row range [begin, end); nullopt = unrestricted.
    std::optional<std::uint64_t> row_begin;
    std::optional<std::uint64_t> row_end;
    std::vector<column_condition> conditions; // AND-combined

    [[nodiscard]] bool empty() const
    {
        return !row_begin && !row_end && conditions.empty();
    }

    [[nodiscard]] bool row_in_range(std::uint64_t row) const
    {
        if (row_begin && row < *row_begin) return false;
        if (row_end && row >= *row_end) return false;
        return true;
    }

    // Provenance line, e.g. "rows [100,200) & col 3 > 0.5 & col 7 is_finite".
    [[nodiscard]] std::string describe() const;
};

inline bool condition_matches(const column_condition& c, double x)
{
    switch (c.op) {
        case filter_op::lt:        return x < c.value;
        case filter_op::le:        return x <= c.value;
        case filter_op::gt:        return x > c.value;
        case filter_op::ge:        return x >= c.value;
        case filter_op::eq:        return x == c.value;
        case filter_op::ne:        return x != c.value;
        case filter_op::is_nan:    return std::isnan(x);
        case filter_op::is_inf:    return std::isinf(x);
        case filter_op::is_finite: return std::isfinite(x);
    }
    std::unreachable();
}

// Column conditions only; the caller applies row_in_range at iteration
// level where the absolute row index is known. Conditions referencing
// columns beyond the row width evaluate false (never matches — surfaced
// to the user by the zero-match result, not by UB).
template <class T>
bool row_matches(const row_filter& f, std::span<const T> row)
{
    for (const column_condition& c : f.conditions) {
        if (c.column >= row.size()) return false;
        if (!condition_matches(c, static_cast<double>(row[c.column]))) return false;
    }
    return true;
}

} // namespace h5b

#endif // COAST_ROW_FILTER_H
